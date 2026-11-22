"""Strict zero-edge temporal cold-source citation retrieval. No API calls."""
import argparse,json,random,time
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from .core import read,write,file_hash
from .citation import CitationGNN,adjacency,negative_targets

ROOT=Path(__file__).resolve().parents[1]
ARMS=['text_cosine','dtm_cosine','combined_cosine','ogbn_gnn','text_gnn','dtm_gnn','combined_gnn']


def temporal_split(years):
    years=np.asarray(years)
    return {'train':np.flatnonzero(years<=2017).tolist(),
            'validation':np.flatnonzero(years==2018).tolist(),
            'test':np.flatnonzero(years>=2019).tolist()}


def prepare(source,out):
    master=read(source/'data/arxiv_master_final.json');nodes=[int(p['node_idx']) for p in master]
    year_all=pd.read_csv(ROOT/'dataset/ogbn_arxiv/raw/node_year.csv.gz',header=None).iloc[:,0].to_numpy()
    years=year_all[nodes].astype(int).tolist();split=temporal_split(years)
    sample=torch.load(source/'data/ogbn_arxiv_16k_ffs_sample.pt',weights_only=False,map_location='cpu')
    if list(map(int,sample['indices']))!=nodes:raise ValueError('Sample alignment mismatch')
    labels=sample['labels'].reshape(-1).tolist()
    local={n:i for i,n in enumerate(nodes)}
    raw=torch.load(source/'data/paper_paper_edges.pt',weights_only=True,map_location='cpu').T.tolist()
    edges=sorted({(local[int(s)],local[int(t)]) for s,t in raw if s!=t})
    warm=set(split['train']);warm_local={g:i for i,g in enumerate(split['train'])}
    train_edges=[(warm_local[s],warm_local[t]) for s,t in edges if s in warm and t in warm]
    truths={}
    for name in ('validation','test'):
        cold=set(split[name]);truth=defaultdict(set)
        for s,t in edges:
            if s in cold and t in warm:truth[s].add(warm_local[t])
        truths[name]={g:sorted(v) for g,v in sorted(truth.items())}
        if not truths[name]:raise ValueError('No eligible cold sources')
    vectors={'ogbn':sample['features'].numpy()}
    for name in ('text','dtm','combined'):
        meta=read(source/'embeddings'/(name+'_manifest.json'))
        if meta['paper_ids']!=[str(p['paper_id']) for p in master]:raise ValueError('Embedding order mismatch')
        with np.load(source/'embeddings'/(name+'.npz'),allow_pickle=False) as f:vectors[name]=f['vectors']
    info={'schema':1,'task':'new paper -> pre-2018 papers, held-out references as truth',
        'protocol':'strict zero-edge cold-source; training graph, feature matrix and positive/negative losses contain only pre-2018 nodes',
        'node_split':{k:len(v) for k,v in split.items()},'training_citation_edges':len(train_edges),
        'eligible_sources':{k:len(v) for k,v in truths.items()},
        'unscored_no_old_references':{k:len(split[k])-len(truths[k]) for k in truths},
        'held_out_references':{k:sum(map(len,v.values())) for k,v in truths.items()},
        'candidate_count':len(warm),'candidate_policy':'all training papers; 2018+ papers never appear as target candidates or training negatives',
        'split_years':{'train':'<=2017','validation':'2018','test':'>=2019'},
        'labels':'observed directed citations, not LLM 0/1/2 judgments','additional_api_calls':0,
        'fair_comparison':'text_gnn/dtm_gnn/combined_gnn share Qwen model, 2560 input dimensions and identical GNN architecture; ogbn_gnn is an additional 128-dimensional reference',
        'limitations':['Simulated unseen nodes from a historical snapshot, not newly collected 2026 publications',
            'Publication years available; citation observation timestamps are unavailable',
            'DTM and Qwen encoders are pretrained; unseen means unseen by the task-specific GNN',
            'Evaluates new papers finding old references, not new candidate papers being discovered by user queries',
            'Only citations into the old candidate corpus are scored; sources without such references are counted but not scored']}
    write(out/'data_audit.json',info)
    write(out/'split.json',{'node_indices':nodes,'years':years,'partition_global_rows':split,
        'train_edges_local':train_edges,'truth_global_source_to_train_local_target':truths})
    return master,nodes,years,labels,split,train_edges,truths,vectors


def top_order(score,stable_ids,k):
    """Exact deterministic top-k without sorting the entire corpus."""
    k=min(k,len(score));threshold=np.partition(score,len(score)-k)[len(score)-k]
    strict=np.flatnonzero(score>threshold);tied=np.flatnonzero(score==threshold)
    tied=tied[np.argsort(stable_ids[tied],kind='stable')][:k-len(strict)]
    picked=np.concatenate([strict,tied])
    return picked[np.lexsort((stable_ids[picked],-score[picked]))]


def ranked_metrics(ranking,truth,k):
    truth=set(truth);hits=[i for i,v in enumerate(ranking[:k]) if v in truth]
    ideal=sum(1/np.log2(i+2) for i in range(min(k,len(truth))))
    return {'recall':len(hits)/len(truth),
        'ndcg':sum(1/np.log2(i+2) for i in hits)/ideal,
        'mrr':1/(hits[0]+1) if hits else 0.}


@torch.no_grad()
def score_queries(source_vectors,target_vectors,truth,source_rows,candidate_global,nodes,labels,years,ks,details=True):
    keys=sorted(truth);source_index={g:i for i,g in enumerate(source_rows)}
    stable=np.array([nodes[i] for i in candidate_global]);rows=[]
    for start in range(0,len(keys),128):
        batch=keys[start:start+128]
        index=torch.tensor([source_index[g] for g in batch],device=source_vectors.device)
        scores=(source_vectors[index] @ target_vectors.T).float().cpu().numpy()
        for g,score in zip(batch,scores):
            ranking=top_order(score,stable,max(ks)).tolist()
            for k in ks:
                row={'source_global_row':g,'source_node_idx':nodes[g],'year':years[g],'category':labels[g],'k':k,
                    **ranked_metrics(ranking,truth[g],k)}
                if details:row.update(truth_node_indices=[nodes[candidate_global[t]] for t in truth[g]],
                    ranking_node_indices=[nodes[candidate_global[t]] for t in ranking[:k]])
                rows.append(row)
    return rows


def summary(rows):
    out={}
    for k in sorted({r['k'] for r in rows}):
        selected=[r for r in rows if r['k']==k];cats=sorted({r['category'] for r in selected})
        out[str(k)]={'sources':len(selected),'represented_categories':len(cats)}
        for m in ('recall','ndcg','mrr'):
            out[str(k)][m]=float(np.mean([r[m] for r in selected]))
            out[str(k)]['category_macro_'+m]=float(np.mean([np.mean([r[m] for r in selected if r['category']==c]) for c in cats]))
    return out


def train(feature,split,train_edges,truths,nodes,labels,years,args,seed,folder):
    random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    device=torch.device(args.device)
    # Cold features are not even present in the training tensors or graph.
    x=F.normalize(torch.from_numpy(feature[split['train']].copy()).to(device),dim=1)
    xv=F.normalize(torch.from_numpy(feature[split['validation']].copy()).to(device),dim=1)
    adj=adjacency(train_edges,len(x),device);isolated_v=adjacency([],len(xv),device)
    model=CitationGNN(x.shape[1],128,len(x),False).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=1e-4)
    positive=torch.tensor(train_edges,device=device).T;rng=random.Random(seed)
    best=-1.;best_epoch=0;state=None;history=[]
    for epoch in range(1,args.epochs+1):
        model.train();optimizer.zero_grad();source,target=model(x,adj)
        negative=torch.tensor(negative_targets(positive[0].tolist(),train_edges,len(x),rng),device=device)
        loss=F.softplus((source[positive[0]]*target[negative]).sum(1)-(source[positive[0]]*target[positive[1]]).sum(1)).mean()
        loss.backward();optimizer.step()
        if epoch==1 or epoch%5==0 or epoch==args.epochs:
            model.eval()
            with torch.no_grad():
                _,warm_target=model(x,adj);cold_source,_=model(xv,isolated_v)
                measured=score_queries(cold_source,warm_target,truths['validation'],split['validation'],split['train'],nodes,labels,years,[20],False)
            value=summary(measured)['20']['ndcg']
            history.append({'epoch':epoch,'loss':float(loss.item()),'validation_ndcg20':value})
            if value>best:
                best=value;best_epoch=epoch;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            if epoch==1 or epoch%25==0:print(f'{folder.name}: epoch={epoch} loss={loss.item():.4f} val_ndcg20={value:.4f}',flush=True)
    model.load_state_dict(state);model.eval()
    xt=F.normalize(torch.from_numpy(feature[split['test']].copy()).to(device),dim=1)
    with torch.no_grad():
        _,warm_target=model(x,adj);cold_source,_=model(xt,adjacency([],len(xt),device))
        rows=score_queries(cold_source,warm_target,truths['test'],split['test'],split['train'],nodes,labels,years,[10,20])
    write(folder/'per_paper.json',rows);write(folder/'history.json',history);torch.save(state,folder/'model.pt')
    result={'seed':seed,'best_epoch':best_epoch,'validation_ndcg20':best,'test':summary(rows)}
    write(folder/'summary.json',result)
    del model,optimizer,x,xv,xt,adj,isolated_v
    if torch.cuda.is_available():torch.cuda.empty_cache()
    return result,rows


def aggregate(out,seeds):
    results={};per={}
    for arm in ARMS:
        summaries=[];per[arm]=[]
        for seed in seeds:
            folder=out/f'seed_{seed}'/arm
            summaries.append(read(folder/'summary.json'));per[arm].append(read(folder/'per_paper.json'))
        results[arm]={'by_seed':summaries,'test':{}}
        for k in (10,20):
            results[arm]['test'][str(k)]={}
            for metric in ('recall','ndcg','mrr','category_macro_recall','category_macro_ndcg','category_macro_mrr'):
                v=[s['test'][str(k)][metric] for s in summaries]
                results[arm]['test'][str(k)][metric]={'mean':float(np.mean(v)),'seed_std':float(np.std(v,ddof=1)) if len(v)>1 else 0.}
    deltas={}
    for arm in ARMS:
        if arm=='text_gnn':continue
        for k in (10,20):
            for metric in ('recall','ndcg','mrr'):
                matrix=[]
                for si,seed in enumerate(seeds):
                    base={r['source_node_idx']:r[metric] for r in per['text_gnn'][si] if r['k']==k}
                    candidate={r['source_node_idx']:r[metric] for r in per[arm][si] if r['k']==k}
                    if base.keys()!=candidate.keys():raise ValueError('Unpaired cold sources')
                    matrix.append([candidate[q]-base[q] for q in sorted(base)])
                d=np.asarray(matrix);rng=np.random.default_rng(42);draw=[]
                for _ in range(2000):
                    ss=rng.integers(0,len(seeds),size=len(seeds));qq=rng.integers(0,d.shape[1],size=d.shape[1])
                    draw.append(float(d[ss][:,qq].mean()))
                deltas[f'{arm}/{metric}@{k}']={'delta':float(d.mean()),'ci95':np.quantile(draw,[.025,.975]).tolist(),
                    'resampling':'paired seeds and source papers','seeds':len(seeds),'sources':d.shape[1]}
    write(out/'summary.json',{'seeds':seeds,'arms':results,'paired_deltas_vs_text_gnn':deltas,'primary_metric':'NDCG@20'})
    print('FINAL NDCG@20',{arm:round(results[arm]['test']['20']['ndcg']['mean'],5) for arm in ARMS},flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'output/balanced_benchmark_20261117')
    parser.add_argument('--output',type=Path,default=ROOT/'output/cold_start_temporal_20261117')
    parser.add_argument('--seeds',nargs='+',type=int,default=[42,43,44]);parser.add_argument('--epochs',type=int,default=200)
    parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu');args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    master,nodes,years,labels,split,edges,truths,vectors=prepare(args.source,args.output)
    manifest={'source':str(args.source),'seeds':args.seeds,'epochs':args.epochs,'device':args.device,'hidden':128,'lr':.001,'eval_every':5,
        'primary_metric':'NDCG@20','model':'same shared-weight CitationGNN; isolated cold sources use self-loop only',
        'source_sha256':{n:file_hash(args.source/n) for n in ['data/arxiv_master_final.json','data/paper_paper_edges.pt','embeddings/text_manifest.json','embeddings/dtm_manifest.json','embeddings/combined_manifest.json']}}
    mp=args.output/'run_manifest.json'
    if mp.exists() and read(mp)!=manifest:raise ValueError('Run configuration changed; choose a new output folder')
    write(mp,manifest);print(json.dumps(read(args.output/'data_audit.json'),indent=2),flush=True)
    for seed in args.seeds:
        for arm in ARMS:
            folder=args.output/f'seed_{seed}'/arm;folder.mkdir(parents=True,exist_ok=True)
            if (folder/'summary.json').exists() and (folder/'per_paper.json').exists():
                print('Reusing '+str(folder),flush=True);continue
            name=arm.rsplit('_',1)[0];feature=vectors[name]
            if arm.endswith('_cosine'):
                x=F.normalize(torch.from_numpy(feature[split['train']].copy()).to(args.device),dim=1)
                xt=F.normalize(torch.from_numpy(feature[split['test']].copy()).to(args.device),dim=1)
                rows=score_queries(xt,x,truths['test'],split['test'],split['train'],nodes,labels,years,[10,20])
                write(folder/'per_paper.json',rows);write(folder/'summary.json',{'seed':seed,'test':summary(rows),'training':'none'})
                del x,xt
            else:train(feature,split,edges,truths,nodes,labels,years,args,seed,folder)
    aggregate(args.output,args.seeds)

if __name__=='__main__':main()
