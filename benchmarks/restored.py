"""Restore imported balanced corpus and run offline Qwen benchmarks.
No API calls; incoming labels and intent plans are reused unchanged.
"""
import argparse,gc,json,os,shutil,sys,time
from collections import Counter,defaultdict
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile,ZIP_STORED
import numpy as np
import pandas as pd
import torch
from .core import Graph,digest,file_hash,metrics,normalize_rows,prior_from_scores,read,write
from .citation import dtm_text,fingerprint,split_edges,train_arm,summarize as citation_summary

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'output/balanced_benchmark_20261117'
sys.path.insert(0,str(ROOT/'code/embeddings'))
sys.path.insert(0,str(ROOT/'code/retrieval'))
sys.path.insert(0,str(ROOT/'code/data_analysis'))
from local_embeddings import IDENTITY,LocalEmbedder
from run_nl_recommendation import build_embedding_query,intent_terms,rerank,load_catalog
from normalize_dtm_vocabulary import unique


def validate_judgments(queries,judgments,nodes):
    byq={q['query_id']:q for q in queries}
    if len(byq)!=len(queries) or len({j['query_id'] for j in judgments})!=len(judgments):
        raise ValueError('Duplicate query IDs')
    if set(byq)!={j['query_id'] for j in judgments}:
        raise ValueError('Missing query/judgment pairs')
    anchor_splits=defaultdict(set)
    for q in queries:
        if int(q['anchor_node_idx']) not in nodes or q['split'] not in ('train','validation','test'):
            raise ValueError('Invalid query source/split')
        anchor_splits[int(q['anchor_node_idx'])].add(q['split'])
    if any(len(v)>1 for v in anchor_splits.values()):
        raise ValueError('Anchor leaks across splits')
    for j in judgments:
        candidates=list(map(int,j['candidate_node_indices']))
        labels=j['labels']
        if len(set(candidates))!=len(candidates) or not set(candidates)<=nodes:
            raise ValueError('Duplicate/unknown candidate node')
        if len(labels)!=len(candidates) or {int(r['node_idx']) for r in labels}!=set(candidates):
            raise ValueError('Incomplete or duplicate judgments')
        if any(type(r['relevance']) is not int or r['relevance'] not in (0,1,2) for r in labels):
            raise ValueError('Invalid relevance')
        if j['query']!=byq[j['query_id']]['query']:
            raise ValueError('Query/judgment text mismatch')
    return dict(Counter(q['split'] for q in queries))


def restore(out):
    data=out/'data';data.mkdir(parents=True,exist_ok=True)
    if (out/'data_audit.json').exists():
        audit=read(out/'data_audit.json')
        for relative,expected in audit['source_sha256'].items():
            if file_hash(ROOT/relative)!=expected:raise ValueError('Imported source changed; choose a new output directory: '+relative)
        if not all((data/n).exists() for n in ('arxiv_master_final.json','ogbn_arxiv_16k_ffs_sample.pt','paper_paper_edges.pt','author_paper_edges.pt','build_hetero_graph_v2.pt','queries.json','judgments.json')):
            raise ValueError('Incomplete restored data; inspect before reusing the run directory')
        print('Reusing validated restored dataset',flush=True);return
    incoming=ROOT/'output/dtm_luna_16k'
    papers=read(incoming/'sample_manifest.json')['papers']
    nodes=[int(p['node_idx']) for p in papers];node_set=set(nodes)
    if len(nodes)!=16000 or len(node_set)!=len(nodes):raise ValueError('Expected unique 16k corpus')
    mappings=pd.read_csv(ROOT/'dataset/ogbn_arxiv/mapping/nodeidx2paperid.csv.gz')
    idmap=dict(zip(mappings.iloc[:,0],mappings.iloc[:,1].astype(str)))
    if any(str(p['paper_id'])!=idmap[int(p['node_idx'])] for p in papers):raise ValueError('OGB paper ID mismatch')
    dtm_rows=read(incoming/'gpt-5.6-luna.json');dtms={int(p['node_idx']):p for p in dtm_rows}
    authors={int(p['node_idx']):p for p in read(ROOT/'output/author_data_openalex.json')}
    if len(dtms)!=len(dtm_rows) or set(dtms)!=node_set or set(authors)!=node_set:raise ValueError('Incoming files refer to different corpus')
    master=[]
    for i,p in enumerate(papers):
        node=int(p['node_idx']);d=dtms[node];a=authors[node]
        if str(d['paper_id'])!=str(p['paper_id']) or str(a['paper_id'])!=str(p['paper_id']):raise ValueError('DTM/author ID mismatch')
        if d['title'].strip()!=p['title'].strip() or d['abstract'].strip()!=p['abstract'].strip():raise ValueError('DTM source text mismatch')
        raw=d.get('knowledge',{}) if d.get('status')=='success' else {}
        k={kind:list(dict.fromkeys(str(t).strip() for t in raw.get(kind,[]) if str(t).strip())) for kind in ('domain','task','method')}
        k['normalized']={kind:unique(k[kind],kind) for kind in ('domain','task','method')}
        master.append({**p,'node_idx':node,'local_idx':i,'authors':a['authors'],'knowledge':k,
            'dtm':{'model':d.get('model'),'status':d.get('status'),'error':d.get('error')}})
    counts={kind:Counter(t for p in master for t in p['knowledge']['normalized'][kind]) for kind in ('domain','task','method')}
    meta={kind+'s':{t:i+1 for i,t in enumerate(sorted(t for t,n in counts[kind].items() if n>=2))} for kind in counts}
    for p in master:
        p['knowledge']['graph']={kind:[t for t in p['knowledge']['normalized'][kind] if t in meta[kind+'s']] for kind in counts}
    all_labels=pd.read_csv(ROOT/'dataset/ogbn_arxiv/raw/node-label.csv.gz',header=None).iloc[:,0].to_numpy()
    labels=all_labels[nodes].astype(np.int64)
    from importlib.util import spec_from_file_location,module_from_spec
    spec=spec_from_file_location('balanced_sampler',ROOT/'code/data_collection/forest_fire.py');mod=module_from_spec(spec);spec.loader.exec_module(mod)
    expected=mod.balanced_quotas(all_labels.tolist(),len(nodes))
    actual=dict(sorted(Counter(labels.tolist()).items()))
    if actual!=expected:raise ValueError(f'Imported corpus does not satisfy balanced quotas: {actual}')
    print('Balanced quotas verified; reading OGB features',flush=True)
    features=pd.read_csv(ROOT/'dataset/ogbn_arxiv/raw/node-feat.csv.gz',header=None,dtype=np.float32).to_numpy()[nodes]
    sample={'indices':nodes,'features':torch.from_numpy(features.copy()),'labels':torch.from_numpy(labels[:,None]),
        'sampling':{'method':'imported_balanced_sample','category_counts':actual,'source':str(incoming/'sample_manifest.json')}}
    source_edges=pd.read_csv(ROOT/'dataset/ogbn_arxiv/raw/edge.csv.gz',header=None).to_numpy(dtype=np.int64)
    keep=np.isin(source_edges[:,0],nodes)&np.isin(source_edges[:,1],nodes)
    citations=source_edges[keep]
    amap=read(ROOT/'output/author_remapping.json')['id_to_idx']
    if len(set(amap.values()))!=len(amap):raise ValueError('Duplicate author remapping indices')
    ap=[]
    for p in master:
        for a in p['authors']:
            if a['author_id'] not in amap:raise ValueError('Missing author mapping')
            ap.append((int(amap[a['author_id']]),p['node_idx']))
    ap=sorted(set(ap))
    write(data/'arxiv_master_final.json',master);write(data/'knowledge_meta.json',meta)
    torch.save(sample,data/'ogbn_arxiv_16k_ffs_sample.pt')
    torch.save(torch.from_numpy(citations.T.copy()),data/'paper_paper_edges.pt')
    torch.save(torch.tensor(ap,dtype=torch.long).T,data/'author_paper_edges.pt')
    # Rebuild topic assignments in the same node-ID order as features.
    from torch_geometric.data import HeteroData
    local={n:i for i,n in enumerate(nodes)}
    g=HeteroData();g['paper'].x=sample['features'];g['paper'].y=sample['labels'];g['paper'].num_nodes=len(nodes)
    g['author'].num_nodes=max(amap.values())+1;g['topic'].num_nodes=40
    g['paper','cites','paper'].edge_index=torch.tensor([(local[int(s)],local[int(t)]) for s,t in citations],dtype=torch.long).T
    g['author','writes','paper'].edge_index=torch.tensor([(a,local[n]) for a,n in ap],dtype=torch.long).T
    g['paper','has_topic','topic'].edge_index=torch.stack([torch.arange(len(nodes)),torch.from_numpy(labels)])
    attrs=[]
    for p in master:
        row=[]
        for kind in ('domain','task','method'):
            terms=p['knowledge']['graph'][kind][:2];row.extend([meta[kind+'s'][t] for t in terms]+[0]*(2-len(terms)))
        attrs.append(row)
    g['paper','has_topic','topic'].edge_attr=torch.tensor(attrs)
    torch.save(g,data/'build_hetero_graph_v2.pt')
    source=ROOT/'output/nl_recommendation_data'
    queries=read(source/'queries.json');judgments=read(source/'judgments.json')
    splits=validate_judgments(queries,judgments,node_set)
    for name in ('queries.json','judgments.json','query_intents.gpt-5_6-luna.json'):
        shutil.copy2(source/name,data/name)
    old_sample=torch.load(ROOT/'subdataset/ogbn_arxiv_16k_ffs_sample.pt',map_location='cpu',weights_only=False)
    old_nodes={int(n) for n in old_sample['indices']}
    label_counts=Counter(v['relevance'] for j in judgments for v in j['labels'])
    info={'papers':len(nodes),'category_counts':actual,'balanced_quota_match':True,'new_nodes_vs_previous':len(node_set-old_nodes),
        'citation_edges':len(citations),'author_edges':len(ap),'dtm_status':dict(Counter(p['dtm']['status'] for p in master)),
        'queries':len(queries),'query_splits':splits,'labels':sum(label_counts.values()),'label_distribution':dict(label_counts),
        'label_model':dict(Counter(j.get('model') for j in judgments)),
        'evaluation_notes':['Imported labels retained unchanged. Original generator shows DTM to judge and forces anchor relevance=2.',
         'Main natural language evaluation reranks fixed fully judged candidate pools; not full corpus retrieval.',
         'Test sources fixed before ranking. Validation/train labels not used for tuning.',
         'Secondary anchor-excluded scores quantify dependence on the query-generating source.'],
        'source_sha256':{str(p.relative_to(ROOT)):file_hash(p) for p in [incoming/'sample_manifest.json',incoming/'gpt-5.6-luna.json',source/'queries.json',source/'judgments.json']}}
    write(out/'data_audit.json',info)
    print(json.dumps({k:v for k,v in info.items() if k not in ('source_sha256','category_counts')},ensure_ascii=False,indent=2),flush=True)


def encode(out):
    from .nl_labeled import encode_text_rows
    data=out/'data';master=read(data/'arxiv_master_final.json');emb=out/'embeddings';emb.mkdir(exist_ok=True)
    queries=read(data/'queries.json');test=[q for q in queries if q['split']=='test']
    plans=read(data/'query_intents.gpt-5_6-luna.json')
    if any(q['query_id'] not in plans['intents'] for q in test):raise ValueError('Missing imported test intent plan')
    encoder=LocalEmbedder();models={}
    for arm in ('dtm','text','combined'):
        if arm=='dtm':texts=[dtm_text(p) for p in master]
        else:
            texts=[f'Title: {p["title"]}\nAbstract: {p["abstract"][:1800]}' for p in master]
            if arm=='combined':texts=[t+'\nD/T/M: '+'; '.join(term for k in ('domain','task','method') for term in p['knowledge'][k]) for t,p in zip(texts,master)]
        signature=digest({'model':IDENTITY,'nodes':[p['node_idx'] for p in master],'texts':texts})
        target=emb/(arm+'.npz');mp=emb/(arm+'_manifest.json')
        if target.exists() and mp.exists():
            if read(mp)['signature']!=signature:raise ValueError('Embedding input mismatch')
            print('Reusing completed '+arm+' bundle',flush=True);continue
        folder=ROOT/'output/nl_eval_embedding_cache'/('text_rows' if arm=='text' else 'restored_'+arm+'_rows')
        print('Encoding '+arm+' '+str(len(texts))+' papers',flush=True)
        values,truncated=encode_text_rows(encoder,texts,folder,IDENTITY)
        if arm=='dtm':
            values[[not any(p['knowledge'][k] for k in ('domain','task','method')) for p in master]]=0
            dtm_folder=emb/'citation_dtm';dtm_folder.mkdir(exist_ok=True)
            torch.save({'vectors':torch.from_numpy(values),'signature':fingerprint({'nodes':[p['node_idx'] for p in master],'texts':texts,'model':IDENTITY}),'model':IDENTITY},dtm_folder/'dtm.pt')
        np.savez_compressed(target,vectors=values)
        write(mp,{'signature':signature,'model':IDENTITY,'paper_ids':[str(p['paper_id']) for p in master],'truncated_documents':truncated,'input':'D/T/M only' if arm=='dtm' else 'Title+abstract[:1800]'+(' + D/T/M' if arm=='combined' else '')})
    terms=sorted({t for q in test for t in intent_terms(plans['intents'][q['query_id']])})
    raw,_=encode_text_rows(encoder,[q['query'] for q in test],ROOT/'output/nl_eval_embedding_cache/query_rows',IDENTITY,True)
    rewritten,_=encode_text_rows(encoder,[build_embedding_query(plans['intents'][q['query_id']]) for q in test],ROOT/'output/nl_eval_embedding_cache/query_rows',IDENTITY,True)
    tv,_=encode_text_rows(encoder,terms,ROOT/'output/nl_eval_embedding_cache/query_rows',IDENTITY,True)
    np.savez_compressed(emb/'queries.npz',raw=raw,rewritten=rewritten,terms=tv)
    write(emb/'queries_manifest.json',{'model':IDENTITY,'query_ids':[q['query_id'] for q in test],'query_hash':digest([q['query'] for q in test]),'terms':terms,'intent_model':plans['model']})
    with np.load(emb/'combined.npz',allow_pickle=False) as v:
        torch.save({'model':IDENTITY,'node_indices':torch.tensor([p['node_idx'] for p in master]),'paper_ids':[str(p['paper_id']) for p in master],
            'embeddings':torch.from_numpy(v['vectors']),'corpus_signature':__import__('hashlib').sha256(''.join(f'{p["node_idx"]}:{p["paper_id"]}\n' for p in master).encode()).hexdigest()},emb/'paper_text_embeddings.pt')
    del encoder;gc.collect();torch.cuda.empty_cache()
    print('All new corpus and query embeddings complete',flush=True)


def bootstrap(rows,baseline,seed=42):
    result={}
    for arm in sorted({r['arm'] for r in rows}-{baseline}):
        for k in sorted({r['k'] for r in rows}):
            for metric in ('recall','ndcg','mrr'):
                a={r['query_id']:r[metric] for r in rows if r['arm']==baseline and r['k']==k}
                b={r['query_id']:r[metric] for r in rows if r['arm']==arm and r['k']==k}
                if a.keys()!=b.keys():raise ValueError('Different paired evaluation sources')
                # Queries from the same source paper are correlated: resample source groups.
                source={r['query_id']:r['anchor_node_idx'] for r in rows}
                groups=sorted({source[q] for q in a});values=[np.array([b[q]-a[q] for q in a if source[q]==g]) for g in groups]
                rng=np.random.default_rng(seed);draw=[]
                for _ in range(2000):
                    picked=rng.integers(0,len(groups),size=len(groups));draw.append(float(np.concatenate([values[i] for i in picked]).mean()))
                result[f'{arm}/{metric}@{k}']={'delta':float(np.mean([b[q]-a[q] for q in a])),'ci95':np.quantile(draw,[.025,.975]).tolist(),'bootstrap_unit':'anchor_paper','groups':len(groups)}
    return result


def nl(out):
    data=out/'data';emb=out/'embeddings';dest=out/'natural_language';dest.mkdir(exist_ok=True)
    papers=read(data/'arxiv_master_final.json');ids=[str(p['paper_id']) for p in papers];nodes=[p['node_idx'] for p in papers]
    local={n:i for i,n in enumerate(nodes)};node_to_id=dict(zip(nodes,ids));category=torch.load(data/'ogbn_arxiv_16k_ffs_sample.pt',weights_only=False)['labels'].reshape(-1).tolist()
    queries=[q for q in read(data/'queries.json') if q['split']=='test'];judgments={j['query_id']:j for j in read(data/'judgments.json')}
    plans=read(data/'query_intents.gpt-5_6-luna.json')['intents'];qm=read(emb/'queries_manifest.json')
    if qm['query_ids']!=[q['query_id'] for q in queries] or qm['query_hash']!=digest([q['query'] for q in queries]):raise ValueError('Query vector mismatch')
    with np.load(emb/'queries.npz') as f:raw=f['raw'];rewritten=f['rewritten'];terms=f['terms']
    vectors={}
    for arm in ('text','dtm','combined'):
        with np.load(emb/(arm+'.npz')) as f:vectors[arm]=f['vectors']
    ce=torch.load(data/'paper_paper_edges.pt',weights_only=True).T.tolist();ae=torch.load(data/'author_paper_edges.pt',weights_only=True).T.tolist()
    graph=Graph(None,ids,citation_edges=[(node_to_id[s],node_to_id[t]) for s,t in ce],author_edges=[(str(a),node_to_id[p]) for a,p in ae])
    # Use the application's own normalized metadata loader for the current-ranker arm.
    app_papers,app_vectors,_=load_catalog(data/'arxiv_master_final.json',emb/'paper_text_embeddings.pt')
    termvectors={t:torch.from_numpy(v.copy()) for t,v in zip(qm['terms'],terms)}
    rows=[];rankings=[]
    for qi,q in enumerate(queries):
        j=judgments[q['query_id']];candidate=[local[int(n)] for n in j['candidate_node_indices']];labels={node_to_id[int(v['node_idx'])]:v['relevance'] for v in j['labels']}
        scores={arm:values@raw[qi] for arm,values in vectors.items()}
        prior=prior_from_scores(scores['combined'],.05);seed=np.zeros(len(ids));picked=sorted(range(len(ids)),key=lambda i:(-prior[i],ids[i]))[:50];seed[picked]=prior[picked];seed/=seed.sum()
        for arm,c,a in (('citation',True,False),('author',False,True),('both',True,True)):
            mass,it,converged=graph.diffuse(seed,c,a)
            if not converged:raise ValueError('PPR did not converge')
            scores[arm]=.6*prior+.4*mass
        rank={arm:sorted(candidate,key=lambda i:(-v[i],ids[i])) for arm,v in scores.items()}
        plan=plans[q['query_id']]
        found=rerank(plan,torch.from_numpy(rewritten[qi].copy()),termvectors,app_papers,app_vectors,{},len(candidate),len(candidate),30,.04,0.,.08,.03,.12,.30,.45,.04,restrict_to=candidate)
        rank['intent']=[v['catalog_index'] for v in found]
        for protocol in ('pooled','pooled_without_anchor'):
            excluded=node_to_id[int(q['anchor_node_idx'])] if protocol=='pooled_without_anchor' else None
            judgments_used={pid:v for pid,v in labels.items() if pid!=excluded}
            rank_used=rank
            if excluded is not None:
                # Excluded sources cannot seed diffusion; they may remain graph intermediates.
                anchor=local[int(q['anchor_node_idx'])];allowed=np.arange(len(ids))!=anchor
                clean_prior=np.zeros(len(ids));clean_prior[allowed]=prior_from_scores(scores['combined'][allowed],.05)
                picked=sorted(np.flatnonzero(allowed),key=lambda i:(-clean_prior[i],ids[i]))[:50]
                clean_seed=np.zeros(len(ids));clean_seed[picked]=clean_prior[picked];clean_seed/=clean_seed.sum()
                rank_used=dict(rank)
                for arm,c,a in (('citation',True,False),('author',False,True),('both',True,True)):
                    mass,_,converged=graph.diffuse(clean_seed,c,a)
                    if not converged:raise ValueError('Anchor-excluded PPR did not converge')
                    v=.6*clean_prior+.4*mass
                    rank_used[arm]=sorted(candidate,key=lambda i:(-v[i],ids[i]))
            for arm,order in rank_used.items():
                ranked=[ids[i] for i in order if ids[i]!=excluded]
                rankings.append({'query_id':q['query_id'],'protocol':protocol,'arm':arm,'ranked_paper_ids':ranked})
                for k in (5,10,20):
                    rows.append({'query_id':q['query_id'],'anchor_node_idx':int(q['anchor_node_idx']),'category':category[local[int(q['anchor_node_idx'])]],'protocol':protocol,'arm':arm,'k':k,**metrics(ranked,judgments_used,k)})
        print(f'Natural language {qi+1}/{len(queries)}',flush=True)
    result={}
    for protocol in ('pooled','pooled_without_anchor'):
        subset=[r for r in rows if r['protocol']==protocol];avg={}
        for arm in vectors.keys()|{'citation','author','both','intent'}:
            avg[arm]={str(k):{m:float(np.mean([r[m] for r in subset if r['arm']==arm and r['k']==k])) for m in ('recall','strong_recall','ndcg','mrr','judged_fraction')} for k in (5,10,20)}
        per_query_labels=[]
        for q in queries:
            values=[v['relevance'] for v in judgments[q['query_id']]['labels'] if protocol=='pooled' or int(v['node_idx'])!=int(q['anchor_node_idx'])]
            per_query_labels.append(values)
        result[protocol]={'queries':len(queries),'metrics':avg,'paired_deltas_vs_text':bootstrap(subset,'text'),
            'zero_positive_queries':sum(not any(v>0 for v in values) for values in per_query_labels),
            'zero_strong_queries':sum(not any(v==2 for v in values) for values in per_query_labels),
            'judged_candidates':sum(len(v) for v in per_query_labels),
            'represented_source_categories':len({r['category'] for r in subset})}
    write(dest/'metrics.json',rows);write(dest/'rankings.json',rankings);write(dest/'summary.json',result)
    write(dest/'manifest.json',{'split':'test','test_queries':len(queries),'unique_test_anchors':len({q['anchor_node_idx'] for q in queries}),'embedding_model':IDENTITY,
        'label_models':dict(Counter(judgments[q['query_id']].get('model') for q in queries)),'intent_model':qm['intent_model'],'mode':'fully judged fixed-candidate pooled reranking',
        'notes':read(out/'data_audit.json')['evaluation_notes'],'input_sha256':{n:file_hash(data/n) for n in ('queries.json','judgments.json','arxiv_master_final.json')}})
    print('Natural language NDCG@10', {p:{a:round(v['10']['ndcg'],4) for a,v in r['metrics'].items()} for p,r in result.items()},flush=True)


def citation(out,device=None):
    from .citation import load_inputs
    data=out/'data';dest=out/'citation';dest.mkdir(exist_ok=True)
    master,x,labels,edges,outside=load_inputs(data/'arxiv_master_final.json',data/'ogbn_arxiv_16k_ffs_sample.pt',data/'paper_paper_edges.pt')
    split=split_edges(edges,42);write(dest/'split.json',{'node_ids':[p['node_idx'] for p in master],'edges':split})
    args=SimpleNamespace(seed=42,device=device or ('cuda' if torch.cuda.is_available() else 'cpu'),hidden=128,lr=.001,epochs=200,eval_every=5,selection_k=20,ks=[10,20])
    with np.load(out/'embeddings/dtm.npz') as f:dtm=torch.from_numpy(f['vectors'])
    results={};allrows=[]
    for arm in ('ogbn','dtm','graph_only'):
        print('Citation training '+arm,flush=True)
        rows,selection,state=train_arm(dtm if arm=='dtm' else x,split,labels,args,arm=='graph_only')
        write(dest/(arm+'_per_paper.json'),rows);torch.save(state,dest/(arm+'_model.pt'))
        results[arm]={'test':citation_summary(rows),**selection};write(dest/'summary.json',results)
        allrows.extend({'query_id':str(r['source']),'anchor_node_idx':r['source'],'arm':arm,**r} for r in rows)
    write(dest/'paired_deltas_vs_ogbn.json',bootstrap(allrows,'ogbn'))
    write(dest/'manifest.json',{'seed':42,'epochs':200,'split_edges':{k:len(v) for k,v in split.items()},'corpus_papers':len(master),'outside_edges':outside,'bootstrap_unit':'source_paper','training_device':args.device,
        'protocol':'same directed pair-group split 80/10/10; train-only message passing; validation model selection; full corpus test candidates','data_audit_sha256':file_hash(out/'data_audit.json')})


def activate(out):
    if (out/'activation.json').exists():
        print('Dataset already activated',flush=True);return
    data=out/'data';emb=out/'embeddings'
    # Verify all components and both experiments exist before modifying active paths.
    for p in (out/'natural_language/summary.json',out/'citation/summary.json',emb/'paper_text_embeddings.pt'):
        if not p.exists():raise ValueError('Activation requires completed benchmarks: '+str(p))
    appcache=emb/'app_cache';appcache.mkdir(exist_ok=True)
    queries=[q for q in read(data/'queries.json') if q['split']=='test'];plans=read(data/'query_intents.gpt-5_6-luna.json')['intents'];qm=read(emb/'queries_manifest.json')
    if qm['query_ids']!=[q['query_id'] for q in queries]:raise ValueError('Application query cache alignment mismatch')
    with np.load(emb/'queries.npz') as qv:
        for name,key,query_texts in [('query_embeddings.pt','raw',{q['query_id']:q['query'] for q in queries}),
             ('intent_query_embeddings.gpt-5_6-luna.pt','rewritten',{q['query_id']:build_embedding_query(plans[q['query_id']]) for q in queries})]:
            torch.save({'model':IDENTITY,'queries':query_texts,'vectors':{q['query_id']:torch.from_numpy(v.copy()) for q,v in zip(queries,qv[key])}},appcache/name)
        torch.save({'model':IDENTITY,'vectors':{t:torch.from_numpy(v.copy()) for t,v in zip(qm['terms'],qv['terms'])}},appcache/'intent_term_embeddings.pt')
    catalog=torch.load(emb/'paper_text_embeddings.pt',weights_only=True,map_location='cpu')
    current_master=read(data/'arxiv_master_final.json');ordered=sorted(current_master,key=lambda p:int(p['node_idx']))
    index_by_node={int(n):i for i,n in enumerate(catalog['node_indices'].tolist())}
    combined_texts=[f'Title: {p["title"]}\nAbstract: {p["abstract"][:1800]}\nD/T/M: '+'; '.join(term for k in ('domain','task','method') for term in p['knowledge'][k]) for p in ordered]
    checkpoint_signature=__import__('hashlib').sha256(json.dumps(combined_texts,ensure_ascii=False).encode()).hexdigest()
    torch.save({'model':IDENTITY,'signature':checkpoint_signature,'vectors':catalog['embeddings'][[index_by_node[int(p['node_idx'])] for p in ordered]]},emb/'local_checkpoint.pt')
    tm=read(emb/'text_manifest.json');write(appcache/'text_manifest.json',{'signature':tm['signature'],'model':IDENTITY,
        'truncated_papers':tm['truncated_documents'],'input':tm['input'],'paper_ids':tm['paper_ids'],'timing_note':'Restored from completed benchmark corpus; not a fresh timing measurement'})
    write(appcache/'manifest.json',{'paper_count':len(catalog['paper_ids']),'corpus_signature':catalog['corpus_signature'],
        'master_path':str(ROOT/'subdataset/arxiv_master_final.json'),'dtm_path':str(ROOT/'output/dtm_luna_16k/gpt-5.6-luna.json'),
        'seed':42,'embedding_model':IDENTITY,'restored_benchmark':str(out)})
    copies=[(data/n,ROOT/'subdataset'/n) for n in ('arxiv_master_final.json','ogbn_arxiv_16k_ffs_sample.pt','paper_paper_edges.pt','author_paper_edges.pt','build_hetero_graph_v2.pt')]
    copies.extend([(data/'knowledge_meta.json',ROOT/'output/knowledge_meta.json'),
        (emb/'paper_text_embeddings.pt',ROOT/'output/nl_recommendation_qwen4b_nf4/paper_text_embeddings.pt'),
        (emb/'paper_text_embeddings.pt',ROOT/'output/nl_recommendation_data/paper_text_embeddings.pt'),
        (emb/'citation_dtm/dtm.pt',ROOT/'output/citation_dtm_embeddings/dtm.pt'),
        (emb/'local_checkpoint.pt',ROOT/'output/nl_recommendation_qwen4b_nf4/local_checkpoint.pt'),
        (emb/'text.npz',ROOT/'output/nl_eval_embedding_cache/text_bundle.npz'),
        (appcache/'text_manifest.json',ROOT/'output/nl_eval_embedding_cache/text_manifest.json')])
    copies.extend((appcache/n,ROOT/'output/nl_recommendation_data'/n) for n in ('query_embeddings.pt','intent_query_embeddings.gpt-5_6-luna.pt','intent_term_embeddings.pt','manifest.json'))
    backup=out/'previous_active_dataset.zip'
    if backup.exists():raise ValueError('Backup already exists without activation marker; inspect interrupted activation')
    with ZipFile(backup,'w',ZIP_STORED) as archive:
        for _,dst in copies:
            if dst.exists():archive.write(dst,dst.relative_to(ROOT))
    changed=[]
    for src,dst in copies:
        dst.parent.mkdir(parents=True,exist_ok=True);temp=dst.with_suffix(dst.suffix+'.restore_tmp');shutil.copyfile(src,temp);temp.replace(dst);changed.append(str(dst.relative_to(ROOT)))
    info={'active_dataset':str(out),'backup':str(backup),'changed_files':changed,'node_order':'imported sample_manifest order','all_ids_and_category_alignment_verified':True}
    write(out/'activation.json',info);write(ROOT/'benchmarks/active_dataset.json',info)
    print('New balanced dataset activated with recoverable backup',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--stage',choices=['restore','encode','nl','citation','activate','all'],default='all');parser.add_argument('--citation-device',choices=['cpu','cuda']);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    for name,fn in [('restore',restore),('encode',encode),('nl',nl),('citation',citation),('activate',activate)]:
        if args.stage in (name,'all'):
            if name=='citation':fn(args.output,args.citation_device)
            else:fn(args.output)

if __name__=='__main__':main()
