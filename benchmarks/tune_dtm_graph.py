"""Select on validation, freeze, evaluate once on test. No online LLM/embeddings."""
import argparse,itertools,sys,time
from pathlib import Path
import numpy as np
import torch
from .core import Graph,read,write,metrics,file_hash
from .lexical_dtm_graph import Index
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code/retrieval'))
from dtm_graph_search import DTMIndex,graph_scores,rank_scores

LEXICAL={
 'legacy':None,
 'normalized_equal':dict(weights=[1.,1.,1.],phrase_weight=0.),
 'task_method':dict(weights=[.5,1.5,1.5],phrase_weight=0.),
 'phrases_equal':dict(weights=[1.,1.,1.],phrase_weight=1.),
 'phrases_task_method':dict(weights=[.5,1.5,1.5],phrase_weight=1.),
}


def main(stage):
    torch.set_num_threads(1)
    src=ROOT/'output/balanced_benchmark_20261117';data=src/'data';out=src/'optimized_dtm_graph'
    papers=read(data/'arxiv_master_final.json');ids=[str(p['paper_id']) for p in papers];local={p['node_idx']:i for i,p in enumerate(papers)}
    idx=DTMIndex(papers);legacy=Index(papers)
    edges=torch.load(data/'paper_paper_edges.pt',weights_only=True,map_location='cpu').T.tolist()
    graph=Graph(None,ids,relations=('citation',),citation_edges=[(ids[local[s]],ids[local[t]]) for s,t in edges])
    split='validation' if stage=='tune' else 'test'
    qs=[q for q in read(data/'queries.json') if q['split']==split];qids={q['query_id'] for q in qs}
    judgments={j['query_id']:j for j in read(data/'judgments.json') if j['query_id'] in qids}
    allqs=read(data/'queries.json');assert not ({q['anchor_node_idx'] for q in allqs if q['split']=='test'} & {q['anchor_node_idx'] for q in allqs if q['split']=='validation'})
    contexts=[]
    for q in qs:
        j=judgments[q['query_id']];anchor=local[int(q['anchor_node_idx'])]
        pool=[local[int(n)] for n in j['candidate_node_indices'] if int(n)!=int(q['anchor_node_idx'])]
        labels={ids[local[int(v['node_idx'])]]:v['relevance'] for v in j['labels'] if int(v['node_idx'])!=int(q['anchor_node_idx'])}
        uniform=np.ones(len(ids));uniform[anchor]=0;uniform/=uniform.sum();centrality,_,ok=graph.diffuse(uniform,True,False);assert ok
        contexts.append((q,anchor,pool,labels,centrality))
    def lexical(query,name):return legacy.search({'rewritten_query':query},False)[0] if name=='legacy' else idx.search(query,**LEXICAL[name])
    def measure(config,details=False):
        values=[];rows=[];full=[];times=[]
        for q,anchor,pool,labels,centrality in contexts:
            start=time.perf_counter();ls=lexical(q['query'],config['lexical']);gs=graph_scores(graph,ls,excluded=anchor,centrality=centrality,**config['graph'])
            scores=(1-config['lexical_mix'])*gs/max(gs.max(),1e-12)+config['lexical_mix']*ls
            ranked=rank_scores(scores,pool) if ls.max()>0 else []
            met=metrics([ids[i] for i in ranked],labels,10);values.append(met['ndcg'])
            if details:
                allowed=np.arange(len(ids))!=anchor;order=np.flatnonzero(allowed);order=order[np.argsort(-scores[order],kind='stable')[:20]] if ls.max()>0 else []
                times.append((time.perf_counter()-start)*1000)
                rows.append(dict(query_id=q['query_id'],anchor=q['anchor_node_idx'],**met))
                top=[ids[i] for i in order];full.append(dict(query_id=q['query_id'],top20=top,judged_fraction_at_10=sum(p in labels for p in top[:10])/min(len(top),10) if len(top) else 0.))
        if details:return rows,full,times
        return float(np.mean(values))
    baseline=dict(lexical='legacy',graph=dict(seed_count=50,restart=.2,temperature=.05,debias=0.),lexical_mix=0.)
    if stage=='tune':
        write(out/'protocol.json',dict(primary_metric='validation anchor-excluded NDCG@10',lexical_configs=LEXICAL,graph_grid=dict(seed_count=[10,50,150],restart=[.2,.5],temperature=[.05,.2],debias=[0.,.5]),lexical_mix=[0.,.2],test_tuning=False,notes=['Select lexical stage with default PPR, then tune graph settings for selected lexical stage.','Global centrality for degree-bias correction computed without qrels; excluded anchor is not a restart seed.','Test set has been explored in earlier sessions, so it is not a new untouched holdout.']))
        trials=[]
        for name in LEXICAL:
            config=dict(baseline,lexical=name);value=measure(config);trials.append(dict(stage='lexical',config=config,ndcg=value));print(name,value,flush=True)
        best=max(trials,key=lambda v:v['ndcg']);selected_lex=best['config']['lexical']
        for count,restart,temp,debias,mix in itertools.product([10,50,150],[.2,.5],[.05,.2],[0.,.5],[0.,.2]):
            config=dict(lexical=selected_lex,graph=dict(seed_count=count,restart=restart,temperature=temp,debias=debias),lexical_mix=mix)
            value=measure(config);trials.append(dict(stage='graph',config=config,ndcg=value))
            if value>best['ndcg']:best=trials[-1]
        write(out/'validation_trials.json',trials);write(out/'selected.json',dict(config=best['config'],validation_ndcg=best['ndcg'],baseline_validation_ndcg=trials[0]['ndcg'],selection_split='validation',validation_queries=len(qs)))
        print('FROZEN',best,flush=True)
    else:
        selected=read(out/'selected.json');frozen_hash=file_hash(out/'selected.json')
        configs={'baseline':baseline,'optimized':selected['config'],'lexical_only':dict(selected['config'],lexical_mix=1.),'lexical_improvement_only':dict(baseline,lexical=selected['config']['lexical'])}
        results={};allrows={}
        for name,config in configs.items():
            rows,full,times=measure(config,True);allrows[name]=rows
            results[name]=dict(config=config,metrics_at_10={m:float(np.mean([r[m] for r in rows])) for m in ('ndcg','recall','strong_recall','mrr')},cpu_ms_median=float(np.median(times)),cpu_ms_p95=float(np.quantile(times,.95)),full_corpus_judged_fraction=float(np.mean([r['judged_fraction_at_10'] for r in full])))
            write(out/(name+'_test_metrics.json'),rows);write(out/(name+'_full_rankings.json'),full)
        old={r['query_id']:r for r in read(src/'graph_without_query_llm/metrics.json') if r['protocol']=='pooled_without_anchor' and r['arm']=='raw_seed_graph_only' and r['k']==10}
        for r in allrows['baseline']:assert abs(r['ndcg']-old[r['query_id']]['ndcg'])<1e-7
        a=allrows['baseline'];b={r['query_id']:r for r in allrows['optimized']};ds=np.array([b[r['query_id']]['ndcg']-r['ndcg'] for r in a]);anchors=np.array([r['anchor'] for r in a]);groups=np.unique(anchors);rng=np.random.default_rng(42)
        draws=[np.concatenate([ds[anchors==g] for g in rng.choice(groups,len(groups))]).mean() for _ in range(2000)]
        write(out/'test_summary.json',dict(selected_sha256=frozen_hash,selection=selected,results=results,paired_ndcg=dict(delta=float(ds.mean()),ci95=np.quantile(draws,[.025,.975]).tolist(),win_tie_loss=[int((ds>1e-10).sum()),int((abs(ds)<=1e-10).sum()),int((ds< -1e-10).sum())]),baseline_reproduction='passed'))
        assert file_hash(out/'selected.json')==frozen_hash
        print(results,flush=True);print('paired delta',ds.mean(),np.quantile(draws,[.025,.975]),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['tune','test'],required=True);main(parser.parse_args().stage)
