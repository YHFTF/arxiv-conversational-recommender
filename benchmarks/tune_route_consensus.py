"""Validate three-route consensus at top3; freeze weights before test scoring."""
import sys
from pathlib import Path
import numpy as np
import torch,joblib
from .core import Graph,read,write,metrics,file_hash
from .lexical_dtm_graph import Index
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code/retrieval'))
from dtm_graph_search import DTMIndex
from dtm_rank_features import Features
from query_constraints import GuardedSearch


def rerank(rows,strength=0.,minimum=1):
    rows=[r for r in rows if minimum==1 or r['route_count']>=minimum]
    return sorted(rows,key=lambda r:(-sum(v=='reported_in_abstract' for v in r['conditions'].values()),-(r['base_score']+strength*r['consensus_score']),r['index']))


def main():
    torch.set_num_threads(1)
    src=ROOT/'output/balanced_benchmark_20261117';out=src/'route_consensus';data=src/'data'
    papers=read(data/'arxiv_master_final.json');ids=[str(p['paper_id']) for p in papers];local={p['node_idx']:i for i,p in enumerate(papers)}
    edges=torch.load(data/'paper_paper_edges.pt',weights_only=True,map_location='cpu').T.tolist();graph=Graph(None,ids,relations=('citation',),citation_edges=[(ids[local[s]],ids[local[t]]) for s,t in edges])
    extractor=Features(papers,Index(papers),DTMIndex(papers),graph);sel=read(src/'learned_dtm_graph/selected.json');model=joblib.load(src/'learned_dtm_graph'/sel['model_file']);engine=GuardedSearch(papers,extractor,model)
    qs=read(data/'queries.json');js={j['query_id']:j for j in read(data/'judgments.json')}
    assert not ({q['anchor_node_idx'] for q in qs if q['split']=='validation'}&{q['anchor_node_idx'] for q in qs if q['split']=='test'})
    configs=[dict(name='baseline',strength=0.,minimum=1)]+[dict(name=f'soft_{s}',strength=s,minimum=1) for s in (.005,.015,.03)]+[dict(name='intersection_2',strength=0.,minimum=2),dict(name='intersection_3',strength=0.,minimum=3)]
    write(out/'protocol.json',dict(configs=configs,selection='validation strong precision@3, then NDCG@3, then lower weight; intersections diagnostic only',graph_agreement_discount=.25,text_graph_agreement_discount=.5,rrf_constant=60,route_depth=200,notes=['same previously observed benchmark, not a fresh holdout','fixed judged pools; actual global route membership, not membership computed within label pool','No labels read by retrieval engine; labels used only to evaluate rankings.']))
    allresults={};allrows={}
    for split in ('validation','test'):
        if split=='test':
            frozen=read(out/'selected.json');locked=file_hash(out/'selected.json')
        rows=[]
        for q in [q for q in qs if q['split']==split]:
            anchor=local[int(q['anchor_node_idx'])];j=js[q['query_id']];pool=[local[int(n)] for n in j['candidate_node_indices'] if int(n)!=int(q['anchor_node_idx'])]
            labels={ids[local[int(v['node_idx'])]]:v['relevance'] for v in j['labels'] if int(v['node_idx'])!=int(q['anchor_node_idx'])}
            found=engine.search(q['query'],top_k=len(pool),excluded=anchor,candidate_pool=pool)['results']
            for c in configs:
                ranked=rerank(found,c['strength'],c['minimum']);pids=[ids[r['index']] for r in ranked]
                rows.append(dict(query_id=q['query_id'],anchor=q['anchor_node_idx'],arm=c['name'],**metrics(pids,labels,3),strong_precision3=sum(labels.get(p,0)==2 for p in pids[:3])/3,precision3=sum(labels.get(p,0)>0 for p in pids[:3])/3,all3_direct=len(pids)>=3 and all(labels.get(p,0)==2 for p in pids[:3]),returned3=len(pids)>=3))
        avg={c['name']:{m:float(np.mean([r[m] for r in rows if r['arm']==c['name']])) for m in ('ndcg','precision3','strong_precision3','all3_direct','returned3')} for c in configs}
        allresults[split]=avg;allrows[split]=rows;write(out/(split+'_metrics.json'),rows)
        if split=='validation':
            # Equivalent integer relevance counts can differ at float summation
            # precision; don't let 5e-17 override the secondary metric.
            best=max(configs[:4],key=lambda c:(round(avg[c['name']]['strong_precision3'],12),round(avg[c['name']]['ndcg'],12),-c['strength']))
            write(out/'selected.json',dict(**best,validation=avg[best['name']],model_sha256=file_hash(src/'learned_dtm_graph'/sel['model_file'])))
            print('FROZEN',best,avg,flush=True)
        else:assert file_hash(out/'selected.json')==locked
    chosen=read(out/'selected.json')['name'];rs=allrows['test'];base={r['query_id']:r for r in rs if r['arm']=='baseline'};after=[r for r in rs if r['arm']==chosen]
    ds=np.array([r['ndcg']-base[r['query_id']]['ndcg'] for r in after]);anchors=np.array([r['anchor'] for r in after]);groups=np.unique(anchors);rng=np.random.default_rng(42);draws=[np.concatenate([ds[anchors==g] for g in rng.choice(groups,len(groups))]).mean() for _ in range(2000)]
    write(out/'summary.json',dict(results=allresults,selected=read(out/'selected.json'),paired_ndcg3=dict(delta=float(ds.mean()),ci95=np.quantile(draws,[.025,.975]).tolist())))
    print('TEST',allresults['test'],flush=True)
    from .review_live_dtm_ranker import REQUESTS
    engine.consensus_strength=read(out/'selected.json')['strength'];live=[]
    for name,query in REQUESTS:
        if not name.endswith('_ko'):continue
        result=engine.search(query,top_k=3)
        for r in result['results']:
            i=r.pop('index');r.update(paper_id=ids[i],title=papers[i]['title'],abstract=papers[i]['abstract'])
        live.append(dict(name=name,query=query,**result))
    write(out/'live_results.json',live)


if __name__=='__main__':main()
