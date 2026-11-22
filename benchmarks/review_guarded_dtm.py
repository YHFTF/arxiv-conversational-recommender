"""Compare guarded live retrieval; export unknown hard negatives without inventing labels."""
import sys,time
from pathlib import Path
import numpy as np
import torch,joblib
from .core import Graph,read,write,file_hash,metrics
from .lexical_dtm_graph import Index
from .review_live_dtm_ranker import REQUESTS
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code/retrieval'))
from dtm_graph_search import DTMIndex
from dtm_rank_features import Features
from query_constraints import GuardedSearch


def main(stage,use_text_bm25=False,requests=None,output_name=None,consensus_strength=0.):
    torch.set_num_threads(1)
    src=ROOT/'output/balanced_benchmark_20261117';data=src/'data';out=src/('guarded_text_bm25' if use_text_bm25 else 'guarded_dtm_graph')
    if output_name is not None:out=src/output_name
    papers=read(data/'arxiv_master_final.json');ids=[str(p['paper_id']) for p in papers];local={p['node_idx']:i for i,p in enumerate(papers)}
    edges=torch.load(data/'paper_paper_edges.pt',weights_only=True,map_location='cpu').T.tolist();graph=Graph(None,ids,relations=('citation',),citation_edges=[(ids[local[s]],ids[local[t]]) for s,t in edges])
    extractor=Features(papers,Index(papers),DTMIndex(papers),graph)
    selected=read(src/'learned_dtm_graph/selected.json');modelpath=src/'learned_dtm_graph'/selected['model_file'];assert file_hash(modelpath)==selected['model_sha256'];model=joblib.load(modelpath)
    engine=GuardedSearch(papers,extractor,model,use_text_bm25=use_text_bm25,consensus_strength=consensus_strength)
    write(out/'search_manifest.json',dict(consensus_strength=consensus_strength,auxiliary_text_bm25=use_text_bm25,text_index_array_bytes=engine.text_index.array_bytes if use_text_bm25 else 0,ranking='evidence tier then equal RRF(model,text), k=60 plus selected consensus bonus' if use_text_bm25 else 'evidence tier then model',model_retrained=False))
    write(out/'metadata.json',engine.metadata)
    if stage in ('live','check'):
        result=[]
        for name,query in (REQUESTS if requests is None else requests):
            start=time.perf_counter();r=engine.search(query,top_k=5)
            for v in r['results']:
                i=v.pop('index');v.update(paper_id=ids[i],title=papers[i]['title'],abstract=papers[i]['abstract'])
            result.append(dict(name=name,query=query,search_ms=(time.perf_counter()-start)*1000,**r));print(name,[v['title'] for v in r['results']],flush=True)
        write(out/'live_results.json',result)
    if stage in ('benchmark','check'):
        qs=[q for q in read(data/'queries.json') if q['split']=='test'];js={j['query_id']:j for j in read(data/'judgments.json')};rows=[];forced=[]
        # Score full actual pipeline first; only then intersect the existing fixed pool.
        # Filtered-out labeled candidates contribute missing results, not a smaller IDCG.
        for q in qs:
            anchor=local[int(q['anchor_node_idx'])];r=engine.search(q['query'],top_k=len(papers),excluded=anchor);j=js[q['query_id']]
            labels={ids[local[int(v['node_idx'])]]:v['relevance'] for v in j['labels'] if int(v['node_idx'])!=int(q['anchor_node_idx'])}
            pool=set(labels);ranked=[ids[v['index']] for v in r['results'] if ids[v['index']] in pool]
            rows.append(dict(query_id=q['query_id'],anchor=q['anchor_node_idx'],eligible_count=r['eligible_count'],**metrics(ranked,labels,10)))
            pool=[local[int(v)] for v in j['candidate_node_indices'] if int(v)!=int(q['anchor_node_idx'])]
            r2=engine.search(q['query'],top_k=len(pool),excluded=anchor,candidate_pool=pool)
            forced.append(dict(query_id=q['query_id'],anchor=q['anchor_node_idx'],**metrics([ids[v['index']] for v in r2['results']],labels,10)))
        write(out/'test_metrics.json',rows);write(out/'forced_pool_metrics.json',forced);write(out/'test_summary.json',dict(full_retrieval_intersected_metrics_at_10={m:float(np.mean([r[m] for r in rows])) for m in ('ndcg','recall','strong_recall','mrr')},fixed_pool_metrics_at_10={m:float(np.mean([r[m] for r in forced])) for m in ('ndcg','recall','strong_recall','mrr')},notes=['Full retrieval intersected with fixed pool is a diagnostic only; use fixed_pool for prior reranking comparisons.','Filtered-out positive documents remain in IDCG and recall denominator.','Unknown full-corpus documents are not assigned relevance=0.']))
    elif stage=='mine':
        queries=[q for q in read(data/'queries.json') if q['split']=='train'];js={j['query_id']:j for j in read(data/'judgments.json')};known=[];unknown=[];start=time.perf_counter()
        for n,q in enumerate(queries):
            r=engine.search(q['query'],top_k=20,excluded=local[int(q['anchor_node_idx'])]);j=js[q['query_id']];labels={int(v['node_idx']):v['relevance'] for v in j['labels']}
            for v in r['results']:
                i=v['index'];row=dict(query_id=q['query_id'],paper_id=ids[i],node_idx=papers[i]['node_idx'],query=q['query'],title=papers[i]['title'],abstract=papers[i]['abstract'],score=v['score'])
                if papers[i]['node_idx'] in labels:
                    row['grade']=labels[papers[i]['node_idx']];known.append(row)
                else:
                    row['grade']=None;unknown.append(row)
            if (n+1)%100==0:print('Mine train',n+1,flush=True)
        write(out/'known_retrieved_train.json',known);write(out/'unjudged_train_review.json',unknown)
        write(out/'mining_summary.json',dict(train_queries=len(queries),known_pairs=len(known),known_negative_pairs=sum(r['grade']==0 for r in known),unjudged_pairs=len(unknown),seconds=time.perf_counter()-start,policy='Only existing query-specific train labels may be reused; unknown results are not negatives and require review.'))
        print('Known',len(known),'unjudged',len(unknown),flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['live','benchmark','mine','check'],required=True);p.add_argument('--text-bm25',action='store_true');args=p.parse_args();main(args.stage,args.text_bm25)
