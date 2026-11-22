"""Search with validation-selected D/T/M + citation settings on CPU."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from benchmarks.core import Graph,read,file_hash
from benchmarks.lexical_dtm_graph import Index
from dtm_graph_search import DTMIndex,graph_scores


def main():
    parser=argparse.ArgumentParser(description='CPU D/T/M and citation search; no online LLM or embeddings')
    base=ROOT/'output/balanced_benchmark_20261117'
    parser.add_argument('--query',required=True)
    parser.add_argument('--top-k',type=int,default=3)
    parser.add_argument('--no-text-bm25',action='store_true',help='Disable auxiliary title/abstract BM25 for comparison')
    parser.add_argument('--no-route-consensus',action='store_true',help='Disable validation-selected overlap bonus')
    parser.add_argument('--mode',choices=['baseline','learned','guarded'],default='guarded')
    parser.add_argument('--config',type=Path,default=None,help='Explicit manual graph configuration; default is the original baseline')
    parser.add_argument('--data-dir',type=Path,default=base/'data')
    args=parser.parse_args()
    if args.top_k<1:parser.error('--top-k must be positive')
    torch.set_num_threads(1)
    config=read(args.config)['config'] if args.config else dict(lexical='legacy',graph=dict(seed_count=50,restart=.2,temperature=.05,debias=0.),lexical_mix=0.)
    papers=read(args.data_dir/'arxiv_master_final.json')
    ids=[str(p['paper_id']) for p in papers];local={p['node_idx']:i for i,p in enumerate(papers)}
    if config['lexical']=='legacy' or args.mode in ('learned','guarded'):index=Index(papers)
    else:
        from benchmarks.tune_dtm_graph import LEXICAL
        index=DTMIndex(papers)
    edges=torch.load(args.data_dir/'paper_paper_edges.pt',weights_only=True,map_location='cpu').T.tolist()
    graph=Graph(None,ids,relations=('citation',),citation_edges=[(ids[local[s]],ids[local[t]]) for s,t in edges])
    centrality=None;extractor=None;model=None
    if args.mode in ('learned','guarded'):
        import joblib
        from dtm_rank_features import Features
        selection=read(base/'learned_dtm_graph/selected.json')
        model_path=base/'learned_dtm_graph'/selection['model_file']
        if file_hash(model_path)!=selection['model_sha256']:raise ValueError('Ranker model hash mismatch')
        model=joblib.load(model_path)
        extractor=Features(papers,index,DTMIndex(papers),graph)
    if config['graph']['debias']:
        centrality,_,ok=graph.diffuse(np.ones(len(ids))/len(ids),True,False)
        if not ok:raise RuntimeError('Centrality did not converge')
    start=time.perf_counter()
    if args.mode=='guarded':
        from query_constraints import GuardedSearch
        strength=0.
        consensus_path=base/'route_consensus/selected.json'
        if consensus_path.exists() and not args.no_route_consensus and not args.no_text_bm25:
            consensus=read(consensus_path)
            if consensus['model_sha256']!=selection['model_sha256']:raise ValueError('Consensus configuration belongs to another ranker')
            strength=float(consensus['strength'])
            if not np.isfinite(strength) or strength<0:raise ValueError('Invalid consensus weight')
        engine=GuardedSearch(papers,extractor,model,use_text_bm25=not args.no_text_bm25,consensus_strength=strength)
        start=time.perf_counter()
        result=engine.search(args.query,top_k=args.top_k)
        for row in result['results']:
            i=row.pop('index');row.update(paper_id=ids[i],title=papers[i]['title'],dtm=papers[i].get('knowledge',{}))
        print(json.dumps(dict(query=args.query,mode='guarded',route_consensus_strength=strength,search_ms=(time.perf_counter()-start)*1000,**result),ensure_ascii=False,indent=2))
        return
    if args.mode=='learned':
        state=extractor.prepare(args.query);lexical=state['lex'];mass=state['mass'];scores=np.full(len(ids),-np.inf)
        # Sparse retrieval proposes <=400 candidates; learned ranker then orders them.
        if lexical.max()>0:
            candidates=sorted(set(np.argsort(-lexical,kind='stable')[:200])|set(np.argsort(-mass,kind='stable')[:200]))
            scores[candidates]=model.predict(extractor.rows(state,candidates))
        config=dict(mode='learned',model=selection['name'],candidate_limit_per_retriever=200)
    else:
        lexical=index.search({'rewritten_query':args.query},False)[0] if config['lexical']=='legacy' else index.search(args.query,**LEXICAL[config['lexical']])
        mass=graph_scores(graph,lexical,centrality=centrality,**config['graph'])
        scores=(1-config['lexical_mix'])*mass/max(mass.max(),1e-12)+config['lexical_mix']*lexical
    available=np.flatnonzero(np.isfinite(scores))
    order=available[np.argsort(-scores[available],kind='stable')[:args.top_k]] if lexical.max()>0 else []
    print(json.dumps(dict(query=args.query,config=config,search_ms=(time.perf_counter()-start)*1000,results=[dict(paper_id=ids[i],title=papers[i]['title'],dtm=papers[i].get('knowledge',{}),score=float(scores[i])) for i in order]),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
