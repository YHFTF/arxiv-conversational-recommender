"""No online LLM/embedding: raw-query lexical seeds versus pure graph ranking."""
import time
from pathlib import Path
import numpy as np
import torch
from .core import Graph,read,write,metrics
from .lexical_dtm_graph import Index,rank

ROOT=Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(1)
    src=ROOT/'output/balanced_benchmark_20261117';data=src/'data';out=src/'graph_without_query_llm'
    papers=read(data/'arxiv_master_final.json');ids=[str(p['paper_id']) for p in papers];local={p['node_idx']:i for i,p in enumerate(papers)}
    idx=Index(papers)
    edges=torch.load(data/'paper_paper_edges.pt',weights_only=True,map_location='cpu').T.tolist()
    graph=Graph(None,ids,relations=('citation',),citation_edges=[(ids[local[s]],ids[local[t]]) for s,t in edges])
    neighbors=[set(graph.citation.indices[graph.citation.indptr[i]:graph.citation.indptr[i+1]]) for i in range(len(ids))]
    qs=[q for q in read(data/'queries.json') if q['split']=='test'];js={j['query_id']:j for j in read(data/'judgments.json')}
    variants=['raw_bm25','raw_seed_graph_only','raw_graph_bonus','raw_graph_both','global_graph_only']
    write(out/'manifest.json',dict(query_rewriting=False,query_intent_cache_read=False,embedding_usage=False,online_llm_usage=False,dtm='existing offline LLM-extracted metadata retained for lexical seeds',arms=dict(raw_bm25='Original query BM25 against DTM',raw_seed_graph_only='Original-query BM25 positive top50 seeds, then PPR ONLY ranking',raw_graph_bonus='Original BM25 score + .1 max-normalized PPR',raw_graph_both='Original BM25 + .1 PPR - .1 direct selected-neighbor penalty',global_graph_only='Uniform restart PPR; query-independent graph centrality diagnostic'),restart=.2,seed_temperature=.05,no_seed_policy='abstain; empty ranking, zero relevance metrics',protocol='same fixed judged pools, plus unjudged full-corpus diagnostics',notes=['No structured plans, rewritten queries or embedding vectors are read.','Original benchmark queries and labels are still LLM-generated; this is not a human-gold evaluation.','Global graph baseline uses no DTM but cannot respond to query meaning.']))
    rows=[];full=[];timings=[];coverage=[]
    for qi,q in enumerate(qs):
        qid=q['query_id'];j=js[qid];labels={ids[local[int(v['node_idx'])]]:v['relevance'] for v in j['labels']};cand=[local[int(n)] for n in j['candidate_node_indices']]
        start=time.perf_counter();score,signal=idx.search({'rewritten_query':q['query']},False);lexms=(time.perf_counter()-start)*1000
        coverage.append(dict(query_id=qid,has_lexical_seed=bool(signal),positive_match_documents=int((score>0).sum()),ascii_token_count=len(__import__('re').findall(r'[A-Za-z]+',q['query']))))
        for protocol in ('pooled','pooled_without_anchor'):
            anchor=local[int(q['anchor_node_idx'])] if protocol.endswith('without_anchor') else None
            allowed=[i for i in range(len(ids)) if i!=anchor];pool=[i for i in cand if i!=anchor];lab={p:g for p,g in labels.items() if anchor is None or p!=ids[anchor]}
            start=time.perf_counter();top=sorted((i for i in allowed if score[i]>0),key=lambda i:-score[i])[:50];seed=np.zeros(len(ids))
            if top:
                v=score[top];v=np.exp((v-v.max())/.05);seed[top]=v/v.sum();mass,_,ok=graph.diffuse(seed,True,False);assert ok;bonus=mass/mass.max()
            else:bonus=np.zeros(len(ids))
            pprms=(time.perf_counter()-start)*1000
            start=time.perf_counter();uniform=np.zeros(len(ids));uniform[allowed]=1/len(allowed);globalmass,_,ok=graph.diffuse(uniform,True,False);assert ok;globalms=(time.perf_counter()-start)*1000
            for arm in variants:
                isglobal=arm=='global_graph_only';active=bool(top) or isglobal
                base=globalmass if isglobal else bonus if arm=='raw_seed_graph_only' else score
                bw=.1 if arm in ('raw_graph_bonus','raw_graph_both') else 0.;pw=.1 if arm=='raw_graph_both' else 0.
                ranked=rank(pool,base,bonus,neighbors,bw,pw) if active else []
                for k in (5,10,20):rows.append(dict(query_id=qid,anchor=q['anchor_node_idx'],protocol=protocol,arm=arm,k=k,**metrics([ids[i] for i in ranked],lab,k)))
                start=time.perf_counter();short=rank(allowed,base,bonus,neighbors,bw,0.,200) if active else [];fr=rank(short,base,bonus,neighbors,bw,pw,20);rankms=(time.perf_counter()-start)*1000
                topids=[ids[i] for i in fr[:10]];positive={p for p,g in lab.items() if g>0}
                full.append(dict(query_id=qid,protocol=protocol,arm=arm,top10=topids,judged_fraction=sum(p in lab for p in topids)/len(topids) if topids else 0.,known_positive_coverage=len(set(topids)&positive)/len(positive) if positive else 0.))
                ms=rankms+(globalms if isglobal else lexms+(pprms if arm!='raw_bm25' else 0.))
                timings.append(dict(query_id=qid,protocol=protocol,arm=arm,total_ms=ms))
        print('Query',qi+1,flush=True)
    previous=read(src/'lexical_dtm_graph/summary.json');summary={}
    for protocol in ('pooled','pooled_without_anchor'):
        result={}
        for arm in variants:
            rr=[r for r in rows if r['protocol']==protocol and r['arm']==arm and r['k']==10];tt=[r['total_ms'] for r in timings if r['protocol']==protocol and r['arm']==arm];ff=[r for r in full if r['protocol']==protocol and r['arm']==arm]
            result[arm]=dict(metrics_at_10={m:float(np.mean([r[m] for r in rr])) for m in ('ndcg','recall','strong_recall','mrr')},cpu_ms_median=float(np.median(tt)),cpu_ms_p95=float(np.quantile(tt,.95)),full_corpus_judged_fraction=float(np.mean([r['judged_fraction'] for r in ff])))
        summary[protocol]=dict(variants=result,previous_llm_enhanced_variants=previous[protocol]['variants'],embedding_baselines=previous[protocol]['stored_embedding_baselines'])
    summary['coverage']=dict(queries=len(qs),queries_with_seeds=sum(c['has_lexical_seed'] for c in coverage),queries_without_seeds=sum(not c['has_lexical_seed'] for c in coverage))
    write(out/'metrics.json',rows);write(out/'full_corpus_diagnostics.json',full);write(out/'timings.json',timings);write(out/'coverage.json',coverage);write(out/'summary.json',summary)
    print(summary['pooled_without_anchor']['variants']);print(summary['coverage'])


if __name__=='__main__':main()
