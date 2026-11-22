"""CPU D/T/M lexical retrieval: no document/query embeddings or model loading."""
import math,re,sys,time,platform
from collections import Counter
from pathlib import Path
import numpy as np
from scipy import sparse
import torch
from .core import Graph,read,write,metrics
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code/data_analysis'))
from normalize_dtm_vocabulary import canonicalize
STOP=set('a an the of for to in on and or with by from is are as at that this using use based without'.split())

def tokens(text):
    return [w for w in canonicalize(text,'method').split() if w not in STOP]


class Index:
    def __init__(self,papers):
        self.tags=[];self.sets=[];self.fieldsets=[];docs=[]
        for p in papers:
            k=p.get('knowledge') or {};fields={}
            for field in ('domain','task','method'):
                values=[]
                for source in (k,k.get('normalized',{}),k.get('graph',{})):
                    if isinstance(source,dict):values.extend(source.get(field,[]))
                fields[field]=list(dict.fromkeys(canonicalize(v,field) for v in values))
            self.tags.append(fields)
            self.fieldsets.append({f:set(tokens(' '.join(values))) for f,values in fields.items()})
            doc=tokens(' '.join(t for values in fields.values() for t in values))
            docs.append(Counter(doc));self.sets.append(set(doc))
        self.vocab={t:i for i,t in enumerate(sorted({t for d in docs for t in d}))}
        rr=[];cc=[];vv=[];length=np.array([sum(d.values()) for d in docs],dtype=float);avg=max(float(length.mean()),1.)
        for i,d in enumerate(docs):
            for t,tf in d.items():rr.append(i);cc.append(self.vocab[t]);vv.append(tf)
        tf=sparse.coo_matrix((vv,(rr,cc)),shape=(len(docs),len(self.vocab)),dtype=float).tocsr()
        df=np.asarray((tf>0).sum(axis=0)).ravel();idf=np.log(1+(len(docs)-df+.5)/(df+.5))
        co=tf.tocoo();den=co.data+1.2*(1-.75+.75*length[co.row]/avg)
        self.bm=sparse.coo_matrix((idf[co.col]*co.data*2.2/den,(co.row,co.col)),shape=tf.shape).tocsr()

    def search(self,plan,structured):
        counts=Counter(tokens(plan['rewritten_query']))
        if structured:
            for f in ('domain','task','method'):
                for t in plan.get(f+'_terms',[]):counts.update(tokens(t))
        q=np.zeros(len(self.vocab))
        for t,count in counts.items():
            if t in self.vocab:q[self.vocab[t]]=min(count,3)
        score=np.asarray(self.bm@q).ravel();mx=float(score.max())
        if mx:score/=mx
        if not structured:return score,mx
        def cov(ts,doc):
            return len(ts&doc)/len(ts) if ts else 0.
        must_terms=[set(tokens(t)) for t in plan.get('must_terms',[])]
        should_terms=[set(tokens(t)) for t in plan.get('should_terms',[])]
        exclude_terms=[set(tokens(t)) for t in plan.get('exclude_terms',[])]
        typed_terms=[(f,set(tokens(t))) for f in ('domain','task','method') for t in plan.get(f+'_terms',[])]
        for i,doc in enumerate(self.sets):
            must=[cov(t,doc) for t in must_terms]
            should=[cov(t,doc) for t in should_terms]
            exc=[cov(t,doc) for t in exclude_terms]
            if must:score[i]+=.08*np.mean(np.log(np.maximum(must,1e-6)))
            if should:score[i]+=.03*np.mean(should)
            if exc:score[i]-=.12*max(exc)
            typed=[cov(t,self.fieldsets[i][f]) for f,t in typed_terms]
            if typed:score[i]+=.04*np.mean(typed)
        return score,mx


def rank(pool,score,bonus,neighbors,bw,pw,limit=None):
    # Stable score/id order; penalty matches existing direct-neighbor diversity idea.
    remaining=sorted(pool);selected=[]
    if not pw:return sorted(remaining,key=lambda i:-(score[i]+bw*bonus[i]))[:limit]
    while remaining and (limit is None or len(selected)<limit):
        best=max(remaining,key=lambda i:score[i]+bw*bonus[i]-pw*any(s in neighbors[i] for s in selected))
        selected.append(best);remaining.remove(best)
    return selected


def main():
    torch.set_num_threads(1);start=time.perf_counter()
    src=ROOT/'output/balanced_benchmark_20261117';out=src/'lexical_dtm_graph';data=src/'data'
    papers=read(data/'arxiv_master_final.json');ids=[str(p['paper_id']) for p in papers];local={p['node_idx']:i for i,p in enumerate(papers)}
    idx=Index(papers)
    edges=torch.load(data/'paper_paper_edges.pt',weights_only=True,map_location='cpu').T.tolist()
    graph=Graph(None,ids,relations=('citation',),citation_edges=[(ids[local[s]],ids[local[t]]) for s,t in edges])
    neighbors=[set(graph.citation.indices[graph.citation.indptr[i]:graph.citation.indptr[i+1]]) for i in range(len(ids))]
    setup=time.perf_counter()-start
    qs=[q for q in read(data/'queries.json') if q['split']=='test'];js={j['query_id']:j for j in read(data/'judgments.json')};plans=read(data/'query_intents.gpt-5_6-luna.json')['intents']
    variants={'rewrite_bm25':(False,0.,0.),'structured':(True,0.,0.),'graph_bonus':(True,.1,0.),'graph_penalty':(True,0.,.1),'graph_both':(True,.1,.1)}
    write(out/'manifest.json',dict(corpus=len(ids),query_count=len(qs),variants=variants,model_loading=False,embedding_usage=False,query_plans='cached LLM plans; API time excluded',bm25=dict(k1=1.2,b=.75),graph='undirected citation PPR; top50 lexical seeds; restart .2; max-normalized bonus',warnings=['Labels originally generated with DTM visible; may favor DTM-based methods.','Fixed-pool metrics do not measure full-corpus retrieval quality.','Full-corpus judged fraction and known-positive coverage are diagnostics only.','Single-host timings include Python scoring; no concurrent server load.'],hardware=platform.processor()))
    rows=[];full=[];timings=[]
    for q in qs:
        qid=q['query_id'];plan=plans[qid];j=js[qid];labels={ids[local[int(v['node_idx'])]]:v['relevance'] for v in j['labels']};cand=[local[int(n)] for n in j['candidate_node_indices']]
        t=time.perf_counter();simple,_=idx.search(plan,False);structured,signal=idx.search(plan,True);lexms=(time.perf_counter()-t)*1000
        for protocol in ('pooled','pooled_without_anchor'):
            anchor=local[int(q['anchor_node_idx'])] if protocol.endswith('without_anchor') else None
            allowed=[i for i in range(len(ids)) if i!=anchor];pool=[i for i in cand if i!=anchor];lab={p:g for p,g in labels.items() if anchor is None or p!=ids[anchor]}
            t=time.perf_counter();seed=np.zeros(len(ids));top=sorted(allowed,key=lambda i:-structured[i])[:50]
            if signal:
                v=structured[top];v=np.exp((v-v.max())/.05);seed[top]=v/v.sum();mass,_,ok=graph.diffuse(seed,True,False);assert ok;bonus=mass/mass.max()
            else:bonus=np.zeros(len(ids))
            pprms=(time.perf_counter()-t)*1000
            for name,(use,bw,pw) in variants.items():
                score=structured if use else simple
                ranked=rank(pool,score,bonus,neighbors,bw,pw)
                for k in (5,10,20):rows.append(dict(query_id=qid,anchor=q['anchor_node_idx'],protocol=protocol,arm=name,k=k,**metrics([ids[i] for i in ranked],lab,k)))
                # Restrict greedy work to initial top 200 by combined score: exact
                # first 20 for a nonnegative fixed penalty <= .1 is not guaranteed;
                # explicitly a practical retrieval pipeline, not fixed-pool path.
                t=time.perf_counter();short=rank(allowed,score,bonus,neighbors,bw,0.,200);fr=rank(short,score,bonus,neighbors,bw,pw,20);rms=(time.perf_counter()-t)*1000
                positive={p for p,g in lab.items() if g>0};topids=[ids[i] for i in fr[:10]]
                full.append(dict(query_id=qid,protocol=protocol,arm=name,top10=topids,judged_fraction=sum(p in lab for p in topids)/len(topids),known_positive_coverage=len(set(topids)&positive)/len(positive) if positive else 0.))
                timings.append(dict(query_id=qid,protocol=protocol,arm=name,lexical_ms=lexms,ppr_ms=pprms if bw else 0.,ranking_ms=rms,total_ms=lexms+(pprms if bw else 0.)+rms))
        print('Query',len({r['query_id'] for r in rows}),flush=True)
    summary={}
    previous=read(src/'natural_language/summary.json')
    for protocol in ('pooled','pooled_without_anchor'):
        result={}
        for arm in variants:
            rr=[r for r in rows if r['protocol']==protocol and r['arm']==arm and r['k']==10];ff=[r for r in full if r['protocol']==protocol and r['arm']==arm];tt=[r['total_ms'] for r in timings if r['protocol']==protocol and r['arm']==arm]
            result[arm]=dict(metrics_at_10={m:float(np.mean([r[m] for r in rr])) for m in ('ndcg','recall','strong_recall','mrr')},full_corpus_judged_fraction=float(np.mean([r['judged_fraction'] for r in ff])),known_positive_coverage=float(np.mean([r['known_positive_coverage'] for r in ff])),cpu_ms_median=float(np.median(tt)),cpu_ms_p95=float(np.quantile(tt,.95)))
        summary[protocol]=dict(variants=result,stored_embedding_baselines={arm:previous[protocol]['metrics'][arm]['10'] for arm in ('combined','intent')})
    summary['resources']=dict(setup_seconds=setup,dtm_bm25_sparse_bytes=sum(a.nbytes for a in (idx.bm.data,idx.bm.indices,idx.bm.indptr)),citation_transition_sparse_bytes=sum(a.nbytes for a in (graph.citation.data,graph.citation.indices,graph.citation.indptr)),vocabulary=len(idx.vocab),notes='Sparse arrays only; excludes Python objects, runtime, and metadata RAM.')
    write(out/'metrics.json',rows);write(out/'full_corpus_diagnostics.json',full);write(out/'timings.json',timings);write(out/'summary.json',summary)
    print(summary)


if __name__=='__main__':main()
