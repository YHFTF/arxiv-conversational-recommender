"""Offline citation bonus/diversity-penalty experiment; no service changes."""
import sys
from pathlib import Path
import numpy as np
import torch
from .core import Graph, read, write, metrics, digest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'code/retrieval'),str(ROOT/'code/embeddings')]
from run_nl_recommendation import load_catalog,relevance_scores


def main():
    torch.set_num_threads(2)
    src=ROOT/'output/balanced_benchmark_20261117';data=src/'data';emb=src/'embeddings';out=src/'graph_score_ablation'
    qs=[q for q in read(data/'queries.json') if q['split']=='test']
    js={j['query_id']:j for j in read(data/'judgments.json')}
    plans=read(data/'query_intents.gpt-5_6-luna.json')['intents'];qm=read(emb/'queries_manifest.json')
    assert qm['query_ids']==[q['query_id'] for q in qs]
    assert qm['query_hash']==digest([q['query'] for q in qs])
    papers,vectors,_=load_catalog(data/'arxiv_master_final.json',emb/'paper_text_embeddings.pt')
    ids=[p['paper_id'] for p in papers];local={p['node_idx']:i for i,p in enumerate(papers)}
    qv=np.load(emb/'queries.npz');tv={t:torch.from_numpy(v.copy()) for t,v in zip(qm['terms'],qv['terms'])}
    edges=torch.load(data/'paper_paper_edges.pt',weights_only=True).T.tolist()
    graph=Graph(None,ids,relations=('citation',),citation_edges=[(ids[local[s]],ids[local[t]]) for s,t in edges])
    adjacency=graph.citation.copy();adjacency.data[:]=1
    neighbors=[set(adjacency.indices[adjacency.indptr[i]:adjacency.indptr[i+1]]) for i in range(len(ids))]
    variants={'baseline':(0.,0.)}
    for w in (.02,.04,.08):
        for name,b,p in [('bonus',w,0.),('penalty',0.,w),('both',w,w)]:variants[f'{name}_{w:.2f}']=(b,p)
    write(out/'manifest.json',dict(variants=variants,primary_weight=.04,bonus='weight * PPR / maximum corpus PPR',penalty='weight if candidate directly cites or is cited by any already selected paper',ppr_seed='Top 50 by current intent score, softmax temperature .05',restart=.2,protocol='fixed fully judged candidates; no candidate expansion',notes=['Exploratory test-set analysis; sensitivity weights not tuned on labels.','Anchor excluded from candidates and PPR seeds in anchor-excluded protocol.','Direct citation pair fraction is a structural statistic, not a semantic diversity metric.']))
    rows=[];rankings=[]
    for qi,q in enumerate(qs):
        j=js[q['query_id']];candidates=[local[int(n)] for n in j['candidate_node_indices']]
        labels={ids[local[int(v['node_idx'])]]:v['relevance'] for v in j['labels']}
        score=relevance_scores(plans[q['query_id']],torch.from_numpy(qv['rewritten'][qi].copy()),tv,papers,vectors,list(range(len(ids))),.04,.08,.03,.12,.30,.45,.04)[0].numpy()
        for protocol in ('pooled','pooled_without_anchor'):
            anchor=local[int(q['anchor_node_idx'])] if protocol=='pooled_without_anchor' else None
            allowed=[i for i in range(len(ids)) if i!=anchor];pool=[i for i in candidates if i!=anchor]
            top=sorted(allowed,key=lambda i:-float(score[i]))[:50];seed=np.zeros(len(ids));values=score[top].astype(float);values=np.exp((values-values.max())/.05);seed[top]=values/values.sum()
            mass,steps,converged=graph.diffuse(seed,True,False);assert converged
            bonus=mass/mass.max()
            lab={p:g for p,g in labels.items() if anchor is None or p!=ids[anchor]}
            for name,(bw,pw) in variants.items():
                remaining=list(pool);selected=[]
                while remaining:
                    best=max(remaining,key=lambda i:float(score[i])+bw*bonus[i]-pw*any(n in neighbors[i] for n in selected))
                    selected.append(best);remaining.remove(best)
                ranked=[ids[i] for i in selected]
                rankings.append(dict(query_id=q['query_id'],protocol=protocol,arm=name,ranked_paper_ids=ranked))
                for k in (5,10,20):
                    topk=selected[:k];pairs=sum(j in neighbors[i] for ix,i in enumerate(topk) for j in topk[ix+1:]);den=len(topk)*(len(topk)-1)/2
                    rows.append(dict(query_id=q['query_id'],anchor=q['anchor_node_idx'],protocol=protocol,arm=name,k=k,citation_pair_fraction=pairs/den if den else 0.,**metrics(ranked,lab,k)))
    old={(r['query_id'],r['protocol'],r['k']):r for r in read(src/'natural_language/metrics.json') if r['arm']=='intent'}
    for r in rows:
        if r['arm']=='baseline':assert all(abs(r[m]-old[(r['query_id'],r['protocol'],r['k'])][m])<1e-7 for m in ('ndcg','recall','mrr'))
    summary={}
    for protocol in ('pooled','pooled_without_anchor'):
        by={arm:{r['query_id']:r for r in rows if r['protocol']==protocol and r['arm']==arm and r['k']==10} for arm in variants}
        result={}
        for arm,d in by.items():
            avg={m:float(np.mean([r[m] for r in d.values()])) for m in ('ndcg','recall','strong_recall','mrr','citation_pair_fraction')}
            ds=np.array([d[q]['ndcg']-by['baseline'][q]['ndcg'] for q in d]);anchors=np.array([r['anchor'] for r in d.values()]);groups=np.unique(anchors);rng=np.random.default_rng(42)
            draws=[np.concatenate([ds[anchors==g] for g in rng.choice(groups,len(groups))]).mean() for _ in range(2000)]
            result[arm]=dict(metrics_at_10=avg,ndcg_delta=float(ds.mean()),ci95=np.quantile(draws,[.025,.975]).tolist(),wins=int((ds>1e-10).sum()),ties=int((abs(ds)<=1e-10).sum()),losses=int((ds< -1e-10).sum()))
        summary[protocol]=result
    write(out/'metrics.json',rows);write(out/'rankings.json',rankings);write(out/'summary.json',summary)
    for protocol,s in summary.items():
        print(protocol)
        for arm in ('baseline','bonus_0.04','penalty_0.04','both_0.04'):print(arm,s[arm])
    print('Baseline reproduction passed; saved',out)


if __name__=='__main__':main()
