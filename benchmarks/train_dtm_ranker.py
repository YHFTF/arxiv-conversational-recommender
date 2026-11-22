"""Train on train labels; choose small ranker on validation; freeze before test."""
import argparse,sys,time
from pathlib import Path
import joblib
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor,ExtraTreesRegressor
from sklearn.linear_model import Ridge
from .core import Graph,read,write,metrics,file_hash
from .lexical_dtm_graph import Index
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code/retrieval'))
from dtm_graph_search import DTMIndex
from dtm_rank_features import Features,FEATURES


def main(stage):
    torch.set_num_threads(1)
    src=ROOT/'output/balanced_benchmark_20261117';data=src/'data';out=src/'learned_dtm_graph';out.mkdir(exist_ok=True)
    if stage=='features':
        papers=read(data/'arxiv_master_final.json');ids=[str(p['paper_id']) for p in papers];local={p['node_idx']:i for i,p in enumerate(papers)}
        legacy=Index(papers);index=DTMIndex(papers)
        edges=torch.load(data/'paper_paper_edges.pt',weights_only=True,map_location='cpu').T.tolist();graph=Graph(None,ids,relations=('citation',),citation_edges=[(ids[local[s]],ids[local[t]]) for s,t in edges])
        extractor=Features(papers,legacy,index,graph);queries=read(data/'queries.json');js={j['query_id']:j for j in read(data/'judgments.json')}
        splits={s:{q['anchor_node_idx'] for q in queries if q['split']==s} for s in ('train','validation','test')}
        assert all(not(splits[a]&splits[b]) for a,b in (('train','validation'),('train','test'),('validation','test')))
        arrays=[];metadata=[]
        for n,q in enumerate(queries):
            j=js[q['query_id']];anchor=local[int(q['anchor_node_idx'])];candidates=sorted(local[int(v)] for v in j['candidate_node_indices'] if int(v)!=q['anchor_node_idx'])
            state=extractor.prepare(q['query'],excluded=anchor);x=extractor.rows(state,candidates)
            labels={local[int(v['node_idx'])]:v['relevance'] for v in j['labels']}
            arrays.append(x);metadata.extend(dict(query_id=q['query_id'],anchor=q['anchor_node_idx'],split=q['split'],paper_id=ids[i],grade=labels[i],baseline=float(state['norm'][i])) for i in candidates)
            if (n+1)%50==0:print('Features',n+1,'/',len(queries),flush=True)
        np.savez_compressed(out/'signals.npz',X=np.concatenate(arrays));write(out/'signal_rows.json',metadata)
        write(out/'protocol.json',dict(features=FEATURES,no_embedding=True,no_online_llm=True,training='train labels only; source anchors excluded; no source overlap across splits',selection='validation anchor-excluded NDCG@10',test='report once after freezing; previously explored benchmark, not new holdout',label_notes='Existing LLM labels; original judge saw DTM; not human gold'))
        return
    rows=read(out/'signal_rows.json');x=np.load(out/'signals.npz')['X'];train=np.array([r['split']=='train' for r in rows]);validation=np.array([r['split']=='validation' for r in rows]);test=np.array([r['split']=='test' for r in rows]);y=np.array([2**r['grade']-1 for r in rows],dtype=float)
    def evaluate(pred,mask):
        selected=[(r,float(pred[i])) for i,r in enumerate(rows) if mask[i]];groups={}
        for r,p in selected:groups.setdefault(r['query_id'],[]).append((r,p))
        result=[]
        for qid,group in groups.items():
            ranked=sorted(group,key=lambda v:-v[1]);labels={r['paper_id']:r['grade'] for r,_ in group}
            result.append(dict(query_id=qid,anchor=group[0][0]['anchor'],**metrics([r['paper_id'] for r,_ in ranked],labels,10)))
        return result
    if stage=='train':
        models={
          'ridge':Ridge(alpha=10.),
          'hist_small':HistGradientBoostingRegressor(max_iter=80,max_leaf_nodes=7,l2_regularization=10.,learning_rate=.05,early_stopping=False,random_state=42),
          'hist_medium':HistGradientBoostingRegressor(max_iter=120,max_leaf_nodes=15,l2_regularization=10.,learning_rate=.05,early_stopping=False,random_state=42),
          'extra_trees':ExtraTreesRegressor(n_estimators=120,max_depth=8,min_samples_leaf=20,max_features=.8,n_jobs=1,random_state=42)}
        trials=[]
        for name,model in models.items():
            model.fit(x[train],y[train]);pred=np.zeros(len(rows));pred[validation]=model.predict(x[validation]);v=evaluate(pred,validation);score=float(np.mean([r['ndcg'] for r in v]));trials.append(dict(name=name,validation_ndcg=score));joblib.dump(model,out/(name+'.joblib'));print(name,score,flush=True)
        best=max(trials,key=lambda v:v['validation_ndcg']);write(out/'validation_trials.json',trials);write(out/'selected.json',dict(**best,model_file=best['name']+'.joblib',model_sha256=file_hash(out/(best['name']+'.joblib')),train_queries=len({r['query_id'] for r in rows if r['split']=='train'}),validation_queries=len({r['query_id'] for r in rows if r['split']=='validation'})))
        return
    selected=read(out/'selected.json');assert file_hash(out/selected['model_file'])==selected['model_sha256'];model=joblib.load(out/selected['model_file']);pred=np.zeros(len(rows));start=time.perf_counter();pred[test]=model.predict(x[test]);predict_ms=(time.perf_counter()-start)*1000
    result=evaluate(pred,test);baseline=evaluate(np.array([r['baseline'] for r in rows]),test)
    old={r['query_id']:r for r in read(src/'graph_without_query_llm/metrics.json') if r['protocol']=='pooled_without_anchor' and r['arm']=='raw_seed_graph_only' and r['k']==10}
    assert all(abs(r['ndcg']-old[r['query_id']]['ndcg'])<1e-7 for r in baseline)
    a={r['query_id']:r for r in baseline};ds=np.array([r['ndcg']-a[r['query_id']]['ndcg'] for r in result]);anchors=np.array([r['anchor'] for r in result]);groups=np.unique(anchors);rng=np.random.default_rng(42);draws=[np.concatenate([ds[anchors==g] for g in rng.choice(groups,len(groups))]).mean() for _ in range(2000)]
    summary=dict(selected=selected,baseline={m:float(np.mean([r[m] for r in baseline])) for m in ('ndcg','recall','strong_recall','mrr')},learned={m:float(np.mean([r[m] for r in result])) for m in ('ndcg','recall','strong_recall','mrr')},paired_ndcg=dict(delta=float(ds.mean()),ci95=np.quantile(draws,[.025,.975]).tolist(),win_tie_loss=[int((ds>1e-10).sum()),int((abs(ds)<=1e-10).sum()),int((ds< -1e-10).sum())]),model_bytes=(out/selected['model_file']).stat().st_size,test_candidate_predict_ms=predict_ms,baseline_reproduction='passed')
    write(out/'test_metrics.json',result);write(out/'test_summary.json',summary);print(summary,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['features','train','test'],required=True);main(p.parse_args().stage)
