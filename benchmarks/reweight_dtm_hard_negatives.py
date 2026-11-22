"""Emphasize only already-judged train negatives found by real retrieval."""
from pathlib import Path
import joblib,numpy as np
from sklearn.base import clone
from .core import read,write,metrics,file_hash


def main():
    src=Path(__file__).resolve().parents[1]/'output/balanced_benchmark_20261117'
    original=src/'learned_dtm_graph';out=src/'guarded_dtm_graph'
    rows=read(original/'signal_rows.json');x=np.load(original/'signals.npz')['X'];y=np.array([2**r['grade']-1 for r in rows],dtype=float)
    mined=read(out/'known_retrieved_train.json');hard={(r['query_id'],r['paper_id']) for r in mined if r['grade']==0}
    train=np.array([r['split']=='train' for r in rows]);val=np.array([r['split']=='validation' for r in rows]);test=np.array([r['split']=='test' for r in rows])
    assert all(r['split']=='train' for r in rows if (r['query_id'],r['paper_id']) in hard)
    selected=read(original/'selected.json');base=joblib.load(original/selected['model_file'])
    def evaluate(model,mask):
        pred=model.predict(x[mask]);groups={}
        for r,p in zip([r for r,m in zip(rows,mask) if m],pred):groups.setdefault(r['query_id'],[]).append((r,p))
        result=[]
        for qid,g in groups.items():
            ranked=sorted(g,key=lambda v:-v[1]);labels={r['paper_id']:r['grade'] for r,_ in g}
            result.append(dict(query_id=qid,**metrics([r['paper_id'] for r,_ in ranked],labels,10)))
        return result
    def average(rs):return {m:float(np.mean([r[m] for r in rs])) for m in ('ndcg','recall','mrr')}
    initial=average(evaluate(base,val));best=base;bestval=initial['ndcg'];bestweight=1.;trials=[dict(weight=1.,validation=initial)]
    for weight in (2.,4.):
        weights=np.array([weight if (r['query_id'],r['paper_id']) in hard else 1. for r in rows])
        model=clone(base);model.fit(x[train],y[train],sample_weight=weights[train]);v=average(evaluate(model,val));trials.append(dict(weight=weight,validation=v))
        if v['ndcg']>bestval:best=model;bestval=v['ndcg'];bestweight=weight
    # Freeze before any test prediction; unknown retrieved documents never receive labels.
    joblib.dump(best,out/'ranker.joblib');write(out/'selected.json',dict(name='hard_negative_weighted' if bestweight!=1 else selected['name'],model_file='ranker.joblib',model_sha256=file_hash(out/'ranker.joblib'),negative_weight=bestweight,validation_ndcg=bestval,known_train_hard_negatives=len(hard),unknowns_used_as_negatives=0))
    write(out/'retraining_trials.json',trials)
    result=dict(initial_validation=initial,selected_validation_ndcg=bestval,selected_negative_weight=bestweight,hard_negative_pairs=len(hard),baseline_test=average(evaluate(base,test)),selected_test=average(evaluate(best,test)),notes='Feature weights trained on train only; validation selects weight; cached original signals used; report is fixed-pool model-only, separate from guarded live search.')
    write(out/'retraining_summary.json',result);print(result)


if __name__=='__main__':main()
