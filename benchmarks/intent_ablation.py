"""Offline component ablation on the restored fixed-candidate test set."""
import itertools
import sys
from pathlib import Path
import numpy as np
import torch
from .core import read, write, metrics, digest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code/retrieval'))
sys.path.insert(0, str(ROOT / 'code/embeddings'))
from run_nl_recommendation import load_catalog, query_signal_scores, dtm_bonus


def main(document_vectors='combined'):
    torch.set_num_threads(2)
    source = ROOT / 'output/balanced_benchmark_20261117'
    dest = source / ('intent_ablation' if document_vectors == 'combined' else 'intent_ablation_dtm')
    data, emb = source / 'data', source / 'embeddings'
    qs = [q for q in read(data/'queries.json') if q['split'] == 'test']
    js = {j['query_id']: j for j in read(data/'judgments.json')}
    plans = read(data/'query_intents.gpt-5_6-luna.json')['intents']
    qm = read(emb/'queries_manifest.json')
    assert qm['query_ids'] == [q['query_id'] for q in qs]
    assert qm['query_hash'] == digest([q['query'] for q in qs])
    papers, vectors, _ = load_catalog(data/'arxiv_master_final.json', emb/'paper_text_embeddings.pt')
    if document_vectors == 'dtm':
        manifest = read(emb/'dtm_manifest.json')
        assert manifest['paper_ids'] == [p['paper_id'] for p in papers]
        with np.load(emb/'dtm.npz') as f:
            vectors = torch.nn.functional.normalize(torch.from_numpy(f['vectors'].copy()).float(),dim=1)
    local = {p['node_idx']: i for i,p in enumerate(papers)}
    ids = [p['paper_id'] for p in papers]
    qv = np.load(emb/'queries.npz')
    tv = {t: torch.from_numpy(v.copy()) for t,v in zip(qm['terms'],qv['terms'])}
    flags = ['rewrite','must','should','exclude','dtm']
    variants = {''.join(map(str,bits)):dict(zip(flags,bits)) for bits in itertools.product((0,1),repeat=5)}
    rows=[]; examples=[]
    for qi,q in enumerate(qs):
        plan=plans[q['query_id']]; j=js[q['query_id']]
        candidates=[local[int(n)] for n in j['candidate_node_indices']]
        labels={ids[local[int(v['node_idx'])]]:v['relevance'] for v in j['labels']}
        cv=vectors[candidates]
        base={k:torch.mv(cv,torch.nn.functional.normalize(torch.from_numpy(qv[k][qi].copy()).float(),dim=0)) for k in ('raw','rewritten')}
        adjustments={}
        for name,weights in [('must',(.08,0,0)),('should',(0,.03,0)),('exclude',(0,0,.12))]:
            adjustments[name]=query_signal_scores(plan,tv,cv,*weights,.30,.45,.04)[0]
        adjustments['dtm']=.04*dtm_bonus(plan,papers,candidates)
        for key,f in variants.items():
            scores=base['rewritten' if f['rewrite'] else 'raw'].clone()
            for name in adjustments:
                if f[name]: scores+=adjustments[name]
            # Match production rerank: stable candidate order for tied scores.
            order=sorted(range(len(candidates)),key=lambda i:-float(scores[i]))
            for protocol in ('pooled','pooled_without_anchor'):
                excluded=ids[local[int(q['anchor_node_idx'])]] if protocol.endswith('without_anchor') else None
                lab={p:v for p,v in labels.items() if p!=excluded}
                ranked=[ids[candidates[i]] for i in order if ids[candidates[i]]!=excluded]
                for k in (5,10,20):
                    rows.append(dict(query_id=q['query_id'],anchor=q['anchor_node_idx'],protocol=protocol,variant=key,k=k,**metrics(ranked,lab,k)))
        examples.append(dict(query_id=q['query_id'],query=q['query'],rewritten_query=plan['rewritten_query'],must_terms=plan['must_terms'],exclude_terms=plan['exclude_terms']))
    # Verify endpoints reproduce previously stored results before interpretation.
    old=read(source/'natural_language/metrics.json')
    oldmap={(r['query_id'],r['protocol'],r['arm'],r['k']):r for r in old}
    for r in rows:
        if r['variant'] not in ('00000','11111'): continue
        if document_vectors == 'dtm' and r['variant'] == '11111': continue
        baseline = ('combined' if document_vectors == 'combined' else 'dtm') if r['variant']=='00000' else 'intent'
        before=oldmap[(r['query_id'],r['protocol'],baseline,r['k'])]
        assert all(abs(r[m]-before[m])<1e-7 for m in ('ndcg','recall','mrr')), (r,before)
    summary={}
    for protocol in ('pooled','pooled_without_anchor'):
        sub=[r for r in rows if r['protocol']==protocol and r['k']==10]
        by={v:{r['query_id']:r for r in sub if r['variant']==v} for v in variants}
        av={v:{m:float(np.mean([r[m] for r in d.values()])) for m in ('ndcg','recall','strong_recall','mrr')} for v,d in by.items()}
        pairs=[('00000','10000')]+[('10000','1'+''.join('1' if i==j else '0' for i in range(4))) for j in range(4)]+[('11111'[:i]+'0'+'11111'[i+1:],'11111') for i in range(5)]
        comparisons=[]
        for a,b in pairs:
            groups=sorted(set(r['anchor'] for r in by[a].values()))
            result=dict(before=a,after=b)
            for m in ('ndcg','recall'):
                ds=np.array([by[b][q][m]-by[a][q][m] for q in by[a]])
                anchors=np.array([r['anchor'] for r in by[a].values()])
                gv=[ds[anchors==g] for g in groups]; rng=np.random.default_rng(42)
                draws=[float(np.concatenate([gv[i] for i in rng.integers(len(gv),size=len(gv))]).mean()) for _ in range(2000)]
                result[m]=dict(delta=float(ds.mean()),ci95=np.quantile(draws,[.025,.975]).tolist(),wins=int((ds>1e-10).sum()),ties=int((abs(ds)<=1e-10).sum()),losses=int((ds< -1e-10).sum()))
            comparisons.append(result)
        summary[protocol]=dict(metrics_at_10=av,comparisons=comparisons)
    write(dest/'metrics.json',rows)
    write(dest/'summary.json',summary)
    write(dest/'queries.json',examples)
    write(dest/'manifest.json',dict(document_vectors=document_vectors,flags=flags,variants=variants,queries=len(qs),bootstrap_groups=24,bootstrap_repetitions=2000,endpoint_validation='passed (DTM: baseline only)' if document_vectors=='dtm' else 'passed',notes=['Fixed judged candidates; no candidate-generation ablation.','Rewrite includes appended Required concepts; cannot separate using available cached vectors.','Exploratory test-set analysis; no tuning or independent confirmation.']))
    for p,s in summary.items():
        print(p)
        for v in ['00000','10000','11000','10100','10010','10001','11111','10111','11011','11101','11110']:
            print(v, s['metrics_at_10'][v])
        print('comparisons',s['comparisons'])


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--document-vectors',choices=['combined','dtm'],default='combined')
    main(parser.parse_args().document_vectors)
