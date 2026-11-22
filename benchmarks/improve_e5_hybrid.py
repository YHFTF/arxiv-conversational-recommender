"""Train-only hybrid learning, validation-only selection, then frozen test."""
import argparse
import itertools
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor

from .core import Graph, digest, file_hash, metrics, read, write
from .lexical_dtm_graph import Index

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'code/retrieval'))
sys.path.insert(0, str(ROOT/'code/embeddings'))
from dtm_graph_search import DTMIndex
from dtm_rank_features import Features
from query_constraints import GuardedSearch
from hybrid_search import HybridSearch, FEATURE_NAMES, improved_plan
from e5_small import E5Small

SRC = ROOT/'output/balanced_benchmark_20261117'
OUT = ROOT/'output/e5_small/improved'


def load_engine(encoder=None, config=None, model=None):
    papers = read(SRC/'data/arxiv_master_final.json')
    ids = [str(p['paper_id']) for p in papers]
    local = {int(p['node_idx']): i for i, p in enumerate(papers)}
    edges = torch.load(SRC/'data/paper_paper_edges.pt', weights_only=True, map_location='cpu').T.tolist()
    graph = Graph(None, ids, relations=('citation',), citation_edges=[(ids[local[s]], ids[local[t]]) for s, t in edges])
    selected = read(SRC/'learned_dtm_graph/selected.json')
    path = SRC/'learned_dtm_graph'/selected['model_file']
    assert file_hash(path) == selected['model_sha256']
    legacy = GuardedSearch(papers, Features(papers, Index(papers), DTMIndex(papers), graph),
        joblib.load(path), consensus_strength=.015)
    dm = read(ROOT/'output/e5_small/documents_manifest.json')
    assert dm['paper_ids'] == ids
    engine = HybridSearch(legacy, np.load(ROOT/'output/e5_small/documents.npz')['vectors'],
        encoder=encoder, config=config, model=model)
    return engine, papers


def evaluate(records, config, model=None):
    engine = HybridSearch.__new__(HybridSearch)
    engine.config = config
    engine.model = model
    result = []
    for r in records:
        indices, _ = engine.rank(r['state'], r['pool'], config=config)
        pids = [r['ids'][i] for i in indices]
        labels = r['labels']
        result.append(dict(query_id=r['query_id'], anchor=r['anchor'],
            **metrics(pids, labels, 3),
            precision3=sum(labels.get(p, 0) > 0 for p in pids[:3])/3,
            strong_precision3=sum(labels.get(p, 0) == 2 for p in pids[:3])/3,
            returned3=len(pids) >= 3))
    return result


def average(rows):
    return {k: float(np.mean([r[k] for r in rows])) for k in
        ('ndcg', 'precision3', 'strong_precision3', 'returned3')}


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    qs = read(SRC/'data/queries.json')
    live = read(ROOT/'output/e5_small/live_results.json')
    inputs = [q['query'] for q in qs] + [q['query'] for q in live]
    canonical = [improved_plan(q).positive_query for q in inputs]
    signature = digest(dict(queries=inputs, canonical=canonical,
        documents=read(ROOT/'output/e5_small/documents_manifest.json')['signature']))
    path = OUT/'query_vectors.npz'
    if path.exists() and read(OUT/'queries_manifest.json')['signature'] == signature:
        qv = np.load(path)['vectors']
    else:
        encoder = E5Small('cpu', threads=4)
        qv = []
        for i, (q, c) in enumerate(zip(inputs, canonical)):
            qv.append(encoder.encode([q, c], query=True, batch_size=1).numpy())
            if (i+1) % 50 == 0:
                print('Encoded questions', i+1, '/', len(inputs), flush=True)
        qv = np.stack(qv)
        np.savez_compressed(path, vectors=qv)
        write(OUT/'queries_manifest.json', dict(signature=signature,
            queries=inputs, canonical=canonical))
        del encoder
    engine, papers = load_engine()
    ids = [str(p['paper_id']) for p in papers]
    local = {int(p['node_idx']): i for i, p in enumerate(papers)}
    js = {j['query_id']: j for j in read(SRC/'data/judgments.json')}
    splits = {s: {q['anchor_node_idx'] for q in qs if q['split'] == s}
              for s in ('train', 'validation', 'test')}
    assert all(not (splits[a] & splits[b]) for a, b in itertools.combinations(splits, 2))
    records = []
    for qi, query in enumerate(inputs):
        if qi < len(qs):
            q = qs[qi]
            excluded = local[int(q['anchor_node_idx'])]
            j = js[q['query_id']]
            pool = [local[int(n)] for n in j['candidate_node_indices'] if int(n) != int(q['anchor_node_idx'])]
            labels = {ids[local[int(v['node_idx'])]]: v['relevance'] for v in j['labels']
                      if int(v['node_idx']) != int(q['anchor_node_idx'])}
        else:
            excluded = None
            pool = []
            labels = {}
        state = engine.prepare(query, qv[qi], excluded=excluded)
        global_candidates = engine.candidates(state)
        keep = np.array(sorted(set(global_candidates) | set(pool)), dtype=int)
        mapping = {int(v): i for i, v in enumerate(keep)}
        compact = dict(plan=state['plan'], features=state['features'][keep],
            eligible=state['eligible'][keep], facets=state['facets'][keep],
            evidence=state['evidence'][keep],
            ranks={name: values[keep] for name, values in state['ranks'].items()})
        record = dict(query=query, state=compact, indices=keep,
            ids=[ids[i] for i in keep], pool=[mapping[i] for i in pool],
            candidates=[mapping[int(i)] for i in global_candidates], labels=labels)
        if qi < len(qs):
            record.update(query_id=q['query_id'], split=q['split'], anchor=q['anchor_node_idx'])
        else:
            record.update(query_id=live[qi-len(qs)]['name'], split='live', anchor=None)
        records.append(record)
        if (qi+1) % 25 == 0:
            print('Prepared signals', qi+1, '/', len(inputs), flush=True)
    torch.save(records, OUT/'signals.pt')
    write(OUT/'protocol.json', dict(date='2026-11-18', features=FEATURE_NAMES,
        signature=signature, counts={s: sum(r['split']==s for r in records)
        for s in ('train','validation','test','live')},
        training='train only; anchors excluded; source-disjoint splits',
        selection='validation NDCG@3, then precision@3, then strong precision@3',
        semantic_depth=100, lexical_depth=200, hard_filters='explicit exclusions only',
        topic_order='confirmed requested facet coverage, then requested evidence, then fused score; no positive-topic rejection',
        caveats=['Repeatedly observed silver-label fixed pools, not a new blind holdout.',
            'Test labels are not used to train or select parameters.',
            'Positive topic requirements are soft evidence, not guaranteed constraints.',
            'No online LLM, GPU inference, or new external API calls.']))
    print('Prepared all signals', flush=True)


def train():
    torch.set_num_threads(4)
    records = torch.load(OUT/'signals.pt', weights_only=False)
    training = [r for r in records if r['split']=='train']
    validation = [r for r in records if r['split']=='validation']
    xx, yy, weights = [], [], []
    for r in training:
        pool = [i for i in r['pool'] if r['state']['eligible'][i]]
        xx.append(r['state']['features'][pool])
        yy.extend(2**r['labels'][r['ids'][i]]-1 for i in pool)
        weights.extend([1/max(len(pool),1)]*len(pool))
    x = np.concatenate(xx)
    weights = np.asarray(weights); weights *= len(weights)/weights.sum()
    models = {}
    for name, leaves, count in [('hist7',7,80),('hist15',15,120)]:
        model = HistGradientBoostingRegressor(max_iter=count, max_leaf_nodes=leaves,
            l2_regularization=10, learning_rate=.05, early_stopping=False, random_state=42)
        model.fit(x, np.asarray(yy), sample_weight=weights)
        models[name] = model
        joblib.dump(model, OUT/(name+'.joblib'))
    trials = []
    for sw, cw, tp in itertools.product((.5,1.,2.), (0.,.5), (0.,.35,.7)):
        c = dict(depth=100, semantic_weight=sw, canonical_weight=cw,
            topic_penalty=tp, evidence_bonus=.03, learned_weight=0.)
        trials.append(dict(config=c, model=None, validation=average(evaluate(validation,c))))
    best_rrf = max(trials, key=lambda t: tuple(t['validation'][k] for k in ('ndcg','precision3','strong_precision3')))
    for name, model in models.items():
        for weight in (.25,.5,.75,1.):
            c = dict(best_rrf['config'], learned_weight=weight)
            trials.append(dict(config=c, model=name+'.joblib', validation=average(evaluate(validation,c,model))))
    chosen = max(trials, key=lambda t: tuple(t['validation'][k] for k in ('ndcg','precision3','strong_precision3')))
    if chosen['model']:
        chosen['model_sha256'] = file_hash(OUT/chosen['model'])
    chosen['protocol_sha256'] = file_hash(OUT/'protocol.json')
    write(OUT/'validation_trials.json', trials)
    write(OUT/'selected.json', chosen)
    print('FROZEN', chosen, flush=True)


def test():
    torch.set_num_threads(4)
    selected = read(OUT/'selected.json'); locked = file_hash(OUT/'selected.json')
    assert file_hash(OUT/'protocol.json') == selected['protocol_sha256']
    model = joblib.load(OUT/selected['model']) if selected['model'] else None
    if model is not None:
        assert file_hash(OUT/selected['model']) == selected['model_sha256']
    records = torch.load(OUT/'signals.pt', weights_only=False)
    rows = evaluate([r for r in records if r['split']=='test'], selected['config'], model)
    summary = dict(selected=selected, test=average(rows))
    old = read(ROOT/'output/e5_small/existing_pipeline/metrics.json')
    summary['paired'] = {}
    for arm in ('existing','e5','e5_existing_union'):
        base = {r['query_id']:r for r in old if r['arm']==arm}
        ds = np.array([r['ndcg']-base[r['query_id']]['ndcg'] for r in rows])
        anchors = np.array([r['anchor'] for r in rows]); groups = np.unique(anchors)
        rng = np.random.default_rng(42)
        draws = [np.concatenate([ds[anchors==g] for g in rng.choice(groups,len(groups))]).mean() for _ in range(2000)]
        summary['paired'][arm] = dict(delta=float(ds.mean()),ci95=np.quantile(draws,[.025,.975]).tolist(),
            win_tie_loss=[int((ds>1e-10).sum()),int((abs(ds)<=1e-10).sum()),int((ds< -1e-10).sum())])
    write(OUT/'test_metrics.json', rows)
    write(OUT/'test_summary.json', summary)
    engine, papers = load_engine(config=selected['config'], model=model)
    live = []
    ranker = HybridSearch.__new__(HybridSearch); ranker.config=selected['config'];ranker.model=model
    for r in records:
        if r['split'] != 'live':
            continue
        ranked, scores = ranker.rank(r['state'],r['candidates'])
        result = []
        plan = r['state']['plan']
        for i,score in zip(ranked[:3],scores[:3]):
            gi = int(r['indices'][i]); p=papers[gi];m=engine.metadata[gi]
            result.append(dict(paper_id=str(p['paper_id']),title=p['title'],abstract=p['abstract'],
                score=float(score),facet_coverage=float(r['state']['facets'][i]),
                missing_facets=sorted(set(plan.facets)-(set(m['facets'])|set(m['text_facets'])))))
        live.append(dict(name=r['query_id'],query=r['query'],plan=plan.to_dict(),results=result))
    write(OUT/'live_results.json',live)
    assert file_hash(OUT/'selected.json')==locked
    print(summary,flush=True)


def deploy():
    chosen = read(OUT/'selected.json')
    tested = read(OUT/'test_summary.json')
    if tested['selected'] != chosen or file_hash(OUT/'protocol.json') != chosen['protocol_sha256']:
        raise ValueError('Selected configuration has not completed matching evaluation')
    if chosen['model'] and file_hash(OUT/chosen['model']) != chosen['model_sha256']:
        raise ValueError('Selected model hash mismatch')
    write(OUT/'deployment.json', dict(version='e5_hybrid_v2',
        selected_sha256=file_hash(OUT/'selected.json'),
        document_signature=read(ROOT/'output/e5_small/documents_manifest.json')['signature'],
        source_hashes={p:file_hash(ROOT/p) for p in
                      ('code/retrieval/hybrid_search.py','code/embeddings/e5_small.py')},
        date='2026-11-18', selection='validation only; qualitative condition-order correction documented'))
    print('Activated evaluated hybrid configuration',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['prepare','train','test','deploy'],required=True)
    args=parser.parse_args()
    {'prepare':prepare,'train':train,'test':test,'deploy':deploy}[args.stage]()
