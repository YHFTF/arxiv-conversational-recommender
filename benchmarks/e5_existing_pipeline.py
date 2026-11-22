"""Frozen E5 top-ten + existing CPU pipeline ablation; no retraining/tuning."""
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import torch

from .core import Graph, digest, file_hash, metrics, read, write
from .lexical_dtm_graph import Index

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code/retrieval'))
from dtm_graph_search import DTMIndex
from dtm_rank_features import Features
from query_constraints import GuardedSearch


def merge_ranks(guarded, semantic):
    """Equal RRF of existing final rank and E5 rank, after existing eligibility.

    Keep the existing evidence tier. Never treat a missing embedding feature
    as a feature that the trained lexical ranker has learned.
    """
    sr = {int(i): r for r, i in enumerate(semantic, 1)}
    fused = []
    for r, row in enumerate(guarded, 1):
        row = dict(row)
        row['e5_rank'] = sr.get(row['index'])
        row['hybrid_score'] = 1 / (60 + r)
        if row['e5_rank'] is not None:
            row['hybrid_score'] += 1 / (60 + row['e5_rank'])
        fused.append(row)
    return sorted(fused, key=lambda r: (
        -sum(v == 'reported_in_abstract' for v in r['conditions'].values()),
        -r['hybrid_score'], r['index']))


def main():
    torch.set_num_threads(4)
    src = ROOT / 'output/balanced_benchmark_20261117'
    cache = ROOT / 'output/e5_small'
    out = cache / 'existing_pipeline'
    papers = read(src / 'data/arxiv_master_final.json')
    ids = [str(p['paper_id']) for p in papers]
    local = {int(p['node_idx']): i for i, p in enumerate(papers)}
    dm = read(cache / 'documents_manifest.json')
    assert dm['paper_ids'] == ids
    qs = [q for q in read(src / 'data/queries.json') if q['split'] == 'test']
    live = read(cache / 'live_results.json')
    inputs = [q['query'] for q in qs] + [q['query'] for q in live]
    assert digest(inputs) == read(cache / 'evaluation_manifest.json')['query_hash']
    dv = np.load(cache / 'documents.npz')['vectors']
    qv = np.load(cache / 'query_vectors.npz')['vectors']
    assert dv.shape == (len(ids), 384) and len(qv) == len(inputs)
    edges = torch.load(src / 'data/paper_paper_edges.pt', weights_only=True,
                       map_location='cpu').T.tolist()
    graph = Graph(None, ids, relations=('citation',), citation_edges=[
        (ids[local[s]], ids[local[t]]) for s, t in edges])
    extractor = Features(papers, Index(papers), DTMIndex(papers), graph)
    selected = read(src / 'learned_dtm_graph/selected.json')
    model_path = src / 'learned_dtm_graph' / selected['model_file']
    consensus = read(src / 'route_consensus/selected.json')['strength']
    engine = GuardedSearch(papers, extractor, joblib.load(model_path),
                           consensus_strength=consensus)
    write(out / 'protocol.json', dict(
        date='2026-11-18', depth=10, model_sha256=file_hash(model_path),
        document_signature=dm['signature'], consensus_strength=consensus,
        fusion='equal RRF k=60, retain existing evidence tier',
        tuning=False, retraining=False, llm_calls=False,
        notes=[
            'Same existing 48-question fixed silver-label pools; source excluded.',
            'Pooled top ten is selected within judged pool, not global top ten.',
            'Live results use the whole 16000-document corpus.',
            'Top-ten-first filter has no backfill; may return fewer than three.',
            'Hybrid unions E5 top ten with existing candidates before eligibility.',
            'Original question E5 vectors cached; existing glossary query rules retained.',
            'No extra graph reseeding from E5; existing DTM-seeded PPR retained.',
            'Timings exclude cached query encoding and initialization.']))
    js = {j['query_id']: j for j in read(src / 'data/judgments.json')}
    rows = []
    live_rows = []
    timings = []
    for qi, query in enumerate(inputs):
        start = time.perf_counter()
        score = dv @ qv[qi]
        if qi < len(qs):
            q = qs[qi]
            excluded = local[int(q['anchor_node_idx'])]
            j = js[q['query_id']]
            pool = [local[int(n)] for n in j['candidate_node_indices']
                    if int(n) != int(q['anchor_node_idx'])]
        else:
            excluded = None
            pool = None
        available = pool if pool is not None else range(len(ids))
        semantic = sorted(available, key=lambda i: (-float(score[i]), i))
        top10 = semantic[:10]
        old = engine.search(query, top_k=len(ids), excluded=excluded,
                            candidate_pool=pool)['results']
        reranked = engine.search(query, top_k=10, excluded=excluded,
                                 candidate_pool=top10)['results']
        union = sorted(set(top10) | {r['index'] for r in old})
        combined = engine.search(query, top_k=len(ids), excluded=excluded,
                                  candidate_pool=union)['results']
        hybrid = merge_ranks(combined, top10)
        arms = dict(existing=[r['index'] for r in old], e5=semantic,
                    e5_top10_existing=[r['index'] for r in reranked],
                    e5_existing_union=[r['index'] for r in hybrid])
        timings.append((time.perf_counter() - start) * 1000)
        if qi < len(qs):
            labels = {ids[local[int(v['node_idx'])]]: v['relevance']
                      for v in j['labels'] if int(v['node_idx']) != int(q['anchor_node_idx'])}
            for arm, ranked in arms.items():
                pids = [ids[i] for i in ranked]
                rows.append(dict(query_id=q['query_id'], anchor=q['anchor_node_idx'],
                    arm=arm, **metrics(pids, labels, 3),
                    precision3=sum(labels.get(p, 0) > 0 for p in pids[:3]) / 3,
                    strong_precision3=sum(labels.get(p, 0) == 2 for p in pids[:3]) / 3,
                    returned3=len(pids) >= 3))
        else:
            def export(indices):
                return [dict(paper_id=ids[i], title=papers[i]['title'],
                             abstract=papers[i]['abstract']) for i in indices[:3]]
            live_rows.append(dict(name=live[qi-len(qs)]['name'], query=query,
                results={arm: export(ix) for arm, ix in arms.items()},
                e5_top10_survivors=len(reranked),
                hybrid_details=[{k: v for k, v in r.items() if k != 'index'}
                                for r in hybrid[:3]]))
        print(f'Finished {qi+1}/{len(inputs)}', flush=True)
    summary = {arm: {m: float(np.mean([r[m] for r in rows if r['arm'] == arm]))
                         for m in ('ndcg', 'precision3', 'strong_precision3', 'returned3')}
               for arm in arms}
    previous = read(src / 'route_consensus/test_metrics.json')
    previous = {r['query_id']: r for r in previous if r['arm'] == 'soft_0.015'}
    for r in rows:
        if r['arm'] == 'existing':
            assert abs(r['ndcg'] - previous[r['query_id']]['ndcg']) < 1e-10
    summary['baseline_reproduction'] = 'passed'
    paired = {}
    for control in ('existing', 'e5'):
        base = {r['query_id']: r for r in rows if r['arm'] == control}
        for arm in ('e5_top10_existing', 'e5_existing_union'):
            after = [r for r in rows if r['arm'] == arm]
            ds = np.array([r['ndcg'] - base[r['query_id']]['ndcg'] for r in after])
            anchors = np.array([r['anchor'] for r in after]); groups = np.unique(anchors)
            rng = np.random.default_rng(42)
            draws = [np.concatenate([ds[anchors == g] for g in rng.choice(groups, len(groups))]).mean()
                     for _ in range(2000)]
            paired[arm + '_minus_' + control] = dict(delta=float(ds.mean()),
                ci95=np.quantile(draws, [.025, .975]).tolist(),
                win_tie_loss=[int((ds > 1e-10).sum()), int((abs(ds) <= 1e-10).sum()), int((ds < -1e-10).sum())])
    summary['paired_ndcg3'] = paired
    summary['live_returned3'] = {
        arm: sum(len(x['results'][arm]) == 3 for x in live_rows) for arm in arms}
    summary['live_query_count'] = len(live_rows)
    summary['all_four_arms_ms_median_cached_queries'] = float(np.median(timings))
    write(out / 'metrics.json', rows)
    write(out / 'live_results.json', live_rows)
    write(out / 'summary.json', summary)
    print(summary, flush=True)


if __name__ == '__main__':
    main()
