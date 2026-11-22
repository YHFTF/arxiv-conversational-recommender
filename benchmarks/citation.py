"""Citation prediction with fixed 80/10/10 pair splits and D/T/M attributes.
Run: python -m benchmarks.citation --help
"""
import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
KINDS = ('domain', 'task', 'method')


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def dtm_text(row):
    knowledge = row.get('knowledge', {})
    for kind in KINDS:
        if not isinstance(knowledge.get(kind), list) or any(not isinstance(v, str) for v in knowledge[kind]):
            raise ValueError(f'Invalid D/T/M: {row.get("node_idx")} {kind}')
    return '\n'.join(f'{kind.title()}: ' + '; '.join(knowledge[kind]) for kind in KINDS)


def split_edges(edges, seed):
    """Keep both directions of a pair together: symmetrizing train cannot leak test."""
    edges = sorted({(int(s), int(t)) for s, t in edges if s != t})
    groups = defaultdict(list)
    for s, t in edges:
        groups[min(s, t), max(s, t)].append((s, t))
    keys = sorted(groups)
    if len(keys) < 10:
        raise ValueError('At least 10 distinct citation pairs required')
    random.Random(seed).shuffle(keys)
    a, b = int(len(keys) * .8), int(len(keys) * .1)
    return {name: [edge for key in part for edge in groups[key]]
            for name, part in zip(('train', 'validation', 'test'), (keys[:a], keys[a:a+b], keys[a+b:]))}


def load_inputs(master_path, sample_path, citation_path):
    master = json.loads(master_path.read_text(encoding='utf-8'))
    nodes = [int(r['node_idx']) for r in master]
    if len(set(nodes)) != len(nodes) or len({str(r['paper_id']) for r in master}) != len(nodes):
        raise ValueError('Duplicate paper/node IDs')
    sample = torch.load(sample_path, map_location='cpu', weights_only=False)
    source = {int(n): i for i, n in enumerate(sample['indices'])}
    if not set(nodes) <= source.keys():
        raise ValueError('Master nodes absent from OGBN sample')
    order = [source[n] for n in nodes]
    x = sample['features'][order].float()
    labels = sample['labels'].reshape(-1)[order].tolist()
    raw = torch.load(citation_path, map_location='cpu', weights_only=False)
    if not isinstance(raw, torch.Tensor) or raw.ndim != 2 or raw.shape[0] != 2:
        raise ValueError('Expected citation tensor [2,E] in original node_idx space')
    local = {n: i for i, n in enumerate(nodes)}
    edges = [(local[s], local[t]) for s, t in raw.T.tolist() if s in local and t in local]
    return master, x, labels, edges, raw.shape[1] - len(edges)


def prepare_dtm(master, folder):
    import sys
    sys.path.insert(0, str(ROOT / 'code' / 'embeddings'))
    from local_embeddings import LocalEmbedder, IDENTITY
    texts = [dtm_text(r) for r in master]
    signature = fingerprint({'nodes': [r['node_idx'] for r in master], 'texts': texts, 'model': IDENTITY})
    bundle = folder / 'dtm.pt'
    if bundle.exists():
        saved = torch.load(bundle, map_location='cpu', weights_only=True)
        if saved['signature'] != signature:
            raise ValueError('D/T/M cache input/model mismatch; choose a new cache folder')
        vectors = saved['vectors']
    else:
        encoder = LocalEmbedder()
        for row, text in zip(master, texts):
            if len(encoder.tokenizer(text)['input_ids']) > 512:
                raise ValueError(f'D/T/M exceeds 512 tokens: {row["node_idx"]}')
        folder.mkdir(parents=True, exist_ok=True)
        rows_dir = folder / 'rows'
        rows_dir.mkdir(exist_ok=True)
        rows = []
        for i, text in enumerate(texts):
            path = rows_dir / (fingerprint({'text': text, 'model': IDENTITY}) + '.pt')
            if path.exists():
                vector = torch.load(path, map_location='cpu', weights_only=True)
            else:
                vector = encoder.encode([text], query=False)[0]
                temporary = path.with_suffix('.tmp')
                torch.save(vector, temporary)
                temporary.replace(path)
            if vector.shape != (2560,) or not torch.isfinite(vector).all() or vector.norm() == 0:
                raise ValueError(f'Invalid D/T/M embedding: {path}')
            rows.append(vector)
            if (i+1) % 100 == 0:
                print(f'D/T/M embedding {i+1}/{len(texts)}', flush=True)
        vectors = torch.stack(rows)
        # Empty attributes receive no synthetic semantic signal.
        vectors[[not any(r['knowledge'][k] for k in KINDS) for r in master]] = 0
        temporary = bundle.with_suffix('.tmp')
        torch.save({'signature': signature, 'vectors': vectors, 'model': IDENTITY}, temporary)
        temporary.replace(bundle)
    if vectors.shape != (len(master), 2560) or not torch.isfinite(vectors).all():
        raise ValueError('Invalid D/T/M bundle shape/values')
    return vectors, {'signature': signature, 'model': IDENTITY}


def adjacency(train, n, device):
    pairs = {(s, t) for s, t in train} | {(t, s) for s, t in train} | {(i, i) for i in range(n)}
    edge = torch.tensor(sorted(pairs), device=device).T
    degree = torch.bincount(edge[0], minlength=n).float()
    weight = (degree[edge[0]] * degree[edge[1]]).rsqrt()
    return torch.sparse_coo_tensor(edge, weight, (n, n)).coalesce()


class CitationGNN(nn.Module):
    def __init__(self, dimension, hidden, n, graph_only=False):
        super().__init__()
        self.node_ids = nn.Embedding(n, hidden) if graph_only else None
        self.project = nn.Linear(hidden if graph_only else dimension, hidden)
        self.conv = nn.Linear(hidden, hidden)
        self.source = nn.Linear(hidden, hidden, bias=False)
        self.target = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x, adj):
        if self.node_ids is not None:
            x = self.node_ids.weight
        h = F.relu(self.project(x))
        h = F.relu(self.conv(torch.sparse.mm(adj, h))) + h
        h = torch.sparse.mm(adj, h)
        return self.source(h), self.target(h)


def negative_targets(sources, train, n, generator):
    # Strict train-only negative filtering; unobserved true citations may be sampled.
    known = defaultdict(set)
    for s, t in train:
        known[s].add(t)
    result = []
    for s in sources:
        if len(known[s] | {s}) >= n:
            raise ValueError('No negative candidate for source')
        t = generator.randrange(n)
        while t == s or t in known[s]:
            t = generator.randrange(n)
        result.append(t)
    return result


def evaluate(source, target, truth, excluded, ks, labels):
    positives, seen = defaultdict(set), defaultdict(set)
    for s, t in truth:
        positives[s].add(t)
    for s, t in excluded:
        seen[s].add(t)
    rows = []
    # One source at a time bounds evaluation memory and produces per-paper examples.
    for s in sorted(positives):
        scores = (source[s] @ target.T).detach().cpu().numpy()
        blocked = seen[s] | {s}
        scores[list(blocked)] = -np.inf
        eligible = np.flatnonzero(np.isfinite(scores))
        ranking = eligible[np.lexsort((eligible, -scores[eligible]))][:max(ks)].tolist()
        relevant = positives[s] - blocked
        if not relevant:
            continue
        for k in ks:
            hits = [i for i, t in enumerate(ranking[:k]) if t in relevant]
            ideal = sum(1/np.log2(i+2) for i in range(min(k, len(relevant))))
            rows.append({'source': s, 'category': labels[s], 'k': k,
                         'recall': len(hits)/len(relevant),
                         'ndcg': sum(1/np.log2(i+2) for i in hits)/ideal,
                         'mrr': 1/(hits[0]+1) if hits else 0.,
                         'truth': sorted(relevant), 'ranking': ranking[:k]})
    return rows


def summarize(rows):
    result = {}
    for k in sorted({r['k'] for r in rows}):
        subset = [r for r in rows if r['k'] == k]
        categories = sorted({r['category'] for r in subset})
        result[str(k)] = {'queries': len(subset), 'represented_categories': len(categories)}
        for metric in ('recall', 'ndcg', 'mrr'):
            result[str(k)][metric] = float(np.mean([r[metric] for r in subset]))
            result[str(k)]['category_macro_' + metric] = float(np.mean([
                np.mean([r[metric] for r in subset if r['category'] == c]) for c in categories]))
    return result


def train_arm(x, split, labels, args, graph_only=False):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    x = F.normalize(x.to(device), dim=1)
    adj = adjacency(split['train'], len(x), device)
    model = CitationGNN(x.shape[1], args.hidden, len(x), graph_only).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    edges = torch.tensor(split['train'], device=device).T
    rng = random.Random(args.seed)
    best, state, best_epoch = -1., None, 0
    for epoch in range(1, args.epochs+1):
        model.train()
        optimizer.zero_grad()
        source, target = model(x, adj)
        neg = torch.tensor(negative_targets(edges[0].tolist(), split['train'], len(x), rng), device=device)
        positive_score = (source[edges[0]] * target[edges[1]]).sum(1)
        negative_score = (source[edges[0]] * target[neg]).sum(1)
        loss = F.softplus(negative_score - positive_score).mean()
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                s, t = model(x, adj)
                rows = evaluate(s, t, split['validation'], split['train'], [args.selection_k], labels)
            score = summarize(rows)[str(args.selection_k)]['ndcg']
            print(f'epoch={epoch} loss={loss.item():.5f} validation_ndcg={score:.5f}', flush=True)
            if score > best:
                best, best_epoch = score, epoch
                state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        s, t = model(x, adj)
        # Validation links are now known and masked, but never added to the graph.
        rows = evaluate(s, t, split['test'], split['train']+split['validation'], args.ks, labels)
    return rows, {'best_epoch': best_epoch, 'validation_ndcg': best}, state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--master', type=Path, default=ROOT/'subdataset/arxiv_master_final.json')
    parser.add_argument('--sample', type=Path, default=ROOT/'subdataset/ogbn_arxiv_16k_ffs_sample.pt')
    parser.add_argument('--citations', type=Path, default=ROOT/'subdataset/paper_paper_edges.pt')
    parser.add_argument('--cache', type=Path, default=ROOT/'output/citation_dtm_embeddings')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--arms', nargs='+', choices=['ogbn', 'dtm', 'graph_only'], default=['ogbn','dtm','graph_only'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--eval-every', type=int, default=5)
    parser.add_argument('--selection-k', type=int, default=20)
    parser.add_argument('--ks', type=int, nargs='+', default=[10,20])
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    if min(args.epochs, args.hidden, args.eval_every, args.selection_k, *args.ks) < 1 or args.lr <= 0:
        parser.error('Training settings and K must be positive')
    if args.output.exists():
        parser.error('Output must be a new directory')
    master, ogbn, labels, edges, outside = load_inputs(args.master,args.sample,args.citations)
    split = split_edges(edges, args.seed)
    args.output.mkdir(parents=True)
    write(args.output/'split.json', {'node_ids':[r['node_idx'] for r in master], 'edges':split})
    audit = {'papers':len(master), 'ogbn_shape':list(ogbn.shape), 'category_counts':dict(sorted(__import__('collections').Counter(labels).items())),
             'split_edges':{k:len(v) for k,v in split.items()}, 'outside_corpus_edges':outside,
             'empty_dtm':{k:sum(not r['knowledge'].get(k) for r in master) for k in KINDS},
             'protocol':'Directed citation prediction; unordered pairs split 80/10/10; train-only symmetric message passing; self and observed citations excluded; full corpus candidates; train-only negative filtering.',
             'input_hash':fingerprint({'master':master,'edges':sorted(set(edges)),'ogbn':hashlib.sha256(ogbn.numpy().tobytes()).hexdigest()}),
             'split_hash':fingerprint(split), 'settings':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
             'torch':torch.__version__}
    write(args.output/'manifest.json',audit)
    print(json.dumps(audit,ensure_ascii=False,indent=2),flush=True)
    if args.audit_only:
        return
    dtm = None
    if 'dtm' in args.arms or args.prepare_only:
        dtm, provenance = prepare_dtm(master,args.cache)
        audit['dtm_embedding'] = provenance
        write(args.output/'manifest.json',audit)
    if args.prepare_only:
        return
    results, per_arm = {}, {}
    for arm in args.arms:
        print(f'Training {arm}',flush=True)
        x = dtm if arm == 'dtm' else ogbn
        rows, selection, state = train_arm(x,split,labels,args,arm=='graph_only')
        write(args.output/f'{arm}_per_paper.json',rows)
        torch.save(state,args.output/f'{arm}_model.pt')
        per_arm[arm] = rows
        results[arm] = {'test':summarize(rows),**selection}
        write(args.output/'summary.json',results)
    if 'ogbn' in per_arm and 'dtm' in per_arm:
        deltas = {}
        for k in args.ks:
            a = {r['source']:r for r in per_arm['ogbn'] if r['k']==k}
            b = {r['source']:r for r in per_arm['dtm'] if r['k']==k}
            if a.keys() != b.keys():
                raise ValueError('Paired evaluation sources differ')
            for metric in ('recall','ndcg','mrr'):
                d = np.array([b[s][metric]-a[s][metric] for s in sorted(a)])
                rng = np.random.default_rng(args.seed)
                samples = [float(rng.choice(d,len(d),replace=True).mean()) for _ in range(2000)]
                deltas[f'{metric}@{k}'] = {'dtm_minus_ogbn':float(d.mean()), 'paired_bootstrap_95':np.quantile(samples,[.025,.975]).tolist()}
        write(args.output/'paired_deltas.json',deltas)


if __name__ == '__main__':
    main()
