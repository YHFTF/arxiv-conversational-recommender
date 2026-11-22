"""CPU E5/lexical/graph retrieval with soft topic checks and trained fusion."""
import re
import numpy as np
from dtm_graph_search import words
from query_constraints import parse_query, build_metadata

FEATURE_NAMES = [
    'old_ranker', 'semantic_original', 'semantic_canonical', 'semantic_rank',
    'canonical_rank', 'text_bm25', 'text_rank', 'model_rank',
    'facet_coverage', 'missing_facets', 'title_coverage', 'abstract_coverage',
    'evidence_coverage', 'title_facet_coverage',
]


def canonical_query(query):
    """Normalize common grammatical variants; never invent new user conditions."""
    query = re.sub(r'\bclassif(?:y|ying|ication of)\s+(?:the\s+)?nodes\b', 'node classification', query, flags=re.I)
    query = re.sub(r'\bnode\s+(?:label(?:ling|ing)|prediction)\b', 'node classification', query, flags=re.I)
    query = re.sub(r'\bdifferential(?:ly)?[- ]priv(?:acy|ate)\b', 'differential privacy', query, flags=re.I)
    query = re.sub(r'\b(?:decentralized|distributed|collaborative)\s+(?:deep\s+)?(?:training|learning)\b', 'collaborative learning', query, flags=re.I)
    query = re.sub(r'노드(?:를|의)?\s*분류', '노드 분류', query)
    query = re.sub(r'확률적\s*경사\s*하강법', '확률적 경사 하강법', query)
    query = re.sub(r'차등\s*프라이버시', '차등 프라이버시', query)
    return query


def improved_plan(query):
    plan = parse_query(canonical_query(query))
    plan.original = query
    if re.search(r'graph neural network|graph convolution|message passing', plan.positive_query):
        plan.facets.append('graph_learning')
    # Preserve the original multilingual meaning for E5; the canonical lexical
    # query remains a limited glossary and is not claimed to be translation.
    plan.facets = list(dict.fromkeys(plan.facets))
    return plan


def _rank(values, allowed):
    result = np.zeros(len(values))
    order = np.flatnonzero(allowed)
    order = order[np.argsort(-values[order], kind='stable')]
    order = order[values[order] > 0]
    result[order] = np.arange(1, len(order) + 1)
    return result


def _rrf(ranks):
    return np.divide(61., 60. + ranks, out=np.zeros_like(ranks), where=ranks > 0)


class HybridSearch:
    def __init__(self, legacy, document_vectors, encoder=None, config=None, model=None):
        self.legacy = legacy
        self.papers = legacy.papers
        self.extractor = legacy.extractor
        self.encoder = encoder
        self.config = config or dict(depth=100, semantic_weight=1., canonical_weight=.5,
            topic_penalty=.5, evidence_bonus=.05, learned_weight=0.)
        self.model = model
        self.vectors = np.asarray(document_vectors, dtype=np.float32)
        if self.vectors.shape != (len(self.papers), 384) or not np.isfinite(self.vectors).all():
            raise ValueError('E5 document vector shape or values are invalid')
        if not np.allclose(np.linalg.norm(self.vectors, axis=1), 1., atol=1e-4):
            raise ValueError('E5 document vectors must be normalized')
        self.metadata = build_metadata(self.papers, include_text=True)
        self.title_tokens = []
        self.abstract_tokens = []
        self.title_facets = []
        for p, m in zip(self.papers, self.metadata):
            title = ' '.join(words(p.get('title', '')))
            text = ' '.join(words(p.get('title', '') + ' ' + p.get('abstract', '')))
            if re.search(r'graph (?:neural|convolution|network embedding)|message passing|node classification|neighbor.{0,30}aggregat', text):
                m['text_facets'].append('graph_learning')
            neural_tags = ' '.join((p.get('knowledge') or {}).get('method', []))
            if re.search(r'graphsage|\bgcn\b|\bgat\b|graph (?:neural|convolution|attention)', text+' '+neural_tags.casefold()):
                m['text_facets'].extend(['graph_learning', 'neural_network'])
            if re.search(r'distributed (?:deep )?(?:learning|training)|decentralized (?:learning|training)|privacy preserving deep learning', text):
                m['text_facets'].append('collaborative_learning')
            # Expand common naming variants in the document text, too.
            normalized = ' '.join(words(canonical_query(text)))
            if 'differential privacy' in normalized:
                m['text_facets'].append('differential_privacy')
            m['nonconvex_text'] = 'nonconvex' in text
            m['convex_title'] = bool(re.search(r'\bconvex\b', title)) and 'nonconvex' not in title
            self.title_tokens.append(set(title.split()))
            self.abstract_tokens.append(set(text.split()))
            self.title_facets.append(set(build_metadata([dict(title=p.get('title', ''), abstract='', knowledge={})], include_text=True)[0]['text_facets']))

    def prepare(self, query, query_vectors=None, excluded=None):
        plan = improved_plan(query)
        if query_vectors is None:
            if self.encoder is None:
                raise ValueError('An E5 encoder is required for uncached queries')
            query_vectors = self.encoder.encode([query, plan.positive_query], query=True).numpy()
        qv = np.asarray(query_vectors)
        if qv.shape != (2, 384) or not np.isfinite(qv).all():
            raise ValueError('Two finite E5 query vectors are required')
        raw, canonical = self.vectors @ qv[0], self.vectors @ qv[1]
        eligible = np.ones(len(self.papers), dtype=bool)
        facets = []
        title_facets = []
        evidence = []
        for i, m in enumerate(self.metadata):
            supported = set(m['facets']) | set(m['text_facets'])
            facets.append(len(set(plan.facets) & supported) / max(len(plan.facets), 1) if plan.facets else 1.)
            title_facets.append(len(set(plan.facets) & self.title_facets[i]) / max(len(plan.facets), 1))
            evidence.append(sum(m['evidence'].get(e, False) for e in plan.evidence_requests) / max(len(plan.evidence_requests), 1))
            if 'survey' in plan.exclusions and m['survey']:
                eligible[i] = False
            if 'attack' in plan.exclusions and m['attack']:
                eligible[i] = False
            if 'convex_only' in plan.exclusions and m['convex_title'] and not m['nonconvex_text']:
                eligible[i] = False
        if excluded is not None:
            eligible[excluded] = False
        state = self.extractor.prepare(plan.positive_query, excluded=excluded, lexical_mask=eligible)
        text = self.legacy.text_index.search(plan.positive_query)
        text[~eligible] = 0
        ranks = dict(e5=_rank(raw, eligible), canonical=_rank(canonical, eligible),
            text=_rank(text, eligible), dtm=_rank(state['lex'], eligible),
            graph=_rank(state['mass'], eligible))
        model_mask = np.zeros(len(self.papers), dtype=bool)
        for name, values in ranks.items():
            model_mask |= (values > 0) & (values <= (300 if name in ('e5', 'canonical') else 200))
        all_indices = np.flatnonzero(model_mask & eligible)
        old = np.zeros(len(self.papers))
        if len(all_indices):
            old[all_indices] = self.legacy.model.predict(self.extractor.rows(state, all_indices))
        ranks['model'] = _rank(old - old.min() + 1e-9, model_mask & eligible)
        qw = set(words(plan.positive_query))
        title_cov = np.array([len(qw & ws) / max(len(qw), 1) for ws in self.title_tokens])
        abstract_cov = np.array([len(qw & ws) / max(len(qw), 1) for ws in self.abstract_tokens])
        facet = np.asarray(facets)
        x = np.column_stack([old, raw, canonical, _rrf(ranks['e5']), _rrf(ranks['canonical']),
            text, _rrf(ranks['text']), _rrf(ranks['model']), facet,
            (1-facet)*len(plan.facets), title_cov, abstract_cov, evidence, title_facets])
        return dict(plan=plan, eligible=eligible, ranks=ranks, features=x,
            evidence=np.asarray(evidence), facets=facet)

    def candidates(self, state, config=None):
        config = config or self.config
        ranks = state['ranks']
        hit = np.zeros(len(self.papers), dtype=bool)
        for name in ('dtm', 'text', 'graph'):
            hit |= (ranks[name] > 0) & (ranks[name] <= 200)
        for name in ('e5', 'canonical'):
            hit |= (ranks[name] > 0) & (ranks[name] <= config['depth'])
        return np.flatnonzero(hit & state['eligible'])

    def rank(self, state, candidates, config=None, model=None):
        config = config or self.config
        candidates = np.asarray(candidates, dtype=int)
        candidates = candidates[state['eligible'][candidates]]
        if not len(candidates):
            return [], np.array([])
        ranks = state['ranks']
        base = (_rrf(ranks['model']) + _rrf(ranks['text'])
            + config['semantic_weight'] * _rrf(ranks['e5'])
            + config['canonical_weight'] * _rrf(ranks['canonical']))
        # Positive conditions are relevance signals rather than exclusion rules.
        base *= 1 - config['topic_penalty'] * (1-state['facets'])
        base += config['evidence_bonus'] * state['evidence']
        selected_model = model if model is not None else self.model
        if config['learned_weight']:
            if selected_model is None:
                raise ValueError('Selected hybrid configuration requires its trained ranker')
            pred = selected_model.predict(state['features'][candidates])
            base = base[candidates] / (2 + config['semantic_weight'] + config['canonical_weight'])
            scores = (1-config['learned_weight'])*base + config['learned_weight']*np.clip(pred/3, 0, 1)
        else:
            scores = base[candidates]
        # Missing a positive phrase is not proof of irrelevance. Keep those
        # candidates, but prefer candidates with more confirmed requested topics.
        order = np.lexsort((candidates, -scores, -state['evidence'][candidates],
                            -state['facets'][candidates]))
        return candidates[order].tolist(), scores[order]

    def search(self, query, top_k=3, excluded=None, candidate_pool=None, query_vectors=None):
        state = self.prepare(query, query_vectors=query_vectors, excluded=excluded)
        candidates = self.candidates(state) if candidate_pool is None else candidate_pool
        ranked, scores = self.rank(state, candidates)
        plan = state['plan']
        result = []
        for i, score in zip(ranked[:top_k], scores[:top_k]):
            m = self.metadata[i]
            missing = sorted(set(plan.facets) - (set(m['facets']) | set(m['text_facets'])))
            result.append(dict(index=i, score=float(score),
                route_ranks={name: int(values[i]) for name, values in state['ranks'].items()
                             if name != 'model' and 0 < values[i] <= (self.config['depth'] if name in ('e5','canonical') else 200)},
                conditions={e:'reported_in_abstract' if m['evidence'].get(e) else 'unknown' for e in plan.evidence_requests},
                evidence={e:m['evidence_snippets'].get(e) for e in plan.evidence_requests},
                missing_facets=missing, facet_coverage=float(state['facets'][i])))
        return dict(plan=plan.to_dict(), results=result, eligible_count=int(state['eligible'].sum()),
            reason=None if result else 'No candidates remain after explicit exclusions.',
            engine='e5_hybrid_v2')
