import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'code/retrieval'))
from hybrid_search import HybridSearch, improved_plan
from query_constraints import GuardedSearch
from dtm_graph_search import DTMIndex
from dtm_rank_features import Features
from benchmarks.core import Graph
from benchmarks.lexical_dtm_graph import Index


class ConstantModel:
    def predict(self, x):
        return np.ones(len(x))


def make_engine(papers):
    ids = [str(i) for i in range(len(papers))]
    graph = Graph(None, ids, relations=('citation',), citation_edges=[])
    features = Features(papers, Index(papers), DTMIndex(papers), graph)
    legacy = GuardedSearch(papers, features, ConstantModel())
    vectors = np.eye(384, dtype=np.float32)[:len(papers)]
    return HybridSearch(legacy, vectors), vectors


class HybridTests(unittest.TestCase):
    def test_grammatical_variants_preserve_node_task_and_exclusions(self):
        p = improved_plan('Graph neural networks for classifying nodes. Exclude surveys.')
        self.assertIn('node_classification', p.facets)
        self.assertIn('graph_learning', p.facets)
        self.assertIn('survey', p.exclusions)
        self.assertNotIn('survey', p.positive_query)

    def test_missing_positive_phrase_does_not_discard_semantic_candidate(self):
        engine, vectors = make_engine([
            dict(title='Private gradient exchange', abstract='Institutions exchange noisy updates.', knowledge={}),
            dict(title='Storage repair', abstract='Repair storage nodes.', knowledge={'task':['storage repair']})])
        result = engine.search('collaborative learning with differential privacy',
            query_vectors=np.stack([vectors[0], vectors[0]]))
        self.assertEqual(result['results'][0]['index'], 0)
        self.assertGreater(len(result['results'][0]['missing_facets']), 0)

    def test_explicit_exclusion_and_source_mask_survive_semantic_scores(self):
        engine, vectors = make_engine([
            dict(title='A survey of FPGA inference', abstract='An overview.', knowledge={}),
            dict(title='FPGA inference engine', abstract='Implemented neural inference.', knowledge={}),
            dict(title='FPGA inference hardware', abstract='Measured neural accelerator.', knowledge={})])
        result = engine.search('FPGA inference, not a survey.', excluded=1,
            query_vectors=np.stack([vectors[0], vectors[0]]), candidate_pool=[0,1,2])
        self.assertEqual([r['index'] for r in result['results']], [2])

    def test_convex_title_excluded_but_nonconvex_not_confused(self):
        engine, vectors = make_engine([
            dict(title='Convex stochastic optimization', abstract='Convex functions.', knowledge={}),
            dict(title='Nonconvex stochastic optimization', abstract='Nonconvex objectives.', knowledge={})])
        result = engine.search('nonconvex stochastic gradient descent, not convex functions',
            query_vectors=np.stack([vectors[0], vectors[0]]), candidate_pool=[0,1])
        self.assertEqual([r['index'] for r in result['results']], [1])

    def test_topic_evidence_precedes_high_semantic_score(self):
        engine, vectors = make_engine([
            dict(title='Power-efficient CPU', abstract='We report 5 W and 10 TOPS.', knowledge={}),
            dict(title='FPGA neural network accelerator', abstract='We report 7 W and 12 TOPS.', knowledge={})])
        result = engine.search('FPGA neural network inference with power consumption and throughput',
            query_vectors=np.stack([vectors[0], vectors[0]]), candidate_pool=[0,1])
        self.assertEqual(result['results'][0]['index'], 1)
        self.assertEqual(result['results'][1]['index'], 0)
        self.assertIn('fpga', result['results'][1]['missing_facets'])

    def test_graphsage_name_can_support_graph_neural_intent(self):
        engine, vectors = make_engine([
            dict(title='Inductive node classification', abstract='GraphSAGE aggregates node features.', knowledge={}),
            dict(title='Image neural network', abstract='Image classification.', knowledge={})])
        result = engine.search('Graph neural networks for classifying nodes',
            query_vectors=np.stack([vectors[1],vectors[1]]), candidate_pool=[0,1])
        self.assertEqual(result['results'][0]['index'], 0)
        self.assertEqual(result['results'][0]['missing_facets'], [])

    def test_empty_eligible_set_returns_no_synthetic_recommendations(self):
        engine, vectors = make_engine([
            dict(title='A survey of FPGA inference', abstract='An overview.', knowledge={})])
        result = engine.search('FPGA, exclude surveys', query_vectors=np.stack([vectors[0],vectors[0]]))
        self.assertEqual(result['results'], [])


if __name__ == '__main__':
    unittest.main()
