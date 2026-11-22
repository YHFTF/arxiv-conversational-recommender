import sys
import unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code/retrieval'))
from dtm_graph_search import words,DTMIndex,graph_scores,rank_scores
from dtm_rank_features import Features,FEATURES
from benchmarks.core import Graph
from benchmarks.lexical_dtm_graph import Index


class FakeGraph:
    def diffuse(self,seed,*args,**kwargs):
        self.seed=seed
        return seed.copy(),1,True


class DTMGraphTests(unittest.TestCase):
    def test_rank_features_are_finite_without_labels_or_text(self):
        papers=[{'knowledge':{'task':['node classification'],'method':['graph neural network']}},{'knowledge':{'task':['image classification'],'method':['convolutional neural network']}}]
        graph=Graph(None,['a','b'],relations=('citation',),citation_edges=[('a','b')])
        features=Features(papers,Index(papers),DTMIndex(papers),graph)
        state=features.prepare('graph neural network node classification',excluded=0)
        values=features.rows(state,[0,1])
        self.assertEqual(values.shape,(2,len(FEATURES)))
        self.assertTrue(np.isfinite(values).all())

    def test_ties_do_not_depend_on_candidate_input_order(self):
        scores=np.array([.1,.2,.2,0.])
        self.assertEqual(rank_scores(scores,[3,2,1,0]),[1,2,0,3])
        self.assertEqual(rank_scores(scores,[0,1,2,3]),[1,2,0,3])

    def test_aliases_and_opposite_conditions(self):
        self.assertEqual(words('GNN'),words('graph neural networks'))
        self.assertEqual(words('그래프 신경망'),words('graph neural network'))
        self.assertNotEqual(words('non-convex'),words('convex'))

    def test_title_and_abstract_cannot_change_scores(self):
        a=[{'title':'graph','abstract':'graph','knowledge':{'method':['node classification']}}]
        b=[{'title':'other','abstract':'other','knowledge':a[0]['knowledge']}]
        self.assertTrue(np.array_equal(DTMIndex(a).search('node classification'),DTMIndex(b).search('node classification')))
        self.assertEqual(float(DTMIndex(a).search('unseenword').sum()),0.)

    def test_excluded_anchor_never_becomes_seed(self):
        graph=FakeGraph()
        graph_scores(graph,np.array([1.,.8,.2]),seed_count=1,excluded=0)
        self.assertEqual(graph.seed[0],0.)
        self.assertEqual(graph.seed[1],1.)

    def test_no_match_does_not_fabricate_graph_ranking(self):
        graph=FakeGraph()
        self.assertTrue(np.array_equal(graph_scores(graph,np.zeros(3)),np.zeros(3)))
        self.assertFalse(hasattr(graph,'seed'))


if __name__=='__main__':unittest.main()
