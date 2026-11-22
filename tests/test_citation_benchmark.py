import unittest
from types import SimpleNamespace
import torch
from benchmarks.citation import split_edges, evaluate, summarize, dtm_text, train_arm, negative_targets
import random

class CitationTests(unittest.TestCase):
    def test_pair_split_no_reverse_leakage(self):
        edges = [(i,j) for i in range(8) for j in range(8) if i != j]
        parts = split_edges(edges+edges,42)
        pairs = [{tuple(sorted(e)) for e in p} for p in parts.values()]
        self.assertFalse(pairs[0]&pairs[1] or pairs[0]&pairs[2] or pairs[1]&pairs[2])
        self.assertEqual(sum(map(len,parts.values())),len(edges))
        self.assertEqual(parts,split_edges(edges,42))

    def test_mask_and_multiple_positives(self):
        s = torch.tensor([[1.,0.]]*5)
        t = torch.tensor([[9.,0.],[8.,0.],[7.,0.],[6.,0.],[5.,0.]])
        rows = evaluate(s,t,[(0,2),(0,3)],[(0,1)],[1,2],[0]*5)
        self.assertEqual(rows[0]['ranking'],[2])
        self.assertEqual(rows[0]['recall'],.5)
        self.assertEqual(rows[1]['recall'],1.)
        self.assertEqual(rows[1]['ndcg'],1.)

    def test_dtm_only_input(self):
        row = {'title':'SECRET','abstract':'SECRET','knowledge':{'domain':['AI'],'task':['Reasoning'],'method':['GNN']}}
        self.assertNotIn('SECRET',dtm_text(row))

    def test_negative_self_and_known_excluded(self):
        values = negative_targets([0]*20,[(0,1)],3,random.Random(1))
        self.assertEqual(set(values),{2})

    def test_train_checkpoint_and_test(self):
        split = split_edges([(i,(i+j)%12) for i in range(12) for j in (1,2,3)],42)
        args = SimpleNamespace(seed=42,device='cpu',hidden=8,lr=.001,epochs=2,eval_every=1,selection_k=3,ks=[3])
        for graph_only in (False,True):
            rows,selection,state = train_arm(torch.randn(12,16),split,[i%2 for i in range(12)],args,graph_only)
            self.assertTrue(rows)
            self.assertTrue(state)
            self.assertGreaterEqual(selection['best_epoch'],1)
            self.assertTrue(0<=summarize(rows)['3']['recall']<=1)

if __name__=='__main__':
    unittest.main()
