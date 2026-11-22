import unittest
import numpy as np
from benchmarks.cold_start import temporal_split,top_order,ranked_metrics

class ColdStartTests(unittest.TestCase):
    def test_time_split_is_disjoint(self):
        split=temporal_split([2015,2017,2018,2019,2020])
        self.assertEqual(split,{'train':[0,1],'validation':[2],'test':[3,4]})
        self.assertFalse(set(split['train'])&set(split['test']))

    def test_exact_topk_tie_break(self):
        scores=np.array([.5,.9,.5,.5]);ids=np.array([30,40,10,20])
        self.assertEqual(top_order(scores,ids,3).tolist(),[1,2,3])
        self.assertEqual(top_order(scores,ids,4).tolist(),[1,2,3,0])

    def test_isolated_new_node_does_not_change_old_embeddings(self):
        import torch
        from benchmarks.citation import CitationGNN,adjacency
        torch.manual_seed(1)
        model=CitationGNN(8,8,3,False).eval()
        warm=torch.randn(3,8);cold=torch.randn(1,8)
        source,target=model(warm,adjacency([(0,1)],3,'cpu'))
        extended_source,extended_target=model(torch.cat([warm,cold]),adjacency([(0,1)],4,'cpu'))
        isolated_source,_=model(cold,adjacency([],1,'cpu'))
        torch.testing.assert_close(target,extended_target[:3])
        torch.testing.assert_close(isolated_source[0],extended_source[3])

    def test_recall_uses_all_hidden_references(self):
        m=ranked_metrics([3,1,0],{0,1},2)
        self.assertEqual(m['recall'],.5);self.assertEqual(m['mrr'],.5)
        self.assertLess(m['ndcg'],.5)

if __name__=='__main__':unittest.main()
