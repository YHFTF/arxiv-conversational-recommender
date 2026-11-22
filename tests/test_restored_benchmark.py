import unittest
from benchmarks.restored import validate_judgments,bootstrap

class RestoredDataTests(unittest.TestCase):
    def fixture(self):
        q=[{'query_id':'q0','query':'query','anchor_node_idx':10,'split':'test'}]
        j=[{'query_id':'q0','query':'query','candidate_node_indices':[10,11],
            'labels':[{'node_idx':10,'relevance':2},{'node_idx':11,'relevance':0}]}]
        return q,j

    def test_complete_import(self):
        q,j=self.fixture();self.assertEqual(validate_judgments(q,j,{10,11}),{'test':1})

    def test_unknown_candidate_and_duplicate_label_fail(self):
        q,j=self.fixture()
        with self.assertRaises(ValueError):validate_judgments(q,j,{10})
        j[0]['labels']=[{'node_idx':10,'relevance':2}]*2
        with self.assertRaises(ValueError):validate_judgments(q,j,{10,11})

    def test_source_cannot_cross_splits(self):
        q,j=self.fixture();q.append({**q[0],'query_id':'q1','split':'train'});j.append({**j[0],'query_id':'q1'})
        with self.assertRaises(ValueError):validate_judgments(q,j,{10,11})

    def test_application_ndcg_uses_full_judged_pool(self):
        from run_nl_recommendation import ndcg_at_k
        self.assertLess(ndcg_at_k([0,1],2,[2,1,0]),.2)
        self.assertAlmostEqual(ndcg_at_k([2,1],2,[2,1,0]),1.)

    def test_cluster_bootstrap_uses_actual_cutoff_and_source(self):
        rows=[]
        for i in range(4):
            for arm,value in [('a',.2),('b',.3)]:
                rows.append({'query_id':str(i),'anchor_node_idx':i//2,'arm':arm,'k':10,
                    'recall':value,'ndcg':value,'mrr':value})
        result=bootstrap(rows,'a')
        self.assertEqual(len(result),3)
        self.assertEqual(result['b/ndcg@10']['groups'],2)
        self.assertAlmostEqual(result['b/ndcg@10']['delta'],.1)
        self.assertAlmostEqual(result['b/ndcg@10']['ci95'][0],.1)

if __name__=='__main__':unittest.main()
