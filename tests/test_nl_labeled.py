import unittest
import random
from collections import Counter
from benchmarks.nl_labeled import choose_anchors, validate_labels, validate_query, QUERY_INSTRUCTIONS, JUDGE_INSTRUCTIONS

class NaturalLanguageEvalTests(unittest.TestCase):
    def test_sampling_unique_and_balanced(self):
        papers=[{'title':'T','abstract':'A'} for _ in range(90)]
        labels=[i%3 for i in range(90)]
        anchors=choose_anchors(papers,labels,30,42)
        self.assertEqual(len(set(anchors)),30)
        self.assertEqual(Counter(labels[i] for i in anchors),{0:10,1:10,2:10})
        self.assertEqual(anchors,choose_anchors(papers,labels,30,42))

    def test_missing_duplicate_out_of_range_labels_rejected(self):
        valid={'labels':[{'id':i,'relevance':i,'reason':'evidence'} for i in range(3)]}
        validate_labels(valid,[0,1,2])
        for invalid in ({'labels':valid['labels'][:2]}, {'labels':[valid['labels'][0]]*3}, {'labels':[{'id':0,'relevance':3,'reason':'bad'}]}):
            with self.assertRaises(ValueError):
                validate_labels(invalid,[0,1,2] if len(invalid['labels'])>1 else [0])

    def test_query_validation(self):
        with self.assertRaises(ValueError):
            validate_query({'query':''})
        validate_query({'query':'Find research about graph learning methods.'})

if __name__=='__main__':
    unittest.main()
