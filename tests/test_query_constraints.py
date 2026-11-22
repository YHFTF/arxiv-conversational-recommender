import sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code/retrieval'))
from query_constraints import parse_query,build_metadata,GuardedSearch,route_agreement
from dtm_graph_search import DTMIndex
from dtm_rank_features import Features
from benchmarks.core import Graph
from benchmarks.lexical_dtm_graph import Index


class ConstantModel:
    def predict(self,x):return np.ones(len(x))


class ConstraintTests(unittest.TestCase):
    def test_consensus_discounts_dependent_graph_route(self):
        self.assertGreater(route_agreement({'dtm':1,'text':1}),route_agreement({'dtm':1,'graph':1}))
        self.assertGreater(route_agreement({'dtm':1,'text':1}),route_agreement({'dtm':200,'text':200,'graph':200}))
        self.assertEqual(route_agreement({'text':1}),0.)
    def test_text_bm25_recovers_a_document_missing_dtm(self):
        papers=[dict(title='Node classification with graph neural networks',abstract='We aggregate node features for node classification.',knowledge={}),dict(title='Storage repair',abstract='Repair storage nodes.',knowledge={'task':['storage repair']})]
        graph=Graph(None,['a','b'],relations=('citation',),citation_edges=[])
        extractor=Features(papers,Index(papers),DTMIndex(papers),graph)
        before=GuardedSearch(papers,extractor,ConstantModel(),use_text_bm25=False).search('node classification')
        after=GuardedSearch(papers,extractor,ConstantModel(),use_text_bm25=True).search('node classification')
        self.assertEqual(before['results'],[])
        self.assertEqual([r['index'] for r in after['results']],[0])
        self.assertGreater(after['results'][0]['text_bm25_score'],0)
        self.assertEqual(after['results'][0]['route_ranks'],{'text':1})
        self.assertEqual(after['results'][0]['route_count'],1)

    def test_negative_words_are_not_positive_search_terms(self):
        p=parse_query('FPGA neural network inference, not a survey.')
        self.assertNotIn('survey',p.positive_query)
        self.assertIn('survey',p.exclusions)
        self.assertIn('fpga',p.facets)

    def test_korean_morphology_and_exclusions(self):
        p=parse_query('노드 속성을 활용한 반지도 노드 분류 연구를 찾아줘. 서베이는 제외해줘.')
        self.assertIn('node classification',p.positive_query)
        self.assertIn('node attribute',p.positive_query)
        self.assertIn('survey',p.exclusions)

    def test_korean_exclusion_before_positive_request(self):
        p=parse_query('서베이 빼고 FPGA 신경망 추론 구현을 찾아줘')
        self.assertIn('survey',p.exclusions)
        self.assertIn('fpga',p.facets)
        self.assertIn('neural_network',p.facets)

    def test_not_sharing_does_not_erase_privacy_requirement(self):
        p=parse_query('collaborative learning not sharing raw data with differential privacy')
        self.assertIn('differential_privacy',p.facets)

    def test_without_surveys_is_an_exclusion(self):
        p=parse_query('FPGA neural network implementations without surveys')
        self.assertIn('survey',p.exclusions)
        self.assertNotIn('survey',p.positive_query)

    def test_korean_review_exclusion(self):
        p=parse_query('리뷰는 빼고 FPGA 구현을 찾아줘')
        self.assertIn('survey',p.exclusions)
        self.assertIn('fpga',p.facets)

    def test_nonconvex_is_preserved_while_convex_excluded(self):
        p=parse_query('SGD on nonconvex functions, not a result restricted to convex functions.')
        self.assertIn('nonconvex',p.facets)
        self.assertIn('stochastic_gradient',p.facets)
        self.assertIn('convex_only',p.exclusions)

    def test_not_only_does_not_remove_positive_clause(self):
        p=parse_query('not only node classification but also link prediction')
        self.assertIn('node_classification',p.facets)
        self.assertEqual(p.exclusions,[])

    def test_missing_power_evidence_is_unknown(self):
        m=build_metadata([dict(title='FPGA accelerator',abstract='We improve throughput.',knowledge={})])[0]
        self.assertFalse(m['evidence']['power'])

    def test_end_to_end_excludes_survey_and_unrelated_dtm(self):
        papers=[dict(title='A survey of FPGA neural inference',abstract='Review.',knowledge={'method':['FPGA'],'task':['Neural network inference']}),dict(title='FPGA accelerator',abstract='We report 5 W and 10 TOPS.',knowledge={'method':['FPGA'],'task':['Neural network inference']}),dict(title='Storage repair',abstract='Recover nodes.',knowledge={'task':['Distributed storage repair']})]
        graph=Graph(None,['a','b','c'],relations=('citation',),citation_edges=[('a','b'),('b','c')])
        extractor=Features(papers,Index(papers),DTMIndex(papers),graph)
        engine=GuardedSearch(papers,extractor,ConstantModel())
        r=engine.search('FPGA neural network inference with power consumption, not a survey.')
        self.assertEqual([v['index'] for v in r['results']],[1])
        self.assertEqual(r['results'][0]['conditions']['power'],'reported_in_abstract')

    def test_unrecognized_exclusion_is_reported(self):
        p=parse_query('graph methods, excluding bananas')
        self.assertTrue(any('Unrecognized exclusion' in w for w in p.warnings))


if __name__=='__main__':unittest.main()
