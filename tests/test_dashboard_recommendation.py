import gzip
import json,time,unittest
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock
from dashboard.cpu_recommendation import RecommendationService, paper_citation_counts
from dashboard.semantic_refinement import ClarificationRequired, Criterion, InvalidOllamaOutput, SemanticRefiner


class FakeEngine:
    def search(self,query,top_k):
        return {'plan':{'positive_query':query,'exclusions':[],'warnings':[]},'results':[{'index':0,'route_ranks':{'text':1},'conditions':{},'evidence':{}}],'reason':None}


def accept_initial_candidates(query, rows, papers):
    return {row['index']: papers[row['index']]['title'] for row in rows}


class RecommendationTests(unittest.TestCase):
    def service(self,loader,refiner=None):
        if refiner is None:
            refiner = Mock(review_initial=accept_initial_candidates)
        service=RecommendationService('.','output',loader=loader,refiner=refiner)
        self.addCleanup(service.executor.shutdown,wait=True)
        return service

    def finished(self,service,jid):
        limit=time.monotonic()+3
        while time.monotonic()<limit:
            job=service.get(jid)
            if job['status'] in ('done','clarify','error'):return job
            time.sleep(.01)
        self.fail('worker did not finish')

    def test_reuses_engine_and_returns_real_card_fields(self):
        loads=[]
        def loader():
            loads.append(1)
            return FakeEngine(), [dict(paper_id='123', title='Actual paper', abstract='Evidence.',
                                       knowledge={'task': ['classification']}, citation_count=0,
                                       citation_scope='ogbn_arxiv')]
        s=self.service(loader)
        for query in ('first','second'):
            result=self.finished(s,s.submit(query)['id'])
            self.assertEqual(result['status'],'done')
            self.assertEqual(result['result']['results'][0]['title'],'Actual paper')
            self.assertEqual(result['result']['results'][0]['citation_count'], 0)
            self.assertEqual(result['result']['results'][0]['citation_scope'], 'ogbn_arxiv')
            self.assertNotIn('index',result['result']['results'][0])
            self.assertEqual(result['result']['recommended_authors'], [])
        self.assertEqual(len(loads),1)

    def test_citations_count_distinct_incoming_papers_with_the_available_scope(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'edge.csv.gz'
            nodes = {10, 20, 30}
            sample_edges = [(10, 20), (10, 20)]
            counts, scope = paper_citation_counts(path, nodes, sample_edges)
            self.assertEqual(scope, 'sample')
            self.assertEqual(counts, {20: 1})

            with gzip.open(path, 'wt') as handle:
                handle.write('10,20\n10,20\n99,20\n20,10\n10,10\n30,99\n')
            counts, scope = paper_citation_counts(path, nodes, sample_edges)
            self.assertEqual(scope, 'ogbn_arxiv')
            self.assertEqual(counts, {20: 2, 10: 1})
            self.assertEqual(counts[30], 0)

    def test_ranks_authors_by_distinct_candidate_papers_in_one_search(self):
        first = {'author_id': 'A1', 'author_name': 'Same Name'}
        frequent = {'author_id': 'A2', 'author_name': 'Same Name'}
        second = {'author_id': 'A3', 'author_name': 'Other Researcher'}
        assignments = [[first, first], [second], [], [frequent], [frequent], [], [], [], [], [frequent]]
        papers = [dict(paper_id=str(i), title=f'Paper {i}', authors=authors)
                  for i, authors in enumerate(assignments)]
        search_limits = []

        class Engine:
            def search(self, query, top_k):
                search_limits.append(top_k)
                return {'results': [{'index': i} for i in range(top_k)]}

        service = self.service(lambda: (Engine(), papers))
        job = self.finished(service, service.submit('query')['id'])
        self.assertEqual(job['status'], 'done', job)
        result = job['result']
        self.assertEqual(search_limits, [10])
        self.assertEqual([p['paper_id'] for p in result['results']], ['0', '1', '2'])
        authors = result['recommended_authors']
        self.assertEqual([a['author_id'] for a in authors], ['A2', 'A1', 'A3'])
        self.assertEqual([a['related_paper_count'] for a in authors], [3, 1, 1])
        self.assertEqual([p['paper_rank'] for p in authors[0]['evidence_papers']], [4, 5, 10])

    def test_initial_search_reviews_only_ten_and_returns_only_supported_papers(self):
        search_limits = []

        class Engine:
            def search(self, query, top_k):
                search_limits.append(top_k)
                return {'results': [{'index': i} for i in range(top_k)]}

        papers = [dict(paper_id=str(i), title=f'FPGA paper {i}', abstract='The device is 8.3x faster.')
                  for i in range(12)]
        for i in (4, 8, 10):
            papers[i]['abstract'] = 'We measured 120 frames per second.'
        refiner = SemanticRefiner()
        refiner._chat = Mock(side_effect=[
            {'matches': [{'id': 0, 'quote': papers[0]['abstract']},
                         {'id': 4, 'quote': papers[4]['abstract']}]},
            {'matches': [{'id': 8, 'quote': papers[8]['abstract']},
                         {'id': 10, 'quote': papers[10]['abstract']}]},
        ])
        service = self.service(lambda: (Engine(), papers), refiner)
        job = self.finished(service, service.submit('실제로 FPS를 측정한 FPGA 논문만 보여줘.')['id'])
        self.assertEqual(job['status'], 'done', job)
        self.assertEqual(search_limits, [10])
        batches = [[paper['id'] for paper in call.args[1]['papers']] for call in refiner._chat.call_args_list]
        self.assertEqual(batches, [list(range(5)), list(range(5, 10))])
        result = job['result']
        self.assertEqual([paper['paper_id'] for paper in result['results']], ['4', '8'])
        self.assertEqual(result['results'][0]['initial_evidence'], papers[4]['abstract'])
        self.assertEqual(result['conversation']['initial_candidate_count'], 10)
        self.assertEqual(result['conversation']['initial_verified_candidate_count'], 2)

    def test_initial_search_with_no_supported_papers_returns_an_empty_result(self):
        refiner = SemanticRefiner()
        refiner._chat = Mock(return_value={'matches': []})
        service = self.service(lambda: (FakeEngine(), [dict(paper_id='0', title='Unrelated paper')]), refiner)
        job = self.finished(service, service.submit('Find an unrelated topic')['id'])
        self.assertEqual(job['status'], 'done', job)
        self.assertEqual(job['result']['results'], [])
        self.assertEqual(job['result']['conversation']['initial_verified_candidate_count'], 0)

    def test_pairwise_ranking_review_requires_evidence_of_the_requested_method(self):
        papers = [{'title': 'Matrix factorization from implicit feedback'},
                  {'title': 'Bayesian personalized ranking from implicit feedback',
                   'abstract': 'We train with BPR-Opt. We learn ranks from click data. This uses implicit feedback.'}]
        refiner = SemanticRefiner()
        refiner._chat = Mock(return_value={'matches': [
            {'id': 0, 'quote': papers[0]['title']},
            {'id': 1, 'quote': 'We train with BPR-Opt... This uses implicit feedback.'},
        ]})
        rows = [{'index': 0}, {'index': 1}]
        matches = refiner.review_initial('pairwise ranking으로 추천을 학습한 논문', rows, papers)
        self.assertEqual(matches, {1: papers[1]['abstract']})

    def test_followups_filter_the_original_pool_and_undo_restores_candidates(self):
        papers = [dict(paper_id=str(i), title=f'Paper {i}', authors=[
            {'author_id': f'A{i}', 'author_name': f'Author {i}'},
        ]) for i in range(5)]
        calls = []

        class Engine(FakeEngine):
            def search(self, query, top_k):
                calls.append(query)
                result = super().search(query, top_k)
                result['results'] = [dict(index=i, conditions={}, evidence={}) for i in range(top_k)]
                return result

        class Refiner:
            def review_initial(self, query, rows, papers):
                return accept_initial_candidates(query, rows, papers)

            def interpret(self, topic, active, query, jid):
                if query == 'undo the last condition':
                    return active[-1].id, None
                return None, Criterion(jid, query, query, query)

            def review(self, criterion, rows, papers, engine):
                eligible = {1, 2, 3, 4} if criterion.text == 'exclude surveys' else {0, 4}
                matches = {row['index']: 'Reported 5 W.' for row in rows if row['index'] in eligible}
                return replace(criterion, matches=matches, reviewed_count=len(rows))

        service = self.service(lambda: (Engine(), papers), Refiner())

        def submit(query=None, previous_id=None, remove_filter=None):
            job = self.finished(service, service.submit(query, previous_id, remove_filter)['id'])
            self.assertEqual(job['status'], 'done', job)
            return job['id'], job['result']

        root_id, root = submit('FPGA 구현 논문을 찾아줘.')
        self.assertEqual([paper['paper_id'] for paper in root['results']], ['0', '1', '2'])
        survey_id, survey = submit('exclude surveys', root_id)
        self.assertEqual(survey['conversation']['candidate_count'], 4)
        power_id, power = submit('실험에서 쓴 전력이 5와트를 넘지 않는 연구 위주로 남겨줄 수 있어?', survey_id)
        self.assertEqual([c['id'] for c in power['conversation']['filters']], [survey_id, power_id])
        self.assertEqual(power['conversation']['initial_candidate_count'], 5)
        self.assertEqual(power['conversation']['candidate_count'], 1)
        self.assertEqual(power['results'][0]['paper_id'], '4')
        self.assertEqual(power['results'][0]['followup_evidence'][-1]['quote'], 'Reported 5 W.')
        self.assertEqual([author['author_id'] for author in power['recommended_authors']], ['A4'])
        _, restored = submit(previous_id=power_id, remove_filter=survey_id)
        self.assertEqual([paper['paper_id'] for paper in restored['results']], ['0', '4'])
        _, undone = submit('undo the last condition', power_id)
        self.assertEqual(undone['conversation']['candidate_count'], 4)
        with self.assertRaises(ValueError):
            service.submit('서베이는 빼줘.', previous_id='expired')
        self.assertEqual(calls, ['FPGA 구현 논문을 찾아줘.'])
        _, independent = submit('A different topic')
        self.assertEqual(independent['conversation']['filters'], [])

    def test_semantic_review_keeps_only_quotes_grounded_in_the_selected_paper(self):
        class Refiner(SemanticRefiner):
            def _chat(self, *args):
                return {'matches': [{'id': 0, 'quote': 'Measured 4 W during inference.'},
                                    {'id': 1, 'quote': 'Measured 4 W during inference.'},
                                    {'id': 2, 'quote': 'Measured 4 W during inference.'}]}

        papers = [{'title': 'Paper A', 'abstract': 'Measured 4 W\n during inference.'},
                  {'title': 'Paper B', 'abstract': 'Measured 8 W.'},
                  {'title': 'Paper C', 'abstract': 'Measured 4 W during inference.'}]
        criterion = Criterion('1', '5W 이하', 'At most 5 W', 'measured power consumption')
        result = Refiner().review(criterion, [{'index': 0}, {'index': 1}], papers, object())
        self.assertEqual(result.matches, {0: 'Measured 4 W during inference.'})
        self.assertEqual(result.reviewed_count, 2)

    def test_retries_truncated_ollama_json_and_resolves_removal_position(self):
        refiner = SemanticRefiner()
        refiner.opener = Mock()
        decision = {'action': 'remove', 'label': '', 'criterion': '', 'search_hint': '', 'remove_position': 2}
        refiner.opener.open.side_effect = [
            BytesIO(b'{"done_reason":"length","message":{"content":"{"}}'),
            BytesIO(json.dumps({'message': {'content': json.dumps(decision)}}).encode()),
        ]
        active = [Criterion('a', 'Condition A', 'A', 'A'), Criterion('b', 'Condition B', 'B', 'B')]
        with self.assertLogs('dashboard.semantic_refinement', level='WARNING'):
            self.assertEqual(refiner.interpret('topic', active, 'undo the last filter', 'c'), ('b', None))
        self.assertEqual(refiner.opener.open.call_count, 2)

    def test_blank_model_label_asks_for_a_reason_and_keeps_context_for_the_reply(self):
        refiner = SemanticRefiner()
        refiner.review_initial = accept_initial_candidates
        refiner._chat = Mock(side_effect=[
            {'action': 'add', 'label': '실험 검증', 'criterion': 'experiment', 'search_hint': ''},
            {'action': 'add', 'label': ' ', 'criterion': 'at most 4 W', 'search_hint': ''},
            {'reason': '어떤 측면이 비슷해야 할지 정해지지 않았어요.',
             'question': '소비 전력과 처리량 중 어떤 측면을 보고 싶으신가요?'},
            {'action': 'add', 'label': '5W 이하', 'criterion': 'at most 5 W', 'search_hint': 'power'},
        ])
        refiner.review = Mock(side_effect=lambda criterion, *args: replace(criterion, matches={0: 'Measured 5 W.'}))
        service = self.service(lambda: (FakeEngine(), [{'paper_id': 'p', 'title': 'FPGA paper'}]), refiner)
        initial = self.finished(service, service.submit('FPGA 논문을 찾아줘.')['id'])
        filtered = self.finished(service, service.submit('실험한 연구만', initial['id'])['id'])
        request = '비슷한 논문을 더 보여줘.'
        clarification = self.finished(service, service.submit(request, filtered['id'])['id'])
        self.assertEqual(clarification['status'], 'clarify', clarification)
        self.assertEqual(refiner.review.call_count, 1)
        explanation = refiner._chat.call_args.args[1]
        self.assertEqual(explanation['request'], request)
        self.assertIn('label', explanation['validation_issue'])
        self.assertEqual(explanation['previous_output']['criterion'], 'at most 4 W')

        answer = self.finished(service, service.submit('소비 전력이 5W 이하인 것', clarification['id'])['id'])
        self.assertEqual(answer['status'], 'done', answer)
        self.assertEqual([item['label'] for item in answer['result']['conversation']['filters']], ['실험 검증', '5W 이하'])
        context = refiner._chat.call_args.args[1]
        self.assertEqual(context['clarification_history'], (
            {'request': request, 'question': clarification['clarification']['question']},
        ))

    def test_an_explicit_clarification_does_not_call_the_model_again(self):
        refiner = SemanticRefiner()
        refiner._chat = Mock(return_value={
            'action': 'clarify', 'reason': '해제할 조건이 여러 개예요.', 'question': '어떤 조건을 해제할까요?',
        })
        with self.assertRaises(ClarificationRequired) as raised:
            refiner.interpret('topic', [], '그 조건을 빼줘.', 'id')
        self.assertEqual(raised.exception.question, '어떤 조건을 해제할까요?')
        refiner._chat.assert_called_once()

    def test_incomplete_model_clarification_still_asks_a_question(self):
        refiner = SemanticRefiner()
        refiner._chat = Mock(side_effect=[InvalidOllamaOutput(), {'reason': '', 'question': ''}])
        with self.assertLogs('dashboard.semantic_refinement', level='WARNING'):
            with self.assertRaises(ClarificationRequired) as raised:
                refiner.interpret('topic', [], '조건을 추가해줘.', 'id')
        self.assertTrue(raised.exception.question.endswith('?'))

    def test_rejects_invalid_queries_without_loading(self):
        s=self.service(lambda:self.fail('must not load'))
        for q in (None,[],12,'   ','x'*2001):
            with self.assertRaises(ValueError):s.submit(q)
        self.assertIsNone(s.get('unknown'))

    def test_missing_artifacts_produce_actionable_error(self):
        def missing():raise FileNotFoundError('missing model')
        s=self.service(missing)
        job=self.finished(s,s.submit('query')['id'])
        self.assertEqual(job['status'],'error')
        self.assertIn('output',job['message'])


if __name__=='__main__':unittest.main()
