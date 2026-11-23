import gzip
import time,unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from dashboard.cpu_recommendation import RecommendationService, paper_citation_counts


class FakeEngine:
    def search(self,query,top_k):
        return {'plan':{'positive_query':query,'exclusions':[],'warnings':[]},'results':[{'index':0,'route_ranks':{'text':1},'conditions':{},'evidence':{}}],'reason':None}


class RecommendationTests(unittest.TestCase):
    def service(self,loader):
        service=RecommendationService('.','output',loader=loader)
        self.addCleanup(service.executor.shutdown,wait=True)
        return service

    def finished(self,service,jid):
        limit=time.monotonic()+3
        while time.monotonic()<limit:
            job=service.get(jid)
            if job['status'] in ('done','error'):return job
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
