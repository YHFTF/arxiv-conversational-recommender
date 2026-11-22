"""Exercise the real dashboard worker and record warm, end-to-end CPU timings."""
import argparse
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
from .core import read, write

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'output/e5_small/improved'
FRESH = [
    ('pairwise_new', 'Methods that recommend items from implicit feedback using pairwise ranking, without requiring star ratings.'),
    ('inductive_new', 'Find graph neural network methods for inductive node classification using attributes and neighboring nodes.'),
    ('neighbors_ko_new', '그래프의 노드 속성과 이웃 정보를 활용해 새로운 노드를 분류하는 연구를 찾아줘.'),
    ('cross_domain_new', 'Cross-domain recommendation through transfer learning to address sparse interactions.'),
    ('nonconvex_new', 'Nonconvex stochastic gradient descent with convergence guarantees.'),
    ('private_distributed_new', 'Differentially private distributed learning where parties keep their local data private. Exclude attack papers.'),
]


def main(port):
    base = f'http://127.0.0.1:{port}'
    def request(path, value=None):
        data = None if value is None else json.dumps(value).encode('utf-8')
        req = urllib.request.Request(base+path, data=data,
            headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.load(response)
    reference = read(OUT/'live_results.json')
    queries = [(r['name'], r['query']) for r in reference] + FRESH
    rows = []
    for i, (name, query) in enumerate(queries):
        start = time.perf_counter()
        job = request('/api/cpu-recommend', dict(query=query))
        while job['status'] not in ('done', 'error'):
            if time.perf_counter()-start > 600:
                raise TimeoutError('Dashboard worker did not complete')
            time.sleep(.1)
            job = request('/api/cpu-recommend/'+job['id'])
        if job['status']=='error':
            raise RuntimeError(job['message'])
        result = job['result']
        assert result['engine']=='e5_hybrid_v2'
        assert len({r['paper_id'] for r in result['results']})==len(result['results'])
        if i < len(reference):
            assert [r['paper_id'] for r in result['results']]==[r['paper_id'] for r in reference[i]['results']], name
        rows.append(dict(name=name, query=query, total_ms=(time.perf_counter()-start)*1000,
            search_ms=result['search_ms'], results=result['results']))
        print('API passed',i+1,'/',len(queries),name,result['search_ms'],'ms',flush=True)
    summary = dict(requests=len(rows), cached_benchmark_reproduction=len(reference),
        fresh_qualitative_queries=len(FRESH),
        cold_first_request_ms=rows[0]['total_ms'],
        warm_search_ms_median=float(np.median([r['search_ms'] for r in rows[1:]])),
        warm_search_ms_p95=float(np.quantile([r['search_ms'] for r in rows[1:]],.95)),
        notes='CPU FP32; actual query encoding and search; sequential requests, no concurrency load. HTTP total includes 100ms polling granularity.')
    write(OUT/'api_smoke_results.json', rows)
    write(OUT/'api_smoke_summary.json', summary)
    print(summary,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8081)
    main(parser.parse_args().port)
