"""Lazy persistent CPU recommendation worker; dashboard startup needs no ML imports."""
import threading,time,uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


PAPER_LIMIT = 3
AUTHOR_PAPER_DEPTH = 10
AUTHOR_LIMIT = 3
AUTHOR_EVIDENCE_LIMIT = 3


def recommend_authors(ranked_papers, papers):
    """Rank authors by distinct papers among the search results."""
    authors = {}
    for rank, row in enumerate(ranked_papers, 1):
        paper = papers[row['index']]
        paper_id = str(paper['paper_id'])
        for author in paper.get('authors') or []:
            author_id = author.get('author_id')
            author_name = author.get('author_name')
            if not author_id or not author_name:
                continue
            entry = authors.setdefault(author_id, dict(
                author_id=author_id, author_name=author_name, evidence={}
            ))
            entry['evidence'].setdefault(paper_id, dict(
                paper_id=paper_id, title=paper.get('title', ''), paper_rank=rank
            ))

    for entry in authors.values():
        evidence = list(entry.pop('evidence').values())
        entry['related_paper_count'] = len(evidence)
        entry['evidence_papers'] = evidence[:AUTHOR_EVIDENCE_LIMIT]

    def ranking(entry):
        return -entry['related_paper_count'], entry['evidence_papers'][0]['paper_rank']

    return sorted(authors.values(), key=ranking)[:AUTHOR_LIMIT]


class RecommendationService:
    def __init__(self,root,output,loader=None):
        self.root=Path(root);self.output=Path(output);self.loader=loader or self._load
        self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='cpu-recommend')
        self.lock=threading.Lock();self.slots=threading.BoundedSemaphore(4)
        self.jobs={};self.engine=None;self.papers=None

    def _load(self):
        import sys
        sys.path.insert(0,str(self.root));sys.path.insert(0,str(self.root/'code/retrieval'))
        import torch,joblib
        from benchmarks.core import Graph,read,file_hash
        from benchmarks.lexical_dtm_graph import Index
        from dtm_graph_search import DTMIndex
        from dtm_rank_features import Features
        from query_constraints import GuardedSearch
        torch.set_num_threads(1)
        base=self.output/'balanced_benchmark_20261117';data=base/'data'
        selected=read(base/'learned_dtm_graph/selected.json');modelpath=base/'learned_dtm_graph'/selected['model_file']
        if file_hash(modelpath)!=selected['model_sha256']:raise ValueError('추천 모델 파일의 해시가 일치하지 않습니다.')
        papers=read(data/'arxiv_master_final.json');ids=[str(p['paper_id']) for p in papers];local={p['node_idx']:i for i,p in enumerate(papers)}
        edges=torch.load(data/'paper_paper_edges.pt',weights_only=True,map_location='cpu').T.tolist()
        graph=Graph(None,ids,relations=('citation',),citation_edges=[(ids[local[s]],ids[local[t]]) for s,t in edges])
        extractor=Features(papers,Index(papers),DTMIndex(papers),graph)
        strength=0.;config=base/'route_consensus/selected.json'
        if config.exists():
            c=read(config)
            if c['model_sha256']!=selected['model_sha256']:raise ValueError('중복 가점 설정과 추천 모델이 일치하지 않습니다.')
            strength=float(c['strength'])
            if not 0<=strength<=1:raise ValueError('중복 가점 설정이 잘못되었습니다.')
        legacy=GuardedSearch(papers,extractor,joblib.load(modelpath),consensus_strength=strength)
        deployment=self.output/'e5_small/improved/deployment.json'
        if deployment.exists():
            import numpy as np
            from hybrid_search import HybridSearch
            sys.path.insert(0,str(self.root/'code/embeddings'))
            from e5_small import E5Small
            hybrid_dir=deployment.parent;deployed=read(deployment)
            if file_hash(hybrid_dir/'selected.json')!=deployed['selected_sha256']:
                raise ValueError('하이브리드 검색 설정의 해시가 일치하지 않습니다.')
            for relative,expected in deployed.get('source_hashes',{}).items():
                if file_hash(self.root/relative)!=expected:
                    raise ValueError('검색 코드가 평가된 버전과 다릅니다. 재평가 후 배포 설정을 갱신해 주세요.')
            chosen=read(hybrid_dir/'selected.json')
            if file_hash(hybrid_dir/'protocol.json')!=chosen['protocol_sha256']:
                raise ValueError('하이브리드 검색 평가 설정이 일치하지 않습니다.')
            manifest=read(self.output/'e5_small/documents_manifest.json')
            if manifest['paper_ids']!=ids or manifest['signature']!=deployed['document_signature']:
                raise ValueError('E5 문서 벡터와 현재 논문 데이터가 일치하지 않습니다.')
            ranker=None
            if chosen['model']:
                rankpath=hybrid_dir/chosen['model']
                if file_hash(rankpath)!=chosen['model_sha256']:
                    raise ValueError('하이브리드 재정렬 모델의 해시가 일치하지 않습니다.')
                ranker=joblib.load(rankpath)
            with np.load(self.output/'e5_small/documents.npz') as vectors:
                document_vectors=vectors['vectors']
            encoder=E5Small('cpu',threads=4,output_dir=self.output)
            if encoder.revision!=manifest['model']['revision']:
                raise ValueError('E5 질문 모델과 문서 벡터의 버전이 일치하지 않습니다.')
            return HybridSearch(legacy,document_vectors,encoder=encoder,
                config=chosen['config'],model=ranker),papers
        return legacy,papers

    def submit(self,query):
        if not isinstance(query,str) or not query.strip() or len(query)>2000:raise ValueError('질문을 1~2,000자로 입력해 주세요.')
        if not self.slots.acquire(blocking=False):raise RuntimeError('검색 요청이 많습니다. 잠시 후 다시 시도해 주세요.')
        jid=uuid.uuid4().hex
        with self.lock:
            if len(self.jobs)>=64:
                for key in list(self.jobs):
                    if self.jobs[key]['status'] in ('done','error'):
                        del self.jobs[key]
                        if len(self.jobs)<64:break
            self.jobs[jid]=dict(id=jid,status='queued',message='검색 대기 중',query=query.strip())
        self.executor.submit(self._work,jid,query.strip())
        return self.get(jid)

    def get(self,jid):
        with self.lock:return dict(self.jobs[jid]) if jid in self.jobs else None

    def _update(self,jid,**values):
        with self.lock:self.jobs[jid].update(values)

    def _work(self,jid,query):
        try:
            if self.engine is None:
                self._update(jid,status='loading',message='논문 색인과 추천 모델을 준비하고 있습니다. 첫 검색은 시간이 걸릴 수 있습니다.')
                self.engine,self.papers=self.loader()
            self._update(jid,status='searching',message='질문에 맞는 논문과 조건 근거를 찾고 있습니다.')
            started=time.perf_counter();result=self.engine.search(query,top_k=AUTHOR_PAPER_DEPTH)
            result['recommended_authors'] = recommend_authors(result['results'], self.papers)
            result['results'] = result['results'][:PAPER_LIMIT]
            for rank,row in enumerate(result['results'],1):
                paper=self.papers[row.pop('index')]
                row.update(rank=rank,paper_id=str(paper['paper_id']),title=paper.get('title',''),abstract=paper.get('abstract',''))
                row['tags']={k:(paper.get('knowledge') or {}).get(k,[]) for k in ('domain','task','method')}
            result['search_ms']=round((time.perf_counter()-started)*1000,1)
            self._update(jid,status='done',message='검색 완료',result=result)
        except Exception as exc:
            message='추천 준비 파일 또는 실행 환경을 확인해 주세요.'
            if isinstance(exc,FileNotFoundError):message='논문 데이터나 학습된 추천 모델이 없습니다. 서버의 output 경로를 확인해 주세요.'
            elif isinstance(exc,ImportError):message='CPU 추천에 필요한 Python 패키지가 없습니다. 프로젝트 ML 의존성을 설치해 주세요.'
            elif isinstance(exc,ValueError):message=str(exc)
            self._update(jid,status='error',message=message)
        finally:self.slots.release()
