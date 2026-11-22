"""Fresh unlabelled requests; collect actual full-corpus recommendations for review."""
import sys,time
from pathlib import Path
import joblib
import numpy as np
import torch
from .core import Graph,read,write,file_hash
from .lexical_dtm_graph import Index
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code/retrieval'))
from dtm_graph_search import DTMIndex
from dtm_rank_features import Features

REQUESTS=[
 ('graph_en','Semi-supervised node classification using node attributes and graph neighborhoods when only a few nodes have labels.'),
 ('graph_ko','라벨이 있는 노드가 적을 때 노드 속성과 주변 연결 정보를 함께 활용하는 반지도 노드 분류 연구를 찾고 싶어.'),
 ('fpga_en','An FPGA implementation of neural network inference with measured power consumption and throughput. Prefer an implemented accelerator, not a survey.'),
 ('fpga_ko','FPGA로 신경망 추론을 가속하고 전력 소모와 처리량을 실제 측정한 구현 논문을 찾아줘. 서베이는 제외해줘.'),
 ('privacy_en','A collaborative learning method that keeps raw training data at each institution and provides a differential privacy guarantee, rather than an attack on privacy.'),
 ('privacy_ko','기관별 원본 데이터를 외부에 보내지 않고 공동 학습하면서 차등 프라이버시 보장을 제공하는 방법을 찾아줘. 공격 논문은 제외해줘.'),
 ('theory_en','A theoretical convergence guarantee for stochastic gradient descent on nonconvex objectives, not a result restricted to convex or strongly convex functions.'),
 ('theory_ko','비볼록 목적함수에 대한 확률적 경사 하강법의 수렴을 증명한 연구를 찾아줘. 볼록 함수에만 적용되는 결과는 제외해줘.'),
 ('recommend_en','A recommendation method using implicit user item interactions that handles the absence of explicit ratings.'),
 ('recommend_ko','별점 같은 명시적 평가 없이 클릭이나 구매 같은 암묵적 사용자 행동으로 추천하는 방법을 찾아줘.'),
]


def main():
    torch.set_num_threads(1)
    src=ROOT/'output/balanced_benchmark_20261117';out=src/'learned_dtm_live_review';data=src/'data'
    write(out/'requests.json',[dict(name=n,query=q) for n,q in REQUESTS])
    papers=read(data/'arxiv_master_final.json');ids=[str(p['paper_id']) for p in papers];local={p['node_idx']:i for i,p in enumerate(papers)}
    edges=torch.load(data/'paper_paper_edges.pt',weights_only=True,map_location='cpu').T.tolist();graph=Graph(None,ids,relations=('citation',),citation_edges=[(ids[local[s]],ids[local[t]]) for s,t in edges])
    extractor=Features(papers,Index(papers),DTMIndex(papers),graph)
    selected=read(src/'learned_dtm_graph/selected.json');modelpath=src/'learned_dtm_graph'/selected['model_file'];assert file_hash(modelpath)==selected['model_sha256'];model=joblib.load(modelpath)
    results=[]
    for name,query in REQUESTS:
        start=time.perf_counter();state=extractor.prepare(query);lex=state['lex'];mass=state['mass']
        if lex.max()>0:
            candidates=sorted(set(np.argsort(-lex,kind='stable')[:200])|set(np.argsort(-mass,kind='stable')[:200]))
            score=model.predict(extractor.rows(state,candidates));order=np.argsort(-score,kind='stable')[:5]
            learned=[candidates[i] for i in order];baseline=np.argsort(-mass,kind='stable')[:5].tolist()
        else:learned=[];baseline=[]
        union=sorted(set(learned)|set(baseline))
        results.append(dict(name=name,query=query,has_lexical_signal=bool(lex.max()),search_ms=(time.perf_counter()-start)*1000,learned=[ids[i] for i in learned],baseline=[ids[i] for i in baseline],papers=[dict(paper_id=ids[i],title=papers[i]['title'],abstract=papers[i]['abstract'],dtm=papers[i].get('knowledge',{})) for i in union]))
    write(out/'results.json',results)
    write(out/'manifest.json',dict(model_sha256=selected['model_sha256'],benchmark_queries_read=False,benchmark_labels_read=False,embeddings_used=False,online_llm_used=False,scope='Ten assistant-authored requests, five English/Korean topic pairs; top5 actual live pipeline candidates',review_limitations='Single-assistant unblinded title/abstract review, not independent human evaluation or full-text verification'))
    for r in results:
        print(r['name'],r['learned'])


if __name__=='__main__':main()
