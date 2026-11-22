"""Fresh-query qualitative audit. No benchmark queries or relevance labels read."""
import json, os, sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'code/retrieval'),str(ROOT/'code/embeddings')]
from local_embeddings import LocalEmbedder
from run_nl_recommendation import load_catalog, build_embedding_query, query_signal_scores

# Written before inspecting retrieval results. Plans are authored by the reviewing
# assistant, not by the unavailable original API model.
CASES=[
 ('graph','노드 라벨이 적은 상황에서 그래프 연결 구조와 노드 특성을 함께 쓰는 반지도 노드 분류 방법을 찾고 싶어.',
  'semi-supervised node classification with few labeled nodes using graph structure and node features',
  ['semi-supervised node classification','graph structure','node features']),
 ('fpga','FPGA에서 신경망 추론을 가속하면서 전력 소모도 평가한 하드웨어 논문을 찾아줘.',
  'FPGA hardware acceleration of neural network inference with evaluation of power consumption',
  ['FPGA','neural network inference','power consumption evaluation']),
 ('privacy','여러 기관이 원본 데이터를 공유하지 않고 모델을 공동 학습하면서 차등 프라이버시를 보장하는 연구를 찾아줘.',
  'collaborative machine learning across institutions without sharing raw data with differential privacy guarantees',
  ['collaborative machine learning','without sharing raw data','differential privacy']),
 ('theory','확률적 경사 하강법에서 비볼록 목적함수의 수렴을 이론적으로 분석한 논문을 찾아줘.',
  'theoretical convergence analysis of stochastic gradient descent for nonconvex objective functions',
  ['stochastic gradient descent','nonconvex optimization','convergence analysis'])
]

def main():
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['TRANSFORMERS_OFFLINE']='1'
    torch.set_num_threads(2)
    out=ROOT/'output/independent_intent_review_20261117';out.mkdir(exist_ok=True)
    plans=[]
    for name,raw,rewrite,must in CASES:
        plans.append(dict(name=name,query=raw,plan=dict(rewritten_query=rewrite,must_terms=must,should_terms=[],exclude_terms=[])))
    (out/'predeclared_queries.json').write_text(json.dumps(plans,ensure_ascii=False,indent=2),encoding='utf-8')
    texts=list(dict.fromkeys(t for c in plans for t in [c['query'],build_embedding_query(c['plan']),*c['plan']['must_terms']]))
    cache=out/'vectors.pt'
    if cache.exists():
        v=torch.load(cache,weights_only=False);assert v['texts']==texts;vec=v['vectors']
    else:
        encoder=LocalEmbedder();vec=encoder.encode(texts,query=True)
        torch.save(dict(texts=texts,vectors=vec),cache)
        del encoder
        torch.cuda.empty_cache()
    lookup=dict(zip(texts,vec))
    base=ROOT/'output/balanced_benchmark_20261117'
    papers,pv,_=load_catalog(base/'data/arxiv_master_final.json',base/'embeddings/paper_text_embeddings.pt')
    results=[]
    for c in plans:
        raw=pv@lookup[c['query']];rewrite=pv@lookup[build_embedding_query(c['plan'])]
        tv={t:lookup[t] for t in c['plan']['must_terms']}
        adjustment,details=query_signal_scores(c['plan'],tv,pv,.08,0,0,.30,.45,.04)
        scores={'raw':raw,'rewrite':rewrite,'rewrite_must':rewrite+adjustment}
        rankings={k:torch.topk(v,5).indices.tolist() for k,v in scores.items()}
        union=sorted(set(i for r in rankings.values() for i in r))
        results.append(dict(name=c['name'],query=c['query'],plan=c['plan'],rankings={k:[papers[i]['paper_id'] for i in r] for k,r in rankings.items()},papers=[dict(papers[i],must_adjustment=float(adjustment[i]),rewrite_score=float(rewrite[i]),**details[i]) for i in union]))
    (out/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Saved',out/'results.json',flush=True)

if __name__=='__main__':main()
