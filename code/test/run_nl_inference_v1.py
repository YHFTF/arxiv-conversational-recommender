import torch
import sys
import os
import json
import argparse
import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'code', 'model'))

from config import *
from utils import (
    load_data, build_knowledge_ids, get_graph_info,
    build_unified_graph, build_initial_features
)
from run_benchmark import get_benchmark_models

def extract_keywords_from_nl(nl_query, api_key):
    """OpenAI API를 사용하여 자연어에서 핵심 키워드와 중요도(가중치)를 추출합니다."""
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    prompt = f"""
    You are an expert AI academic librarian. The user will provide a natural language query describing the type of research papers they want to find.
    Your task is to extract 2 to 4 distinct core keywords or short phrases that are highly likely to appear in relevant academic paper titles or metadata.
    Crucially, you must assign an importance weight (from 1 to 5) to each keyword, where 5 means the keyword is extremely critical to the user's core intent (e.g., a specific dataset, rare method, or core domain), and 1 means it is a secondary or generic term (e.g., 'performance optimization', 'analysis').
    
    Rules:
    - ALL EXTRACTED KEYWORDS MUST BE IN ENGLISH, regardless of the user query's language. Translate them if necessary.
    - Return ONLY a valid JSON object. Keys are the lowercase English keywords, values are integer weights. No markdown blocks, no explanations.
    - Example output: {{"heterogeneous graph": 5, "ogbn-mag": 5, "node classification": 3, "performance": 1}}
    
    User Query: "{nl_query}"
    """
    
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a helpful academic assistant that outputs only JSON."},
            {"role": "user", "content": prompt}
        ],
        "max_completion_tokens": 50,
        "temperature": 0.0,
        "response_format": {"type": "json_object"}
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        content = result['choices'][0]['message']['content'].strip()
        
        # JSON 파싱
        keyword_dict = json.loads(content)
        # 소문자로 정규화 및 가중치 형변환
        keywords_with_weights = {k.strip().lower(): float(v) for k, v in keyword_dict.items() if k.strip()}
        return keywords_with_weights
        
    except Exception as e:
        print(f"\n[ERROR] OpenAI API 호출 또는 파싱 중 오류 발생: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="자연어 이해 기반 논문 추천 시스템 (LLM + Graph Model)")
    parser.add_argument('--model', type=str, default="LightGCN + Knowledge (Ours)", help="사용할 백엔드 그래프 모델")
    parser.add_argument('--top_k', type=int, default=5, help="추천받을 논문 개수")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n[ERROR] OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        return

    print("=" * 80)
    print(f"⚙️ 백엔드 엔진 [{args.model}] 준비 중...")
    
    # 데이터 로드
    data, meta, meta_counts = load_data()
    num_papers, _, _, _ = get_graph_info(data)
    initial_features = build_initial_features(data)
    paper_knowledge_ids = build_knowledge_ids(data, meta)

    all_models = get_benchmark_models(data, meta_counts, initial_features, paper_knowledge_ids)
    
    target_model_info = None
    for m in all_models:
        if m['name'].lower() == args.model.lower():
            target_model_info = m
            break
            
    if not target_model_info:
        print(f"\n[ERROR] '{args.model}' 모델을 찾을 수 없습니다.")
        return

    model = target_model_info['model_builder']().to(DEVICE)
    save_name = target_model_info['name'].replace(' ', '_').replace('+', '').lower()
    save_path = os.path.join(OUTPUT_DIR, f"benchmark_{save_name}.pt")

    if not os.path.exists(save_path):
        print(f"\n[ERROR] '{save_path}' 가중치 파일이 없습니다.")
        return

    model.load_state_dict(torch.load(save_path, weights_only=True))
    model.eval()

    # 메타데이터 로드
    with open(MASTER_FILE, 'r', encoding='utf-8') as f:
        master_list = json.load(f)
        
    print("✅ 준비 완료! 자연어로 논문을 검색해 보세요.")
    print("=" * 80)

    # 인용수(Citation Count) 사전 계산 (Seed 논문 동점자 처리용)
    if ('paper', 'cites', 'paper') in data.edge_types:
        edge_index = data['paper', 'cites', 'paper'].edge_index
        cited_papers = edge_index[1]
        unique_indices, counts = torch.unique(cited_papers, return_counts=True)
        citation_counts = torch.zeros(num_papers, dtype=torch.long)
        citation_counts[unique_indices] = counts.cpu()
    else:
        citation_counts = torch.zeros(num_papers, dtype=torch.long)

    # 대화형 프롬프트 루프
    while True:
        try:
            nl_query = input("\n💬 질문 입력 (종료: 'q' 또는 'exit'): ").strip()
        except KeyboardInterrupt:
            break
            
        if not nl_query:
            continue
        if nl_query.lower() in ['q', 'exit', 'quit']:
            print("프로그램을 종료합니다.")
            break

        print("\n🤖 [STEP 1] LLM 자연어 분석 중...")
        extracted_keywords = extract_keywords_from_nl(nl_query, api_key)
        
        if not extracted_keywords:
            continue
            
        print("   - 💡 LLM이 추출한 가중치 기반 다중 키워드:")
        for kw, w in extracted_keywords.items():
            print(f"      [{w}점] {kw}")

        # 키워드 가중치 기반 후보 논문 스코어링
        paper_scores = []
        for i, item in enumerate(master_list):
            if i >= num_papers: break
            
            title = item['title'].lower()
            k = item.get('knowledge', {})
            
            d_val = k.get('domain', [])
            t_val = k.get('task', [])
            m_val = k.get('method', [])
            
            if isinstance(d_val, list): d_val = ' '.join(d_val)
            if isinstance(t_val, list): t_val = ' '.join(t_val)
            if isinstance(m_val, list): m_val = ' '.join(m_val)
            
            meta_text = f"{d_val} {t_val} {m_val}".lower()
            
            # 가중치 합산 계산
            match_score = sum(weight for kw, weight in extracted_keywords.items() if (kw in title or kw in meta_text))
            
            if match_score > 0:
                paper_scores.append((match_score, citation_counts[i].item(), i, item['title']))

        if not paper_scores:
            print(f"\n❌ 키워드 {list(extracted_keywords.keys())} 중 어느 것도 포함하는 논문을 DB에서 찾을 수 없습니다. 다르게 질문해보세요.")
            continue

        # 1순위: 가중치 점수 합, 2순위: 인용수 내림차순 정렬
        paper_scores.sort(key=lambda x: (x[0], x[1]), reverse=True)
        seed_papers = paper_scores[:3]
        seed_indices = [p[2] for p in seed_papers]

        print("\n🎯 [STEP 2] 기준(Seed) 논문 복수 매칭 완료 (가중치+인용수 반영):")
        for rank, p in enumerate(seed_papers, 1):
            print(f"   {rank}. {p[3]} (키워드 점수: {p[0]}점 | 인용수: {p[1]}회)")

        print("🔄 통합 유사도 계산 중...")
        with torch.no_grad():
            full_edges = build_unified_graph(data)
            out_embeddings = model(full_edges)
            paper_embeddings = out_embeddings[:num_papers]

        # 여러 Seed 논문의 벡터 평균을 내서 하나의 강력한 쿼리 벡터 생성
        seed_vecs = paper_embeddings[seed_indices]
        target_vec = seed_vecs.mean(dim=0).unsqueeze(0)
        
        sim_scores = torch.cosine_similarity(target_vec, paper_embeddings)
        
        # Seed 본인이 추천에 나오는 것을 방지하기 위해 넉넉히 뽑아서 거름
        values, indices = torch.topk(sim_scores, k=args.top_k + len(seed_indices))

        print(f"\n📊 [{target_model_info['name']}] 기반 최종 추천 논문 Top-{args.top_k}")
        print("-" * 100)
        
        count = 0
        for i in range(len(indices)):
            if count >= args.top_k:
                break
                
            idx = indices[i].item()
            # Seed로 사용된 논문은 추천 결과에서 제외
            if idx in seed_indices:
                continue
                
            score = values[i].item()
            rec_item = master_list[idx]
            count += 1
            
            print(f"[{count}] 유사도: {score:.4f} | 제목: {rec_item['title']}")
            k = rec_item.get('knowledge', {})
            print(f"    - Domain: {k.get('domain', 'N/A')} | Task: {k.get('task', 'N/A')} | Method: {k.get('method', 'N/A')}")
            print("-" * 100)

if __name__ == "__main__":
    main()
