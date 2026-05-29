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
from utils_v2 import load_data, get_graph_info, build_unified_graph_v2
from LGCmodel_v4 import ArxivLightGCNV4

META_EMB_PATH = os.path.join(PROJECT_ROOT, 'output', 'knowledge_meta_embeddings.pt')


def extract_semantic_specs_from_nl(nl_query, api_key):
    """OpenAI API를 사용하여 자연어 질문에서 최적의 Domain, Task, Method 카테고리를 도출합니다."""
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    prompt = f"""
    You are an expert AI academic librarian.
    The user is searching for research papers with this query: "{nl_query}"
    
    Your task is to extract the single most relevant academic Domain, Task, and Method from this query.
    Extract them as clear, standardized English academic terms (e.g., 'robotics', 'path planning', 'control method').
    If a category is completely irrelevant, return null.
    
    Return ONLY a valid JSON object with the following keys:
    - "domain": string or null
    - "task": string or null
    - "method": string or null
    
    No explanations, no markdown blocks.
    """
    
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a helpful academic metadata translator that outputs only JSON."},
            {"role": "user", "content": prompt}
        ],
        "max_completion_tokens": 80,
        "temperature": 0.0,
        "response_format": {"type": "json_object"}
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        content = result['choices'][0]['message']['content'].strip()
        return json.loads(content)
    except Exception as e:
        print(f"\n[ERROR] OpenAI API 추출 중 오류 발생: {e}")
        return None


def get_openai_text_embedding_128d(text, api_key):
    """OpenAI의 text-embedding-3-small 모델을 사용하여 텍스트의 128차원 임베딩을 가져옵니다."""
    url = "https://api.openai.com/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    data = {
        "model": "text-embedding-3-small",
        "input": text,
        "dimensions": 128
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        res_json = response.json()
        embedding = res_json['data'][0]['embedding']
        return torch.tensor(embedding, dtype=torch.float32, device=DEVICE)
    except Exception as e:
        print(f"\n[ERROR] OpenAI Embedding API 연동 실패: {e}")
        return None


def find_best_semantic_match(term, vocab_dict, vocab_embs, sorted_keys, api_key, threshold=0.35):
    """
    1. 대소문자 구분 없이 완벽 일치하는 단어가 있다면 즉시 ID 반환 (API 호출 최소화).
    2. 완벽 일치가 없다면 extracted_term의 임베딩을 생성하여, 전체 vocab 임베딩과의 코사인 유사도를 바탕으로 매핑.
    """
    if not term:
        return 0, "N/A", 0.0
    
    term_lower = term.strip().lower()
    
    # 1. 완벽 일치 검사 (대소문자 무시)
    for k, v in vocab_dict.items():
        if k.lower() == term_lower:
            return v, k, 1.0
            
    # 2. 제로샷 시맨틱 벡터 매칭
    term_emb = get_openai_text_embedding_128d(term, api_key)
    if term_emb is None:
        return 0, "Embedding Failed", 0.0
        
    # vocab_embs는 [Num_Vocabs, 128] 형태
    vocab_embs_dev = vocab_embs.to(DEVICE)
    
    # 코사인 유사도 연산
    sims = torch.cosine_similarity(term_emb.unsqueeze(0), vocab_embs_dev)
    max_sim, best_idx = torch.max(sims, dim=0)
    max_sim = max_sim.item()
    best_idx = best_idx.item()
    
    if max_sim >= threshold:
        matched_term = sorted_keys[best_idx]
        matched_id = best_idx + 1 # ID는 1부터 시작하므로
        return matched_id, matched_term, max_sim
    else:
        return 0, f"No Match (Best: {sorted_keys[best_idx]} with {max_sim:.4f})", max_sim


def main():
    parser = argparse.ArgumentParser(description="End-to-End [V5 순수 신경망형] 제로샷 시맨틱 벡터 매치 모델")
    parser.add_argument('--top_k', type=int, default=5, help="추천받을 논문 개수")
    parser.add_argument('--threshold', type=float, default=0.35, help="시맨틱 매칭 유사도 임계값")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n[ERROR] OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        return

    print("=" * 105)
    print("⚙️ [End-to-End V5] [시맨틱 벡터 매칭 엔진] Ours v4 지식 임베딩 엔진 활성화 중...")
    print("=" * 105)
    
    # 1. 데이터 로드
    raw_data, meta, meta_counts = load_data()
    num_papers, _, _, _ = get_graph_info(raw_data)
    
    with open(MASTER_FILE, 'r', encoding='utf-8') as f:
        master_list = json.load(f)

    # 2. 사전 구축된 메타 임베딩 존재 여부 확인
    if not os.path.exists(META_EMB_PATH):
        print(f"\n[!] 사전 구축된 메타 임베딩 파일({META_EMB_PATH})이 없습니다.")
        print("    먼저 'python code/test/precompute_meta_embeddings.py'를 실행하여 메타 임베딩을 빌드해주세요.")
        return

    print(f"💾 사전 빌드된 메타 임베딩 로드 중... ({META_EMB_PATH})")
    vocab_embeddings = torch.load(META_EMB_PATH, map_location=DEVICE)
    
    # 정렬된 키 리스트 준비 (사전 구축 시 정렬 기준과 완전 동일해야 함)
    def get_sorted_keys(vocab_dict):
        sorted_pairs = sorted(vocab_dict.items(), key=lambda x: x[1])
        return [pair[0] for pair in sorted_pairs]

    domain_keys = get_sorted_keys(meta['domains'])
    task_keys = get_sorted_keys(meta['tasks'])
    method_keys = get_sorted_keys(meta['methods'])

    # Ours v4 전용 가중치 복원
    save_path = os.path.join(OUTPUT_DIR, "benchmark_lightgcn__knowledge_(ours)_v2.pt")
    if not os.path.exists(save_path):
        save_path = os.path.join(OUTPUT_DIR, "benchmark_lightgcn__knowledge_(ours).pt")
        
    if not os.path.exists(save_path):
        print(f"\n[ERROR] 학습된 Ours 모델의 가중치를 '{save_path}'에서 찾을 수 없습니다.")
        return

    from utils_v2 import build_knowledge_ids
    paper_knowledge_ids = build_knowledge_ids(raw_data, meta)

    # 모델 선언 및 로드
    model = ArxivLightGCNV4(raw_data, meta_counts, paper_knowledge_ids, knowledge_weight=KNOWLEDGE_WEIGHT).to(DEVICE)
    model.load_state_dict(torch.load(save_path, weights_only=True))
    model.eval()

    print("✅ V5 순수 딥러닝 + 제로샷 시맨틱 매칭 엔진 준비 완료!")
    print("=" * 105)

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

        print("\n🤖 [STEP 1] 트랜스포머(Embedding) 및 GNN 지식 신경망 작동 중...")
        
        # 1. 트랜스포머 API를 통한 128차원 의미 벡터 실시간 인코딩
        raw_text_emb = get_openai_text_embedding_128d(nl_query, api_key)
        if raw_text_emb is None:
            print("   - 텍스트 임베딩 생성에 실패했습니다. 다시 시도해 주세요.")
            continue
            
        # 2. LLM 시맨틱 파싱
        specs = extract_semantic_specs_from_nl(nl_query, api_key)
        if not specs:
            print("   - 시맨틱 카테고리 분석에 실패했습니다. 다시 시도해 주세요.")
            continue
            
        print("   - 💡 LLM 시맨틱 파싱 및 제로샷 벡터 정렬 결과:")
        
        # 3. [핵심] 제로샷 시맨틱 벡터 매칭 수행
        d_term = specs.get('domain')
        t_term = specs.get('task')
        m_term = specs.get('method')
        
        d_id, d_matched, d_sim = find_best_semantic_match(d_term, meta['domains'], vocab_embeddings['domains'], domain_keys, api_key, args.threshold)
        t_id, t_matched, t_sim = find_best_semantic_match(t_term, meta['tasks'], vocab_embeddings['tasks'], task_keys, api_key, args.threshold)
        m_id, m_matched, m_sim = find_best_semantic_match(m_term, meta['methods'], vocab_embeddings['methods'], method_keys, api_key, args.threshold)
        
        print(f"      Domain : {d_term} -> Match: '{d_matched}' (Sim: {d_sim:.4f}) | ID: {d_id}")
        print(f"      Task   : {t_term} -> Match: '{t_matched}' (Sim: {t_sim:.4f}) | ID: {t_id}")
        print(f"      Method : {m_term} -> Match: '{m_matched}' (Sim: {m_sim:.4f}) | ID: {m_id}")

        print("\n🎯 [STEP 2] 모델 지식 임베딩 공간으로 직접 쿼리 번역 및 논문 추천 중...")
        
        with torch.no_grad():
            d_vec = model.domain_emb(torch.tensor([d_id], device=DEVICE))
            t_vec = model.task_emb(torch.tensor([t_id], device=DEVICE))
            m_vec = model.method_emb(torch.tensor([m_id], device=DEVICE))
            
            # OpenAI가 구운 128차원 텍스트 벡터에 GNN 지식 임베딩 합산
            query_embedding = raw_text_emb.unsqueeze(0) + KNOWLEDGE_WEIGHT * (d_vec + t_vec + m_vec)

            # 전체 16000 편 모델 임베딩 로드
            train_unified_edges = build_unified_graph_v2(raw_data)
            out_embeddings = model(train_unified_edges)
            paper_embeddings = out_embeddings[:num_papers]
            
            # 코사인 유사도 연산 (순수 벡터 공간 거리 연산)
            sim_scores = torch.cosine_similarity(query_embedding, paper_embeddings)
            values, indices = torch.topk(sim_scores, k=args.top_k)

        print(f"\n📊 [{args.top_k} 개의 End-to-End 순수 딥러닝 추천 결과 (V3)]")
        print("-" * 105)
        
        for rank in range(args.top_k):
            idx = indices[rank].item()
            score = values[rank].item()
            rec_item = master_list[idx]
            
            print(f"[{rank + 1}] 유사도: {score:.4f} | 제목: {rec_item['title']}")
            k = rec_item.get('knowledge', {})
            print(f"    - Domain: {k.get('domain', 'N/A')} | Task: {k.get('task', 'N/A')} | Method: {k.get('method', 'N/A')}")
            print("-" * 105)


if __name__ == "__main__":
    main()
