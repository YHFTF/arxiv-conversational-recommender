import json
import os
import sys
import torch
import requests
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, SCRIPT_DIR)

from config import *

META_EMB_PATH = os.path.join(PROJECT_ROOT, 'output', 'knowledge_meta_embeddings.pt')

def get_openai_embeddings_batch(texts, api_key, batch_size=1024):
    """OpenAI API를 사용하여 텍스트 리스트의 128차원 임베딩을 배치 단위로 조회합니다."""
    url = "https://api.openai.com/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    embeddings = [None] * len(texts)
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        print(f"   -> 배치 처리 중: {i} ~ {min(i+batch_size, len(texts))} / {len(texts)}")
        
        data = {
            "model": "text-embedding-3-small",
            "input": batch_texts,
            "dimensions": 128
        }
        
        retries = 3
        success = False
        while retries > 0 and not success:
            try:
                response = requests.post(url, headers=headers, json=data, timeout=60)
                response.raise_for_status()
                res_json = response.json()
                
                # 반환된 결과를 index 순서에 맞게 배치
                for item in res_json['data']:
                    idx = item['index']
                    embeddings[i + idx] = item['embedding']
                success = True
            except Exception as e:
                print(f"      [오류 발생] {e}. {retries-1}번 재시도합니다...")
                retries -= 1
                time.sleep(2)
        
        if not success:
            raise RuntimeError("OpenAI Embedding API 배치 조회에 실패했습니다.")
            
    return torch.tensor(embeddings, dtype=torch.float32)


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n[ERROR] OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        return

    print("=" * 80)
    print("🚀 [지식 메타데이터 전체 시맨틱 벡터 사전 구축] 프로세스 시작")
    print("=" * 80)

    # 1. 메타데이터 로드
    with open(META_PATH, 'r', encoding='utf-8') as f:
        meta = json.load(f)

    # 각 메타데이터 카테고리별로 정렬된 키 리스트 추출 (ID 순서대로 정렬)
    # meta['domains'] 등은 { "텍스트": ID } 형태입니다.
    def get_sorted_keys(vocab_dict):
        # ID는 1부터 시작하므로, ID-1 인덱스에 배치할 수 있도록 정렬
        sorted_pairs = sorted(vocab_dict.items(), key=lambda x: x[1])
        return [pair[0] for pair in sorted_pairs]

    domain_keys = get_sorted_keys(meta['domains'])
    task_keys = get_sorted_keys(meta['tasks'])
    method_keys = get_sorted_keys(meta['methods'])

    print(f"🔹 분석 대상 요약:")
    print(f"   - Domains: {len(domain_keys)} 개")
    print(f"   - Tasks  : {len(task_keys)} 개")
    print(f"   - Methods: {len(method_keys)} 개")
    print("-" * 80)

    # 2. 임베딩 사전 구축
    print("\n[1/3] Domains 임베딩 조회 중...")
    domain_embs = get_openai_embeddings_batch(domain_keys, api_key)
    
    print("\n[2/3] Tasks 임베딩 조회 중...")
    task_embs = get_openai_embeddings_batch(task_keys, api_key)
    
    print("\n[3/3] Methods 임베딩 조회 중...")
    method_embs = get_openai_embeddings_batch(method_keys, api_key)

    # 3. 파일 저장
    embedding_data = {
        'domains': domain_embs, # Shape: [Num_Domains, 128]
        'tasks': task_embs,     # Shape: [Num_Tasks, 128]
        'methods': method_embs  # Shape: [Num_Methods, 128]
    }

    torch.save(embedding_data, META_EMB_PATH)
    print("=" * 80)
    print(f"✅ 메타데이터 시맨틱 임베딩 사전 구축 완료!")
    print(f"   💾 저장 경로: {META_EMB_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()
