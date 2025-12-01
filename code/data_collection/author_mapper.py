import torch
import pandas as pd
import requests
import json
import time
import os
import sys

# --- 1. 사용자 설정 ---
# 이메일을 넣으면 OpenAlex가 더 빠르게 처리해줍니다.
USER_EMAIL = "test@example.com" 

# --- 2. 경로 설정 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))

FFS_LOAD_FILE = os.path.join(project_root, 'subdataset', 'ogbn_arxiv_16k_ffs_sample.pt')
NODE_TO_ID_MAP_PATH = os.path.join(project_root, 'dataset', 'ogbn_arxiv', 'mapping', 'nodeidx2paperid.csv')
OUTPUT_DIR = os.path.join(project_root, 'output')
OUTPUT_AUTHOR_FILE = os.path.join(OUTPUT_DIR, 'author_data_openalex.json')

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 안전하게 40개씩 처리
BATCH_SIZE = 40

# --- 3. 데이터 로드 ---
print(f"OpenAlex API 모드 시작... (이메일: {USER_EMAIL})")

_real_torch_load = torch.load
def _torch_load_with_weights_only_false(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _real_torch_load(*args, **kwargs)
torch.load = _torch_load_with_weights_only_false 

try:
    if not os.path.exists(FFS_LOAD_FILE):
        sys.exit(f"샘플 파일 없음: {FFS_LOAD_FILE}")
        
    loaded_data = torch.load(FFS_LOAD_FILE)
    sampled_indices = set(loaded_data['indices']) 
    print(f"FFS 샘플 로드 완료: {len(sampled_indices)}개")

    if not os.path.exists(NODE_TO_ID_MAP_PATH):
        sys.exit(f"매핑 파일 없음: {NODE_TO_ID_MAP_PATH}")

    mapping_df = pd.read_csv(NODE_TO_ID_MAP_PATH)
    mapping_df.columns = [str(c).strip().lower() for c in mapping_df.columns]
    
    if 'node idx' in mapping_df.columns: mapping_df.rename(columns={'node idx': 'idx'}, inplace=True)
    if 'idx' not in mapping_df.columns: 
        mapping_df.rename(columns={mapping_df.columns[0]: 'idx', mapping_df.columns[1]: 'paper id'}, inplace=True)

    target_mapping = mapping_df[mapping_df['idx'].isin(sampled_indices)].copy()
    print(f"매핑 대상: {len(target_mapping)}개")

except Exception as e:
    sys.exit(f"초기화 에러: {e}")


# --- 4. OpenAlex API 함수 (수정됨) ---
def fetch_authors_openalex(node_to_paper_map):
    results = {}
    items = list(node_to_paper_map.items()) 
    total = len(items)
    
    print(f"OpenAlex 호출 시작 (총 {total}건, Batch {BATCH_SIZE})...")
    
    base_url = "https://api.openalex.org/works"
    
    for i in range(0, total, BATCH_SIZE):
        batch_items = items[i : i + BATCH_SIZE]
        
        # [수정된 부분] 'mag:' 접두사 제거! 순수 ID 숫자만 파이프로 연결
        # item[1]이 Paper ID (MAG ID)
        mag_ids_str = "|".join([str(item[1]).strip() for item in batch_items])
        
        params = {
            "filter": f"ids.mag:{mag_ids_str}", # 여기에 ids.mag: 가 이미 있으므로 뒤에는 숫자만 옴
            "per_page": BATCH_SIZE,
            "select": "ids,authorships" 
        }
        
        if USER_EMAIL != "test@example.com":
            params["mailto"] = USER_EMAIL

        try:
            response = requests.get(base_url, params=params, timeout=20)
            
            if response.status_code == 200:
                data = response.json()
                results_list = data.get('results', [])
                
                # 매핑용 임시 딕셔너리
                batch_result_map = {}
                for work in results_list:
                    # 결과에서 MAG ID 찾기 (숫자형일 수 있으므로 문자열로 통일)
                    mag_val = work.get('ids', {}).get('mag')
                    if mag_val:
                        authorships = work.get('authorships', [])
                        names = [a.get('author', {}).get('display_name') for a in authorships]
                        names = [n for n in names if n]
                        batch_result_map[str(mag_val)] = names
                
                # 원본 요청 순서대로 결과 저장
                for idx, pid in batch_items:
                    authors = batch_result_map.get(str(pid), [])
                    results[idx] = authors
                    
            else:
                print(f"API 오류 {response.status_code} (Batch {i}). 요청 URL 확인 필요.")
                # 디버깅을 위해 실패한 URL 일부 출력
                print(f"   (Failed Params: {params['filter'][:50]}...)")
                for idx, _ in batch_items: results[idx] = []
                
        except Exception as e:
            print(f"네트워크/파싱 오류: {e}")
            for idx, _ in batch_items: results[idx] = []
            
        # 진행률
        if (i + BATCH_SIZE) % 1000 < BATCH_SIZE:
            current = min(i + BATCH_SIZE, total)
            print(f"  - {current}/{total} 완료 ({current/total*100:.1f}%)")
            
        time.sleep(0.1) # OpenAlex는 매우 빠르므로 짧은 대기만
        
    return results

# --- 5. 실행 및 저장 ---
node_to_paper = dict(zip(target_mapping['idx'], target_mapping['paper id']))

author_map_result = fetch_authors_openalex(node_to_paper)

final_data = []
for idx, authors in author_map_result.items():
    final_data.append({
        "node_idx": int(idx),
        "paper_id": node_to_paper[idx],
        "authors": authors
    })

found_count = sum(1 for item in final_data if item['authors'])
print("-" * 50)
print(f"수집 완료!")
print(f"총 요청: {len(final_data)}건")
print(f"저자 찾음: {found_count}건 ({found_count/len(final_data)*100:.1f}%)")

with open(OUTPUT_AUTHOR_FILE, 'w', encoding='utf-8') as f:
    json.dump(final_data, f, ensure_ascii=False, indent=4)

print(f"파일 저장됨: {OUTPUT_AUTHOR_FILE}")