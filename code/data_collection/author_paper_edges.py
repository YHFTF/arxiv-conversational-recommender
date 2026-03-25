import json
import os
import torch

# --- 1. 경로 설정 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SUBDATASET_DIR = os.path.join(project_root, 'subdataset')
OUTPUT_DIR = os.path.join(project_root, 'output')

# 모든 데이터의 기준점이 되는 마스터 JSON
MASTER_DATA_FILE = os.path.join(SUBDATASET_DIR, 'arxiv_master_final.json')

# 최종 결과 저장 경로
OUTPUT_AUTHOR_PAPER_FILE = os.path.join(SUBDATASET_DIR, 'author_paper_edges.pt')
AUTHOR_MAPPING_FILE = os.path.join(OUTPUT_DIR, 'author_remapping.json')

def extract_author_paper_edges_from_master():
    print("🚀 [System] 저자-논문 관계 추출 작업을 시작합니다.")
    
    # 디렉토리 존재 확인
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 마스터 데이터 로드
    print(f"📦 마스터 JSON 로딩 중: {MASTER_DATA_FILE}")
    if not os.path.exists(MASTER_DATA_FILE):
        print(f"❌ 에러: {MASTER_DATA_FILE} 파일을 찾을 수 없습니다.")
        return

    with open(MASTER_DATA_FILE, 'r', encoding='utf-8') as f:
        master_data = json.load(f)

    author_paper_pairs = []
    author_info_map = {} # {auth_id: auth_name}
    
    print("🔗 관계 분석 및 저자 인덱싱 중...")
    
    for item in master_data:
        # 마스터 JSON에 정의된 논문 인덱스 (0~15999)
        p_idx = item.get('node_idx')
        if p_idx is None: continue
        
        authors = item.get('authors', [])

        for auth in authors:
            auth_id = None
            auth_name = None

            # 케이스 1: 저자 정보가 딕셔너리 형태일 경우 (id와 name 추출)
            if isinstance(auth, dict):
                auth_id = str(auth.get('author_id', ''))
                # 이름이 없으면 ID로 대체
                auth_name = auth.get('author_name', f"Unknown_{auth_id}")
            
            # 케이스 2: 저자 정보가 단순 문자열(이름)일 경우 (이름을 ID로 활용)
            elif isinstance(auth, str):
                auth_id = auth.strip()
                auth_name = auth.strip()
            
            if auth_id and auth_name:
                # [저자ID, 논문인덱스] 쌍 저장
                author_paper_pairs.append({'auth_id': auth_id, 'paper_idx': p_idx})
                
                # 저자 ID에 대응하는 실명 저장 (중복 방지)
                if auth_id not in author_info_map:
                    author_info_map[auth_id] = auth_name

    if not author_paper_pairs:
        print("⚠️ 경고: 추출된 저자 관계가 없습니다. JSON 형식을 확인하세요.")
        return

    # --- 1. 저자 고유 ID 재매핑 (정렬 후 0부터 번호 부여) ---
    unique_author_ids = sorted(list(author_info_map.keys()))
    author_id_to_idx = {auth_id: i for i, auth_id in enumerate(unique_author_ids)}
    
    # --- 2. 에지 리스트(Tensor) 생성 ---
    edge_list = []
    for pair in author_paper_pairs:
        # 저자 ID를 위에서 만든 새 인덱스로 치환
        auth_idx = author_id_to_idx[pair['auth_id']]
        paper_idx = pair['paper_idx']
        edge_list.append([auth_idx, paper_idx])
    
    # PyTorch Geometric 표준 포맷 [2, E] 텐서로 변환
    # (0행: 저자 인덱스, 1행: 논문 인덱스)
    author_paper_edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    # --- 3. 파일 저장 ---
    # 3-1. 에지 텐서 (.pt)
    torch.save(author_paper_edge_index, OUTPUT_AUTHOR_PAPER_FILE)
    
    # 3-2. 매핑 정보 (.json) - 디버깅 및 결과 확인용
    final_author_map = {
        "id_to_idx": author_id_to_idx,
        "idx_to_name": {i: author_info_map[auth_id] for auth_id, i in author_id_to_idx.items()}
    }
    with open(AUTHOR_MAPPING_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_author_map, f, ensure_ascii=False, indent=4)

    print("\n" + "="*50)
    print(f"✅ [작업 완료]")
    print(f" - 결과 파일 1 (에지): {OUTPUT_AUTHOR_PAPER_FILE}")
    print(f" - 결과 파일 2 (매핑): {AUTHOR_MAPPING_FILE}")
    print("-" * 50)
    print(f" 📊 통계:")
    print(f"  - 고유 저자 수: {len(unique_author_ids):,}명")
    print(f"  - 생성된 연결(Edge) 수: {author_paper_edge_index.size(1):,}개")
    print("="*50)

if __name__ == "__main__":
    extract_author_paper_edges_from_master()