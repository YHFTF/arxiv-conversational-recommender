import json
import os
import pandas as pd
import torch

# 경로 설정
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
AUTHOR_DATA_FILE = os.path.join(project_root, 'output', 'author_data_openalex.json')
NODE_MAPPING_FILE = os.path.join(project_root, 'output', 'node_remapping.json')

OUTPUT_AUTHOR_PAPER_FILE = os.path.join(project_root, 'subdataset', 'author_paper_edges.pt')
AUTHOR_MAPPING_FILE = os.path.join(project_root, 'output', 'author_remapping.json')

def extract_author_paper_edges():
    print("[System] 데이터 로딩 중...")
    
    with open(AUTHOR_DATA_FILE, 'r', encoding='utf-8') as f:
        author_data = json.load(f)
        
    with open(NODE_MAPPING_FILE, 'r', encoding='utf-8') as f:
        node_mapping = json.load(f)

    author_paper_pairs = []
    author_info_map = {} 
    
    print("[System] 저자-논문 관계 추출 시작...")
    for item in author_data:
        # node_idx를 가져와서 문자열로 통일
        raw_paper_id = item.get('node_idx')
        if raw_paper_id is None: continue
        
        paper_id = str(raw_paper_id)
        authors = item.get('authors', [])

        if paper_id not in node_mapping:
            continue
            
        paper_idx = node_mapping[paper_id]
        
        for auth in authors:
            # 저자 정보가 딕셔너리 형태일 경우
            if isinstance(auth, dict):
                auth_id = auth.get('author_id')
                auth_name = auth.get('author_name', f"Unknown_{auth_id}")
            # 저자 정보가 단순 문자열(ID)일 경우
            elif isinstance(auth, str):
                auth_id = auth
                auth_name = f"Author_{auth}" # 이름 정보가 없으므로 ID로 대체
            else:
                continue
            
            if auth_id:
                author_paper_pairs.append({'auth_id': auth_id, 'paper_idx': paper_idx})
                # 중복 방지하며 이름 저장
                if auth_id not in author_info_map or author_info_map[auth_id].startswith("Author_"):
                    author_info_map[auth_id] = auth_name

    if not author_paper_pairs:
        print("[Warning] 추출된 관계가 없습니다. 데이터 형식을 다시 확인해주세요.")
        return

    # 1. 저자 고유 ID 재매핑 (정렬하여 인덱스 부여)
    unique_author_ids = sorted(list(author_info_map.keys()))
    author_mapping = {auth_id: i for i, auth_id in enumerate(unique_author_ids)}
    
    # 2. 에지 리스트 생성
    edge_list = []
    for pair in author_paper_pairs:
        auth_idx = author_mapping[pair['auth_id']]
        edge_list.append([auth_idx, pair['paper_idx']])
    
    # [2, E] 형태의 텐서로 변환
    author_paper_edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    # 3. 결과 저장
    torch.save(author_paper_edge_index, OUTPUT_AUTHOR_PAPER_FILE)
    
    final_author_map = {
        "id_to_idx": author_mapping,
        "idx_to_name": {i: author_info_map[auth_id] for auth_id, i in author_mapping.items()}
    }
    with open(AUTHOR_MAPPING_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_author_map, f, ensure_ascii=False, indent=4)

    print(f"[Success] 저자-논문 연결 완료!")
    print(f"  - 추출된 고유 저자 수: {len(unique_author_ids)}명")
    print(f"  - 생성된 연결(Edge) 수: {author_paper_edge_index.size(1)}개")

if __name__ == "__main__":
    extract_author_paper_edges()