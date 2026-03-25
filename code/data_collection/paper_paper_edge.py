import pandas as pd
import torch
import os
import json
import numpy as np

# --- 경로 설정 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SUBDATASET_DIR = os.path.join(project_root, 'subdataset')
MASTER_DATA_FILE = os.path.join(SUBDATASET_DIR, 'arxiv_master_final.json')
# ⭐ MAG ID와 내부 인덱스를 연결해주는 원본 매핑 파일
MAPPING_CSV = os.path.join(project_root, 'dataset', 'ogbn_arxiv', 'mapping', 'nodeidx2paperid.csv')
ORIGINAL_EDGE_FILE = r"C:\Users\Arachne\OneDrive\Desktop\arxiv-conversational-recommender-main\dataset\ogbn_arxiv\raw\edge.csv.gz"
OUTPUT_EDGE_FILE = os.path.join(SUBDATASET_DIR, 'paper_paper_edges.pt')

def rebuild_sub_edge_index_final():
    print(f"[System] 1. ID 변환 매핑 로드 중...")
    # nodeidx2paperid.csv 로드 (node idx, paper id 컬럼)
    map_df = pd.read_csv(MAPPING_CSV)
    # MAG ID(paper id) -> 내부 인덱스(node idx) 딕셔너리 생성
    mag_to_internal = dict(zip(map_df['paper id'].astype(str), map_df['node idx']))

    print(f"[System] 2. 마스터 JSON 분석 중...")
    with open(MASTER_DATA_FILE, 'r', encoding='utf-8') as f:
        master_data = json.load(f)

    # {내부_인덱스: 새_node_idx} 매핑 생성
    # 예: {104447: 14, ...}
    internal_to_new = {}
    for item in master_data:
        mag_id = str(item['paper_id'])
        new_idx = item['node_idx']
        
        if mag_id in mag_to_internal:
            internal_id = mag_to_internal[mag_id]
            internal_to_new[internal_id] = new_idx

    valid_internal_ids = set(internal_to_new.keys())
    print(f"💡 매핑 완료: {len(internal_to_new)}개 노드 연결됨")

    print(f"[System] 3. 원본 에지 파일 로딩 및 필터링...")
    edges = pd.read_csv(ORIGINAL_EDGE_FILE, compression='gzip', header=None, names=['src', 'dst'])
    
    # 이제 src, dst(내부 인덱스)가 valid_internal_ids에 있는지 확인 가능!
    sub_edges = edges[edges['src'].isin(valid_internal_ids) & edges['dst'].isin(valid_internal_ids)].copy()
    print(f"✅ 필터링 완료: {len(sub_edges):,}개의 인용 관계 발견")

    if len(sub_edges) == 0:
        print("❗ 여전히 0개입니다. 매핑 파일의 컬럼명을 확인해보세요.")
        return

    # 4. 새 인덱스(0~15999)로 최종 변환
    sub_edges['src_new'] = sub_edges['src'].map(internal_to_new)
    sub_edges['dst_new'] = sub_edges['dst'].map(internal_to_new)
    
    edge_index = torch.from_numpy(np.stack([
        sub_edges['src_new'].values, 
        sub_edges['dst_new'].values
    ])).long()
    
    torch.save(edge_index, OUTPUT_EDGE_FILE)
    print(f"🚀 저장 성공: {OUTPUT_EDGE_FILE}")

if __name__ == "__main__":
    rebuild_sub_edge_index_final()