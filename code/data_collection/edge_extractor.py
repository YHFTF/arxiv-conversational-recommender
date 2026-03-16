import pandas as pd
import torch
import os
import json

# --- 경로 설정 (확인하신 경로로 업데이트) ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SAMPLE_NODE_FILE = os.path.join(project_root, 'subdataset', 'ogbn_arxiv_16k_ffs_sample.pt')
# 알려주신 절대 경로를 사용합니다.
ORIGINAL_EDGE_FILE = r"C:\Users\Arachne\OneDrive\Desktop\arxiv-conversational-recommender-main\dataset\ogbn_arxiv\raw\edge.csv.gz"

OUTPUT_EDGE_FILE = os.path.join(project_root, 'subdataset', 'sub_edge_index.pt')
MAPPING_FILE = os.path.join(project_root, 'output', 'node_remapping.json')

def extract_subgraph_edges():
    print(f"[System] 16k 샘플 데이터 로딩 중: {SAMPLE_NODE_FILE}")
    try:
        checkpoint = torch.load(SAMPLE_NODE_FILE)
        # 분석 결과에 따라 'indices' 키에서 리스트 추출
        sample_nodes = checkpoint['indices']
        node_set = set(sample_nodes)
    except Exception as e:
        print(f"[Error] 샘플 노드 로드 실패: {e}")
        return

    print(f"[System] 노드 재매핑 생성 중... (총 {len(sample_nodes)}개)")
    # Original ID -> 0~15999 매핑 (정렬하여 일관성 유지)
    sorted_nodes = sorted(sample_nodes)
    mapping = {int(old_id): i for i, old_id in enumerate(sorted_nodes)}
    
    print(f"[System] 원본 에지 파일 분석 시작: {ORIGINAL_EDGE_FILE}")
    try:
        # csv.gz 파일을 직접 읽음 (헤더 없음, src, dst 컬럼)
        # 데이터가 크므로 필요한 컬럼만 지정하여 메모리 최적화
        edges = pd.read_csv(ORIGINAL_EDGE_FILE, compression='gzip', header=None, names=['src', 'dst'])
    except Exception as e:
        print(f"[Error] 원본 에지 파일 로드 실패: {e}")
        return

    # 출발지와 목적지 노드가 모두 우리 16k 셋에 포함된 경우만 필터링
    print("[System] 엣지 필터링 중... (Induced Subgraph 추출)")
    sub_edges = edges[edges['src'].isin(node_set) & edges['dst'].isin(node_set)].copy()
    
    print(f"[System] 추출된 내부 에지 수: {len(sub_edges)}")
    
    if len(sub_edges) == 0:
        print("[Warning] 추출된 에지가 없습니다. 노드 번호 범위를 다시 확인해주세요.")
        return

    # 새로운 인덱스(0~15999)로 변환
    sub_edges['src_new'] = sub_edges['src'].map(mapping)
    sub_edges['dst_new'] = sub_edges['dst'].map(mapping)
    
    # PyG(PyTorch Geometric) 포맷 [2, E] 텐서 생성
    edge_index = torch.tensor([sub_edges['src_new'].values, 
                               sub_edges['dst_new'].values], dtype=torch.long)
    
    # 결과 저장
    torch.save(edge_index, OUTPUT_EDGE_FILE)
    with open(MAPPING_FILE, 'w', encoding='utf-8') as f:
        json.dump(mapping, f)
        
    print(f"[Success] 에지 추출 및 매핑 완료!")
    print(f"  - 저장된 에지 텐서: {OUTPUT_EDGE_FILE}")
    print(f"  - 저장된 매핑 정보: {MAPPING_FILE}")

if __name__ == "__main__":
    extract_subgraph_edges()