import torch
import os
import json
from torch_geometric.data import HeteroData

# --- 경로 설정 ---
project_root = r"C:\Users\Arachne\OneDrive\Desktop\arxiv-conversational-recommender-main"
SUBDATASET_DIR = os.path.join(project_root, "subdataset")
OUTPUT_DIR = os.path.join(project_root, "output")

HETERO_FILE = os.path.join(SUBDATASET_DIR, "build_hetero_graph_v2.pt")
META_FILE = os.path.join(OUTPUT_DIR, "knowledge_meta.json")
MASTER_FILE = os.path.join(SUBDATASET_DIR, "arxiv_master_final.json")

def verify_v2_6column_knowledge():
    if not all(os.path.exists(f) for f in [HETERO_FILE, META_FILE, MASTER_FILE]):
        print("❌ 에러: 검증에 필요한 파일이 누락되었습니다.")
        return

    # 1. 데이터 로드
    data = torch.load(HETERO_FILE, weights_only=False)
    
    with open(META_FILE, 'r', encoding='utf-8') as f:
        meta = json.load(f)
        
    with open(MASTER_FILE, 'r', encoding='utf-8') as f:
        master_data = json.load(f)
        master_map = {item['node_idx']: item.get('paper_id', item.get('arxiv_id', 'N/A')) for item in master_data}

    # 2. Reverse Map (Idx -> Name)
    d_rev = {idx: name for name, idx in meta.get('domains', {}).items()}
    t_rev = {idx: name for name, idx in meta.get('tasks', {}).items()}
    m_rev = {idx: name for name, idx in meta.get('methods', {}).items()}

    # 3. 엣지 데이터 추출 (Paper -> Topic)
    edge_index = data['paper', 'has_topic', 'topic'].edge_index
    edge_attr = data['paper', 'has_topic', 'topic'].edge_attr # [16000, 6] 예상

    print("\n" + "="*175)
    print(f"🕵️‍♂️ [V2-Extended] 6컬럼 지식 그래프 정밀 검증 ([D1, D2] | [T1, T2] | [M1, M2])")
    print("-" * 175)
    # 헤더 설정
    header = f"{'G_Idx':<6} | {'ID':<12} | {'Top':<3} | {'Domains (Idx1, Idx2)':<45} | {'Tasks (Idx1, Idx2)':<45} | {'Methods (Idx1, Idx2)'}"
    print(header)
    print("-" * 175)

    # --- 수정된 노드 선택 부분 ---
    num_samples = 10  # 확인하고 싶은 랜덤 샘플 개수
    all_paper_indices = edge_index[0].unique() # 에지가 존재하는 논문 인덱스들
    
    # 랜덤하게 10개 추출
    if len(all_paper_indices) >= num_samples:
        random_indices = torch.randperm(len(all_paper_indices))[:num_samples]
        test_nodes = all_paper_indices[random_indices].tolist()
    else:
        test_nodes = all_paper_indices.tolist()
    
    test_nodes.sort() # 보기 편하게 정렬
    # --------------------------
    
    for g_idx in test_nodes:
        mask = (edge_index[0] == g_idx)
        if mask.any():
            idx = mask.nonzero(as_tuple=True)[0][0]
            t_hub = edge_index[1, idx].item()
            attr = edge_attr[idx].tolist() # [D1, D2, T1, T2, M1, M2]
            
            p_id = master_map.get(g_idx, "Unknown")
            
            # 정보 복원 및 포맷팅 함수
            def format_pair(idx1, idx2, rev_map):
                name1 = rev_map.get(idx1, "None")[:18]
                name2 = rev_map.get(idx2, "None")[:18]
                # 인덱스가 -1인 경우 None으로 표시
                val1 = f"{name1}({idx1})" if idx1 != -1 else "(-)"
                val2 = f"{name2}({idx2})" if idx2 != -1 else "(-)"
                return f"{val1}, {val2}"

            d_display = format_pair(attr[0], attr[1], d_rev)
            t_display = format_pair(attr[2], attr[3], t_rev)
            m_display = format_pair(attr[4], attr[5], m_rev)
            
            print(f"{g_idx:<6} | {p_id:<12} | {t_hub:<3} | {d_display:<45} | {t_display:<45} | {m_display}")
        else:
            print(f"{g_idx:<6} | 데이터 없음")

    print("="*175)
    print(f"📊 엣지 속성 텐서 크기: {edge_attr.shape} (모든 논문이 6개의 지식 슬롯 보유)")
    print("="*175)

if __name__ == "__main__":
    verify_v2_6column_knowledge()