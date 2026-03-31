import torch
import json
import os
from torch_geometric.data import HeteroData

# --- 경로 설정 ---
project_root = r"C:\Users\Arachne\OneDrive\Desktop\arxiv-conversational-recommender-main"
SUBDATASET_DIR = os.path.join(project_root, "subdataset")

MASTER_FILE = os.path.join(SUBDATASET_DIR, "arxiv_master_final.json")
SAMPLE_FEAT_FILE = os.path.join(SUBDATASET_DIR, "ogbn_arxiv_16k_ffs_sample.pt")
PP_EDGE_FILE = os.path.join(SUBDATASET_DIR, "paper_paper_edges.pt")
AP_EDGE_FILE = os.path.join(SUBDATASET_DIR, "author_paper_edges.pt")
SAVE_PATH = os.path.join(SUBDATASET_DIR, "build_hetero_graph.pt")

def build_graph_by_node_idx():
    print("\n" + "="*60)
    print("🚀 [1단계] 데이터 정렬 및 행 재배치 시작...")
    print("="*60)

    # 1. 파일 로드
    sample_data = torch.load(SAMPLE_FEAT_FILE, weights_only=False)
    with open(MASTER_FILE, 'r', encoding='utf-8') as f:
        master_list = json.load(f)
    
    # 2. 마스터 기준 설정 (JSON 순서대로 0, 1, 2...)
    target_node_indices = [item['node_idx'] for item in master_list]
    node_to_local = {node_id: i for i, node_id in enumerate(target_node_indices)}

    # 3. 피처 매핑 (indices 키 사용)
    # 🌟 민혁 님의 데이터에 확인된 'indices' 키를 직접 사용합니다.
    if 'indices' in sample_data:
        print("🎯 [Success] 'indices' 키를 사용하여 원본 ID를 매핑합니다.")
        raw_node_ids = sample_data['indices']
        if isinstance(raw_node_ids, torch.Tensor):
            raw_node_ids = raw_node_ids.tolist()
    else:
        print("❌ [Error] 'indices' 키를 찾을 수 없습니다. 다시 확인해주세요.")
        return

    raw_feats = sample_data['features']
    feat_map = {node_id: feat for node_id, feat in zip(raw_node_ids, raw_feats)}
    
    # 마스터 리스트 순서에 맞게 피처 행렬 재배치
    aligned_features = []
    for node_id in target_node_indices:
        aligned_features.append(feat_map.get(node_id, torch.zeros(128)))
    
    paper_x = torch.stack(aligned_features)
    print(f"📊 피처 정렬 완료: {paper_x.shape}")

    # 4. 에지 재매핑 (원본 ID -> 0~15999 번호로 번역)
    print("🔗 에지 재매핑 중...")
    pp_edge_raw = torch.load(PP_EDGE_FILE)
    ap_edge_raw = torch.load(AP_EDGE_FILE)

    def fast_remap(edge_index, is_paper_src, is_paper_dst):
        src, dst = edge_index[0].tolist(), edge_index[1].tolist()
        new_src, new_dst = [], []
        for s, d in zip(src, dst):
            s_idx = node_to_local.get(s, -1) if is_paper_src else s
            d_idx = node_to_local.get(d, -1) if is_paper_dst else d
            if s_idx != -1 and d_idx != -1:
                new_src.append(s_idx)
                new_dst.append(d_idx)
        return torch.tensor([new_src, new_dst], dtype=torch.long)

    # 5. 그래프 조립 (Paper 노드 피처 주입)
    data = HeteroData()
    data['paper'].x = paper_x
    data['paper'].node_id = torch.tensor(target_node_indices)
    data['paper', 'cites', 'paper'].edge_index = fast_remap(pp_edge_raw, True, True)
    data['author', 'writes', 'paper'].edge_index = fast_remap(ap_edge_raw, False, True)

    # Topic 에지 (라벨 기반 연결)
    labels = torch.tensor([item.get('label', 0) for item in master_list])
    paper_indices = torch.arange(len(target_node_indices))
    data['paper', 'has_topic', 'topic'].edge_index = torch.stack([paper_indices, labels], dim=0)
    
    # 6. Author/Topic 초기 벡터(평균) 주입
    print("✍️ 저자 및 토픽 초기 벡터 계산 중...")
    
    # Author 평균
    num_authors = data['author'].num_nodes = int(data['author', 'writes', 'paper'].edge_index[0].max()) + 1
    author_x = torch.zeros((num_authors, 128))
    ap_edge = data['author', 'writes', 'paper'].edge_index
    for i in range(num_authors):
        p_idx = ap_edge[1, ap_edge[0] == i]
        if len(p_idx) > 0: author_x[i] = paper_x[p_idx].mean(dim=0)
    data['author'].x = author_x

    # Topic 평균
    topic_x = torch.zeros((40, 128))
    pt_edge = data['paper', 'has_topic', 'topic'].edge_index
    for i in range(40):
        p_idx = pt_edge[0, pt_edge[1] == i]
        if len(p_idx) > 0: topic_x[i] = paper_x[p_idx].mean(dim=0)
    data['topic'].x = topic_x

    torch.save(data, SAVE_PATH)
    print(f"✅ [Final] 모든 노드가 정렬되고 초기화되었습니다: {SAVE_PATH}")

if __name__ == "__main__":
    build_graph_by_node_idx()