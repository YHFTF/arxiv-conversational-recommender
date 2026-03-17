import torch
import json
import os
from collections import Counter
# 최신 PyTorch 보안 가이드에 따라 HeteroData를 허용 리스트에 추가
from torch_geometric.data import HeteroData
torch.serialization.add_safe_globals([HeteroData])

# --- 경로 설정 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
HETERO_GRAPH_FILE = os.path.join(project_root, 'subdataset', 'hetero_graph.pt')
AUTHOR_MAP_FILE = os.path.join(project_root, 'output', 'author_remapping.json')


def verify_graph_structure():
    if not os.path.exists(HETERO_GRAPH_FILE):
        print(f"[Error] 파일을 찾을 수 없습니다: {HETERO_GRAPH_FILE}")
        return

    # 1. 데이터 로드 (weights_only=False 설정)
    print(f"[System] 그래프 데이터 로딩 중: {HETERO_GRAPH_FILE}")
    data = torch.load(HETERO_GRAPH_FILE, weights_only=False)
    
    with open(AUTHOR_MAP_FILE, 'r', encoding='utf-8') as f:
        author_map = json.load(f)['idx_to_name']
    
    print("\n" + "="*50)
    print("📊 [스타형 이종 그래프 내부 검증 보고서]")
    print("="*50)

    # --- [검증 1: 주제 허브(Topic Hub)] ---
    if ('paper', 'has_topic', 'topic') in data.edge_types:
        pt_edge = data['paper', 'has_topic', 'topic'].edge_index
        topic_counts = Counter(pt_edge[1].tolist())
        print(f"1. 🏢 주제 노드(Topic Hub) 분포 (상위 5개)")
        for t_idx, count in topic_counts.most_common(5):
            print(f"   - Topic {t_idx:2d}번 허브: {count:5,d}개의 논문이 연결됨")
    else:
        print("1. 🏢 [Warning] 'has_topic' 관계를 찾을 수 없습니다.")

def analyze_author_network():
    # 1. 데이터 로드 (보안 설정 포함)
    from torch_geometric.data import HeteroData
    torch.serialization.add_safe_globals([HeteroData])
    data = torch.load(HETERO_GRAPH_FILE, weights_only=False)
    
    with open(AUTHOR_MAP_FILE, 'r', encoding='utf-8') as f:
        auth_map = json.load(f)['idx_to_name']

    # 2. 협업 에지 데이터 추출
    aa_edge = data['author', 'collaborates', 'author'].edge_index
    
    # 3. 주요 지표 계산
    # (1) 저자별 협업자 수 (Degree)
    collab_counts = Counter(aa_edge[0].tolist())
    
    # (2) 가장 강력한 협업 듀오 (Edge Weight 분석)
    # 두 저자가 여러 논문을 같이 썼다면 에지가 중복되어 있음
    pair_counts = Counter()
    for i in range(aa_edge.size(1)):
        u, v = aa_edge[0, i].item(), aa_edge[1, i].item()
        if u < v: # 중복 방지 (A-B만 카운트)
            pair_counts[(u, v)] += 1

    print("\n" + "="*50)
    print("🤝 [저자 협업 네트워크 상세 리포트]")
    print("="*50)

    print("\n🏆 [Top 5 인맥 왕 (가장 많은 동료와 협업)]")
    for auth_idx, count in collab_counts.most_common(5):
        name = auth_map.get(str(auth_idx), "Unknown")
        print(f" - {name:20s}: {count:4d}명의 동료와 연결됨")

    print("\n🔥 [Top 5 찰떡 콤비 (가장 많은 논문을 함께 씀)]")
    for (u, v), count in pair_counts.most_common(5):
        name_u = auth_map.get(str(u), "Unknown")
        name_v = auth_map.get(str(v), "Unknown")
        print(f" - {name_u} & {name_v}: 총 {count}편의 논문 공동 집필")

    print("\n" + "="*50)

if __name__ == "__main__":
    verify_graph_structure()
    analyze_author_network()