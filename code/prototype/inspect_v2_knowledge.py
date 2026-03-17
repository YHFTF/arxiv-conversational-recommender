# 기존 헤테로 그래프에 메소드, 테스크, 도메인 정보를 엣지 속성으로 추가한 그래프
import torch
import json
import os
import random
from torch_geometric.data import HeteroData

def inspect_knowledge():
    # --- 1. 경로 설정 (함수 내부에서 정의) ---
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    HETERO_GRAPH_V2 = os.path.join(project_root, 'subdataset', 'hetero_graph_v2.pt')
    META_FILE = os.path.join(project_root, 'output', 'knowledge_meta.json')

    # 파일 존재 여부 확인
    if not os.path.exists(HETERO_GRAPH_V2):
        print(f"[Error] 그래프 파일을 찾을 수 없습니다: {HETERO_GRAPH_V2}")
        return

    # 2. 데이터 로드 (보안 설정 및 HeteroData 허용)
    torch.serialization.add_safe_globals([HeteroData])
    print(f"[System] 데이터 로딩 중: {HETERO_GRAPH_V2}")
    
    # weights_only=False로 객체 전체 로드
    data = torch.load(HETERO_GRAPH_V2, weights_only=False)
    
    with open(META_FILE, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    
    # 3. 엣지 데이터 추출
    # paper -> has_topic -> topic 관계의 인덱스와 속성
    edge_index = data['paper', 'has_topic', 'topic'].edge_index
    edge_attr = data['paper', 'has_topic', 'topic'].edge_attr
    num_edges = edge_attr.size(0)

    print("="*60)
    print(f"🔬 [V2 그래프 지식 주입 상태 점검 (무작위 5건)]")
    print(f" - 총 소속 엣지 수: {num_edges:,}개")
    print("="*60)

    # 4. 무작위 샘플링 (중복 없는 5개 인덱스)
    sample_indices = random.sample(range(num_edges), 5)

    for i in sample_indices:
        p_idx = edge_index[0, i].item()
        t_idx = edge_index[1, i].item()
        attr = edge_attr[i] # [d_idx, tk_idx, mt_idx]
        
        # 인덱스를 텍스트로 변환 (범위 밖일 경우 대비)
        try:
            domain = meta['domains'][attr[0]] if attr[0] != -1 else "N/A"
            task = meta['tasks'][attr[1]] if attr[1] != -1 else "N/A"
            method = meta['methods'][attr[2]] if attr[2] != -1 else "N/A"
        except IndexError:
            domain, task, method = "Index Error", "Index Error", "Index Error"

        print(f"\n[엣지 ID: {i}] 논문 인덱스: {p_idx}")
        print(f" └ 연결된 카테고리(Topic): {t_idx}번")
        print(f" └ 엣지에 담긴 지식 정보:")
        print(f"   - 도메인(Domain): {domain}")
        print(f"   - 태스크(Task)  : {task}")
        print(f"   - 메소드(Method): {method}")

    print("\n" + "="*60)
    print("✅ 무작위 점검 완료")

if __name__ == "__main__":
    inspect_knowledge()