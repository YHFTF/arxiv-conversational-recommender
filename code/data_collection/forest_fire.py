import networkx as nx
import numpy as np
import random
from collections import deque
import torch
_real_torch_load = torch.load

def _torch_load_with_weights_only_false(*args, **kwargs):
    kwargs.setdefault("weights_only", False)  # 기본값을 False로 되돌림
    return _real_torch_load(*args, **kwargs)

torch.load = _torch_load_with_weights_only_false  # 임시 패치(세션 동안만)
# ogb 로드 코드는 생략 (이미 성공했으므로)

# --- 🚨 데이터셋 로드 가정 (성공한 loader.py 코드 이후) ---
# dataset 변수는 PygNodePropPredDataset(name="ogbn-arxiv") 결과라고 가정합니다.

# 예시를 위해 PyG 객체만 준비 (실제 환경에서는 loader.py에서 로드된 dataset 사용)
from ogb.nodeproppred import PygNodePropPredDataset

# 
# PyG 데이터셋 로딩 (실제 프로젝트에서는 이미 메모리에 로드되어 있을 수 있음)
# OGB 로드 시 필요한 안전 목록 추가 코드는 이미 파일 상단에 있다고 가정합니다.
dataset = PygNodePropPredDataset(name = "ogbn-arxiv")
graph_data = dataset[0] # PyG Data 객체 획득
TOTAL_NODES = graph_data.num_nodes # 169,343개

# --- FFS 파라미터 설정 ---
TARGET_SIZE = 16000     # 목표 샘플 노드 수
PF_VALUE = 0.75         # 번짐 확률 (Burning Probability)

def to_networkx_graph(data):
    """PyG Data 객체를 NetworkX DiGraph(방향성 그래프)로 변환합니다."""
    G = nx.DiGraph() 
    
    # 노드 인덱스를 노드로 추가 (0부터 N-1까지)
    G.add_nodes_from(range(data.num_nodes))

    # 엣지 추가 (edge_index 텐서 사용)
    # 엣지 방향: 인용하는 논문(source) -> 인용 당하는 논문(target)
    source_nodes = data.edge_index[0].tolist()
    target_nodes = data.edge_index[1].tolist()
    edges = list(zip(source_nodes, target_nodes))
    
    G.add_edges_from(edges)
    
    return G

# PyG 그래프를 NetworkX 그래프로 변환
citation_graph = to_networkx_graph(graph_data)
print(f"✅ PyG 객체를 NetworkX DiGraph로 변환 완료. 엣지 수: {citation_graph.number_of_edges()}")

def forest_fire_sampling(graph, target_size, pf=0.75):
    """
    Forest Fire Sampling을 수행하여 목표 노드 수만큼 추출합니다.
    """
    sampled_nodes = set()
    all_nodes = list(graph.nodes())
    p_b = 1.0 - pf # 번짐 실패 확률

    while len(sampled_nodes) < target_size:
        
        # 1. 시작 노드 선택 (아직 샘플링되지 않은 노드 중에서 무작위 선택)
        remaining_nodes = list(set(all_nodes) - sampled_nodes)
        if not remaining_nodes:
            break
            
        start_node = random.choice(remaining_nodes) 
        queue = deque([start_node])
        
        while queue and len(sampled_nodes) < target_size:
            
            current_node = queue.popleft()

            if current_node not in sampled_nodes:
                sampled_nodes.add(current_node)

                # 2. 이웃 노드 획득 (current_node가 인용하는 논문)
                neighbors = list(graph.successors(current_node)) 
                unvisited_neighbors = [n for n in neighbors if n not in sampled_nodes]
                
                if not unvisited_neighbors:
                    continue

                # 3. 기하 분포를 사용한 번짐 수 결정
                num_to_burn = max(0, np.random.geometric(p=p_b) - 1) 
                num_to_burn = min(num_to_burn, len(unvisited_neighbors))

                # 4. 선택된 이웃 노드를 큐에 추가
                burned_neighbors = random.sample(unvisited_neighbors, num_to_burn)
                queue.extend(burned_neighbors)
                
    # 목표 크기에 맞춰 16000개만 반환
    return list(sampled_nodes)[:target_size]

# --- FFS 최종 실행 ---

sampled_node_list = forest_fire_sampling(citation_graph, TARGET_SIZE, pf=PF_VALUE)

print("-" * 50)
print(f"✅ Forest Fire Sampling 완료. 최종 노드 수: **{len(sampled_node_list)}개**")
print(f"사용된 번짐 확률 (Pf): {PF_VALUE}")
print(f"추출된 샘플 노드 인덱스 예시: {sampled_node_list[:5]}")
print("-" * 50)

# FFS 결과 인덱스를 PyTorch Tensor로 변환
sample_indices_tensor = torch.tensor(sampled_node_list, dtype=torch.long)

# 16,000개 샘플의 피처 데이터 (16000 x 128)
sample_features = graph_data.x[sample_indices_tensor] 

# 16,000개 샘플의 레이블 데이터 (16000 x 1)
sample_labels = graph_data.y[sample_indices_tensor]

FILE_NAME = 'ogbn_arxiv_16k_ffs_sample.pt'

torch.save({
    'indices': sampled_node_list,  # 파이썬 리스트 형태로 인덱스 저장
    'features': sample_features,   # 텐서 형태로 피처 저장
    'labels': sample_labels        # 텐서 형태로 레이블 저장
}, FILE_NAME)

print(f"✅ FFS 샘플링 데이터 저장 완료: **{FILE_NAME}**")
print(f"저장된 피처 텐서 크기: {sample_features.shape}")