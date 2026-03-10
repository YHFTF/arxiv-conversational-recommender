import os
# torch.load 기본을 예전처럼 weights_only=False 로 돌리기 (이 스크립트 내에서만)
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

import torch
from torch_geometric.data.data import DataEdgeAttr
from ogb.nodeproppred import PygNodePropPredDataset

# DataEdgeAttr 클래스를 신뢰한다고 명시 (weights_only=True인 경우 대비)
torch.serialization.add_safe_globals([DataEdgeAttr])

dataset = PygNodePropPredDataset(name="ogbn-arxiv")
graph = dataset[0]


# 레이블 데이터
labels = graph.y.squeeze()  # [num_nodes] 형태로 변환

# 고유한 클래스 값들 확인
unique_labels = torch.unique(labels)
print("레이블 종류:", unique_labels.tolist())
print("총 클래스 개수:", len(unique_labels))

# 각 클래스별 개수 확인
counts = torch.bincount(labels)
for i, c in enumerate(counts.tolist()):
    print(f"클래스 {i}: {c}개")
