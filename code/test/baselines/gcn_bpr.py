import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class GCNBPR(nn.Module):
    """GCN + BPR: 표준 Graph Convolutional Network.

    Feature Transformation(nn.Linear) + ReLU 활성화를 포함하는 표준 GCN 구조입니다.
    LightGCN이 이 비선형 변환을 제거한 이유("Less is More")를 검증하고,
    V4의 지식 주입 효과를 대조하기 위한 비교 대상입니다.

    Reference:
        Kipf & Welling, "Semi-Supervised Classification with Graph Convolutional Networks", ICLR 2017
    """
    def __init__(self, total_nodes, initial_features, embedding_dim=128, num_layers=2):
        super().__init__()
        self.total_nodes = total_nodes
        self.embedding = nn.Parameter(initial_features.clone())

        # GCNConv는 내부적으로 Weight Matrix + 대칭 정규화를 수행
        self.convs = nn.ModuleList([
            GCNConv(embedding_dim, embedding_dim) for _ in range(num_layers)
        ])
        # 콜드 스타트 시뮬레이션용 원본 피처
        self.register_buffer('initial_raw_x', initial_features.clone())

    def forward(self, edge_index):
        x = self.embedding
        xs = [x]
        for conv in self.convs:
            x = conv(x, edge_index)
            x = torch.relu(x)
            xs.append(x)
        # LightGCN과 동일하게 레이어별 평균 집계
        return torch.stack(xs, dim=0).mean(dim=0)

    def get_cold_start_embeddings(self, node_ids):
        """본 적 없는 노드에 대해 원본 피처만 반환합니다.
        (전파 없이도 지식 기반 모델과 공정한 비교를 수행하기 위함)
        """
        return self.initial_raw_x[node_ids]
