import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv


class GraphSAGEBPR(nn.Module):
    """GraphSAGE + BPR: 이웃 특징 집계(Neighbor Aggregation) 기반 GNN.

    Mean Aggregation으로 이웃 정보를 수집하여 노드 임베딩을 학습합니다.
    GCN과 다른 집계 패러다임을 대표하며, 다양한 GNN 아키텍처와의
    비교를 위해 포함됩니다.

    Reference:
        Hamilton et al., "Inductive Representation Learning on Large Graphs", NeurIPS 2017
    """
    def __init__(self, total_nodes, initial_features, embedding_dim=128, num_layers=2):
        super().__init__()
        self.total_nodes = total_nodes
        self.embedding = nn.Parameter(initial_features.clone())

        # SAGEConv: 이웃 피처를 Mean/LSTM/Pool 방식으로 집계
        self.convs = nn.ModuleList([
            SAGEConv(embedding_dim, embedding_dim) for _ in range(num_layers)
        ])
        # 콜드 스타트 시뮬레이션용 원본 피처
        self.register_buffer('initial_raw_x', initial_features.clone())

    def forward(self, edge_index):
        x = self.embedding
        xs = [x]
        for conv in self.convs:
            x = conv(x, edge_index)
            xs.append(x)
        # 레이어별 평균 집계
        return torch.stack(xs, dim=0).mean(dim=0)

    def get_cold_start_embeddings(self, node_ids):
        """본 적 없는 노드에 대해 원본 피처만 반환합니다."""
        return self.initial_raw_x[node_ids]
