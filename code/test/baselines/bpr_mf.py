import torch
import torch.nn as nn


class BPRMF(nn.Module):
    """BPR-MF: 그래프 구조를 사용하지 않는 순수 행렬 분해 모델.

    그래프 전파(Message Passing) 없이 노드 임베딩만으로 BPR 학습을 수행합니다.
    GNN 기반 모델과 비교하여 "그래프 구조가 추천에 기여하는 정도"를
    정량적으로 측정하기 위한 베이스라인입니다.

    Reference:
        Rendle et al., "BPR: Bayesian Personalized Ranking from Implicit Feedback", UAI 2009
    """
    def __init__(self, total_nodes, initial_features, embedding_dim=128):
        super().__init__()
        self.total_nodes = total_nodes
        # 공정 비교를 위해 다른 모델과 동일한 초기 피처로 시작
        self.embedding = nn.Parameter(initial_features.clone())
        # 콜드 스타트 시뮬레이션용 원본 피처
        self.register_buffer('initial_raw_x', initial_features.clone())

    def forward(self, edge_index=None):
        # 그래프 구조를 완전히 무시하고 임베딩을 직접 반환
        return self.embedding

    def get_cold_start_embeddings(self, node_ids):
        """본 적 없는 노드에 대해 원본 피처만 반환합니다."""
        return self.initial_raw_x[node_ids]
