import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import LightGCN
from torch_geometric.utils import degree

class ArxivLightGCN(nn.Module):
    def __init__(self, data, embedding_dim=64, num_layers=3):
        super().__init__()
        # 1. 모든 노드의 총 개수 계산 (Paper + Author + Topic)
        self.num_papers = data['paper'].num_nodes
        self.num_authors = data['author'].num_nodes
        self.num_topics = data['topic'].num_nodes
        self.total_nodes = self.num_papers + self.num_authors + self.num_topics

        # 2. 오프셋 설정 (전체 인덱스에서 각 노드 타입의 시작 위치)
        self.offset_author = self.num_papers
        self.offset_topic = self.num_papers + self.num_authors

        # 3. 이종 그래프 에지를 단일 에지 리스트로 통합
        edge_index = self._build_unified_edge_index(data)
        
        # 4. PyG LightGCN 모델 초기화
        self.model = LightGCN(
            num_nodes=self.total_nodes,
            embedding_dim=embedding_dim,
            num_layers=num_layers
        )

    def _build_unified_edge_index(self, data):
        edge_indices = []

        # (1) Paper - Paper (Cites)
        edge_indices.append(data['paper', 'cites', 'paper'].edge_index)

        # (2) Author - Paper (Writes) -> Author 인덱스에 오프셋 추가
        ap_edge = data['author', 'writes', 'paper'].edge_index.clone()
        ap_edge[0] += self.offset_author
        edge_indices.append(ap_edge)
        edge_indices.append(ap_edge.flip(0)) # 무향 그래프화

        # (3) Paper - Topic (Has_Topic) -> Topic 인덱스에 오프셋 추가
        pt_edge = data['paper', 'has_topic', 'topic'].edge_index.clone()
        pt_edge[1] += self.offset_topic
        edge_indices.append(pt_edge)
        edge_indices.append(pt_edge.flip(0)) # 무향 그래프화

        return torch.cat(edge_indices, dim=1)

    def forward(self, edge_index):
        # LightGCN의 특징: 최종 임베딩(e_0, e_1, ..., e_L의 평균) 반환
        return self.model.get_embedding(edge_index)

    def recommend_loss(self, out, pos_edge_index, neg_edge_index):
        # BPR (Bayesian Personalized Ranking) Loss 계산
        return self.model.recommendation_loss(out, pos_edge_index, neg_edge_index)

# --- 실행 예시 ---
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = torch.load('subdataset/build_hetero_graph.pt', weights_only=False).to(device)
    
    # 모델 생성
    model = ArxivLightGCN(data, embedding_dim=128, num_layers=3).to(device)
    unified_edge_index = model._build_unified_edge_index(data).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(1, 101):
        optimizer.zero_grad()
        
        # 전체 노드 임베딩 추출
        out = model(unified_edge_index)
        
        # 포지티브/네거티브 샘플링 (여기서는 예시로 Paper-Paper 인용 관계 학습)
        pos_edge = data['paper', 'cites', 'paper'].edge_index
        # 네거티브 샘플링 로직 필요 (생략)
        
        # loss = model.recommend_loss(...)
        # loss.backward()
        # optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch} 학습 중...")

    print("✅ 학습 완료!")