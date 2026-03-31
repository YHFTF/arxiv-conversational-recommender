import torch
import torch.nn as nn
from torch_geometric.nn.conv import LGConv
from torch_geometric.utils import coalesce, degree

class ArxivLightGCN(nn.Module):
    def __init__(self, data, embedding_dim=128, num_layers=2):
        super(ArxivLightGCN, self).__init__()
        self.num_layers = num_layers
        self.num_papers = data['paper'].num_nodes
        self.num_authors = data['author'].num_nodes
        self.num_topics = data['topic'].num_nodes
        
        self.offset_author = self.num_papers
        self.offset_topic = self.num_papers + self.num_authors
        self.total_nodes = self.num_papers + self.num_authors + self.num_topics

        device = data['paper'].x.device
        # 초기 임베딩 통합
        combined_x = torch.cat([
            data['paper'].x, 
            data['author'].x, 
            data['topic'].x
        ], dim=0).to(device)
        
        self.embedding = nn.Parameter(combined_x)
        
        # 🌟 PyG의 LightGCN 모델 대신 개별 LGConv 레이어를 직접 사용합니다.
        self.convs = nn.ModuleList([LGConv() for _ in range(num_layers)])

    def _build_unified_edge_index(self, data):
        device = self.embedding.device
        edge_list = []

        # 모든 관계를 long(int64)으로 강제 통합
        edge_list.append(data['paper', 'cites', 'paper'].edge_index.to(torch.int64))
        
        ap_edge = data['author', 'writes', 'paper'].edge_index.clone().to(torch.int64)
        ap_edge[0] += self.offset_author
        edge_list.append(ap_edge); edge_list.append(ap_edge.flip(0))

        pt_edge = data['paper', 'has_topic', 'topic'].edge_index.clone().to(torch.int64)
        pt_edge[1] += self.offset_topic
        edge_list.append(pt_edge); edge_list.append(pt_edge.flip(0))

        unified = torch.cat(edge_list, dim=1).to(device)
        unified, _ = coalesce(unified, None, num_nodes=self.total_nodes)
        
        return unified.to(torch.int64)

    def forward(self, edge_index):
        # 🌟 수동 메시지 패싱 루프
        # 내부 gcn_norm 에러를 피하기 위해 연산 전 타입을 확실히 고정합니다.
        edge_index = edge_index.to(torch.int64)
        
        # 정규화 계수(Normalization)를 수동으로 계산 (scatter 에러 지점을 우회)
        row, col = edge_index
        deg = degree(col, self.total_nodes, dtype=self.embedding.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        emb = self.embedding
        embs = [emb]

        for conv in self.convs:
            # 개별 레이어 연산 수행
            emb = conv(emb, edge_index, edge_weight=norm)
            embs.append(emb)

        # 모든 레이어의 결과 평균 (LightGCN의 핵심 공식)
        out = torch.stack(embs, dim=0).mean(dim=0)
        return out

    def get_paper_embeddings(self, out):
        return out[:self.num_papers]