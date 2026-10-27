import torch
import torch.nn as nn
from torch_geometric.nn.conv import LGConv
from torch_geometric.utils import coalesce, degree

class ArxivLightGCNV2(nn.Module):
    def __init__(self, data, meta_counts, embedding_dim=128, num_layers=2):
        super(ArxivLightGCNV2, self).__init__()
        self.num_layers = num_layers
        self.num_papers = data['paper'].num_nodes
        self.num_authors = data['author'].num_nodes
        self.num_topics = data['topic'].num_nodes
        
        self.offset_author = self.num_papers
        self.offset_topic = self.num_papers + self.num_authors
        self.total_nodes = self.num_papers + self.num_authors + self.num_topics

        device = data['paper'].x.device
        paper_x = data['paper'].x

        # --- [자동 피처 생성 로직] ---
        # 1. Author 피처 생성 (논문 벡터 평균)
        if 'x' in data['author']:
            author_x = data['author'].x
        else:
            print(" Author 피처 실시간 생성 중...")
            author_x = torch.zeros((self.num_authors, embedding_dim), device=device)
            ap_edge = data['author', 'writes', 'paper'].edge_index
            author_x.index_add_(0, ap_edge[0], paper_x[ap_edge[1]])
            counts = torch.bincount(ap_edge[0], minlength=self.num_authors).view(-1, 1).float().to(device)
            author_x = author_x / torch.clamp(counts, min=1.0)

        # 2. Topic 피처 생성 (논문 벡터 평균)
        if 'x' in data['topic']:
            topic_x = data['topic'].x
        else:
            print(" Topic 피처 실시간 생성 중...")
            topic_x = torch.zeros((self.num_topics, embedding_dim), device=device)
            pt_edge = data['paper', 'has_topic', 'topic'].edge_index
            topic_x.index_add_(0, pt_edge[1], paper_x[pt_edge[0]])
            counts = torch.bincount(pt_edge[1], minlength=self.num_topics).view(-1, 1).float().to(device)
            topic_x = topic_x / torch.clamp(counts, min=1.0)

        # 모든 노드 임베딩 통합 (nn.Parameter로 학습 가능하게 설정)
        combined_x = torch.cat([paper_x, author_x, topic_x], dim=0).to(device)
        self.embedding = nn.Parameter(combined_x)
        # 콜드 스타트 시뮬레이션용 원본 논문 피처
        self.register_buffer('initial_raw_paper_x', paper_x.clone())

        # --- [지식 속성 임베딩 레이어] ---
        # +1은 -1 패딩(0번 인덱스)을 처리하기 위함입니다.
        self.domain_emb = nn.Embedding(meta_counts['domains'] + 1, embedding_dim, padding_idx=0)
        self.task_emb = nn.Embedding(meta_counts['tasks'] + 1, embedding_dim, padding_idx=0)
        self.method_emb = nn.Embedding(meta_counts['methods'] + 1, embedding_dim, padding_idx=0)

        # LightGCN 레이어 직접 구현용
        self.convs = nn.ModuleList([LGConv() for _ in range(num_layers)])

    def _build_unified_graph(self, data):
        device = self.embedding.device
        edge_list = []
        
        # 1. Paper-Paper (0 ~ 15999)
        # 16000 이상인 인덱스가 있으면 제거 (안전장치)
        pp_edge = data['paper', 'cites', 'paper'].edge_index.to(torch.int64)
        pp_mask = (pp_edge[0] < self.num_papers) & (pp_edge[1] < self.num_papers)
        edge_list.append(pp_edge[:, pp_mask])

        # 2. Author-Paper (Author: 16000 ~ )
        ap_edge = data['author', 'writes', 'paper'].edge_index.clone().to(torch.int64)
        
        #  Author 인덱스 범위 강제 제한 (0 ~ num_authors-1)
        # Paper 인덱스 범위 강제 제한 (0 ~ num_papers-1)
        ap_mask = (ap_edge[0] < self.num_authors) & (ap_edge[1] < self.num_papers)
        ap_edge = ap_edge[:, ap_mask]
        
        # 오프셋 적용
        ap_edge[0] += self.offset_author
        edge_list.append(ap_edge)
        edge_list.append(ap_edge.flip(0))

        # 3. Paper-Topic (Topic: 16000 + Author ~ )
        pt_edge = data['paper', 'has_topic', 'topic'].edge_index.clone().to(torch.int64)
        
        #  Topic 인덱스 범위 강제 제한 (0 ~ 39)
        pt_mask = (pt_edge[0] < self.num_papers) & (pt_edge[1] < self.num_topics)
        pt_edge = pt_edge[:, pt_mask]
        
        # 오프셋 적용
        pt_edge[1] += self.offset_topic
        edge_list.append(pt_edge)
        edge_list.append(pt_edge.flip(0))

        # 4. 통합 및 최종 필터링
        unified_edges = torch.cat(edge_list, dim=1).to(device)
        
        #  coalesce 전 마지막 확인: total_nodes를 넘는 놈이 절대 없도록 함
        final_mask = (unified_edges[0] < self.total_nodes) & (unified_edges[1] < self.total_nodes)
        unified_edges = unified_edges[:, final_mask]

        # 이제 안전하게 coalesce 호출
        unified_edges, _ = coalesce(unified_edges, None, num_nodes=self.total_nodes)
        
        return unified_edges.long()
    
    
    def forward(self, edge_index):
        # 수동 정규화 및 메시지 패싱
        edge_index = edge_index.to(torch.long)
        row, col = edge_index
        deg = degree(col, self.total_nodes, dtype=self.embedding.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        emb = self.embedding
        embs = [emb]
        for conv in self.convs:
            emb = conv(emb, edge_index, edge_weight=norm)
            embs.append(emb)
        
        return torch.stack(embs, dim=0).mean(dim=0)

    def get_cold_start_embeddings(self, node_ids):
        """본 적 없는 노드에 대해 원본 피처만 반환합니다."""
        return self.initial_raw_paper_x[node_ids]

    def get_paper_embeddings(self, out):
        return out[:self.num_papers]