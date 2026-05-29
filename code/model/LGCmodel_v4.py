import torch
import torch.nn as nn
from torch_geometric.nn.conv import LGConv
from torch_geometric.utils import coalesce, degree

class ArxivLightGCNV4(nn.Module):
    def __init__(self, data, meta_counts, paper_knowledge_ids, embedding_dim=128, num_layers=2, knowledge_weight=0.1):
        super(ArxivLightGCNV4, self).__init__()
        self.num_layers = num_layers
        self.knowledge_weight = knowledge_weight
        self.num_papers = data['paper'].num_nodes
        self.num_authors = data['author'].num_nodes
        self.num_topics = data['topic'].num_nodes
        
        self.offset_author = self.num_papers
        self.offset_topic = self.num_papers + self.num_authors
        self.total_nodes = self.num_papers + self.num_authors + self.num_topics

        device = data['paper'].x.device
        paper_x = data['paper'].x

        # --- [V4: 지식 주입 세팅] ---
        # 1. Base 논문 임베딩 (V2처럼 학습 가능하도록 Parameter로 변경)
        self.paper_base_x = nn.Parameter(paper_x)
        # 콜드 스타트용 원본 피처 (학습되지 않은 순수 텍스트 벡터)
        self.register_buffer('paper_raw_x', paper_x.clone())
        
        # 2. 16000 편의 논문에 대응하는 [16000, 3] 지식 ID 배열
        self.register_buffer('paper_knowledge_ids', paper_knowledge_ids.long().to(device))

        # 3. 모델이 실제로 학습할 지식 속성 Embedding (padding_idx=0 이 0이 되므로 모르는 값은 0이 더해짐)
        self.domain_emb = nn.Embedding(meta_counts['domains'] + 1, embedding_dim, padding_idx=0)
        self.task_emb = nn.Embedding(meta_counts['tasks'] + 1, embedding_dim, padding_idx=0)
        self.method_emb = nn.Embedding(meta_counts['methods'] + 1, embedding_dim, padding_idx=0)

        # --- [자동 피처 생성 로직 (V2 로직 유지하되 Parameter로 등록)] ---
        if 'x' in data['author']:
            author_x = data['author'].x
        else:
            print(" Author 피처 실시간 생성 중...")
            author_x = torch.zeros((self.num_authors, embedding_dim), device=device)
            ap_edge = data['author', 'writes', 'paper'].edge_index
            author_x.index_add_(0, ap_edge[0], paper_x[ap_edge[1]])
            counts = torch.bincount(ap_edge[0], minlength=self.num_authors).view(-1, 1).float().to(device)
            author_x = author_x / torch.clamp(counts, min=1.0)

        if 'x' in data['topic']:
            topic_x = data['topic'].x
        else:
            print(" Topic 피처 실시간 생성 중...")
            topic_x = torch.zeros((self.num_topics, embedding_dim), device=device)
            pt_edge = data['paper', 'has_topic', 'topic'].edge_index
            topic_x.index_add_(0, pt_edge[1], paper_x[pt_edge[0]])
            counts = torch.bincount(pt_edge[1], minlength=self.num_topics).view(-1, 1).float().to(device)
            topic_x = topic_x / torch.clamp(counts, min=1.0)

        # Author와 Topic은 지식 임베딩과 분리되어 자체 학습되도록 Parameter 처리
        self.author_emb = nn.Parameter(author_x)
        self.topic_emb = nn.Parameter(topic_x)

        # LightGCN 레이어
        self.convs = nn.ModuleList([LGConv() for _ in range(num_layers)])

    def _build_unified_graph(self, data):
        device = self.paper_base_x.device
        edge_list = []
        
        # 1. Paper-Paper
        pp_edge = data['paper', 'cites', 'paper'].edge_index.to(torch.int64)
        pp_mask = (pp_edge[0] < self.num_papers) & (pp_edge[1] < self.num_papers)
        edge_list.append(pp_edge[:, pp_mask])

        # 2. Author-Paper
        ap_edge = data['author', 'writes', 'paper'].edge_index.clone().to(torch.int64)
        ap_mask = (ap_edge[0] < self.num_authors) & (ap_edge[1] < self.num_papers)
        ap_edge = ap_edge[:, ap_mask]
        ap_edge[0] += self.offset_author
        edge_list.append(ap_edge)
        edge_list.append(ap_edge.flip(0))

        # 3. Paper-Topic
        pt_edge = data['paper', 'has_topic', 'topic'].edge_index.clone().to(torch.int64)
        pt_mask = (pt_edge[0] < self.num_papers) & (pt_edge[1] < self.num_topics)
        pt_edge = pt_edge[:, pt_mask]
        pt_edge[1] += self.offset_topic
        edge_list.append(pt_edge)
        edge_list.append(pt_edge.flip(0))

        unified_edges = torch.cat(edge_list, dim=1).to(device)
        final_mask = (unified_edges[0] < self.total_nodes) & (unified_edges[1] < self.total_nodes)
        unified_edges = unified_edges[:, final_mask]
        unified_edges, _ = coalesce(unified_edges, None, num_nodes=self.total_nodes)
        
        return unified_edges.long()

    def _get_combined_embedding(self):
        # 1. 동적 지식 결합 (Forward 시 계산하여 Gradient를 추적 가능하게 함)
        d_val = self.domain_emb(self.paper_knowledge_ids[:, 0])
        t_val = self.task_emb(self.paper_knowledge_ids[:, 1])
        m_val = self.method_emb(self.paper_knowledge_ids[:, 2])
        
        # 가중치(knowledge_weight)를 적용하여 지식의 영향력 조절
        dynamic_paper_x = self.paper_base_x + self.knowledge_weight * (d_val + t_val + m_val)
        
        # 2. 전체 노드의 시작 값으로 결합
        return torch.cat([dynamic_paper_x, self.author_emb, self.topic_emb], dim=0)
    
    def forward(self, edge_index):
        edge_index = edge_index.to(torch.long)
        row, col = edge_index
        deg = degree(col, self.total_nodes, dtype=self.paper_base_x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # V4 핵심: 매개 파라미터가 여기서 생성되어 역전파를 받음
        emb = self._get_combined_embedding()
        embs = [emb]
        for conv in self.convs:
            emb = conv(emb, edge_index, edge_weight=norm)
            embs.append(emb)
        
        return torch.stack(embs, dim=0).mean(dim=0)

    def get_paper_embeddings(self, out):
        return out[:self.num_papers]

    def get_cold_start_embeddings(self, node_ids):
        """[V4 전용] 콜드 스타트용 임베딩 생성.
        학습된 개별 논문 파라미터(paper_base_x)와 이웃 전파 없이,
        오직 원본 피처와 지식 임베딩만 결합하여 반환합니다.
        """
        node_ids = node_ids.to(self.paper_raw_x.device)
        d_val = self.domain_emb(self.paper_knowledge_ids[node_ids, 0])
        t_val = self.task_emb(self.paper_knowledge_ids[node_ids, 1])
        m_val = self.method_emb(self.paper_knowledge_ids[node_ids, 2])
        
        # 원본 피처 + 가중치 적용된 지식 임베딩
        return self.paper_raw_x[node_ids] + self.knowledge_weight * (d_val + t_val + m_val)
