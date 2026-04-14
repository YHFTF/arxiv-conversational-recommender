import torch
import torch.optim as optim
import json
import os
from torch_geometric.utils import coalesce
from config import *


# ============================================================
# 1. 데이터 로드
# ============================================================
def load_data():
    """그래프 데이터, 메타 정보, 지식 메타 카운트를 로드합니다."""
    print(" 데이터를 로드하는 중...")
    data = torch.load(GRAPH_PATH, weights_only=False).to(DEVICE)

    with open(META_PATH, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    meta_counts = {
        'domains': len(meta['domains']),
        'tasks': len(meta['tasks']),
        'methods': len(meta['methods'])
    }

    return data, meta, meta_counts


def build_knowledge_ids(data, meta):
    """V4+ 모델용 논문별 지식 ID 텐서([num_papers, 3])를 생성합니다."""
    print(" 지식(Knowledge) ID 텐서를 구축하는 중...")
    with open(MASTER_FILE, 'r', encoding='utf-8') as f:
        master_list = json.load(f)

    num_papers = data['paper'].num_nodes
    paper_knowledge_ids = torch.zeros((num_papers, 3), dtype=torch.long)

    for i, item in enumerate(master_list):
        if i >= num_papers: break
        k_dict = item.get('knowledge', {})

        d_val = k_dict.get('domain', None)
        if isinstance(d_val, list) and len(d_val) > 0: d_val = d_val[0]
        t_val = k_dict.get('task', None)
        if isinstance(t_val, list) and len(t_val) > 0: t_val = t_val[0]
        m_val = k_dict.get('method', None)
        if isinstance(m_val, list) and len(m_val) > 0: m_val = m_val[0]

        paper_knowledge_ids[i] = torch.tensor([
            meta['domains'].get(d_val, 0) if d_val else 0,
            meta['tasks'].get(t_val, 0) if t_val else 0,
            meta['methods'].get(m_val, 0) if m_val else 0
        ])

    return paper_knowledge_ids


# ============================================================
# 2. 그래프 구축
# ============================================================
def get_graph_info(data):
    """그래프의 노드 수 정보를 반환합니다."""
    num_papers = data['paper'].num_nodes
    num_authors = data['author'].num_nodes
    num_topics = data['topic'].num_nodes
    total_nodes = num_papers + num_authors + num_topics
    return num_papers, num_authors, num_topics, total_nodes


def build_unified_graph(data):
    """이종(Heterogeneous) 그래프를 동종(Homogeneous) 통합 그래프로 변환합니다.

    Paper-Paper(인용), Author-Paper(저술), Paper-Topic(분류) 에지를
    하나의 통합 에지 인덱스로 결합합니다.
    """
    num_papers, num_authors, num_topics, total_nodes = get_graph_info(data)
    offset_author = num_papers
    offset_topic = num_papers + num_authors

    edge_list = []

    # Paper - Paper (인용)
    pp_edge = data['paper', 'cites', 'paper'].edge_index.to(torch.int64)
    pp_mask = (pp_edge[0] < num_papers) & (pp_edge[1] < num_papers)
    edge_list.append(pp_edge[:, pp_mask])

    # Author - Paper (양방향)
    ap_edge = data['author', 'writes', 'paper'].edge_index.clone().to(torch.int64)
    ap_mask = (ap_edge[0] < num_authors) & (ap_edge[1] < num_papers)
    ap_edge = ap_edge[:, ap_mask]
    ap_edge[0] += offset_author
    edge_list.append(ap_edge)
    edge_list.append(ap_edge.flip(0))

    # Paper - Topic (양방향)
    pt_edge = data['paper', 'has_topic', 'topic'].edge_index.clone().to(torch.int64)
    pt_mask = (pt_edge[0] < num_papers) & (pt_edge[1] < num_topics)
    pt_edge = pt_edge[:, pt_mask]
    pt_edge[1] += offset_topic
    edge_list.append(pt_edge)
    edge_list.append(pt_edge.flip(0))

    unified_edges = torch.cat(edge_list, dim=1).to(DEVICE)
    final_mask = (unified_edges[0] < total_nodes) & (unified_edges[1] < total_nodes)
    unified_edges = unified_edges[:, final_mask]
    unified_edges, _ = coalesce(unified_edges, None, num_nodes=total_nodes)

    return unified_edges.long()


# ============================================================
# 3. 에지 분할
# ============================================================
def split_edges(full_edges):
    """고정 시드(SEED)로 에지를 Train/Val/Test로 분할합니다.

    모든 모델이 동일한 데이터 셋에서 평가받기 위해 시드를 고정합니다.
    """
    torch.manual_seed(SEED)
    num_edges = full_edges.size(1)
    indices = torch.randperm(num_edges)

    train_size = int(TRAIN_RATIO * num_edges)
    val_size = int(VAL_RATIO * num_edges)

    train_edges = full_edges[:, indices[:train_size]]
    val_edges = full_edges[:, indices[train_size:train_size + val_size]]
    test_edges = full_edges[:, indices[train_size + val_size:]]

    return train_edges, val_edges, test_edges


# ============================================================
# 4. 초기 피처 생성
# ============================================================
def build_initial_features(data):
    """Paper, Author, Topic의 초기 피처를 생성하고 하나로 결합합니다.

    - Paper: 128차원 word embedding (OGBN-Arxiv 원본)
    - Author: 연결된 논문 벡터의 평균
    - Topic: 연결된 논문 벡터의 평균
    """
    num_papers, num_authors, num_topics, _ = get_graph_info(data)
    paper_x = data['paper'].x
    device = paper_x.device

    # Author 피처
    if 'x' in data['author']:
        author_x = data['author'].x
    else:
        author_x = torch.zeros((num_authors, EMBEDDING_DIM), device=device)
        ap_edge = data['author', 'writes', 'paper'].edge_index
        author_x.index_add_(0, ap_edge[0], paper_x[ap_edge[1]])
        counts = torch.bincount(ap_edge[0], minlength=num_authors).view(-1, 1).float().to(device)
        author_x = author_x / torch.clamp(counts, min=1.0)

    # Topic 피처
    if 'x' in data['topic']:
        topic_x = data['topic'].x
    else:
        topic_x = torch.zeros((num_topics, EMBEDDING_DIM), device=device)
        pt_edge = data['paper', 'has_topic', 'topic'].edge_index
        topic_x.index_add_(0, pt_edge[1], paper_x[pt_edge[0]])
        counts = torch.bincount(pt_edge[1], minlength=num_topics).view(-1, 1).float().to(device)
        topic_x = topic_x / torch.clamp(counts, min=1.0)

    # 전체 노드 통합 피처
    combined = torch.cat([paper_x, author_x, topic_x], dim=0)
    return combined


# ============================================================
# 5. 평가
# ============================================================
def evaluate_ranking(out, edges, num_papers, k=TOP_K, batch_size=EVAL_BATCH_SIZE):
    """All-Item Ranking 평가 (Recall@K, NDCG@K)를 수행합니다.

    16,000개 전체 논문을 후보군으로 놓고, 각 에지의 정답 논문이
    상위 K위 안에 드는지를 측정합니다.
    """
    src, pos_dst = edges[0], edges[1]

    # ======================================================
    # [수정] 평가 대상은 '논문 -> 논문' 연결로만 한정합니다!
    mask = (src < num_papers) & (pos_dst < num_papers)
    src = src[mask]
    pos_dst = pos_dst[mask]

    # 평가할 에지가 없으면 0.0 반환
    if src.size(0) == 0: 
        return 0.0, 0.0
    # ======================================================

    paper_embeddings = out[:num_papers]
    total_edges = src.size(0)

    all_hits = []
    all_ndcgs = []

    for i in range(0, total_edges, batch_size):
        end = min(i + batch_size, total_edges)
        batch_src = src[i:end]
        batch_pos_dst = pos_dst[i:end]

        batch_src_embs = out[batch_src]
        all_scores = torch.matmul(batch_src_embs, paper_embeddings.t())

        _, indices = torch.sort(all_scores, dim=1, descending=True)
        rankings = (indices == batch_pos_dst.unsqueeze(1)).nonzero(as_tuple=True)[1]

        hits = (rankings < k).float()
        ndcg = (1.0 / torch.log2(rankings.float() + 2.0))
        ndcg[rankings >= k] = 0.0

        all_hits.append(hits)
        all_ndcgs.append(ndcg)

    if not all_hits:  # 안전 장치: 루프를 돌았지만 데이터가 없을 경우
        return 0.0, 0.0

    final_hits = torch.cat(all_hits).mean().item()
    final_ndcg = torch.cat(all_ndcgs).mean().item()

    return final_hits, final_ndcg


def evaluate_cold_start(model, test_out, test_edges, num_papers, k=TOP_K, batch_size=EVAL_BATCH_SIZE):
    """신규 노드(Cold-Start) 상황을 시뮬레이션하여 평가를 수행합니다.
    쿼리 노드의 ID 정보(Identity)를 배제하고 오직 특징과 지식만 사용합니다.
    """
    src, pos_dst = test_edges[0], test_edges[1]

    # ======================================================
    # [수정] 평가 대상은 '논문 -> 논문' 연결로만 한정합니다!
    mask = (src < num_papers) & (pos_dst < num_papers)
    src = src[mask]
    pos_dst = pos_dst[mask]

    # 평가할 에지가 없으면 0.0 반환
    if src.size(0) == 0: 
        return 0.0, 0.0
    # ======================================================

    # 추천 후보군은 일반적인 학습된 임베딩 사용
    paper_embeddings = test_out[:num_papers]
    total_edges = src.size(0)

    all_hits = []
    all_ndcgs = []

    for i in range(0, total_edges, batch_size):
        end = min(i + batch_size, total_edges)
        batch_src = src[i:end]
        batch_pos_dst = pos_dst[i:end]

        # V4 등의 모델에서 구현한 콜드 스타트용 임베딩 생성
        if hasattr(model, 'get_cold_start_embeddings'):
            batch_src_embs = model.get_cold_start_embeddings(batch_src)
        else:
            # 지원하지 않는 모델은 0 벡터 또는 기존 에러 처리 (BPR-MF 등)
            batch_src_embs = torch.zeros((batch_src.size(0), paper_embeddings.size(1)), device=paper_embeddings.device)

        all_scores = torch.matmul(batch_src_embs, paper_embeddings.t())

        _, indices = torch.sort(all_scores, dim=1, descending=True)
        rankings = (indices == batch_pos_dst.unsqueeze(1)).nonzero(as_tuple=True)[1]

        hits = (rankings < k).float()
        ndcg = (1.0 / torch.log2(rankings.float() + 2.0))
        ndcg[rankings >= k] = 0.0

        all_hits.append(hits)
        all_ndcgs.append(ndcg)

    if not all_hits:  # 안전 장치: 루프를 돌았지만 데이터가 없을 경우
        return 0.0, 0.0

    final_hits = torch.cat(all_hits).mean().item()
    final_ndcg = torch.cat(all_ndcgs).mean().item()

    return final_hits, final_ndcg


# ============================================================
# 6. 통합 학습 루프
# ============================================================
def train_and_evaluate(model, train_edges, val_edges, test_edges, num_papers,
                       model_name="Model", save_path=None):
    """통일된 BPR 학습 + 평가를 수행합니다.

    두 가지 모드(General, Cold-Start)의 성능을 모두 측정합니다.
    """
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_val_ndcg = -float('inf')

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()

        out = model(train_edges)

        pos_src, pos_dst = train_edges[0], train_edges[1]
        neg_dst = torch.randint(0, model.total_nodes, (pos_src.size(0),), device=DEVICE)

        pos_scores = (out[pos_src] * out[pos_dst]).sum(dim=-1)
        neg_scores = (out[pos_src] * out[neg_dst]).sum(dim=-1)
        bpr_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-15).mean()

        bpr_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out = model(train_edges)
            val_recall, val_ndcg = evaluate_ranking(val_out, val_edges, num_papers)

            if val_ndcg > best_val_ndcg:
                best_val_ndcg = val_ndcg
                if save_path:
                    torch.save(model.state_dict(), save_path)

        if epoch % EVAL_INTERVAL == 0 or epoch == 1:
            print(f"  [{model_name}] Epoch {epoch:3d} | BPR Loss: {bpr_loss.item():.4f} "
                  f"| Val R@{TOP_K}: {val_recall:.4f} | Val N@{TOP_K}: {val_ndcg:.4f}")

    # Best 모델 복원 후 최종 테스트 (General + Cold-Start)
    if save_path and os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, weights_only=True))

    model.eval()
    with torch.no_grad():
        test_out = model(train_edges)
        
        # 1. 일반 성능 (Transductive)
        test_recall, test_ndcg = evaluate_ranking(test_out, test_edges, num_papers)
        
        # 2. 콜드 스타트 성능 (Inductive Simulation)
        cs_recall, cs_ndcg = evaluate_cold_start(model, test_out, test_edges, num_papers)
        
        print(f"  [OK] [{model_name}] Test R@{TOP_K}: {test_recall:.4f} | N@{TOP_K}: {test_ndcg:.4f}")
        print(f"  [CS] [{model_name}] Cold-Start R@{TOP_K}: {cs_recall:.4f} | N@{TOP_K}: {cs_ndcg:.4f}")

    return test_recall, test_ndcg, cs_recall, cs_ndcg