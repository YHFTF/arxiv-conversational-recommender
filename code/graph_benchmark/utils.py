import torch
import torch.optim as optim
import json
import os
import copy
from torch_geometric.utils import coalesce
from config import *


# ============================================================
# 1. 데이터 로드 및 Zero-Leakage 분할
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
        # The raw Luna output is retained for traceability, while ``graph``
        # contains conservative normalized terms that occur often enough to
        # learn a useful knowledge embedding.  Fall back for older datasets.
        k_dict = k_dict.get('graph', k_dict.get('normalized', k_dict))

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


def split_data_v2(data):
    """[핵심 개선] 데이터 누수를 완전히 배제하고 진정한 콜드 스타트 환경을 조성하기 위해 에지를 엄격히 분할합니다.

    1. 고정된 시드(SEED)로 전체 논문 노드 중 10%를 'Cold-Start 논문 노드'로 선정합니다.
    2. 이 Cold-Start 논문들과 관련된 모든 인용 에지를 'test_cold_edges'로 완벽 분리하여 학습/검증에서 제외합니다.
    3. Cold-Start 논문과 관련되지 않은 나머지 'Warm-Start 인용 에지'만을 대상으로 8:1:1 비율로 학습/검증/테스트 인용 에지를 나눕니다.
    4. 저술(writes) 및 토픽(has_topic) 에지 역시 Cold-Start 논문과 연계된 것을 완전히 지워 누수를 제거한 뒤, 100% 학습용 컨텍스트 그래프로 제공합니다.
    """
    print(f"\n[SPLIT v2] Zero-Leakage 데이터 격리 및 진정한 콜드 스타트 분할 실행... (Seed={SEED})")
    
    num_papers = data['paper'].num_nodes
    num_authors = data['author'].num_nodes
    num_topics = data['topic'].num_nodes
    
    # 1. Cold-Start 대상 논문 선정 (10%)
    torch.manual_seed(SEED)
    shuffled_papers = torch.randperm(num_papers)
    num_cold = int(num_papers * 0.1)
    
    cold_papers = shuffled_papers[:num_cold].to(DEVICE)
    cold_mask = torch.zeros(num_papers, dtype=torch.bool, device=DEVICE)
    cold_mask[cold_papers] = True
    
    print(f"  - 전체 논문 수: {num_papers} | Warm 논문: {num_papers - num_cold} | Cold 논문 (10% 격리): {num_cold}")
    
    # 2. 인용 에지 (Paper - Cites - Paper) 분할
    pp_edge = data['paper', 'cites', 'paper'].edge_index.to(torch.int64)
    
    # Cold 논문이 소스 또는 타겟이 되는 에지 마스킹
    src_in_cold = cold_mask[pp_edge[0]]
    dst_in_cold = cold_mask[pp_edge[1]]
    is_cold_pp = src_in_cold | dst_in_cold
    
    cold_pp_edges = pp_edge[:, is_cold_pp]
    warm_pp_edges = pp_edge[:, ~is_cold_pp]
    
    print(f"  - 원본 인용 에지: {pp_edge.size(1)} | Warm 인용 에지: {warm_pp_edges.size(1)} | Cold 격리 인용 에지: {cold_pp_edges.size(1)}")
    
    # Warm 인용 에지만을 대상으로 8:1:1 분할
    num_warm_pp = warm_pp_edges.size(1)
    indices = torch.randperm(num_warm_pp)
    
    train_size = int(TRAIN_RATIO * num_warm_pp)
    val_size = int(VAL_RATIO * num_warm_pp)
    
    train_pp_edges = warm_pp_edges[:, indices[:train_size]]
    val_pp_edges = warm_pp_edges[:, indices[train_size:train_size + val_size]]
    test_pp_edges = warm_pp_edges[:, indices[train_size + val_size:]]
    
    # 3. 저술 에지 (Author - Writes - Paper) 필터링
    ap_edge = data['author', 'writes', 'paper'].edge_index.to(torch.int64)
    # paper(ap_edge[1])가 Cold 논문인 에지 마스킹 및 제거
    ap_in_cold = cold_mask[ap_edge[1]]
    train_ap_edges = ap_edge[:, ~ap_in_cold]
    
    # 4. 토픽 에지 (Paper - Has_Topic - Topic) 필터링
    pt_edge = data['paper', 'has_topic', 'topic'].edge_index.to(torch.int64)
    # paper(pt_edge[0])가 Cold 논문인 에지 마스킹 및 제거
    pt_in_cold = cold_mask[pt_edge[0]]
    train_pt_edges = pt_edge[:, ~pt_in_cold]
    
    print(f"  - 저술 에지 (Cold 제거): {ap_edge.size(1)} -> {train_ap_edges.size(1)}")
    print(f"  - 토픽 에지 (Cold 제거): {pt_edge.size(1)} -> {train_pt_edges.size(1)}")
    
    # 5. 깨끗하게 격리된 train_data 객체 동적 생성 (Deepcopy 대용)
    from torch_geometric.data import HeteroData
    train_data = HeteroData()
    
    train_data['paper'].num_nodes = num_papers
    train_data['paper'].x = data['paper'].x
    
    train_data['author'].num_nodes = num_authors
    if 'x' in data['author']:
        train_data['author'].x = data['author'].x
        
    train_data['topic'].num_nodes = num_topics
    if 'x' in data['topic']:
        train_data['topic'].x = data['topic'].x
        
    # 엄격하게 격리된 에지만 주입
    train_data['paper', 'cites', 'paper'].edge_index = train_pp_edges
    train_data['author', 'writes', 'paper'].edge_index = train_ap_edges
    train_data['paper', 'has_topic', 'topic'].edge_index = train_pt_edges
    
    # 검증: train_data에 Cold-Start 논문 관련 에지가 진짜 없는지 Assert 검사 (Zero-Leakage 안전장치)
    assert not torch.any(cold_mask[train_data['paper', 'cites', 'paper'].edge_index[0]]), "Leakage detected in train_pp_edges src"
    assert not torch.any(cold_mask[train_data['paper', 'cites', 'paper'].edge_index[1]]), "Leakage detected in train_pp_edges dst"
    assert not torch.any(cold_mask[train_data['author', 'writes', 'paper'].edge_index[1]]), "Leakage detected in train_ap_edges paper"
    assert not torch.any(cold_mask[train_data['paper', 'has_topic', 'topic'].edge_index[0]]), "Leakage detected in train_pt_edges paper"
    print("  [PASS] Zero-Leakage 검증 완료: 학습용 그래프에 Cold 논문 관련 에지가 전혀 존재하지 않습니다.")
    
    return train_data, val_pp_edges, test_pp_edges, cold_pp_edges, cold_mask


# ============================================================
# 2. 통합 그래프 및 피처 구축 (train_data 기준 - 누수 차단)
# ============================================================
def get_graph_info(train_data):
    """그래프의 노드 수 정보를 반환합니다."""
    num_papers = train_data['paper'].num_nodes
    num_authors = train_data['author'].num_nodes
    num_topics = train_data['topic'].num_nodes
    total_nodes = num_papers + num_authors + num_topics
    return num_papers, num_authors, num_topics, total_nodes


def build_unified_graph_v2(train_data):
    """학습용 train_data를 기반으로 완벽히 누수가 격리된 통합 그래프를 구축합니다."""
    num_papers, num_authors, num_topics, total_nodes = get_graph_info(train_data)
    offset_author = num_papers
    offset_topic = num_papers + num_authors

    edge_list = []

    # Paper - Paper (인용)
    pp_edge = train_data['paper', 'cites', 'paper'].edge_index.to(torch.int64)
    pp_mask = (pp_edge[0] < num_papers) & (pp_edge[1] < num_papers)
    edge_list.append(pp_edge[:, pp_mask])

    # Author - Paper (양방향)
    ap_edge = train_data['author', 'writes', 'paper'].edge_index.clone().to(torch.int64)
    ap_mask = (ap_edge[0] < num_authors) & (ap_edge[1] < num_papers)
    ap_edge = ap_edge[:, ap_mask]
    ap_edge[0] += offset_author
    edge_list.append(ap_edge)
    edge_list.append(ap_edge.flip(0))

    # Paper - Topic (양방향)
    pt_edge = train_data['paper', 'has_topic', 'topic'].edge_index.clone().to(torch.int64)
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


def build_initial_features_v2(train_data):
    """[핵심 개선] 오직 train_data(학습 그래프) 내의 연결만으로 Author와 Topic 피처를 생성하여 데이터 누수를 완벽 차단합니다."""
    num_papers, num_authors, num_topics, _ = get_graph_info(train_data)
    paper_x = train_data['paper'].x
    device = paper_x.device

    # Author 피처 (학습용 writes 에지만 반영)
    if 'x' in train_data['author']:
        author_x = train_data['author'].x
    else:
        author_x = torch.zeros((num_authors, EMBEDDING_DIM), device=device)
        ap_edge = train_data['author', 'writes', 'paper'].edge_index
        author_x.index_add_(0, ap_edge[0], paper_x[ap_edge[1]])
        counts = torch.bincount(ap_edge[0], minlength=num_authors).view(-1, 1).float().to(device)
        author_x = author_x / torch.clamp(counts, min=1.0)

    # Topic 피처 (학습용 has_topic 에지만 반영)
    if 'x' in train_data['topic']:
        topic_x = train_data['topic'].x
    else:
        topic_x = torch.zeros((num_topics, EMBEDDING_DIM), device=device)
        pt_edge = train_data['paper', 'has_topic', 'topic'].edge_index
        topic_x.index_add_(0, pt_edge[1], paper_x[pt_edge[0]])
        counts = torch.bincount(pt_edge[1], minlength=num_topics).view(-1, 1).float().to(device)
        topic_x = topic_x / torch.clamp(counts, min=1.0)

    # 전체 노드 통합 피처
    combined = torch.cat([paper_x, author_x, topic_x], dim=0)
    return combined


# ============================================================
# 3. 평가
# ============================================================
def _citation_sets(edges, num_papers):
    """Return ``source -> cited-paper set`` for paper-to-paper edges only."""
    src, dst = edges[0], edges[1]
    mask = (src < num_papers) & (dst < num_papers)
    result = {}
    for source, target in zip(src[mask].detach().cpu().tolist(), dst[mask].detach().cpu().tolist()):
        result.setdefault(source, set()).add(target)
    return result


def evaluate_ranking_v2(out, edges, num_papers, train_edges=None, k=TOP_K,
                        batch_size=EVAL_BATCH_SIZE, query_embeddings=None):
    """Evaluate citation retrieval per source paper, not per individual edge.

    Each source has one ranked candidate list and all of its held-out citations are
    relevance labels for that list.  Citations already observed in training are
    removed from the candidate list, preventing both leakage and an inflated
    Recall@K. ``query_embeddings`` is used by the cold-start evaluation.
    """
    positives_by_source = _citation_sets(edges, num_papers)
    seen_by_source = _citation_sets(train_edges, num_papers) if train_edges is not None else {}
    if not positives_by_source:
        return 0.0, 0.0

    paper_embeddings = out[:num_papers]
    query_embeddings = paper_embeddings if query_embeddings is None else query_embeddings
    device = paper_embeddings.device
    recalls, ndcgs = [], []
    sources = sorted(positives_by_source)

    for start in range(0, len(sources), batch_size):
        batch_sources = sources[start:start + batch_size]
        source_ids = torch.tensor(batch_sources, device=device, dtype=torch.long)
        scores = torch.matmul(query_embeddings[source_ids], paper_embeddings.t())

        # Exclude only known train citations.  Held-out labels remain eligible.
        for row, source in enumerate(batch_sources):
            seen = seen_by_source.get(source, set())
            if seen:
                scores[row, torch.tensor(sorted(seen), device=device)] = -torch.inf

        topk = torch.topk(scores, k=min(k, num_papers), dim=1).indices.cpu().tolist()
        for source, ranking in zip(batch_sources, topk):
            # A duplicated edge can appear across random splits; it is already
            # known at train time and therefore cannot be a valid test label.
            relevant = positives_by_source[source] - seen_by_source.get(source, set())
            if not relevant:
                continue
            hits = [rank for rank, paper_id in enumerate(ranking) if paper_id in relevant]
            recalls.append(len(hits) / len(relevant))
            dcg = sum(1.0 / torch.log2(torch.tensor(rank + 2.0)).item() for rank in hits)
            ideal_count = min(len(relevant), k)
            idcg = sum(1.0 / torch.log2(torch.tensor(rank + 2.0)).item()
                       for rank in range(ideal_count))
            ndcgs.append(dcg / idcg if idcg else 0.0)

    if not recalls:
        return 0.0, 0.0
    return sum(recalls) / len(recalls), sum(ndcgs) / len(ndcgs)


def evaluate_cold_start_v2(model, test_out, cold_edges, cold_mask, num_papers,
                           train_edges=None, k=TOP_K, batch_size=EVAL_BATCH_SIZE):
    """[핵심 개선] 진정한 신규 노드(Cold-Start) 상황을 평가합니다.

    학습 시 물리적으로 완벽 차단된 Cold-Start 논문 노드를 쿼리(`src`)로 삼고,
    이들이 가리키는 기존 논문(`pos_dst`)에 대한 추천 정확도를 측정합니다.
    """
    src, pos_dst = cold_edges[0], cold_edges[1]

    # 논문 -> 논문 에지만 한정
    mask = (src < num_papers) & (pos_dst < num_papers)
    src = src[mask]
    pos_dst = pos_dst[mask]

    # 쿼리 노드가 실제로 Cold-Start 논문인 에지만 필터링 (진정한 Cold 쿼리)
    cold_query_mask = cold_mask[src]
    src = src[cold_query_mask]
    pos_dst = pos_dst[cold_query_mask]

    if src.size(0) == 0:
        return 0.0, 0.0
    # Build a full query matrix so the common source-grouped evaluator can be
    # reused.  Cold sources receive text/knowledge-only embeddings.
    query_embeddings = test_out[:num_papers].clone()
    cold_sources = torch.unique(src)
    if hasattr(model, 'get_cold_start_embeddings'):
        query_embeddings[cold_sources] = model.get_cold_start_embeddings(cold_sources)
    else:
        query_embeddings[cold_sources] = 0
    return evaluate_ranking_v2(test_out, torch.stack([src, pos_dst]), num_papers,
                               train_edges=train_edges, k=k, batch_size=batch_size,
                               query_embeddings=query_embeddings)


# ============================================================
# 4. 통합 학습 루프
# ============================================================
def sample_paper_negatives(pos_src, train_pp_edges, num_papers):
    """Sample one non-cited Paper per positive citation.

    The lookup uses encoded paper-pair IDs, so negatives can never be Author or
    Topic nodes and can never be an observed positive citation for that source.
    """
    device = pos_src.device
    edge_hashes = torch.unique(
        train_pp_edges[0].to(device=device, dtype=torch.long) * num_papers
        + train_pp_edges[1].to(device=device, dtype=torch.long)
    ).sort().values
    negatives = torch.randint(num_papers, (pos_src.numel(),), device=device)

    # Sparse citation graphs converge quickly.  Keep resampling only invalid
    # positions, preserving a vectorized GPU path.
    invalid = torch.ones_like(negatives, dtype=torch.bool)
    while invalid.any():
        candidate_hashes = pos_src[invalid].long() * num_papers + negatives[invalid].long()
        locations = torch.searchsorted(edge_hashes, candidate_hashes)
        is_positive = torch.zeros_like(locations, dtype=torch.bool)
        in_bounds = locations < edge_hashes.numel()
        is_positive[in_bounds] = edge_hashes[locations[in_bounds]] == candidate_hashes[in_bounds]
        invalid_indices = invalid.nonzero(as_tuple=True)[0]
        invalid.fill_(False)
        invalid[invalid_indices[is_positive]] = True
        if invalid.any():
            negatives[invalid] = torch.randint(num_papers, (invalid.sum().item(),), device=device)
    return negatives


def train_and_evaluate_v2(model, train_data, val_edges, test_edges, cold_edges, cold_mask,
                          num_papers, model_name="Model", save_path=None, force_retrain=False):
    """통일된 BPR 학습 + 평가를 수행합니다 (v2 Zero-Leakage)."""
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_val_ndcg = -float('inf')
    checkpoint_written = False

    # 학습용 통합 그래프 추출
    train_unified_edges = build_unified_graph_v2(train_data)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # P0: never evaluate a stale train_v4/checkpoint result as a new benchmark.
    # A checkpoint is only a transient best epoch from *this* benchmark run.
    if save_path and os.path.exists(save_path):
        print(f"  [OVERWRITE] 기존 체크포인트를 재사용하지 않고 새 실험으로 덮어씁니다: {os.path.basename(save_path)}")

    train_pp_edges = train_data['paper', 'cites', 'paper'].edge_index
    # Do not turn a held-out real citation into a false negative.  These edges
    # are used only for negative filtering, never for message passing, scoring,
    # or model selection.
    all_positive_citations = torch.cat(
        [train_pp_edges, val_edges, test_edges, cold_edges], dim=1)
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()

        # GNN 모델의 경우 학습 그래프로 완전히 격리된 train_unified_edges 전달
        out = model(train_unified_edges)
        pos_src, pos_dst = train_pp_edges[0], train_pp_edges[1]
        neg_dst = sample_paper_negatives(pos_src, all_positive_citations, num_papers)

        pos_scores = (out[pos_src] * out[pos_dst]).sum(dim=-1)
        neg_scores = (out[pos_src] * out[neg_dst]).sum(dim=-1)
        bpr_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-15).mean()

        bpr_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out = model(train_unified_edges)
            val_recall, val_ndcg = evaluate_ranking_v2(
                val_out, val_edges, num_papers, train_edges=train_pp_edges)

            if val_ndcg > best_val_ndcg:
                best_val_ndcg = val_ndcg
                if save_path:
                    torch.save(model.state_dict(), save_path)
                    checkpoint_written = True

        if epoch % EVAL_INTERVAL == 0 or epoch == 1:
            print(f"  [{model_name}] Epoch {epoch:3d} | BPR Loss: {bpr_loss.item():.4f} "
                  f"| Val R@{TOP_K}: {val_recall:.4f} | Val N@{TOP_K}: {val_ndcg:.4f}")

    # Best 모델 복원 후 최종 테스트 (General + True Cold-Start)
    if save_path and checkpoint_written:
        model.load_state_dict(torch.load(save_path, weights_only=True))
    elif save_path:
        print("  [WARN] 이번 실행에서 validation checkpoint가 생성되지 않아 마지막 epoch를 평가합니다.")

    model.eval()
    with torch.no_grad():
        test_out = model(train_unified_edges)
        
        # 1. 일반 성능 (Transductive - Warm)
        test_recall, test_ndcg = evaluate_ranking_v2(
            test_out, test_edges, num_papers, train_edges=train_pp_edges)
        
        # 2. 콜드 스타트 성능 (True Inductive Simulation)
        cs_recall, cs_ndcg = evaluate_cold_start_v2(
            model, test_out, cold_edges, cold_mask, num_papers, train_edges=train_pp_edges)
        
        print(f"  [OK] [{model_name}] Test R@{TOP_K}: {test_recall:.4f} | N@{TOP_K}: {test_ndcg:.4f}")
        print(f"  [CS] [{model_name}] Cold-Start R@{TOP_K}: {cs_recall:.4f} | N@{TOP_K}: {cs_ndcg:.4f}")

    return test_recall, test_ndcg, cs_recall, cs_ndcg
