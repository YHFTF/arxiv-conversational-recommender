import torch
import torch.nn as nn
import torch.optim as optim
import os, sys, json
import numpy as np

# 현재 스크립트 위치가 code/model 이므로 상위 2단계로 올라가면 프로젝트 루트입니다.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(project_root, 'code', 'model'))
sys.path.append(os.path.join(project_root, 'code', 'test'))
from LGCmodel_v4 import ArxivLightGCNV4
from utils_v2 import evaluate_ranking_v2, sample_paper_negatives

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GRAPH_PATH = os.path.join(project_root, 'subdataset', 'build_hetero_graph_v2.pt')
output_dir = os.path.abspath(os.getenv('OUTPUT_DIR', os.path.join(project_root, 'output')))
META_PATH = os.path.join(output_dir, 'knowledge_meta.json')
MASTER_FILE = os.path.join(project_root, 'subdataset', 'arxiv_master_final.json')
SAVE_PATH = os.path.join(output_dir, 'lightgcn_v4_knowledge_bpr.pt')

def train_v4_knowledge_bpr():
    print(f"🖥️ 사용 장치: {DEVICE}")
    
    # 1. 데이터 로드
    data = torch.load(GRAPH_PATH, weights_only=False).to(DEVICE)
    with open(META_PATH, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    meta_counts = {
        'domains': len(meta['domains']),
        'tasks': len(meta['tasks']),
        'methods': len(meta['methods'])
    }

    with open(MASTER_FILE, 'r', encoding='utf-8') as f:
        master_list = json.load(f)
        
    print("🧠 지식(Knowledge) 텐서 구축 중...")
    num_papers = data['paper'].num_nodes
    paper_knowledge_ids = torch.zeros((num_papers, 3), dtype=torch.long)
    for i, item in enumerate(master_list):
        if i >= num_papers: break
        k_dict = item.get('knowledge', {})
        
        # 도메인 파싱
        # 데이터가 배열일 가능성 대비 (임시로 첫 번째 속성만 사용)
        d_val = k_dict.get('domain', None)
        if isinstance(d_val, list) and len(d_val) > 0: d_val = d_val[0]
        
        t_val = k_dict.get('task', None)
        if isinstance(t_val, list) and len(t_val) > 0: t_val = t_val[0]
            
        m_val = k_dict.get('method', None)
        if isinstance(m_val, list) and len(m_val) > 0: m_val = m_val[0]
        
        d_id = meta['domains'].get(d_val, 0) if d_val else 0
        t_id = meta['tasks'].get(t_val, 0) if t_val else 0
        m_id = meta['methods'].get(m_val, 0) if m_val else 0
        
        paper_knowledge_ids[i] = torch.tensor([d_id, t_id, m_id])

    # 2. V4 모델 초기화 (에지 통합)
    model = ArxivLightGCNV4(data, meta_counts, paper_knowledge_ids).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    # Citation recommendation is trained/evaluated only on Paper -> Paper
    # citations.  Author/Topic relations remain context in the GNN graph, but
    # must never be used as BPR positives or negatives.
    citation_edges = data['paper', 'cites', 'paper'].edge_index.to(DEVICE)
    num_edges = citation_edges.size(1)
    
    indices = torch.randperm(num_edges)
    train_size = int(0.8 * num_edges)
    val_size = int(0.1 * num_edges)
    
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]
    
    train_edges = citation_edges[:, train_idx]
    val_edges = citation_edges[:, val_idx]
    test_edges = citation_edges[:, test_idx]

    # Keep Author/Topic context, but remove validation/test citations from
    # message passing as well as from the BPR objective.
    train_graph_data = data.clone()
    train_graph_data['paper', 'cites', 'paper'].edge_index = train_edges
    full_edges = model._build_unified_graph(train_graph_data)

    print(f"📊 [V4 Knowledge BPR] 에지 분할 완료 | Train: {train_edges.size(1)} | Val: {val_edges.size(1)} | Test: {test_edges.size(1)}")

    # 3. 학습 루프
    best_val_ndcg = -float('inf')
    checkpoint_written = False
    
    for epoch in range(1, 101):
        model.train()
        optimizer.zero_grad()
        
        out = model(full_edges)
        
        pos_src, pos_dst = train_edges[0], train_edges[1]
        # Filter against every real citation, including held-out labels, so a
        # true citation can never be presented as a BPR negative.
        neg_dst = sample_paper_negatives(pos_src, citation_edges, data['paper'].num_nodes)
        
        pos_scores = (out[pos_src] * out[pos_dst]).sum(dim=-1)
        neg_scores = (out[pos_src] * out[neg_dst]).sum(dim=-1)
        
        bpr_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-15).mean()
        
        bpr_loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_out = model(full_edges)
            val_recall_20, val_ndcg_20 = evaluate_ranking_v2(
                val_out, val_edges, num_papers=data['paper'].num_nodes,
                train_edges=train_edges, k=20)
            
            if val_ndcg_20 > best_val_ndcg:
                best_val_ndcg = val_ndcg_20
                torch.save(model.state_dict(), SAVE_PATH)
                checkpoint_written = True

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Train BPR Loss: {bpr_loss.item():.4f} | Val Recall@20: {val_recall_20:.4f} | Val NDCG@20: {val_ndcg_20:.4f}")

    # 4. 종합 테스트
    if checkpoint_written:
        model.load_state_dict(torch.load(SAVE_PATH, weights_only=True))
    else:
        print("[WARN] 이번 실행에서 validation checkpoint가 생성되지 않아 마지막 epoch를 평가합니다.")
    model.eval()
    with torch.no_grad():
        # Recompute output after restoring the selected checkpoint; ``out``
        # from the last epoch is not the best model's representation.
        test_out = model(full_edges)
        test_recall_20, test_ndcg_20 = evaluate_ranking_v2(
            test_out, test_edges, num_papers=data['paper'].num_nodes,
            train_edges=train_edges, k=20)
        print("\n" + "="*60)
        print(f"🏁 [V4 Knowledge BPR] 최종 테스트 결과 (Unseen Data 10%)")
        print(f"    : 16,000개 모든 논문 대상 All-Item Ranking 테스트 진행")
        print(f" - ⭐ Test Recall@20: {test_recall_20:.4f}")
        print(f" - ⭐ Test NDCG@20:   {test_ndcg_20:.4f}")
        print("="*60)

if __name__ == "__main__":
    train_v4_knowledge_bpr()
