import torch
import torch.nn as nn
import torch.optim as optim
import os, sys, json
import numpy as np

# 현재 스크립트 위치가 code/model 이므로 상위 2단계로 올라가면 프로젝트 루트입니다.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(project_root, 'code', 'model'))
from LGCmodel_v4 import ArxivLightGCNV4

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GRAPH_PATH = os.path.join(project_root, 'subdataset', 'build_hetero_graph_v2.pt')
META_PATH = os.path.join(project_root, 'output', 'knowledge_meta.json')
MASTER_FILE = os.path.join(project_root, 'subdataset', 'arxiv_master_final.json')
SAVE_PATH = os.path.join(project_root, 'output', 'lightgcn_v4_knowledge_bpr.pt')

def evaluate_ranking_all_papers(out, edges, num_papers, k=20, batch_size=4096):
    src, pos_dst = edges[0], edges[1]
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
        
    final_hits = torch.cat(all_hits).mean().item()
    final_ndcg = torch.cat(all_ndcgs).mean().item()
    
    return final_hits, final_ndcg

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

    full_edges = model._build_unified_graph(data)
    num_edges = full_edges.size(1)
    
    indices = torch.randperm(num_edges)
    train_size = int(0.8 * num_edges)
    val_size = int(0.1 * num_edges)
    
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]
    
    train_edges = full_edges[:, train_idx]
    val_edges = full_edges[:, val_idx]
    test_edges = full_edges[:, test_idx]

    print(f"📊 [V4 Knowledge BPR] 에지 분할 완료 | Train: {train_edges.size(1)} | Val: {val_edges.size(1)} | Test: {test_edges.size(1)}")

    # 3. 학습 루프
    best_val_ndcg = -float('inf')
    
    for epoch in range(1, 101):
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
            val_recall_20, val_ndcg_20 = evaluate_ranking_all_papers(out, val_edges, num_papers=data['paper'].num_nodes, k=20)
            
            if val_ndcg_20 > best_val_ndcg:
                best_val_ndcg = val_ndcg_20
                torch.save(model.state_dict(), SAVE_PATH)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Train BPR Loss: {bpr_loss.item():.4f} | Val Recall@20: {val_recall_20:.4f} | Val NDCG@20: {val_ndcg_20:.4f}")

    # 4. 종합 테스트
    model.load_state_dict(torch.load(SAVE_PATH))
    model.eval()
    with torch.no_grad():
        test_recall_20, test_ndcg_20 = evaluate_ranking_all_papers(out, test_edges, num_papers=data['paper'].num_nodes, k=20)
        print("\n" + "="*60)
        print(f"🏁 [V4 Knowledge BPR] 최종 테스트 결과 (Unseen Data 10%)")
        print(f"    : 16,000개 모든 논문 대상 All-Item Ranking 테스트 진행")
        print(f" - ⭐ Test Recall@20: {test_recall_20:.4f}")
        print(f" - ⭐ Test NDCG@20:   {test_ndcg_20:.4f}")
        print("="*60)

if __name__ == "__main__":
    train_v4_knowledge_bpr()
