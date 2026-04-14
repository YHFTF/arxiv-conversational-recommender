import torch
import json
import os
import sys

# 경로 설정
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(project_root, 'code', 'model'))

from LGCmodel_v2 import ArxivLightGCNV2
from LGCmodel_v4 import ArxivLightGCNV4

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GRAPH_PATH = os.path.join(project_root, 'subdataset', 'build_hetero_graph_v2.pt')
META_PATH = os.path.join(project_root, 'output', 'knowledge_meta.json')
MASTER_FILE = os.path.join(project_root, 'subdataset', 'arxiv_master_final.json')

def evaluate_ranking_all_papers(out, edges, num_papers=16000, k=20, batch_size=4096):
    src, pos_dst = edges[0], edges[1]
    paper_embeddings = out[:num_papers]
    total_edges = src.size(0)
    
    all_hits = []
    all_ndcgs = []
    
    # 메모리 초과를 방지하기 위해 배치 단위 연산
    for i in range(0, total_edges, batch_size):
        end = min(i + batch_size, total_edges)
        batch_src = src[i:end]
        batch_pos_dst = pos_dst[i:end]
        
        # [batch_size, emb_dim]와 [16000, emb_dim]의 행렬 곱셈 연산 (모든 논문과 유사도 계산)
        batch_src_embs = out[batch_src]
        all_scores = torch.matmul(batch_src_embs, paper_embeddings.t())
        
        # 순위 측정 (정답 논문 인덱스가 줄을 선 결과에서 몇 번째 칸에 위치하는지 계산)
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

def main():
    print(f"🖥️ 사용 장치: {DEVICE}")
    print("🔄 데이터 및 그래프를 로드하는 중...")
    
    data = torch.load(GRAPH_PATH, weights_only=False).to(DEVICE)
    with open(META_PATH, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    meta_counts = {'domains': len(meta['domains']), 'tasks': len(meta['tasks']), 'methods': len(meta['methods'])}
    
    # 평가의 공정성을 위해 난수 시드(Seed) 고정
    # 모든 모델이 똑같이 분할된 동일한 Test Edge에서 시험을 치루도록 보장
    torch.manual_seed(42)
    
    # V2, V3, V4 모두 graph 구조(unified_edges)는 사실상 동일
    temp_model = ArxivLightGCNV2(data, meta_counts).to(DEVICE)
    full_edges = temp_model._build_unified_graph(data)
    
    num_edges = full_edges.size(1)
    indices = torch.randperm(num_edges)
    train_size = int(0.8 * num_edges)
    val_size = int(0.1 * num_edges)
    
    test_idx = indices[train_size + val_size:]
    test_edges = full_edges[:, test_idx]
    
    # 임시 모델 모델 삭제로 메모리 확보
    del temp_model

    models_to_test = [
        {"name": "V2 (MSE Loss)", "file": "lightgcn_v2_split_trained.pt", "version": "v2"},
        {"name": "V3 (BPR Loss)", "file": "lightgcn_v3_bpr_trained.pt", "version": "v3"},
        {"name": "V4 (Knowledge BPR)", "file": "lightgcn_v4_knowledge_bpr.pt", "version": "v4"}
    ]
    
    results = []
    paper_knowledge_ids = None

    for m_info in models_to_test:
        model_path = os.path.join(project_root, 'output', m_info['file'])
        if not os.path.exists(model_path):
            print(f"⏭️ {m_info['file']} 파일이 존재하지 않아 건너뜁니다.")
            continue
            
        print(f"▶️ 평가 진행 중: {m_info['name']}")
        
        # 모델 구조(껍데기) 로드
        if m_info['version'] == 'v4':
            if paper_knowledge_ids is None:
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
            model = ArxivLightGCNV4(data, meta_counts, paper_knowledge_ids).to(DEVICE)
        else:
            model = ArxivLightGCNV2(data, meta_counts).to(DEVICE)
            
        # 가중치(.pt) 삽입
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
        
        with torch.no_grad():
            out = model(full_edges)
            test_recall_20, test_ndcg_20 = evaluate_ranking_all_papers(out, test_edges, num_papers=data['paper'].num_nodes, k=20)
            
        results.append({
            "name": m_info['name'],
            "recall": test_recall_20,
            "ndcg": test_ndcg_20
        })

    print("\n" + "="*65)
    print("🏆 버전에 따른 통합 성능 평가 리포트 (동일 조건 Test Data)")
    print("="*65)
    # 한글 지원을 위해서 파이썬 f-string 포맷팅을 단순하게 사용
    print(f"| {'모델 버전명':<22} | {'Recall@20 (%)':<15} | {'NDCG@20':<12} |")
    print("-" * 65)
    for res in results:
        recall_pct = f"{res['recall']*100:.2f}%"
        print(f"| {res['name']:<26} | {recall_pct:<15} | {res['ndcg']:<12.4f} |")
    print("="*65)

if __name__ == "__main__":
    main()
