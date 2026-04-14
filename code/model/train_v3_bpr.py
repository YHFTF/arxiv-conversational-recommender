import torch
import torch.nn as nn
import torch.optim as optim
import os, sys, json
import numpy as np

# 1. 경로 및 모델 로드
# 현재 스크립트 위치가 code/model 이므로 상위 2단계로 올라가면 프로젝트 루트입니다.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(project_root, 'code', 'model'))
from LGCmodel_v2 import ArxivLightGCNV2 # v2 지식 확장 모델

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GRAPH_PATH = os.path.join(project_root, 'subdataset', 'build_hetero_graph_v2.pt')
META_PATH = os.path.join(project_root, 'output', 'knowledge_meta.json')
SAVE_PATH = os.path.join(project_root, 'output', 'lightgcn_v3_bpr_trained.pt')

def evaluate_ranking(out, edges, k=20, num_negatives=99):
    """
    Validation 및 Test를 위한 정량 평가 함수.
    1개의 Positive Edge(정답) 대비 num_negatives 개의 무작위 오답(Negative Edge)을 
    생성하여 랭킹 및 변별력을 테스트합니다.
    """
    src, pos_dst = edges[0], edges[1]
    batch_size = src.size(0)
    
    # 1. 정답(Positive) 스코어 계산: (N, 1)
    pos_scores = (out[src] * out[pos_dst]).sum(dim=-1).unsqueeze(1)
    
    # 2. 오답(Negative) 스코어 계산: (N, num_negatives)
    # out.size(0)은 전체 노드 개수(total_nodes)입니다.
    neg_dst = torch.randint(0, out.size(0), (batch_size, num_negatives), device=out.device)
    neg_scores = (out[src].unsqueeze(1) * out[neg_dst]).sum(dim=-1)
    
    # 3. 랭킹 병합 및 정렬 
    # scores: (N, 1 + num_negatives) - 0번 인덱스가 정답 위치
    scores = torch.cat([pos_scores, neg_scores], dim=1)
    
    # 내림차순 정렬했을 때 정답(0번째)이 위치하는 랭크 번호 찾기
    _, indices = torch.sort(scores, dim=1, descending=True)
    rankings = (indices == 0).nonzero(as_tuple=True)[1] # 0이 1등, 1이 2등...
    
    # Recall@K (K등 안에 들었는가?)
    hits = (rankings < k).float().mean().item()
    
    # NDCG@K (들어갔을 때 얼마나 앞쪽에 있는가 가중치 점수)
    ndcg = (1.0 / torch.log2(rankings.float() + 2.0))
    ndcg[rankings >= k] = 0.0
    ndcg = ndcg.mean().item()
    
    return hits, ndcg

def train_v3_bpr():
    print(f"🖥️ 사용 장치: {DEVICE}")
    
    # 데이터 및 메타 정보 로드
    data = torch.load(GRAPH_PATH, weights_only=False).to(DEVICE)
    with open(META_PATH, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    meta_counts = {
        'domains': len(meta['domains']),
        'tasks': len(meta['tasks']),
        'methods': len(meta['methods'])
    }

    # 2. 모델 초기화
    model = ArxivLightGCNV2(data, meta_counts).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    # 3. 통합 에지 구축 및 8:1:1 분할
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

    print(f"📊 [V3 BPR] 에지 분할 완료 | Train: {train_edges.size(1)} | Val: {val_edges.size(1)} | Test: {test_edges.size(1)}")

    # 4. 학습 루프
    best_val_ndcg = -float('inf') # 이번엔 Recall/NDCG가 높을수록 좋음
    
    for epoch in range(1, 101): # 보통 BPR 학습은 조금 더 필요할 수 있어 100 에포크로 늘려줍니다
        model.train()
        optimizer.zero_grad()
        
        # 메시지 패싱 (Train Data Leakage 차단 유지)
        out = model(train_edges)
        
        # BPR Loss (Pairwise Training) 계산
        pos_src, pos_dst = train_edges[0], train_edges[1]
        
        # 무작위 오답 노드 샘플링
        neg_dst = torch.randint(0, model.total_nodes, (pos_src.size(0),), device=DEVICE)
        
        pos_scores = (out[pos_src] * out[pos_dst]).sum(dim=-1)
        neg_scores = (out[pos_src] * out[neg_dst]).sum(dim=-1)
        
        # 정답 점수는 올리고 오답 점수는 내리는 Pairwise 방식을 Sigmoid(-log)에 태움
        # 1e-15 는 log(0)으로 터지는 값을 방지하기 위함
        bpr_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-15).mean()
        
        bpr_loss.backward()
        optimizer.step()
        
        # Validation 평가
        model.eval()
        with torch.no_grad():
            # Evaluation에서는 BPR Loss 점수가 아닌, 명확한 추천 순위 지표를 구함
            val_recall_20, val_ndcg_20 = evaluate_ranking(out, val_edges, k=20)
            
            # 최고 성능 달성 시 저장 기준을 Loss -> NDCG로 변경
            if val_ndcg_20 > best_val_ndcg:
                best_val_ndcg = val_ndcg_20
                torch.save(model.state_dict(), SAVE_PATH)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Train BPR Loss: {bpr_loss.item():.4f} | Val Recall@20: {val_recall_20:.4f} | Val NDCG@20: {val_ndcg_20:.4f}")

    # 5. 최종 테스트
    model.load_state_dict(torch.load(SAVE_PATH))
    model.eval()
    with torch.no_grad():
        test_recall_20, test_ndcg_20 = evaluate_ranking(out, test_edges, k=20)
        print("\n" + "="*60)
        print(f"🏁 [V3 BPR] 최종 테스트 결과 (Unseen Data 10%)")
        print("    : 테스트 과정에서 정답 1개 vs 오답 99개의 랭킹 테스트 진행")
        print(f" - ⭐ Test Recall@20: {test_recall_20:.4f}")
        print(f" - ⭐ Test NDCG@20:   {test_ndcg_20:.4f}")
        print("="*60)

if __name__ == "__main__":
    train_v3_bpr()
