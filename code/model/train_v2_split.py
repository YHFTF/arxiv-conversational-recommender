import torch
import torch.nn as nn
import torch.optim as optim
import os, sys, json

# 1. 경로 및 모델 로드
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(project_root, 'code', 'model'))
from LGCmodel_v2 import ArxivLightGCNV2 # v2 지식 확장 모델

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GRAPH_PATH = os.path.join(project_root, 'subdataset', 'build_hetero_graph_v2.pt')
META_PATH = os.path.join(project_root, 'output', 'knowledge_meta.json')
SAVE_PATH = os.path.join(project_root, 'output', 'lightgcn_v2_split_trained.pt')

def train_v2_with_split():
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
    # V2의 _build_unified_graph 로직을 사용하여 모든 타입의 에지를 가져옴
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

    print(f"📊 [V2] 에지 분할 완료 | Train: {train_edges.size(1)} | Val: {val_edges.size(1)} | Test: {test_edges.size(1)}")

    # 4. 학습 루프
    best_val_loss = float('inf')
    
    for epoch in range(1, 51):
        model.train()
        optimizer.zero_grad()
        
        # 🌟 중요: 오직 학습용 에지로만 메시지 패싱 (Data Leakage 방지)
        out = model(train_edges)
        
        # Train Loss 계산
        pos_src, pos_dst = train_edges[0], train_edges[1]
        pos_scores = (out[pos_src] * out[pos_dst]).sum(dim=-1)
        train_loss = -torch.log(torch.sigmoid(pos_scores)).mean()
        
        train_loss.backward()
        optimizer.step()
        
        # Validation 평가
        model.eval()
        with torch.no_grad():
            v_src, v_dst = val_edges[0], val_edges[1]
            v_scores = (out[v_src] * out[v_dst]).sum(dim=-1)
            val_loss = -torch.log(torch.sigmoid(v_scores)).mean()
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), SAVE_PATH)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Train Loss: {train_loss.item():.4f} | Val Loss: {val_loss.item():.4f}")

    # 5. 최종 테스트
    model.load_state_dict(torch.load(SAVE_PATH))
    model.eval()
    with torch.no_grad():
        t_src, t_dst = test_edges[0], test_edges[1]
        t_scores = (out[t_src] * out[t_dst]).sum(dim=-1)
        test_loss = -torch.log(torch.sigmoid(t_scores)).mean()
        print("\n" + "="*50)
        print(f"🏁 [V2] 최종 테스트 결과 (Unseen Data)")
        print(f" - Test Loss: {test_loss.item():.4f}")
        print("="*50)

if __name__ == "__main__":
    train_v2_with_split()