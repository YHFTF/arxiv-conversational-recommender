import torch
import torch.nn as nn
import torch.optim as optim
import os, sys
from torch_geometric.data import HeteroData

# 모델 클래스 로드 (v3 구조 중심 모델)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(project_root, 'code', 'model'))
from LGCmodel import ArxivLightGCN # v3에서 썼던 기본 모델

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GRAPH_PATH = os.path.join(project_root, 'subdataset', 'build_hetero_graph.pt')
SAVE_PATH = os.path.join(project_root, 'output', 'lightgcn_split_trained.pt')

def train_with_split():
    print(f"🖥️ 사용 장치: {DEVICE}")
    data = torch.load(GRAPH_PATH, weights_only=False).to(DEVICE)
    
    # 1. 모델 초기화 및 통합 에지 생성
    model = ArxivLightGCN(data, embedding_dim=128, num_layers=2).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    # 전체 에지 구축 (Cites, Writes, Has_Topic 통합)
    full_edges = model._build_unified_edge_index(data)
    num_edges = full_edges.size(1)
    
    # 2. 데이터 8:1:1 분할 (인덱스 셔플)
    indices = torch.randperm(num_edges)
    train_size = int(0.8 * num_edges)
    val_size = int(0.1 * num_edges)
    
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]
    
    train_edges = full_edges[:, train_idx]
    val_edges = full_edges[:, val_idx]
    test_edges = full_edges[:, test_idx]

    print(f"📊 에지 분할 완료 | Train: {train_edges.size(1)} | Val: {val_edges.size(1)} | Test: {test_edges.size(1)}")

    # 3. 학습 루프
    best_val_loss = float('inf')
    
    for epoch in range(1, 51): 
        model.train()
        optimizer.zero_grad()
        
        # 전체 그래프에서 메시지 패싱 (Transductive 설정)
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
            # 🌟 평가는 학습된 임베딩(train_edges 기반)으로 검증용 에지를 예측합니다.
            v_src, v_dst = val_edges[0], val_edges[1]
            v_scores = (out[v_src] * out[v_dst]).sum(dim=-1)
            val_loss = -torch.log(torch.sigmoid(v_scores)).mean()
            
            # 최적 모델 저장 (Early Stopping 대용)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), SAVE_PATH)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Train Loss: {train_loss.item():.4f} | Val Loss: {val_loss.item():.4f}")

    # 5. Final Test (최종 성능 확인)
    model.load_state_dict(torch.load(SAVE_PATH))
    model.eval()
    with torch.no_grad():
        t_src, t_dst = test_edges[0], test_edges[1]
        t_scores = (out[t_src] * out[t_dst]).sum(dim=-1)
        test_loss = -torch.log(torch.sigmoid(t_scores)).mean()
        print("\n" + "="*50)
        print(f"🏁 최종 테스트 결과 (Unseen Data)")
        print(f" - Test Loss: {test_loss.item():.4f}")
        print("="*50)

if __name__ == "__main__":
    train_with_split()