import torch
import torch.nn as nn
import torch.optim as optim
import os, sys
from torch_geometric.data import HeteroData

# 모델 클래스 로드 (LGCmodel.py가 같은 폴더에 있다고 가정)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(project_root, 'code', 'model'))
from LGCmodel import ArxivLightGCN

# --- 설정 ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GRAPH_PATH = os.path.join(project_root, 'subdataset', 'build_hetero_graph.pt')
SAVE_PATH = os.path.join(project_root, 'output', 'lightgcn_trained.pt')

def train():
    print(f"🖥️ 사용 장치: {DEVICE}")
    
    # 1. 정렬된 v3 데이터 로드
    data = torch.load(GRAPH_PATH, weights_only=False).to(DEVICE)
    print(f"📦 그래프 로드 완료: {GRAPH_PATH}")

    # 2. 모델 초기화
    model = ArxivLightGCN(data, embedding_dim=128, num_layers=2).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    # 통합 에지 인덱스 생성
    unified_edges = model._build_unified_edge_index(data)
    
    # 3. 학습 루프 (BPR Loss 또는 Simple Contrastive Loss)
    # 여기서는 추천의 기본인 임베딩 유사도 극대화를 위해 간단한 재구성 학습을 진행합니다.
    model.train()
    print("🚀 학습 시작...")
    
    for epoch in range(1, 51):
        optimizer.zero_grad()
        
        # LightGCN 전파
        out = model(unified_edges)
        
        # 간단한 자가 학습 (Self-supervised): 연결된 노드끼리 임베딩이 비슷해지도록
        # (실제 추천 엔진에서는 BPR Loss를 쓰지만, 프로토타입은 가볍게 시작합니다)
        pos_src = unified_edges[0]
        pos_dst = unified_edges[1]
        
        # 내적(Dot Product)을 통한 유사도 계산
        pos_scores = (out[pos_src] * out[pos_dst]).sum(dim=-1)
        loss = -torch.log(torch.sigmoid(pos_scores)).mean()
        
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f}")

    # 4. 결과 저장
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"✅ 학습 완료 및 모델 저장: {SAVE_PATH}")

if __name__ == "__main__":
    train()