import torch
import torch.nn as nn
import os
import sys
import numpy as np

# 모델 파일 경로 설정
project_root = r"C:\Users\Arachne\OneDrive\Desktop\arxiv-conversational-recommender-main"
sys.path.append(os.path.join(project_root, 'code', 'model'))

# LGCmodel.py에서 클래스 가져오기
try:
    from LGCmodel import ArxivLightGCN
except ImportError:
    print("❌ LGCmodel.py를 찾을 수 없습니다. 경로를 확인하세요.")

# --- 1. 설정 및 장치 준비 ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GRAPH_PATH = os.path.join(project_root, 'subdataset', 'build_hetero_graph.pt')
OUTPUT_DIR = os.path.join(project_root, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 2. 데이터 로드 함수 ---
def load_data():
    if not os.path.exists(GRAPH_PATH):
        raise FileNotFoundError(f"❌ 그래프 파일을 찾을 수 없습니다: {GRAPH_PATH}")
    
    print(f"📦 그래프 로드 중: {GRAPH_PATH}")
    # weights_only=False는 PyTorch 최신 버전 호환성을 위해 필요합니다.
    data = torch.load(GRAPH_PATH, weights_only=False)
    return data.to(device)

# --- 3. 메인 학습 함수 (이게 'main'입니다!) ---
def main():
    print(f"🖥️  사용 장치: {device}")
    
    # 데이터 로드
    data = load_data()
    
    # 모델 초기화
    # 인자: data, embedding_dim=64, num_layers=3
    model = ArxivLightGCN(data, embedding_dim=64, num_layers=3).to(device)
    
    # 통합 에지 인덱스 생성 (LGCmodel 내의 메서드 호출)
    unified_edge_index = model._build_unified_edge_index(data).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    
    # 학습 타겟: Paper -> Paper 인용 관계
    pos_edge_index = data['paper', 'cites', 'paper'].edge_index
    num_papers = data['paper'].num_nodes

    print("\n" + "="*50)
    print(f"🚀 LightGCN 학습 시작 (총 {pos_edge_index.size(1):,}개 관계)")
    print("="*50)

    model.train()
    for epoch in range(1, 101):
        optimizer.zero_grad()
        
        # 1. 모든 노드 임베딩 추출 (전파 수행)
        out = model(unified_edge_index)
        
        # 2. 네거티브 샘플링 (랜덤하게 연결되지 않은 논문 선택)
        neg_edge_index = torch.randint(0, num_papers, pos_edge_index.size(), device=device)
        
        # 3. BPR Loss 계산
        # LightGCN 모델 내부의 recommendation_loss 활용
        loss = model.model.recommendation_loss(
            out, 
            pos_edge_index, 
            neg_edge_index
        )
        
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch [{epoch:3d}/100] | Loss: {loss.item():.4f}")

    # --- 4. 학습 결과 저장 ---
    SAVE_PATH = os.path.join(OUTPUT_DIR, 'lightgcn_trained_model.pt')
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"\n✅ 학습 완료 및 모델 저장: {SAVE_PATH}")

# --- 5. 실행부 ---
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"❌ 실행 중 에러 발생:\n{traceback.format_exc()}")