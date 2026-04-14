import torch
import torch.nn as nn
import torch.optim as optim
import os, sys, json

# 경로 설정
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(project_root, 'code', 'model'))
from LGCmodel_v2 import ArxivLightGCNV2

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GRAPH_PATH = os.path.join(project_root, 'subdataset', 'build_hetero_graph_v2.pt')
SAVE_PATH = os.path.join(project_root, 'output', 'lightgcn_v2_trained.pt')
META_PATH = os.path.join(project_root, 'output', 'knowledge_meta.json')

def train():
    print(f"🖥️ 장치: {DEVICE}")
    data = torch.load(GRAPH_PATH, weights_only=False).to(DEVICE)
    
    with open(META_PATH, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    meta_counts = {'domains': len(meta['domains']), 'tasks': len(meta['tasks']), 'methods': len(meta['methods'])}

    model = ArxivLightGCNV2(data, meta_counts).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    # 엣지 통합 (V2 전용 로직)
    unified_edges = model._build_unified_graph(data)
    
    model.train()
    print("🚀 V2 지식 확장 모델 학습 가동...")
    for epoch in range(1, 51):
        optimizer.zero_grad()
        out = model(unified_edges)
        
        # BPR 기반 유사도 학습 (Positive Pair)
        pos_src, pos_dst = unified_edges[0], unified_edges[1]
        pos_scores = (out[pos_src] * out[pos_dst]).sum(dim=-1)
        loss = -torch.log(torch.sigmoid(pos_scores)).mean()
        
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f}")

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"✅ 모델 저장 완료: {SAVE_PATH}")

if __name__ == "__main__":
    train()