import torch
import json
import os
import sys

# 경로 설정
project_root = r"C:\Users\Arachne\OneDrive\Desktop\arxiv-conversational-recommender-main"
sys.path.append(os.path.join(project_root, 'code', 'model'))
from LGCmodel_v2 import ArxivLightGCNV2

# 파일 경로
GRAPH_PATH = os.path.join(project_root, 'subdataset', 'build_hetero_graph_v2.pt')
MODEL_PATH = os.path.join(project_root, 'output', 'lightgcn_v2_trained.pt')
MASTER_FILE = os.path.join(project_root, 'subdataset', 'arxiv_master_final.json')
META_PATH = os.path.join(project_root, 'output', 'knowledge_meta.json')

def recommend_v2(target_title, top_k=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 메타데이터 및 데이터 로드
    with open(META_PATH, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    meta_counts = {'domains': len(meta['domains']), 'tasks': len(meta['tasks']), 'methods': len(meta['methods'])}

    data = torch.load(GRAPH_PATH, weights_only=False).to(device)
    with open(MASTER_FILE, 'r', encoding='utf-8') as f:
        master_list = json.load(f)
    
    # 2. V2 모델 초기화 및 가중치 로드
    model = ArxivLightGCNV2(data, meta_counts).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()

    # 3. 모든 논문 임베딩 추출
    with torch.no_grad():
        unified_edges = model._build_unified_graph(data)
        all_embeddings = model(unified_edges)
        paper_embeddings = model.get_paper_embeddings(all_embeddings)

    # 4. 제목 검색
    target_idx = -1
    for i, item in enumerate(master_list):
        if target_title.lower() in item['title'].lower():
            target_idx = i
            print(f"\n🎯 검색된 논문: {item['title']} (Idx: {target_idx})")
            break
    
    if target_idx == -1:
        print(f"❌ '{target_title}' 논문을 찾을 수 없습니다.")
        return

    # 5. 코사인 유사도 기반 추천
    target_vec = paper_embeddings[target_idx].unsqueeze(0)
    sim_scores = torch.cosine_similarity(target_vec, paper_embeddings)
    values, indices = torch.topk(sim_scores, k=top_k + 1)
    
    print(f"\n📚 [V2-Extended] '{target_title}' 기반 추천 리스트:")
    print("=" * 100)
    
    for i in range(1, len(indices)):
        idx = indices[i].item()
        score = values[i].item()
        rec_item = master_list[idx]
        
        print(f"[{i}] 유사도: {score:.4f} | {rec_item['title']}")
        print(f"    - 핵심기법: {rec_item['knowledge']['method']}")
        print(f"    - 연구분야: {rec_item['knowledge']['domain']}")
        print("-" * 100)

if __name__ == "__main__":
    # 아까 성공했던 "evasion attacks"를 다시 넣어 품질 차이를 느껴보세요!
    recommend_v2("evasion attacks")