import torch
import json
import os
import sys

# 경로 설정
project_root = r"C:\Users\Arachne\OneDrive\Desktop\arxiv-conversational-recommender-main"
sys.path.append(os.path.join(project_root, 'code', 'model'))
from LGCmodel import ArxivLightGCN

# 파일 경로
GRAPH_PATH = os.path.join(project_root, 'subdataset', 'build_hetero_graph.pt')
MODEL_PATH = os.path.join(project_root, 'output', 'lightgcn_v3_trained.pt')
MASTER_FILE = os.path.join(project_root, 'subdataset', 'arxiv_master_final.json')

def recommend(target_title, top_k=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 데이터 및 모델 로드
    data = torch.load(GRAPH_PATH, weights_only=False).to(device)
    with open(MASTER_FILE, 'r', encoding='utf-8') as f:
        master_list = json.load(f)
    
    model = ArxivLightGCN(data).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()

    # 2. 모든 논문의 최종 임베딩 추출
    with torch.no_grad():
        unified_edges = model._build_unified_edge_index(data)
        all_embeddings = model(unified_edges)
        paper_embeddings = model.get_paper_embeddings(all_embeddings) # [16000, 128]

    # 3. 입력한 제목에 해당하는 인덱스 찾기 (수정본)
    target_idx = -1
    matched_titles = []
    
    for i, item in enumerate(master_list):
        if target_title.lower() in item['title'].lower():
            matched_titles.append((i, item['title']))
    
    if not matched_titles:
        print(f"❌ '{target_title}'을(를) 포함하는 논문을 찾을 수 없습니다.")
        # 팁: 아까 뽑은 샘플 제목 중 하나를 넣어보세요!
        return
    else:
        # 여러 개가 검색되면 가장 첫 번째 것을 타겟으로 잡습니다.
        target_idx, full_title = matched_titles[0]
        print(f"\n🎯 검색된 논문 ({len(matched_titles)}건 중 선택): {full_title} (Idx: {target_idx})")

    # 4. 코사인 유사도 계산
    target_vec = paper_embeddings[target_idx].unsqueeze(0) # [1, 128]
    # Cosine Similarity = (A·B) / (||A||*||B||)
    sim_scores = torch.cosine_similarity(target_vec, paper_embeddings)
    
    # 자기 자신 제외하고 상위 K개 추출
    values, indices = torch.topk(sim_scores, k=top_k + 1)
    
    print(f"\n📚 '{target_title}'와(과) 유사한 추천 논문 리스트:")
    print("-" * 100)
    
    for i in range(1, len(indices)): # 0번은 자기 자신
        idx = indices[i].item()
        score = values[i].item()
        rec_item = master_list[idx]
        
        print(f"[{i}] 유사도: {score:.4f} | 제목: {rec_item['title']}")
        print(f"    └ 도메인: {rec_item['knowledge']['domain']}")
        print(f"    └ 주요기법: {rec_item['knowledge']['method']}")
        print("-" * 100)

if __name__ == "__main__":
    # 어제 테스트했던 Booking.com 논문이나 관심 있는 키워드를 입력해보세요!
    search_query = "evasion attacks" 
    recommend(search_query)