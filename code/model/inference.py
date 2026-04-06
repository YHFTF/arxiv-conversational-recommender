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
MODEL_PATH = os.path.join(project_root, 'output', 'lightgcn_trained.pt')
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
    
 # 1. 그래프 데이터에서 인용수(In-degree) 계산
    # 'cites' 에지의 [0]은 인용하는 논문, [1]은 인용받는 논문입니다.
    if ('paper', 'cites', 'paper') in data.edge_types:
        edge_index = data['paper', 'cites', 'paper'].edge_index
        cited_papers = edge_index[1]  # 인용을 받은(target) 논문 인덱스들
        
        # 전체 논문 수만큼 빈 카운터 생성 (기본값 0)
        num_papers = paper_embeddings.size(0)
        citation_counts = torch.zeros(num_papers, dtype=torch.long)
        
        # 인용받은 횟수 합산 (CPU로 옮겨서 처리하는 게 안정적입니다)
        unique_indices, counts = torch.unique(cited_papers, return_counts=True)
        citation_counts[unique_indices] = counts.cpu()
    else:
        citation_counts = torch.zeros(paper_embeddings.size(0), dtype=torch.long)

    # 2. 결과 출력 부분 수정
    print(f"\n📊 [Split-Trained] '{target_title}' 기반 추천 리스트:")
    print("=" * 115)
    
    for i in range(1, len(indices)):
        idx = indices[i].item()
        score = values[i].item()
        rec_item = master_list[idx]
        
        # .pt 파일에서 추출한 실제 인용수
        graph_cites = citation_counts[idx].item()
        
        # 출력 포맷 (유사도 | 인용수 | 제목)
        print(f"[{i}] 유사도: {score:.4f} | 📈 그래프 인용수: {graph_cites:<4} | 제목: {rec_item['title']}")
        k = rec_item.get('knowledge', {})
        print(f"    - 분야: {k.get('domain', 'N/A')} | 기법: {k.get('method', 'N/A')}")
        print("-" * 115)

if __name__ == "__main__":
    recommend("graph neural networks for social recommendation")