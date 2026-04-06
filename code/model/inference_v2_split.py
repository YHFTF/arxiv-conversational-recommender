import torch
import json
import os
import sys

# 1. 경로 설정 (민혁 연구원님이 지정해주신 경로 준수)
project_root = r"C:\Users\Arachne\OneDrive\Desktop\arxiv-conversational-recommender-main"
sys.path.append(os.path.join(project_root, 'code', 'model'))
from LGCmodel_v2 import ArxivLightGCNV2

# 파일 경로
GRAPH_PATH = os.path.join(project_root, 'subdataset', 'build_hetero_graph_v2.pt')
MODEL_PATH = os.path.join(project_root, 'output', 'lightgcn_v2_split_trained.pt')
MASTER_FILE = os.path.join(project_root, 'subdataset', 'arxiv_master_final.json')
META_PATH = os.path.join(project_root, 'output', 'knowledge_meta.json')

def recommend_v2_split(target_title, top_k=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. 메타데이터 및 데이터 로드
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 에러: {MODEL_PATH} 가중치 파일이 없습니다. v2_split 학습을 먼저 완료하세요.")
        return

    with open(META_PATH, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    meta_counts = {
        'domains': len(meta['domains']), 
        'tasks': len(meta['tasks']), 
        'methods': len(meta['methods'])
    }

    data = torch.load(GRAPH_PATH, weights_only=False).to(device)
    with open(MASTER_FILE, 'r', encoding='utf-8') as f:
        master_list = json.load(f)
    
    # 3. 모델 초기화 및 가중치 로드
    model = ArxivLightGCNV2(data, meta_counts).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()

    # 4. 임베딩 추출 (인퍼런스 시에는 전체 그래프 구조 활용)
    with torch.no_grad():
        # V2 모델의 통합 그래프 빌드 로직 사용
        unified_edges = model._build_unified_graph(data)
        all_embeddings = model(unified_edges)
        # 논문 노드(0~15999) 임베딩만 가져오기
        paper_embeddings = model.get_paper_embeddings(all_embeddings)

    # 5. 제목 검색
    target_idx = -1
    for i, item in enumerate(master_list):
        if target_title.lower() in item['title'].lower():
            target_idx = i
            print(f"\n🎯 검색된 논문: {item['title']} (Idx: {target_idx})")
            break
    
    if target_idx == -1:
        print(f"❌ '{target_title}' 논문을 찾을 수 없습니다.")
        return

    # 6. 코사인 유사도 기반 추천
    target_vec = paper_embeddings[target_idx].unsqueeze(0)
    sim_scores = torch.cosine_similarity(target_vec, paper_embeddings)
    
    # 자기 자신 제외하고 Top-K
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
    recommend_v2_split("graph neural networks for social recommendation")