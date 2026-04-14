import torch
import json
import os
import sys

# 경로 설정
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(project_root, 'code', 'model'))
from LGCmodel_v2 import ArxivLightGCNV2

# 파일 경로
GRAPH_PATH = os.path.join(project_root, 'subdataset', 'build_hetero_graph_v2.pt')
MODEL_PATH = os.path.join(project_root, 'output', 'lightgcn_v3_bpr_trained.pt')
MASTER_FILE = os.path.join(project_root, 'subdataset', 'arxiv_master_final.json')
META_PATH = os.path.join(project_root, 'output', 'knowledge_meta.json')

def recommend_v3(target_title, top_k=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 메타데이터 및 데이터 로드
    with open(META_PATH, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    meta_counts = {'domains': len(meta['domains']), 'tasks': len(meta['tasks']), 'methods': len(meta['methods'])}

    # weights_only=False for PyG heterodata backward compatibility usually
    data = torch.load(GRAPH_PATH, weights_only=False).to(device)
    with open(MASTER_FILE, 'r', encoding='utf-8') as f:
        master_list = json.load(f)
    
    # 2. V2 모델 초기화 및 BPR 가중치 로드
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
            print(f"\n🎯 입력된 논문: {item['title']} (Idx: {target_idx})")
            break
    
    if target_idx == -1:
        print(f"❌ '{target_title}' 논문을 찾을 수 없습니다. 다른 키워드로 검색해보세요.")
        return

    # 5. 코사인 유사도 기반 추천
    target_vec = paper_embeddings[target_idx].unsqueeze(0)
    sim_scores = torch.cosine_similarity(target_vec, paper_embeddings)
    values, indices = torch.topk(sim_scores, k=top_k + 1)
    
    # 1. 그래프 데이터에서 인용수(In-degree) 계산
    if ('paper', 'cites', 'paper') in data.edge_types:
        edge_index = data['paper', 'cites', 'paper'].edge_index
        cited_papers = edge_index[1]
        
        num_papers = paper_embeddings.size(0)
        citation_counts = torch.zeros(num_papers, dtype=torch.long)
        
        unique_indices, counts = torch.unique(cited_papers, return_counts=True)
        citation_counts[unique_indices] = counts.cpu()
    else:
        citation_counts = torch.zeros(paper_embeddings.size(0), dtype=torch.long)

    # 2. 결과 출력
    print(f"\n📊 [V3 BPR] 모델 추천 리스트:")
    print("=" * 115)
    
    for i in range(1, len(indices)):
        idx = indices[i].item()
        score = values[i].item()
        rec_item = master_list[idx]
        
        graph_cites = citation_counts[idx].item()
        
        print(f"[{i}] 유사도: {score:.4f} | 📈 그래프 인용수: {graph_cites:<4} | 제목: {rec_item['title']}")
        k = rec_item.get('knowledge', {})
        print(f"    - 분야: {k.get('domain', 'N/A')} | 세부Task: {k.get('task', 'N/A')} | 기법: {k.get('method', 'N/A')}")
        print("-" * 115)

if __name__ == "__main__":
    # 커맨드라인 인자가 있으면 그걸 검색어로 사용, 없으면 v2와 동일한 기본값 사용
    query = sys.argv[1] if len(sys.argv) > 1 else "graph neural networks for social recommendation"
    recommend_v3(query)
