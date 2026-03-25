import torch
import os
import sys
import pandas as pd

# 1. 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(project_root)

from LGCmodel import HeteroLightGCN 

def run_paper_recommendation():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 경로 설정
    GRAPH_PATH = os.path.join(project_root, 'subdataset', 'hetero_graph.pt')
    MODEL_PATH = os.path.join(project_root, 'output', 'best_lgc_model_baseline.pth')
    TITLE_TSV = os.path.join(project_root, 'subdataset', 'titleabs.tsv')

    # 데이터 로드
    from torch_geometric.data import HeteroData
    torch.serialization.add_safe_globals([HeteroData])
    data = torch.load(GRAPH_PATH, weights_only=False).to(device)

    # 제목 데이터 로드
    try:
        titles_df = pd.read_csv(TITLE_TSV, sep='\t', names=['paper_id', 'title', 'abstract'])
        paper_titles = titles_df['title'].tolist()
    except:
        paper_titles = [f"Paper ID {i}" for i in range(16000)]

    # 모델 로드
    model = HeteroLightGCN(
        num_authors=data['author'].num_nodes,
        num_topics=data['topic'].num_nodes,
        embedding_dim=64,
        num_layers=3
    ).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    with torch.no_grad():
        out_dict = model(data.x_dict, data.edge_index_dict)
        
        # 🎯 분석할 기준 논문 인덱스 (Bengio가 나왔던 그 논문)
        target_idx = 476
        query_paper_title = paper_titles[target_idx]
        query_emb = out_dict['paper'][target_idx]

        print("\n" + "="*80)
        print(f"🔎 [기준 논문] : {query_paper_title}")
        print("="*80)

        # --- 1. 유사 저자 추천 ---
        author_scores = torch.matmul(out_dict['author'], query_emb)
        auth_values, auth_indices = torch.topk(author_scores, k=5)

        print(f"\n👥 [추천 전문가 TOP 5]")
        for i in range(5):
            print(f" {i+1}위: 저자 ID {auth_indices[i].item():6d} | 점수: {auth_values[i].item():.4f}")

        # --- 2. 유사 논문 추천 (자신 제외) ---
        # 모든 논문 임베딩과 내적 계산
        paper_scores = torch.matmul(out_dict['paper'], query_emb)
        # 자기 자신은 점수가 가장 높으므로 k+1개를 뽑아 첫 번째를 제외
        paper_values, paper_indices = torch.topk(paper_scores, k=6)

        print(f"\n📚 [추천 유사 논문 TOP 5]")
        count = 0
        for i in range(len(paper_indices)):
            idx = paper_indices[i].item()
            if idx == target_idx: continue # 자기 자신 제외
            if count >= 5: break
            
            title = paper_titles[idx] if idx < len(paper_titles) else f"ID {idx}"
            print(f" {count+1}위: {title} (ID: {idx}, 점수: {paper_values[i].item():.4f})")
            count += 1

        print("\n" + "="*80)

if __name__ == "__main__":
    run_paper_recommendation()