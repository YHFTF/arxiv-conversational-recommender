import torch
from torch_geometric.data import HeteroData
import os
from collections import defaultdict

# --- 경로 설정 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SUBDATASET_DIR = os.path.join(project_root, 'subdataset')
FINAL_HETERO_FILE = os.path.join(SUBDATASET_DIR, 'hetero_graph.pt')

def build_star_hetero_graph():
    data = HeteroData()

    print("[System] 1. 기본 물리 데이터 로드 중...")
    pp_edge = torch.load(os.path.join(SUBDATASET_DIR, 'sub_edge_index.pt')) # 인용
    ap_edge = torch.load(os.path.join(SUBDATASET_DIR, 'author_paper_edges.pt')) # 저술
    sample_data = torch.load(os.path.join(SUBDATASET_DIR, 'ogbn_arxiv_16k_ffs_sample.pt')) # 원본
    labels = sample_data['labels'].squeeze() # 각 논문의 카테고리 번호

    # --- 2. 공저자(Collaborates) 관계 추출 ---
    print("[System] 2. 공저자 관계 분석 중 (Clique 추출)...")
    paper_to_authors = defaultdict(list)
    for i in range(ap_edge.size(1)):
        auth_idx = ap_edge[0, i].item()
        paper_idx = ap_edge[1, i].item()
        paper_to_authors[paper_idx].append(auth_idx)
    
    coauthor_edges = []
    for authors in paper_to_authors.values():
        if len(authors) > 1:
            for i in range(len(authors)):
                for j in range(i + 1, len(authors)):
                    # 저자 간의 수평적 연결 (A-B, B-A)
                    coauthor_edges.append([authors[i], authors[j]])
                    coauthor_edges.append([authors[j], authors[i]])
    
    # --- 3. 데이터 주입 (관계망 구축) ---
    print("[System] 3. 주제(Topic) 허브 중심의 관계망 조립 중...")

    # [인용] Paper A -> Paper B (단방향)
    data['paper', 'cites', 'paper'].edge_index = pp_edge
    
    # [저술] Author A -> Paper B (단방향)
    data['author', 'writes', 'paper'].edge_index = ap_edge
    
    # [협업] Author A <-> Author B (양방향/무향)
    if coauthor_edges:
        data['author', 'collaborates', 'author'].edge_index = torch.tensor(coauthor_edges, dtype=torch.long).t().contiguous()
    
    # [소속] Paper A -> Topic T (Hub 연결)
    # 여기서 Topic 노드는 0~39번까지 총 40개의 독립 노드입니다.
    paper_indices = torch.arange(len(labels))
    data['paper', 'has_topic', 'topic'].edge_index = torch.stack([paper_indices, labels], dim=0)

    # --- 4. 노드 명세 정의 ---
    data['paper'].num_nodes = 16000
    data['author'].num_nodes = ap_edge[0].max().item() + 1
    data['topic'].num_nodes = 40 # 40개의 카테고리 노드가 고정된 인덱스로 존재함
    
    # 논문 피처 주입
    if 'features' in sample_data:
        data['paper'].x = sample_data['features']

    # --- 5. 결과 저장 ---
    torch.save(data, FINAL_HETERO_FILE)
    
    print("\n" + "="*50)
    print(f"✅ [스타형 이종 그래프 구축 완료]")
    print(f" - 파일 경로: {FINAL_HETERO_FILE}")
    print("-" * 50)
    print(f" 1. 인용 (Paper -> Paper): {data['paper', 'cites', 'paper'].edge_index.size(1):,}개")
    print(f" 2. 저술 (Author -> Paper): {data['author', 'writes', 'paper'].edge_index.size(1):,}개")
    print(f" 3. 소속 (Paper -> Topic Hub): {data['paper', 'has_topic', 'topic'].edge_index.size(1):,}개")
    print(f" 4. 협업 (Author <-> Author): {data['author', 'collaborates', 'author'].edge_index.size(1):,}개")
    print("="*50)

if __name__ == "__main__":
    build_star_hetero_graph()