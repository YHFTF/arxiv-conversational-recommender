import torch
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
import os
import json
from collections import defaultdict

# --- 1. 경로 설정 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SUBDATASET_DIR = os.path.join(project_root, 'subdataset')
OUTPUT_DIR = os.path.join(project_root, 'output')

# 입력 파일들
PAPER_PAPER_EDGE_FILE = os.path.join(SUBDATASET_DIR, 'sub_edge_index.pt')
AUTHOR_PAPER_EDGE_FILE = os.path.join(SUBDATASET_DIR, 'author_paper_edges.pt')
SAMPLE_DATA_FILE = os.path.join(SUBDATASET_DIR, 'ogbn_arxiv_16k_ffs_sample.pt')

# 출력 파일
FINAL_HETERO_FILE = os.path.join(SUBDATASET_DIR, 'hetero_graph_v3.pt')

def build_integrated_hetero_graph():
    data = HeteroData()

    print("[System] 1. 기본 관계(Edge) 데이터 로드 중...")
    # 논문-논문 인용 (cites)
    pp_edge = torch.load(PAPER_PAPER_EDGE_FILE)
    # 저자-논문 집필 (writes)
    ap_edge = torch.load(AUTHOR_PAPER_EDGE_FILE)
    # 논문-주제 라벨 (has_topic)
    sample_data = torch.load(SAMPLE_DATA_FILE)
    labels = sample_data['labels'].squeeze()
    
    # --- 2. 공저자(Co-authorship) 관계 추출 ---
    print("[System] 2. 공저자(Collaborates) 관계 추출 중...")
    paper_to_authors = defaultdict(list)
    for i in range(ap_edge.size(1)):
        auth_idx = ap_edge[0, i].item()
        paper_idx = ap_edge[1, i].item()
        paper_to_authors[paper_idx].append(auth_idx)
    
    coauthor_edges = []
    for paper_idx, authors in paper_to_authors.items():
        if len(authors) > 1:
            for i in range(len(authors)):
                for j in range(i + 1, len(authors)):
                    # 양방향 에지 추가 (무향 그래프 특성)
                    coauthor_edges.append([authors[i], authors[j]])
                    coauthor_edges.append([authors[j], authors[i]])
    
    if coauthor_edges:
        aa_edge = torch.tensor(coauthor_edges, dtype=torch.long).t().contiguous()
    else:
        aa_edge = torch.empty((2, 0), dtype=torch.long)

    # --- 3. 논문-주제(Topic) 관계 구성 ---
    print("[System] 3. 논문-주제(Has_Topic) 관계 구성 중...")
    paper_indices = torch.arange(len(labels))
    pt_edge = torch.stack([paper_indices, labels], dim=0)

    # --- 4. HeteroData 객체에 데이터 주입 ---
    print("[System] 4. HeteroData 객체 조립 중...")
    
    # 에지 인덱스 설정
    data['paper', 'cites', 'paper'].edge_index = pp_edge
    data['author', 'writes', 'paper'].edge_index = ap_edge
    data['paper', 'has_topic', 'topic'].edge_index = pt_edge
    data['author', 'collaborates', 'author'].edge_index = aa_edge

    # 노드 개수 명시
    data['paper'].num_nodes = 16000
    data['author'].num_nodes = ap_edge[0].max().item() + 1
    data['topic'].num_nodes = 40 # ogbn-arxiv 표준 카테고리 수

    # --- 5. 노드 피처(Features) 설정 (선택 사항) ---
    # FFS 샘플링 시 저장된 128차원 기본 피처가 있다면 로드
    if 'features' in sample_data:
        data['paper'].x = sample_data['features']
    
    # --- 6. 역방향 에지 생성 (Undirected) ---
    # 모델이 Paper -> Author 방향으로도 메시지를 전달받을 수 있게 함
    print("[System] 5. 역방향 에지 생성 및 데이터 최적화...")
    data = T.ToUndirected()(data)

    # --- 7. 최종 결과 저장 ---
    torch.save(data, FINAL_HETERO_FILE)
    
    print("\n" + "="*50)
    print(f"✅ [최종 결과 리포트]")
    print(f" - 저장된 파일: {FINAL_HETERO_FILE}")
    print(f" - 논문 노드: {data['paper'].num_nodes:,}개")
    print(f" - 저자 노드: {data['author'].num_nodes:,}개")
    print(f" - 주제 노드: {data['topic'].num_nodes:,}개")
    print("-" * 50)
    print(f" - 인용 관계 (cites): {data['paper', 'cites', 'paper'].edge_index.size(1):,}개")
    print(f" - 저술 관계 (writes): {data['author', 'writes', 'paper'].edge_index.size(1):,}개")
    print(f" - 소속 관계 (has_topic): {data['paper', 'has_topic', 'topic'].edge_index.size(1):,}개")
    print(f" - 협업 관계 (collaborates): {data['author', 'collaborates', 'author'].edge_index.size(1):,}개")
    print("="*50)

if __name__ == "__main__":
    build_integrated_hetero_graph()