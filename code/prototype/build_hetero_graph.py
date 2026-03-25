import json
import os
import torch
from torch_geometric.data import HeteroData
from collections import defaultdict

# --- 경로 설정 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SUBDATASET_DIR = os.path.join(project_root, 'subdataset')

# 입력 소스 파일들
MASTER_DATA_FILE = os.path.join(SUBDATASET_DIR, 'arxiv_master_final.json')
PP_EDGE_FILE = os.path.join(SUBDATASET_DIR, 'paper_paper_edges.pt')  # 인용 관계
AP_EDGE_FILE = os.path.join(SUBDATASET_DIR, 'author_paper_edges.pt') # 저술 관계
SAMPLE_DATA_FILE = os.path.join(SUBDATASET_DIR, 'ogbn_arxiv_16k_ffs_sample.pt') # 피처/라벨 소스

# 최종 결과 파일 이름 (요청하신 대로 명명)
FINAL_GRAPH_FILE = os.path.join(SUBDATASET_DIR, 'build_hetero_graph.pt')

def assemble_star_graph():
    print("📦 [1/3] 개별 에지 및 피처 데이터 로드 중...")
    
    # 1. 물리적 파일 로드
    if not all(os.path.exists(f) for f in [PP_EDGE_FILE, AP_EDGE_FILE, SAMPLE_DATA_FILE]):
        print("❌ 에러: 필요한 .pt 에지 파일이 누락되었습니다. 에지 추출 코드를 먼저 실행하세요.")
        return

    pp_edge = torch.load(PP_EDGE_FILE)
    ap_edge = torch.load(AP_EDGE_FILE)
    sample_data = torch.load(SAMPLE_DATA_FILE)
    
    data = HeteroData()

    # --- 2. 노드 데이터 정의 및 주입 ---
    print("🛠️ [2/3] 노드 피처 및 기본 관계(Cites, Writes) 조립 중...")
    
    # Paper 노드: 16,000개 고정
    data['paper'].num_nodes = 16000
    data['paper'].x = sample_data['features'] # 128차원 임베딩
    data['paper'].y = sample_data['labels'].squeeze() # 40개 카테고리 라벨

    # Author 노드: 저술 에지에 등장하는 최대 인덱스 기준
    num_authors = ap_edge[0].max().item() + 1
    data['author'].num_nodes = num_authors

    # Topic 노드: 40개 카테고리 허브
    data['topic'].num_nodes = 40

    # 기본 에지 주입
    # [인용] Paper -> Paper
    data['paper', 'cites', 'paper'].edge_index = pp_edge
    
    # [저술] Author -> Paper
    data['author', 'writes', 'paper'].edge_index = ap_edge

    # --- 3. 스타형 구조 확장을 위한 추가 관계 생성 ---
    print("🤝 [3/3] 공저자(Collaborates) 및 토픽(Has_Topic) 관계 생성 중...")
    
    # 3-1. 공저자 관계 (같은 논문을 쓴 저자들끼리 연결)
    paper_to_authors = defaultdict(list)
    for i in range(ap_edge.size(1)):
        a_idx, p_idx = ap_edge[0, i].item(), ap_edge[1, i].item()
        paper_to_authors[p_idx].append(a_idx)
    
    coauthor_edges = []
    for authors in paper_to_authors.values():
        if len(authors) > 1:
            for i in range(len(authors)):
                for j in range(i + 1, len(authors)):
                    # 양방향(무향) 에지로 구성
                    coauthor_edges.append([authors[i], authors[j]])
                    coauthor_edges.append([authors[j], authors[i]])
    
    if coauthor_edges:
        data['author', 'collaborates', 'author'].edge_index = \
            torch.tensor(coauthor_edges, dtype=torch.long).t().contiguous()

    # 3-2. 토픽 관계 (논문들을 40개 주제 허브에 연결)
    # 0~15999번 논문 각각을 자신의 라벨(0~39) 번호에 연결
    paper_indices = torch.arange(data['paper'].num_nodes)
    data['paper', 'has_topic', 'topic'].edge_index = \
        torch.stack([paper_indices, data['paper'].y], dim=0)

    # 최종 결과 저장
    torch.save(data, FINAL_GRAPH_FILE)
    
    print("\n" + "="*50)
    print(f"✅ [이종 그래프 구축 완료]")
    print(f" - 최종 파일: {FINAL_GRAPH_FILE}")
    print("-" * 50)
    print(f" 📊 노드 통계:")
    print(f"  • Paper  : {data['paper'].num_nodes:,}개")
    print(f"  • Author : {data['author'].num_nodes:,}개")
    print(f"  • Topic  : {data['topic'].num_nodes:,}개")
    print(f" 🔗 에지 통계:")
    print(f"  • Cites (P-P)   : {data['paper', 'cites', 'paper'].edge_index.size(1):,}개")
    print(f"  • Writes (A-P)  : {data['author', 'writes', 'paper'].edge_index.size(1):,}개")
    print(f"  • Collab (A-A)  : {data['author', 'collaborates', 'author'].edge_index.size(1):,}개")
    print(f"  • Topic (P-T)   : {data['paper', 'has_topic', 'topic'].edge_index.size(1):,}개")
    print("="*50)

if __name__ == "__main__":
    assemble_star_graph()