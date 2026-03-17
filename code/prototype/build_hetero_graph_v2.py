import torch
from torch_geometric.data import HeteroData
import os
import json
from collections import defaultdict, Counter

# --- 경로 설정 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SUBDATASET_DIR = os.path.join(project_root, 'subdataset')
OUTPUT_DIR = os.path.join(project_root, 'output')

LLM_RESULT_FILE = os.path.join(OUTPUT_DIR, '3llm_extraction_results.json')
NODE_MAP_FILE = os.path.join(OUTPUT_DIR, 'node_remapping.json')
FINAL_HETERO_FILE = os.path.join(SUBDATASET_DIR, 'hetero_graph_v2.pt')

def build_hetero_graph_v2():
    print("[System] 1. 데이터 및 매핑 로드 중...")
    with open(LLM_RESULT_FILE, 'r', encoding='utf-8') as f:
        llm_data = json.load(f)
    with open(NODE_MAP_FILE, 'r', encoding='utf-8') as f:
        node_mapping = json.load(f)
    
    # 40개 표준 카테고리 라벨 로드 (Topic Hub용)
    sample_data = torch.load(os.path.join(SUBDATASET_DIR, 'ogbn_arxiv_16k_ffs_sample.pt'))
    labels = sample_data['labels'].squeeze()

    # --- 2. 도메인/태스크/메소드 전수 조사 ---
    print("[System] 2. 지식 데이터(Domain/Task/Method) 인덱싱 중...")
    domain_list, task_list, method_list = [], [], []
    paper_knowledge = {}

    for item in llm_data:
        if item.get('status') == 'success':
            p_id = str(item['node_idx'])
            if p_id in node_mapping:
                p_idx = node_mapping[p_id]
                d = item.get('domain', [])
                t = item.get('task', [])
                m = item.get('method', [])
                
                domain_list.extend(d)
                task_list.extend(t)
                method_list.extend(m)
                paper_knowledge[p_idx] = {'d': d, 't': t, 'm': m}

    # 고유 사전(Vocabulary) 구축
    def build_vocab(lst):
        unique = sorted(list(set(lst)))
        return {name: i for i, name in enumerate(unique)}, unique

    d_map, d_names = build_vocab(domain_list)
    t_map, t_names = build_vocab(task_list)
    m_map, m_names = build_vocab(method_list)

    print(f" - 추출 완료: Domain({len(d_names)}), Task({len(t_names)}), Method({len(m_names)})")

    # --- 3. 그래프 조립 ---
    data = HeteroData()
    print("[System] 3. 관계망 및 엣지 속성 주입 중...")

    # (1) 기존 엣지 로드
    pp_edge = torch.load(os.path.join(SUBDATASET_DIR, 'sub_edge_index.pt'))
    ap_edge = torch.load(os.path.join(SUBDATASET_DIR, 'author_paper_edges.pt'))
    
    data['paper', 'cites', 'paper'].edge_index = pp_edge
    data['author', 'writes', 'paper'].edge_index = ap_edge

    # (2) [핵심] 소속 엣지 + 지식 속성 (Paper -> Topic)
    paper_indices = []
    topic_indices = []
    edge_attr = [] 

    for p_idx, info in paper_knowledge.items():
        t_idx = labels[p_idx].item()
        
        # 각 리스트의 첫 번째 값만 사용하는 단순화 버전 (추후 멀티 엣지로 확장 가능)
        d_idx = d_map[info['d'][0]] if info['d'] else -1
        tk_idx = t_map[info['t'][0]] if info['t'] else -1
        mt_idx = m_map[info['m'][0]] if info['m'] else -1
        
        paper_indices.append(p_idx)
        topic_indices.append(t_idx)
        edge_attr.append([d_idx, tk_idx, mt_idx])

    data['paper', 'has_topic', 'topic'].edge_index = torch.stack([
        torch.tensor(paper_indices), torch.tensor(topic_indices)
    ], dim=0)
    data['paper', 'has_topic', 'topic'].edge_attr = torch.tensor(edge_attr, dtype=torch.long)

    # (3) 노드 정의
    data['paper'].num_nodes = 16000
    data['author'].num_nodes = ap_edge[0].max().item() + 1
    data['topic'].num_nodes = 40
    data['paper'].x = sample_data['features']

    # --- 4. 공저자(Collaborates) 관계 추출 ---
    print("[System] 4. 공저자 관계 분석 중 (Clique 추출)...")
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
                    
    if coauthor_edges:
        data['author', 'collaborates', 'author'].edge_index = torch.tensor(coauthor_edges, dtype=torch.long).t().contiguous()
    
    # --- 5. 저장 및 최종 리포트 출력 ---
    torch.save(data, FINAL_HETERO_FILE)
    
    knowledge_meta = {'domains': d_names, 'tasks': t_names, 'methods': m_names}
    with open(os.path.join(OUTPUT_DIR, 'knowledge_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(knowledge_meta, f, ensure_ascii=False, indent=4)

    # 엣지 개수 계산
    num_cites = data['paper', 'cites', 'paper'].edge_index.size(1)
    num_writes = data['author', 'writes', 'paper'].edge_index.size(1)
    num_has_topic = data['paper', 'has_topic', 'topic'].edge_index.size(1)
    num_collab = data['author', 'collaborates', 'author'].edge_index.size(1) if coauthor_edges else 0

    print("\n" + "="*50)
    print(f"✅ [V2 지식 주입 그래프 구축 완료]")
    print(f" - 저장 경로: {FINAL_HETERO_FILE}")
    print("-" * 50)
    print(f" 1. 인용 관계 (Paper -> Paper): {num_cites:,}개")
    print(f" 2. 저술 관계 (Author -> Paper): {num_writes:,}개")
    print(f" 3. 소속 관계 (Paper -> Topic): {num_has_topic:,}개 (지식 속성 포함)")
    print(f" 4. 협업 관계 (Author <-> Author): {num_collab:,}개")
    print("="*50)

if __name__ == "__main__":
    build_hetero_graph_v2()