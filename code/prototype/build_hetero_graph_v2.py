import torch
from torch_geometric.data import HeteroData
import os
import json
from collections import defaultdict

# --- 1. 경로 설정 ---
project_root = r"C:\Users\Arachne\OneDrive\Desktop\arxiv-conversational-recommender-main"
SUBDATASET_DIR = os.path.join(project_root, 'subdataset')
OUTPUT_DIR = os.path.join(project_root, 'output')

MASTER_JSON_FILE = os.path.join(SUBDATASET_DIR, 'arxiv_master_final.json')
FINAL_HETERO_FILE = os.path.join(SUBDATASET_DIR, 'build_hetero_graph_v2.pt')
KNOWLEDGE_META_FILE = os.path.join(OUTPUT_DIR, 'knowledge_meta.json')

def build_hetero_graph_v2_extended():
    print("🚀 [Step 1] 데이터 및 리소스 로드 중 (6-Column 확장 모드)...")
    
    if not os.path.exists(MASTER_JSON_FILE):
        print(f"❌ 에러: {MASTER_JSON_FILE} 파일이 없습니다.")
        return

    with open(MASTER_JSON_FILE, 'r', encoding='utf-8') as f:
        master_data = json.load(f)

    sample_pt_path = os.path.join(SUBDATASET_DIR, 'ogbn_arxiv_16k_ffs_sample.pt')
    sample_data = torch.load(sample_pt_path, weights_only=False)
    labels = sample_data['labels'].squeeze()

    # --- 2. 지식 데이터 인덱싱 ---
    domain_list, task_list, method_list = [], [], []
    paper_knowledge = {}

    for item in master_data:
        p_idx = item['node_idx']
        k = item.get('knowledge', {})
        d, t, m = k.get('domain', []), k.get('task', []), k.get('method', [])
        
        domain_list.extend(d)
        task_list.extend(t)
        method_list.extend(m)
        paper_knowledge[p_idx] = {'d': d, 't': t, 'm': m}

    def build_fixed_vocab(lst):
        unique_names = sorted(list(set([str(x).strip() for x in lst if x])))
        return {name: i for i, name in enumerate(unique_names)}

    d_map = build_fixed_vocab(domain_list)
    t_map = build_fixed_vocab(task_list)
    m_map = build_fixed_vocab(method_list)

    # --- 3. 그래프 조립 및 6개 속성 주입 ---
    data = HeteroData()
    print("🔗 [Step 3] 6개 지식 속성 [D1, D2, T1, T2, M1, M2] 주입 중...")

    pp_edge = torch.load(os.path.join(SUBDATASET_DIR, 'paper_paper_edges.pt'), weights_only=False)
    ap_edge = torch.load(os.path.join(SUBDATASET_DIR, 'author_paper_edges.pt'), weights_only=False)
    
    data['paper', 'cites', 'paper'].edge_index = pp_edge
    data['author', 'writes', 'paper'].edge_index = ap_edge

    paper_indices, topic_indices, edge_attr_6 = [], [], []

    for p_idx in range(16000):
        t_idx = int(labels[p_idx].item())
        info = paper_knowledge.get(p_idx, {'d': [], 't': [], 'm': []})
        
        # 각 카테고리별로 최대 2개씩 추출하는 로직 (모자라면 -1 패딩)
        def get_top_two(items, mapping):
            res = []
            for i in range(2):
                if i < len(items) and items[i] in mapping:
                    res.append(mapping[items[i]])
                else:
                    res.append(-1)
            return res

        d_pair = get_top_two(info['d'], d_map)
        t_pair = get_top_two(info['t'], t_map)
        m_pair = get_top_two(info['m'], m_map)
        
        paper_indices.append(p_idx)
        topic_indices.append(t_idx)
        # 6차원 리스트 생성: [D1, D2, T1, T2, M1, M2]
        edge_attr_6.append(d_pair + t_pair + m_pair)

    data['paper', 'has_topic', 'topic'].edge_index = torch.stack([
        torch.tensor(paper_indices), torch.tensor(topic_indices)
    ], dim=0)
    # 엣지 속성을 (16000, 6) 형태로 저장
    data['paper', 'has_topic', 'topic'].edge_attr = torch.tensor(edge_attr_6, dtype=torch.long)

    # 기본 노드 정보 설정
    data['paper'].num_nodes = 16000
    data['paper'].x = sample_data['features']
    data['author'].num_nodes = int(ap_edge[0].max().item() + 1)
    data['topic'].num_nodes = 40

    # --- 4. 공저자 관계 (기존 로직 유지) ---
    paper_to_authors = defaultdict(list)
    for i in range(ap_edge.size(1)):
        paper_to_authors[ap_edge[1, i].item()].append(ap_edge[0, i].item())
    
    coauthor_edges = []
    for authors in paper_to_authors.values():
        if len(authors) > 1:
            for i in range(len(authors)):
                for j in range(i + 1, len(authors)):
                    coauthor_edges.append([authors[i], authors[j]])
                    coauthor_edges.append([authors[j], authors[i]])
    
    if coauthor_edges:
        data['author', 'collaborates', 'author'].edge_index = \
            torch.tensor(coauthor_edges, dtype=torch.long).t().contiguous()
    
    # --- 5. 저장 및 메타파일 생성 ---
    torch.save(data, FINAL_HETERO_FILE)
    
    knowledge_meta = {'domains': d_map, 'tasks': t_map, 'methods': m_map}
    with open(KNOWLEDGE_META_FILE, 'w', encoding='utf-8') as f:
        json.dump(knowledge_meta, f, ensure_ascii=False, indent=4)

    print("\n" + "="*50)
    print(f"✅ [이종 그래프 구축 완료]")
    print(f" - 최종 파일: {FINAL_HETERO_FILE}")
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
    print("\n" + "="*50)
    print(f"✅ [V2-Extended 그래프 구축 완료]")
    print(f" - Edge_Attr Shape: {data['paper', 'has_topic', 'topic'].edge_attr.shape}")
    print(f" - 구성: [D1, D2, T1, T2, M1, M2]")
    print("="*50)

if __name__ == "__main__":
    build_hetero_graph_v2_extended()