import torch
from torch_geometric.data import HeteroData
import os, json
from collections import defaultdict

# --- 1. 경로 설정 ---
project_root = r"C:\Users\Arachne\OneDrive\Desktop\arxiv-conversational-recommender-main"
SUBDATASET_DIR = os.path.join(project_root, 'subdataset')
OUTPUT_DIR = os.path.join(project_root, 'output')

MASTER_JSON_FILE = os.path.join(SUBDATASET_DIR, 'arxiv_master_final.json')
FINAL_HETERO_FILE = os.path.join(SUBDATASET_DIR, 'build_hetero_graph_v2.pt')
KNOWLEDGE_META_FILE = os.path.join(OUTPUT_DIR, 'knowledge_meta.json')
SAMPLE_PT_PATH = os.path.join(SUBDATASET_DIR, 'ogbn_arxiv_16k_ffs_sample.pt')

def build_v4_from_v3_logic():
    print("🚀 [1/4] v3 방식의 인덱스 재배치 시작...")
    
    with open(MASTER_JSON_FILE, 'r', encoding='utf-8') as f:
        master_data = json.load(f)

    # 🌟 v3의 핵심: JSON 순서대로 0~15999 로컬 번호 부여
    node_to_local = {int(item['node_idx']): i for i, item in enumerate(master_data)}
    target_node_indices = [int(item['node_idx']) for item in master_data]

    # 피처 로드 및 정렬 (v3와 동일)
    sample_pt = torch.load(SAMPLE_PT_PATH, weights_only=False)
    source_idx_map = {
        (int(idx.item()) if hasattr(idx, 'item') else int(idx)): i 
        for i, idx in enumerate(sample_pt['indices'])
    }
    
    reordered_features = torch.stack([sample_pt['features'][source_idx_map[rid]] for rid in target_node_indices])
    paper_y = sample_pt['labels'].squeeze()

    # --- 2. 지식 데이터 및 단어장(Vocab) 생성 ---
    domain_list, task_list, method_list = [], [], []
    paper_knowledge = {} 
    for i, item in enumerate(master_data):
        k = item.get('knowledge', {})
        d, t, m = k.get('domain', []), k.get('task', []), k.get('method', [])
        domain_list.extend(d); task_list.extend(t); method_list.extend(m)
        paper_knowledge[i] = {'d': d, 't': t, 'm': m}

    def build_vocab(lst):
        unique = sorted(list(set([str(x).strip() for x in lst if x])))
        return {name: i + 1 for i, name in enumerate(unique)} # 0은 패딩

    d_map, t_map, m_map = build_vocab(domain_list), build_vocab(task_list), build_vocab(method_list)

    # --- 3. 에지 번역 (v3 스타일) ---
    print("🔗 [2/4] 모든 에지를 로컬 인덱스(Dense)로 변환 중...")
    data = HeteroData()
    
    # (1) Paper-Paper (Cites)
    raw_pp = torch.load(os.path.join(SUBDATASET_DIR, 'paper_paper_edges.pt'), weights_only=False)
    new_pp_src, new_pp_dst = [], []
    for s, d in zip(raw_pp[0].tolist(), raw_pp[1].tolist()):
        if s in node_to_local and d in node_to_local:
            new_pp_src.append(node_to_local[s])
            new_pp_dst.append(node_to_local[d])
    data['paper', 'cites', 'paper'].edge_index = torch.tensor([new_pp_src, new_pp_dst], dtype=torch.long)

    # (2) Author-Paper (Writes)
    raw_ap = torch.load(os.path.join(SUBDATASET_DIR, 'author_paper_edges.pt'), weights_only=False)
    new_ap_author, new_ap_paper = [], []
    for a, p in zip(raw_ap[0].tolist(), raw_ap[1].tolist()):
        if p in node_to_local:
            new_ap_author.append(int(a))
            new_ap_paper.append(node_to_local[p])
    data['author', 'writes', 'paper'].edge_index = torch.tensor([new_ap_author, new_ap_paper], dtype=torch.long)

    # --- 4. 🌟 v4의 핵심: 지식 속성(edge_attr) 주입 ---
    print("✍️ [3/4] 6차원 지식 속성 [D1, D2, T1, T2, M1, M2] 생성 중...")
    paper_indices, topic_indices, edge_attr_6 = [], [], []

    for i in range(16000):
        info = paper_knowledge.get(i, {'d': [], 't': [], 'm': []})
        def get_top_two(items, mapping):
            return [mapping[items[j]] if j < len(items) and items[j] in mapping else 0 for j in range(2)]
        
        edge_attr_6.append(get_top_two(info['d'], d_map) + get_top_two(info['t'], t_map) + get_top_two(info['m'], m_map))
        paper_indices.append(i)
        topic_indices.append(int(paper_y[i].item()))

    data['paper', 'has_topic', 'topic'].edge_index = torch.tensor([paper_indices, topic_indices], dtype=torch.long)
    data['paper', 'has_topic', 'topic'].edge_attr = torch.tensor(edge_attr_6, dtype=torch.long)

    # 최종 노드 설정
    data['paper'].x = reordered_features
    data['paper'].num_nodes = 16000
    data['author'].num_nodes = int(data['author', 'writes', 'paper'].edge_index[0].max().item() + 1)
    data['topic'].num_nodes = 40

    torch.save(data, FINAL_HETERO_FILE)
    with open(KNOWLEDGE_META_FILE, 'w', encoding='utf-8') as f:
        json.dump({'domains': d_map, 'tasks': t_map, 'methods': m_map}, f, ensure_ascii=False, indent=4)
    
    print(f"\n✅ [V4 완성] v3의 인덱스 체계 + v2의 지식 속성 통합 완료!")

if __name__ == "__main__":
    build_v4_from_v3_logic()