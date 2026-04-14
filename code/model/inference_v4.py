import torch
import json
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(project_root, 'code', 'model'))
from LGCmodel_v4 import ArxivLightGCNV4

GRAPH_PATH = os.path.join(project_root, 'subdataset', 'build_hetero_graph_v2.pt')
MODEL_PATH = os.path.join(project_root, 'output', 'lightgcn_v4_knowledge_bpr.pt')
MASTER_FILE = os.path.join(project_root, 'subdataset', 'arxiv_master_final.json')
META_PATH = os.path.join(project_root, 'output', 'knowledge_meta.json')

def recommend_v4(target_title, top_k=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(META_PATH, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    meta_counts = {'domains': len(meta['domains']), 'tasks': len(meta['tasks']), 'methods': len(meta['methods'])}

    data = torch.load(GRAPH_PATH, weights_only=False).to(device)
    with open(MASTER_FILE, 'r', encoding='utf-8') as f:
        master_list = json.load(f)

    print("🧠 지식(Knowledge) 텐서 구축 중...")
    num_papers = data['paper'].num_nodes
    paper_knowledge_ids = torch.zeros((num_papers, 3), dtype=torch.long)
    for i, item in enumerate(master_list):
        if i >= num_papers: break
        k_dict = item.get('knowledge', {})
        d_val = k_dict.get('domain', None)
        if isinstance(d_val, list) and len(d_val) > 0: d_val = d_val[0]
        t_val = k_dict.get('task', None)
        if isinstance(t_val, list) and len(t_val) > 0: t_val = t_val[0]
        m_val = k_dict.get('method', None)
        if isinstance(m_val, list) and len(m_val) > 0: m_val = m_val[0]
        
        d_id = meta['domains'].get(d_val, 0) if d_val else 0
        t_id = meta['tasks'].get(t_val, 0) if t_val else 0
        m_id = meta['methods'].get(m_val, 0) if m_val else 0
        paper_knowledge_ids[i] = torch.tensor([d_id, t_id, m_id])

    model = ArxivLightGCNV4(data, meta_counts, paper_knowledge_ids).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()

    with torch.no_grad():
        unified_edges = model._build_unified_graph(data)
        all_embeddings = model(unified_edges)
        paper_embeddings = model.get_paper_embeddings(all_embeddings)

    target_idx = -1
    for i, item in enumerate(master_list):
        if target_title.lower() in item['title'].lower():
            target_idx = i
            print(f"\n🎯 입력된 논문: {item['title']} (Idx: {target_idx})")
            break
    
    if target_idx == -1:
        print(f"❌ '{target_title}' 논문을 찾을 수 없습니다. 다른 키워드로 검색해보세요.")
        return

    target_vec = paper_embeddings[target_idx].unsqueeze(0)
    sim_scores = torch.cosine_similarity(target_vec, paper_embeddings)
    values, indices = torch.topk(sim_scores, k=top_k + 1)
    
    if ('paper', 'cites', 'paper') in data.edge_types:
        edge_index = data['paper', 'cites', 'paper'].edge_index
        cited_papers = edge_index[1]
        num_papers = paper_embeddings.size(0)
        citation_counts = torch.zeros(num_papers, dtype=torch.long)
        unique_indices, counts = torch.unique(cited_papers, return_counts=True)
        citation_counts[unique_indices] = counts.cpu()
    else:
        citation_counts = torch.zeros(paper_embeddings.size(0), dtype=torch.long)

    print(f"\n📊 [V4 Knowledge-Aware BPR] 모델 추천 리스트:")
    print("=" * 115)
    
    for i in range(1, len(indices)):
        idx = indices[i].item()
        score = values[i].item()
        rec_item = master_list[idx]
        graph_cites = citation_counts[idx].item()
        
        print(f"[{i}] 유사도: {score:.4f} | 📈 원본 인용수: {graph_cites:<4} | 제목: {rec_item['title']}")
        k = rec_item.get('knowledge', {})
        print(f"    - 분야: {k.get('domain', 'N/A')} | 세부Task: {k.get('task', 'N/A')} | 기법: {k.get('method', 'N/A')}")
        print("-" * 115)

if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "graph neural networks for social recommendation"
    recommend_v4(query)
