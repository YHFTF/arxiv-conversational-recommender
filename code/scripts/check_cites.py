import torch
import json
import os
import random

# 경로 설정
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MASTER_DATA_FILE = os.path.join(project_root, 'subdataset', 'arxiv_master_final.json')
EDGE_FILE = os.path.join(project_root, 'subdataset', 'paper_paper_edges.pt')

def verify_citations(num_samples=5):
    print("🔍 [검증] 인용 관계 샘플 테스트 시작...")
    
    # 데이터 로드
    with open(MASTER_DATA_FILE, 'r', encoding='utf-8') as f:
        master_data = json.load(f)
    
    # 인덱스를 키로 하는 딕셔너리 생성 (빠른 조회를 위함)
    idx_to_paper = {item['node_idx']: item for item in master_data}
    
    edge_index = torch.load(EDGE_FILE)
    total_edges = edge_index.size(1)

    # 무작위 샘플링
    sample_indices = random.sample(range(total_edges), num_samples)

    print("-" * 50)
    for i, idx in enumerate(sample_indices):
        src_idx = edge_index[0, idx].item() # 인용하는 논문
        dst_idx = edge_index[1, idx].item() # 인용되는 논문

        src_paper = idx_to_paper.get(src_idx, {"title": "Unknown"})
        dst_paper = idx_to_paper.get(dst_idx, {"title": "Unknown"})

        print(f"샘플 {i+1}:")
        print(f"  📄 [인용 주체] (idx: {src_idx}) {src_paper['title']}")
        print(f"  ➡️ [인용 대상] (idx: {dst_idx}) {dst_paper['title']}")
        
        # 간단한 논리적 체크: 두 논문의 키워드가 유사한지 눈으로 확인
        src_kg = set(src_paper.get('knowledge', {}).get('domain', []))
        dst_kg = set(dst_paper.get('knowledge', {}).get('domain', []))
        common = src_kg.intersection(dst_kg)
        
        if common:
            print(f"  ✅ 공통 분야 발견: {common}")
        print("-" * 50)

if __name__ == "__main__":
    verify_citations()