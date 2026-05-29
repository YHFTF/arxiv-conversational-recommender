import torch
import sys
import os
import json
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'code', 'model'))

from config import *
from utils import (
    load_data, build_knowledge_ids, get_graph_info,
    build_unified_graph, build_initial_features
)
from run_benchmark import get_benchmark_models

def main():
    parser = argparse.ArgumentParser(description="학습된 모델을 이용한 논문 추천 추론 시스템")
    parser.add_argument('--query', type=str, required=True, help="검색할 논문 제목의 키워드")
    parser.add_argument('--model', type=str, default="LightGCN + Knowledge (Ours)", help="사용할 모델 이름 (기본값: LightGCN + Knowledge (Ours))")
    parser.add_argument('--top_k', type=int, default=5, help="추천받을 논문 개수")
    args = parser.parse_args()

    print(f"[{args.model}] 모델 로드 및 추론 준비 중...")

    # 데이터 로드
    data, meta, meta_counts = load_data()
    num_papers, _, _, _ = get_graph_info(data)
    initial_features = build_initial_features(data)
    paper_knowledge_ids = build_knowledge_ids(data, meta)

    # 모델 목록 가져오기
    all_models = get_benchmark_models(data, meta_counts, initial_features, paper_knowledge_ids)
    
    target_model_info = None
    for m in all_models:
        if m['name'].lower() == args.model.lower():
            target_model_info = m
            break
            
    if not target_model_info:
        print(f"\n[ERROR] '{args.model}' 모델을 찾을 수 없습니다.")
        print("사용 가능한 모델 목록:")
        for m in all_models:
            print(f"  - {m['name']}")
        return

    # 모델 인스턴스화 및 가중치 로드
    model = target_model_info['model_builder']().to(DEVICE)
    save_name = target_model_info['name'].replace(' ', '_').replace('+', '').lower()
    save_path = os.path.join(OUTPUT_DIR, f"benchmark_{save_name}.pt")

    if not os.path.exists(save_path):
        print(f"\n[ERROR] '{save_path}' 가중치 파일이 없습니다. 먼저 run_benchmark.py를 실행하여 학습을 진행해주세요.")
        return

    model.load_state_dict(torch.load(save_path, weights_only=True))
    model.eval()

    # 원본 메타데이터 로드 (제목 검색용)
    with open(MASTER_FILE, 'r', encoding='utf-8') as f:
        master_list = json.load(f)

    target_idx = -1
    for i, item in enumerate(master_list):
        if i >= num_papers: break
        if args.query.lower() in item['title'].lower():
            target_idx = i
            print(f"\n🎯 입력된 논문: {item['title']} (Idx: {target_idx})")
            break
            
    if target_idx == -1:
        print(f"\n❌ '{args.query}' 키워드를 포함하는 논문을 찾을 수 없습니다.")
        return

    print("\n유사도 계산 중...")
    with torch.no_grad():
        # 통합 그래프 생성 후 임베딩 추출
        full_edges = build_unified_graph(data)
        out_embeddings = model(full_edges)
        paper_embeddings = out_embeddings[:num_papers]

    # 코사인 유사도 계산
    target_vec = paper_embeddings[target_idx].unsqueeze(0)
    sim_scores = torch.cosine_similarity(target_vec, paper_embeddings)
    
    # 자기 자신 제외하고 Top-K
    values, indices = torch.topk(sim_scores, k=args.top_k + 1)

    print(f"\n📊 [{target_model_info['name']}] 기반 추천 논문 Top-{args.top_k}")
    print("=" * 120)
    
    for i in range(1, len(indices)):
        idx = indices[i].item()
        score = values[i].item()
        rec_item = master_list[idx]
        
        print(f"[{i}] 유사도: {score:.4f} | 제목: {rec_item['title']}")
        k = rec_item.get('knowledge', {})
        print(f"    - Domain: {k.get('domain', 'N/A')} | Task: {k.get('task', 'N/A')} | Method: {k.get('method', 'N/A')}")
        print("-" * 120)

if __name__ == "__main__":
    main()
