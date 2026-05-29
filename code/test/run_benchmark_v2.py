"""
통합 벤치마크 성능 평가 시스템 v2 (Zero-Leakage & True Cold-Start)
==================================================================

학습 데이터와 평가 데이터의 완벽한 격리(Data Leakage 방지) 및 
진정한 의미의 신규 논문(Cold-Start) 추천 성능을 공정하게 평가합니다.
"""

import torch
import sys
import os
import json
import argparse
from datetime import datetime

# 경로 설정 (code/test 및 code/model 모듈 접근)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'code', 'model'))

from config import *
from utils_v2 import (
    load_data, build_knowledge_ids, get_graph_info,
    split_data_v2, build_initial_features_v2,
    train_and_evaluate_v2
)
from baselines.bpr_mf import BPRMF
from baselines.gcn_bpr import GCNBPR
from baselines.graphsage_bpr import GraphSAGEBPR
from LGCmodel_v2 import ArxivLightGCNV2
from LGCmodel_v4 import ArxivLightGCNV4


def get_benchmark_models_v2(train_data, meta_counts, train_initial_features, paper_knowledge_ids):
    """v2 벤치마크에 포함할 모델 목록을 반환합니다.

    격리된 train_data 및 train_initial_features를 각 모델 생성자에 넘겨
    모델 내부에서의 데이터 누수(Leakage)까지 완전히 원천 봉쇄합니다.
    """
    _, _, _, total_nodes = get_graph_info(train_data)

    models = [
        # Tier 1: 비그래프 베이스라인 (누수 차단된 피처 주입)
        {
            "name": "BPR-MF",
            "type": "Baseline",
            "data_info": "Cit",
            "model_builder": lambda: BPRMF(total_nodes, train_initial_features),
        },

        # Tier 2: 그래프 기반 베이스라인 (학습 전용 train_data 주입)
        {
            "name": "GCN + BPR",
            "type": "GNN",
            "data_info": "Cit",
            "model_builder": lambda: GCNBPR(total_nodes, train_initial_features),
        },
        {
            "name": "GraphSAGE + BPR",
            "type": "GNN",
            "data_info": "Cit",
            "model_builder": lambda: GraphSAGEBPR(total_nodes, train_initial_features),
        },
        {
            "name": "LightGCN (BPR)",
            "type": "GNN",
            "data_info": "Cit+Net",
            "model_builder": lambda: ArxivLightGCNV2(train_data, meta_counts),
        },

        # Tier 3: LLM 지식 활용 모델 (학습 전용 train_data 및 지식 주입)
        {
            "name": "LightGCN + Knowledge (Ours)",
            "type": "KG-GNN",
            "data_info": "Cit+Net+LLM",
            "model_builder": lambda: ArxivLightGCNV4(train_data, meta_counts, paper_knowledge_ids, knowledge_weight=KNOWLEDGE_WEIGHT),
        },
    ]

    return models


def print_results_table(results):
    """결과를 보기 좋은 테이블 형태로 출력합니다."""
    print("\n" + "=" * 120)
    print("[RESULT] 통합 벤치마크 v2 결과 (Zero-Leakage & True Cold-Start)")
    print(f"   평가 설정: Recall@{TOP_K}, NDCG@{TOP_K} | Seed={SEED} | Epochs={NUM_EPOCHS}")
    print("=" * 120)

    header = f"| {'Model':<28} | {'Type':<8} | {'Data':<12} | {'Recall':>9} | {'NDCG':>9} | {'CS-Rec':>9} | {'CS-NDCG':>9} |"
    sep = "|" + "-" * 30 + "|" + "-" * 10 + "|" + "-" * 14 + "|" + "-" * 11 + "|" + "-" * 11 + "|" + "-" * 11 + "|" + "-" * 11 + "|"
    print(header)
    print(sep)

    best_recall = max(r['recall'] for r in results)
    best_cs_ndcg = max(r['cs_ndcg'] for r in results)

    for res in results:
        r_str = f"{res['recall']*100:.2f}%"
        n_str = f"{res['ndcg']:.4f}"
        csr_str = f"{res['cs_recall']*100:.2f}%"
        csn_str = f"{res['cs_ndcg']:.4f}"

        # 최고 성능 강조
        r_mark = " *" if res['recall'] == best_recall else ""
        csn_mark = " #" if res['cs_ndcg'] == best_cs_ndcg else ""

        print(f"| {res['name']:<28} | {res['type']:<8} | {res['data_info']:<12} "
              f"| {r_str+r_mark:>9} | {n_str:>9} | {csr_str:>9} | {csn_str+csn_mark:>9} |")

    print("=" * 120)
    print("  * Data - Cit: Citation, Net: Author/Topic Network, LLM: LLM-Knowledge")
    print("  * CS: Cold-Start (물리적 격리 신규 논문 10% 대상 테스팅 성능)")

    # 개선율 요약
    ours = next((r for r in results if "Ours" in r['name']), None)
    baselines = [r for r in results if "Ours" not in r['name']]

    if ours and baselines:
        best_bl_ndcg = max(r['ndcg'] for r in baselines)
        best_bl_cs = max(r['cs_ndcg'] for r in baselines)

        print(f"\n[SUMMARY] Ours vs 최고 베이스라인 (Leakage 제거 상태):")
        if best_bl_ndcg > 0:
            ndcg_imp = ((ours['ndcg'] - best_bl_ndcg) / best_bl_ndcg) * 100
            print(f"   일반 성능(NDCG@{TOP_K}) 개선율: {ndcg_imp:+.2f}%")
        if best_bl_cs > 0:
            cs_imp = ((ours['cs_ndcg'] - best_bl_cs) / best_bl_cs) * 100
            print(f"   신규 논문(CS-NDCG) 성능 개선율: {cs_imp:+.2f}%")


def save_results(results):
    """결과를 JSON 파일로 저장합니다."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_path = os.path.join(OUTPUT_DIR, f"benchmark_v2_results_{timestamp}.json")

    save_data = {
        "timestamp": timestamp,
        "device": str(DEVICE),
        "config": {
            "embedding_dim": EMBEDDING_DIM,
            "num_layers": NUM_LAYERS,
            "learning_rate": LEARNING_RATE,
            "num_epochs": NUM_EPOCHS,
            "seed": SEED,
            "top_k": TOP_K,
        },
        "results": results
    }

    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)

    print(f"\n[SAVE] 결과 저장 완료: {result_path}")


def parse_args():
    """argparse를 사용하여 명령줄 인자를 안전하게 처리합니다."""
    parser = argparse.ArgumentParser(description="통합 벤치마크 v2 성능 평가 시스템")
    parser.add_argument('--only', nargs='+', help="특정 모델만 실행할 경우 모델 이름들을 입력하세요.")
    parser.add_argument('--force-retrain', action='store_true', help="기존 학습된 모델이 있더라도 무시하고 새로 학습합니다.")
    args = parser.parse_args()
    return args.only, args.force_retrain


def main():
    print("=" * 105)
    print("[SYSTEM] 통합 벤치마크 v2 성능 평가 시스템 (Zero-Leakage & True Cold-Start)")
    print(f"   실행 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   사용 장치: {DEVICE}")
    print("=" * 105)

    only_models, force_retrain = parse_args()

    # 1. 데이터 로드
    raw_data, meta, meta_counts = load_data()
    num_papers, _, _, _ = get_graph_info(raw_data)

    # 2. Zero-Leakage 데이터 격리 및 분할
    train_data, val_edges, test_edges, cold_edges, cold_mask = split_data_v2(raw_data)

    # 3. 격리된 train_data를 바탕으로 초기 피처 및 지식 ID 생성
    train_initial_features = build_initial_features_v2(train_data)
    paper_knowledge_ids = build_knowledge_ids(train_data, meta)

    # 4. 모델 목록 생성 (격리된 정보 안전 주입)
    all_models = get_benchmark_models_v2(train_data, meta_counts, train_initial_features, paper_knowledge_ids)

    # 특정 모델만 필터링
    if only_models:
        all_models = [m for m in all_models if m['name'] in only_models]

    if not all_models:
        print("\n[ERROR] 실행할 모델이 없습니다. '--only' 옵션에 입력한 이름을 확인해주세요.")
        return

    # 5. 순차적으로 각 모델 학습 + 평가 (v2)
    results = []
    for i, m_info in enumerate(all_models, 1):
        print(f"\n{'=' * 105}")
        print(f"--- [{i}/{len(all_models)}] {m_info['name']} ({m_info['type']}) 시작 ---")
        print(f"{'=' * 105}")

        # 이 시점에서 모델을 인스턴스화하고 GPU 메모리에 올림
        model = m_info['model_builder']().to(DEVICE)
        
        save_name = m_info['name'].replace(' ', '_').replace('+', '').lower() + "_v2"
        save_path = os.path.join(OUTPUT_DIR, f"benchmark_{save_name}.pt")

        recall, ndcg, cs_recall, cs_ndcg = train_and_evaluate_v2(
            model, train_data, val_edges, test_edges, cold_edges, cold_mask,
            num_papers, model_name=m_info['name'], save_path=save_path,
            force_retrain=force_retrain
        )

        results.append({
            "name": m_info['name'],
            "type": m_info['type'],
            "data_info": m_info['data_info'],
            "recall": recall,
            "ndcg": ndcg,
            "cs_recall": cs_recall,
            "cs_ndcg": cs_ndcg
        })

        # 메모리 해제
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 6. 결과 출력 및 저장
    print_results_table(results)
    save_results(results)


if __name__ == "__main__":
    main()
