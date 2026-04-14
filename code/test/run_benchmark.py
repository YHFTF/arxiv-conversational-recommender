"""
통합 벤치마크 성능 평가 시스템
===============================

모든 모델을 동일한 조건(학습률, 에포크, 시드, 평가함수)에서 학습/평가하여
공정한 성능 비교를 수행합니다.

새로운 모델을 추가하는 방법:
    1. baselines/ 폴더에 새 모델 파일 생성 (forward(edge_index)  [N, D] 반환)
    2. 아래 get_benchmark_models() 함수에 딕셔너리 1개 추가
    3. python run_benchmark.py 실행

사용법:
    python run_benchmark.py              # 전체 모델 벤치마크
    python run_benchmark.py --only "BPR-MF" "GCN + BPR"  # 특정 모델만 실행
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
from utils import (
    load_data, build_knowledge_ids, get_graph_info,
    build_unified_graph, split_edges, build_initial_features,
    train_and_evaluate
)
from baselines.bpr_mf import BPRMF
from baselines.gcn_bpr import GCNBPR
from baselines.graphsage_bpr import GraphSAGEBPR
from LGCmodel_v2 import ArxivLightGCNV2
from LGCmodel_v4 import ArxivLightGCNV4


def get_benchmark_models(data, meta_counts, initial_features, paper_knowledge_ids):
    """벤치마크에 포함할 모델 목록을 반환합니다.

    [중요] OOM(메모리 누수)을 방지하기 위해 모델 인스턴스를 바로 생성하지 않고,
    호출 시 생성하도록 람다(lambda) 함수를 사용해 지연 로딩(Lazy Loading)합니다.
    """
    _, _, _, total_nodes = get_graph_info(data)

    models = [
        # Tier 1: 비그래프 베이스라인 
        {
            "name": "BPR-MF",
            "type": "Baseline",
            "knowledge": "",
            "model_builder": lambda: BPRMF(total_nodes, initial_features),
        },

        # Tier 2: 그래프 기반 베이스라인 (LLM 미사용) 
        {
            "name": "GCN + BPR",
            "type": "GNN",
            "knowledge": "",
            "model_builder": lambda: GCNBPR(total_nodes, initial_features),
        },
        {
            "name": "GraphSAGE + BPR",
            "type": "GNN",
            "knowledge": "",
            "model_builder": lambda: GraphSAGEBPR(total_nodes, initial_features),
        },
        {
            "name": "LightGCN (BPR)",
            "type": "GNN",
            "knowledge": "",
            "model_builder": lambda: ArxivLightGCNV2(data, meta_counts),
        },

        # Tier 3: LLM 지식 활용 모델 (Ours) 
        {
            "name": "LightGCN + Knowledge (Ours)",
            "type": "KG-GNN",
            "knowledge": " (LLM)",
            "model_builder": lambda: ArxivLightGCNV4(data, meta_counts, paper_knowledge_ids),
        },

        # 새 모델은 여기에 추가 
        # {
        #     "name": "V5 (새 모델)",
        #     "type": "...",
        #     "knowledge": "...",
        #     "model_builder": lambda: YourModelV5(...),
        # },
    ]

    return models


def print_results_table(results):
    """결과를 보기 좋은 테이블 형태로 출력합니다."""
    print("\n" + "=" * 105)
    print("[RESULT] 통합 벤치마크 결과 (All-Item Ranking, 동일 조건)")
    print(f"   평가 설정: Recall@{TOP_K}, NDCG@{TOP_K} | Seed={SEED} | Epochs={NUM_EPOCHS}")
    print("=" * 105)

    header = f"| {'Model':<28} | {'Type':<8} | {'Recall':>9} | {'NDCG':>9} | {'CS-Rec':>9} | {'CS-NDCG':>9} |"
    sep = "|" + "-" * 30 + "|" + "-" * 10 + "|" + "-" * 11 + "|" + "-" * 11 + "|" + "-" * 11 + "|" + "-" * 11 + "|"
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

        print(f"| {res['name']:<28} | {res['type']:<8} "
              f"| {r_str+r_mark:>9} | {n_str:>9} | {csr_str:>9} | {csn_str+csn_mark:>9} |")

    print("=" * 105)
    print("  * CS: Cold-Start (신규 논문 시뮬레이션 성능)")

    # 개선율 요약
    ours = next((r for r in results if "Ours" in r['name']), None)
    baselines = [r for r in results if "Ours" not in r['name']]

    if ours and baselines:
        best_bl_ndcg = max(r['ndcg'] for r in baselines)
        best_bl_cs = max(r['cs_ndcg'] for r in baselines)

        print(f"\n[SUMMARY] Ours vs 최고 베이스라인:")
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
    result_path = os.path.join(OUTPUT_DIR, f"benchmark_results_{timestamp}.json")

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
    parser = argparse.ArgumentParser(description="통합 벤치마크 성능 평가 시스템")
    parser.add_argument('--only', nargs='+', help="특정 모델만 실행할 경우 모델 이름들을 입력하세요. 예: --only BPR-MF \"GCN + BPR\"")
    args = parser.parse_args()
    return args.only


def main():
    print("=" * 105)
    print("[SYSTEM] 통합 벤치마크 성능 평가 시스템 (Transductive + Cold-Start)")
    print(f"   실행 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   사용 장치: {DEVICE}")
    print("=" * 105)

    only_models = parse_args()

    # 1. 데이터 로드
    data, meta, meta_counts = load_data()
    num_papers, num_authors, num_topics, total_nodes = get_graph_info(data)

    # 2. 통합 그래프 구축 + 에지 분할
    full_edges = build_unified_graph(data)
    train_edges, val_edges, test_edges = split_edges(full_edges)

    # 3. 초기 피처 생성
    initial_features = build_initial_features(data)
    paper_knowledge_ids = build_knowledge_ids(data, meta)

    # 4. 모델 목록 생성
    all_models = get_benchmark_models(data, meta_counts, initial_features, paper_knowledge_ids)

    # 특정 모델만 필터링
    if only_models:
        all_models = [m for m in all_models if m['name'] in only_models]

    # 실행할 모델이 없는 경우 방어 로직
    if not all_models:
        print("\n[ERROR] 실행할 모델이 없습니다. '--only' 옵션에 입력한 이름을 확인해주세요.")
        if only_models:
            print(f"입력하신 이름: {only_models}")
        return

    # 5. 순차적으로 각 모델 학습 + 평가
    results = []
    for i, m_info in enumerate(all_models, 1):
        print(f"\n{'=' * 105}")
        print(f"--- [{i}/{len(all_models)}] {m_info['name']} ({m_info['type']}) 학습 시작 ---")
        print(f"{'=' * 105}")

        # 이 시점에서 모델을 인스턴스화하고 GPU 메모리에 올림 (OOM 방지)
        model = m_info['model_builder']().to(DEVICE)
        
        save_name = m_info['name'].replace(' ', '_').replace('+', '').lower()
        save_path = os.path.join(OUTPUT_DIR, f"benchmark_{save_name}.pt")

        recall, ndcg, cs_recall, cs_ndcg = train_and_evaluate(
            model, train_edges, val_edges, test_edges,
            num_papers, model_name=m_info['name'], save_path=save_path
        )

        results.append({
            "name": m_info['name'],
            "type": m_info['type'],
            "knowledge": m_info['knowledge'],
            "recall": recall,
            "ndcg": ndcg,
            "cs_recall": cs_recall,
            "cs_ndcg": cs_ndcg
        })

        # 학습이 끝난 모델은 확실히 메모리에서 해제
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 6. 결과 출력 및 저장
    print_results_table(results)
    save_results(results)


if __name__ == "__main__":
    main()