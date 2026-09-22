"""카테고리 균형을 보장하는 OGBN-Arxiv Forest Fire 샘플러.

기존 구현은 인용 이웃만 따라가므로 연결이 조밀한 카테고리가 표본을
독점했다. 이 스크립트는 Forest Fire 확장은 유지하면서 OGB의 40개 레이블에
가능한 한 동일한 쿼터를 적용한다. 희소 레이블은 보유한 전량을 쓰고 남은
쿼터는 다른 레이블에 균등 재배분한다.
"""

from __future__ import annotations

import argparse
import os
import random
from collections import Counter, deque
from typing import Dict, Iterable, List

import networkx as nx
import numpy as np
import torch


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "subdataset", "ogbn_arxiv_16k_ffs_sample.pt")


def load_trusted_ogbn_arxiv(dataset_class):
    """OGB가 생성한 PyG 캐시를 PyTorch 2.6+에서 호환되게 읽는다.

    OGB의 ``dataset_pyg.py``는 ``torch.load`` 인자를 전달하지 않는다. PyTorch
    2.6부터 기본값이 ``weights_only=True``로 바뀌어 PyG의 ``Data`` 객체 캐시를
    읽지 못하므로, OGB 공식 데이터셋 생성 중에만 이전 동작을 적용한다. 캐시는
    사용자가 신뢰하는 OGB 데이터 디렉터리의 파일이어야 한다.
    """
    original_torch_load = torch.load

    def load_with_full_pickle(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = load_with_full_pickle
    try:
        return dataset_class(name="ogbn-arxiv", root=os.path.join(PROJECT_ROOT, "dataset"))
    finally:
        torch.load = original_torch_load


def to_networkx_graph(data) -> nx.DiGraph:
    """PyG citation graph를 방향 그래프로 변환한다."""
    graph = nx.DiGraph()
    graph.add_nodes_from(range(data.num_nodes))
    graph.add_edges_from(zip(data.edge_index[0].tolist(), data.edge_index[1].tolist()))
    return graph


def balanced_quotas(labels: Iterable[int], target_size: int) -> Dict[int, int]:
    """가능한 한 모든 카테고리에 같은 수를 배정한다."""
    counts = Counter(labels)
    categories = sorted(counts)
    if target_size <= 0:
        raise ValueError("target_size는 1 이상이어야 합니다.")
    if target_size > sum(counts.values()):
        raise ValueError("target_size가 전체 노드 수보다 큽니다.")

    base, _ = divmod(target_size, len(categories))
    quotas = {category: min(base, counts[category]) for category in categories}
    remaining = target_size - sum(quotas.values())

    # 희소 카테고리와 나머지로 생긴 잔여분을 아직 여유가 있는 카테고리에 순환 배정한다.
    while remaining:
        progressed = False
        for category in categories:
            if remaining == 0:
                break
            if quotas[category] < counts[category]:
                quotas[category] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            raise RuntimeError("카테고리 쿼터를 배정할 수 없습니다.")
    return quotas


def balanced_forest_fire_sampling(
    graph: nx.DiGraph,
    labels: List[int],
    target_size: int,
    pf: float,
    rng: random.Random,
    np_rng: np.random.Generator,
) -> List[int]:
    """카테고리 쿼터를 넘지 않는 Forest Fire 샘플링을 수행한다.

    확장 과정에서 이미 쿼터를 채운 레이블은 건너뛴다. 고립 노드나 희소 연결로
    Forest Fire가 채우지 못한 쿼터는 동일 카테고리의 미선택 노드로 보충한다.
    """
    if not 0 < pf < 1:
        raise ValueError("pf는 0과 1 사이여야 합니다.")
    if len(labels) != graph.number_of_nodes():
        raise ValueError("labels 길이와 그래프 노드 수가 일치해야 합니다.")

    quotas = balanced_quotas(labels, target_size)
    nodes_by_category: Dict[int, List[int]] = {category: [] for category in quotas}
    for node, label in enumerate(labels):
        nodes_by_category[label].append(node)

    sampled: set[int] = set()
    sampled_counts: Counter[int] = Counter()
    p_b = 1.0 - pf

    def can_add(node: int) -> bool:
        return node not in sampled and sampled_counts[labels[node]] < quotas[labels[node]]

    # 모든 카테고리의 seed pool에서 시작하므로 시작점부터 균형을 보장한다.
    while len(sampled) < target_size:
        available_categories = [
            category for category, quota in quotas.items() if sampled_counts[category] < quota
        ]
        if not available_categories:
            break
        category = rng.choice(available_categories)
        candidates = [node for node in nodes_by_category[category] if node not in sampled]
        if not candidates:
            raise RuntimeError(f"카테고리 {category}의 쿼터를 채울 노드가 없습니다.")

        queue = deque([rng.choice(candidates)])
        while queue and len(sampled) < target_size:
            current = queue.popleft()
            if not can_add(current):
                continue
            sampled.add(current)
            sampled_counts[labels[current]] += 1

            eligible_neighbors = [node for node in graph.successors(current) if can_add(node)]
            if eligible_neighbors:
                num_to_burn = min(
                    int(np_rng.geometric(p=p_b) - 1), len(eligible_neighbors)
                )
                if num_to_burn:
                    queue.extend(rng.sample(eligible_neighbors, num_to_burn))

    # FFS가 도달하지 못한 노드를 같은 카테고리 pool에서 보충한다.
    for category, quota in quotas.items():
        missing = quota - sampled_counts[category]
        if missing:
            candidates = [node for node in nodes_by_category[category] if node not in sampled]
            sampled.update(rng.sample(candidates, missing))

    result = list(sampled)
    rng.shuffle(result)  # 저장 순서가 카테고리 순서가 되지 않도록 한다.
    assert len(result) == target_size
    assert Counter(labels[node] for node in result) == quotas
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="카테고리 균형 Forest Fire 샘플링")
    parser.add_argument("--target-size", type=int, default=16000)
    parser.add_argument("--pf", type=float, default=0.75, help="Forest Fire 확장 확률")
    parser.add_argument("--seed", type=int, default=42, help="재현 가능한 난수 시드")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    try:
        from ogb.nodeproppred import PygNodePropPredDataset
    except ImportError as exc:
        raise SystemExit(
            "PygNodePropPredDataset를 사용하려면 torch-geometric이 필요합니다. "
            "현재 설치된 CUDA PyTorch를 유지하려면 다음을 실행하세요:\n"
            "  uv pip install --python venv/bin/python torch-geometric\n"
            "그 후 이 스크립트를 다시 실행하세요."
        ) from exc

    args = parse_args()
    dataset = load_trusted_ogbn_arxiv(PygNodePropPredDataset)
    graph_data = dataset[0]
    labels = graph_data.y.view(-1).tolist()
    graph = to_networkx_graph(graph_data)
    print(f"citation graph: nodes={graph.number_of_nodes():,}, edges={graph.number_of_edges():,}")

    sampled_nodes = balanced_forest_fire_sampling(
        graph, labels, args.target_size, args.pf, random.Random(args.seed), np.random.default_rng(args.seed)
    )
    sample_indices = torch.tensor(sampled_nodes, dtype=torch.long)
    sample_labels = graph_data.y[sample_indices]
    distribution = Counter(sample_labels.view(-1).tolist())
    print(f"균형 샘플링 완료: {len(sampled_nodes):,}개, 카테고리 수={len(distribution)}")
    print("카테고리별 표본 수:", dict(sorted(distribution.items())))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(
        {
            "indices": sampled_nodes,
            "features": graph_data.x[sample_indices],
            "labels": sample_labels,
            "sampling": {
                "method": "balanced_forest_fire",
                "target_size": args.target_size,
                "pf": args.pf,
                "seed": args.seed,
                "category_counts": dict(sorted(distribution.items())),
            },
        },
        args.output,
    )
    print(f"저장 완료: {args.output}")


if __name__ == "__main__":
    main()
