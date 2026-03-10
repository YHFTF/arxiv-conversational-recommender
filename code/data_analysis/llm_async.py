import asyncio
import sys
from typing import Any, Awaitable, Callable, Dict, Iterable, List, AsyncIterator


def _print_progress_bar(done: int, total: int, prefix: str = "") -> None:
    """간단한 텍스트 프로그레스 바 (표준 라이브러리만 사용)."""
    if total <= 0:
        return

    bar_len = 40
    ratio = done / total
    filled = int(bar_len * ratio)
    bar = "#" * filled + "-" * (bar_len - filled)
    sys.stdout.write(f"\r{prefix} [{bar}] {done}/{total}")
    if done >= total:
        sys.stdout.write("\n")
    sys.stdout.flush()


async def iter_with_concurrency(
    items: List[Any],
    worker: Callable[[Any], Awaitable[Any]],
    max_concurrency: int = 10,
    label: str = "Processing",
    enable_progress: bool = True,
) -> AsyncIterator[Any]:
    """
    비동기 LLM 호출 등을 위해 공통으로 사용하는 동시 처리 유틸리티.

    - items: 처리할 작업 리스트 (각 원소는 worker에 그대로 전달됨)
    - worker: async 함수, 시그니처는 `async def worker(item) -> result`
    - max_concurrency: 동시에 실행할 최대 작업 수
    - label: 프로그레스 바 앞에 붙는 텍스트
    - enable_progress: True일 때만 텍스트 프로그레스 바 출력

    사용 예:
        async for result in iter_with_concurrency(items, worker, max_concurrency=10, label="LLM"):
            ...
    """
    total = len(items)
    if total == 0:
        return

    sem = asyncio.Semaphore(max_concurrency)
    done = 0

    async def _wrapped(item: Any) -> Any:
        async with sem:
            return await worker(item)

    tasks = [asyncio.create_task(_wrapped(item)) for item in items]

    if enable_progress:
        _print_progress_bar(0, total, prefix=label)

    for coro in asyncio.as_completed(tasks):
        result = await coro
        done += 1
        if enable_progress:
            _print_progress_bar(done, total, prefix=label)
        yield result


