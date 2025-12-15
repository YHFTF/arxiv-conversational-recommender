import os
import json
import time
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any


@dataclass
class LLMCostRecord:
    """개별 LLM 호출 1건에 대한 토큰 사용량 기록."""

    input_tokens: int
    output_tokens: int
    meta: Optional[Dict[str, Any]] = None


class LLMCostTracker:
    """
    LLM 토큰 사용량 및 비용 산정을 공통으로 처리하는 유틸 클래스.

    - 각 LLM 호출마다 add_call(...) 로 토큰 사용량을 기록
    - finalize(...) 호출 후 summary()/print_summary() 로 집계/출력
    - save_json(...) 으로 별도 비용 리포트 저장 가능
    """

    def __init__(
        self,
        project_size: Optional[int] = None,
        input_cost_per_million: float = 0.15,
        output_cost_per_million: float = 0.60,
    ):
        self.records: List[LLMCostRecord] = []
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.call_count: int = 0

        self.project_size = project_size
        self.input_cost_per_million = input_cost_per_million
        self.output_cost_per_million = output_cost_per_million

        self.start_time: float = time.perf_counter()
        self.end_time: Optional[float] = None

    # --- 기록/집계 ---

    def add_call(
        self,
        input_tokens: int,
        output_tokens: int,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """LLM 호출 1건에 대한 토큰 사용량 기록."""
        rec = LLMCostRecord(
            input_tokens=int(input_tokens or 0),
            output_tokens=int(output_tokens or 0),
            meta=meta or {},
        )
        self.records.append(rec)
        self.total_input_tokens += rec.input_tokens
        self.total_output_tokens += rec.output_tokens
        self.call_count += 1

    def finalize(self) -> None:
        """측정 종료 시점 기록."""
        self.end_time = time.perf_counter()

    # --- 요약/비용 계산 ---

    def _avg_tokens(self) -> Dict[str, float]:
        if self.call_count == 0:
            return {"avg_input": 0.0, "avg_output": 0.0}
        return {
            "avg_input": self.total_input_tokens / self.call_count,
            "avg_output": self.total_output_tokens / self.call_count,
        }

    def _project_tokens(self) -> Dict[str, float]:
        avg = self._avg_tokens()
        if not self.project_size or self.call_count == 0:
            return {"project_input": 0.0, "project_output": 0.0}
        return {
            "project_input": avg["avg_input"] * self.project_size,
            "project_output": avg["avg_output"] * self.project_size,
        }

    def _project_cost(self) -> Dict[str, float]:
        proj = self._project_tokens()
        project_input_tokens = proj["project_input"]
        project_output_tokens = proj["project_output"]

        cost_in = (
            project_input_tokens / 1_000_000 * self.input_cost_per_million
            if project_input_tokens
            else 0.0
        )
        cost_out = (
            project_output_tokens / 1_000_000 * self.output_cost_per_million
            if project_output_tokens
            else 0.0
        )
        return {
            "project_input_cost": cost_in,
            "project_output_cost": cost_out,
            "project_total_cost": cost_in + cost_out,
        }

    def summary(self) -> Dict[str, Any]:
        """토큰/비용/시간 정보를 모두 포함한 요약 정보를 dict로 반환."""
        avg = self._avg_tokens()
        proj_tokens = self._project_tokens()
        proj_cost = self._project_cost()

        elapsed = None
        if self.end_time is not None:
            elapsed = self.end_time - self.start_time

        return {
            "num_calls": self.call_count,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "avg_input_tokens_per_call": avg["avg_input"],
            "avg_output_tokens_per_call": avg["avg_output"],
            "project_size": self.project_size,
            "project_input_tokens": proj_tokens["project_input"],
            "project_output_tokens": proj_tokens["project_output"],
            "project_input_cost_usd": proj_cost["project_input_cost"],
            "project_output_cost_usd": proj_cost["project_output_cost"],
            "project_total_cost_usd": proj_cost["project_total_cost"],
            "elapsed_seconds": elapsed,
        }

    # --- 출력/저장 ---

    def print_summary(self, label: str = "") -> None:
        """콘솔용 요약 출력 (비용/토큰)."""
        s = self.summary()
        header = f"[LLM Cost Summary] {label}".strip()
        print("\n" + "-" * 50)
        print(header)
        print(f"- calls: {s['num_calls']}")
        print(f"- total input tokens:  {s['total_input_tokens']:,}")
        print(f"- total output tokens: {s['total_output_tokens']:,}")
        print(
            f"- avg input / output per call: "
            f"{s['avg_input_tokens_per_call']:.2f} / {s['avg_output_tokens_per_call']:.2f}"
        )

        if s["project_size"]:
            print(f"- project size (samples): {s['project_size']:,}")
            print(
                f"- projected input tokens:  {s['project_input_tokens']:,.0f} "
                f"(cost ${s['project_input_cost_usd']:.2f})"
            )
            print(
                f"- projected output tokens: {s['project_output_tokens']:,.0f} "
                f"(cost ${s['project_output_cost_usd']:.2f})"
            )
            print(
                f"- projected total cost:    ${s['project_total_cost_usd']:.2f} USD"
            )

        if s["elapsed_seconds"] is not None:
            print(f"- elapsed time (loop):     {s['elapsed_seconds']:.2f} seconds")
        print("-" * 50)

    def save_json(self, path: str, extra_meta: Optional[Dict[str, Any]] = None) -> None:
        """비용 관련 상세 로그를 JSON 파일로 저장."""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        data = {
            "summary": self.summary(),
            "records": [asdict(r) for r in self.records],
            "meta": extra_meta or {},
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)


