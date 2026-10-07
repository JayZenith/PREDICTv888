"""MBPP tasks for the agent loop in program.py, and their rewards."""

from __future__ import annotations

import json
from pathlib import Path

import verifiers.v1 as vf

PLACEHOLDER = b"# Write your function here.\n"


class GlyphTaskData(vf.TaskData):
    arm: str
    case_id: str
    trace_prefix: str  # the project directory holding solution.py
    test_code: str


class GlyphTaskConfig(vf.TaskConfig):
    max_trace_tokens: int = 4096
    test_run_cost: float = 0.1  # reward lost per python_test run beyond the first


class GlyphTask(vf.Task[GlyphTaskData, vf.State, GlyphTaskConfig]):
    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        await runtime.write(f"{self.data.trace_prefix}/solution.py", PLACEHOLDER)
        await runtime.write(".glyph/tests.py", self.data.test_code.encode("utf-8"))

    async def finalize(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        # A truncated rollout raises, so it gets no trace and scores 0.
        if trace.is_truncated:
            raise vf.TaskError(f"truncated rollout: {trace.stop_condition or 'generation length'}")
        longest = max((branch.num_total_tokens for branch in trace.branches), default=0)
        if longest > self.config.max_trace_tokens:
            raise vf.TaskError(f"rollout has {longest} tokens, over {self.config.max_trace_tokens}")
        trace.info["glyph"] = json.loads((await runtime.read(".glyph/trace.json")).decode("utf-8"))

    @staticmethod
    def _state(trace: vf.Trace) -> dict:
        return trace.info.get("glyph") or {}

    def _passed(self, trace: vf.Trace) -> bool:
        """The agent ran a passing python_test and ended with FINAL without a protocol error."""
        state = self._state(trace)
        ran_passing_test = any(
            call["tool"] == "python_test" and state["results"][call["id"]]["success"]
            for call in state.get("calls", [])
        )
        return ran_passing_test and state["final_verification"]["success"] and not state["protocol_errors"]

    def _test_runs(self, trace: vf.Trace) -> int:
        return sum(call["tool"] == "python_test" for call in self._state(trace).get("calls", []))

    @vf.reward(weight=1.0)
    async def reward(self, trace: vf.Trace) -> float:
        """Same for both arms: 1 for passing, minus test_run_cost per extra python_test run."""
        if not self._passed(trace):
            return 0.0
        return 1.0 - self.config.test_run_cost * (self._test_runs(trace) - 1)

    @vf.metric
    async def passed(self, trace: vf.Trace) -> float:
        return float(self._passed(trace))

    @vf.metric
    async def test_runs(self, trace: vf.Trace) -> float:
        return float(self._test_runs(trace))


class GlyphTasksetConfig(vf.TasksetConfig):
    data_path: str
    task: GlyphTaskConfig = GlyphTaskConfig()


class GlyphTaskset(vf.Taskset[GlyphTask, GlyphTasksetConfig]):
    def load(self) -> list[GlyphTask]:
        rows = [json.loads(line) for line in Path(self.config.data_path).read_text().splitlines()]
        return [
            GlyphTask(GlyphTaskData(idx=i, name=row["case_id"], prompt=row["prompt"], arm=row["arm"],
                                    case_id=row["case_id"], trace_prefix=row["trace_prefix"],
                                    test_code=row["test_code"]),
                      self.config.task)
            for i, row in enumerate(rows)
        ]


__all__ = ["GlyphTaskset"]
