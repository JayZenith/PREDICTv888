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
    # Arm B reward terms. Off means 0 reward; the value is still logged as a metric.
    prediction_reward: bool = True
    false_match_penalty: bool = True


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

    @vf.reward(weight=1.0)
    async def passed(self, trace: vf.Trace) -> float:
        """1 if the agent ran a passing python_test and ended with FINAL without a protocol error."""
        state = self._state(trace)
        ran_passing_test = any(
            call["tool"] == "python_test" and state["results"][call["id"]]["success"]
            for call in state.get("calls", [])
        )
        ok = ran_passing_test and state["final_verification"]["success"] and not state["protocol_errors"]
        return float(ok)

    def _line_accuracy(self, trace: vf.Trace) -> float:
        """Fraction of predicted values equal to what the code really returns."""
        lines = [
            (target["predicted"][i] if i < len(target["predicted"]) else None) == actual
            for target in self._state(trace).get("prediction_targets", [])
            for i, actual in enumerate(target["actual"])
        ]
        return sum(lines) / len(lines) if lines else 0.0

    def _false_match_rate(self, trace: vf.Trace) -> float:
        """Fraction of PREDICTION blocks that predict every expected value on failing code."""
        targets = self._state(trace).get("prediction_targets", [])
        if not targets:
            return 0.0
        return sum(t["claims_all_match"] and not t["passes"] for t in targets) / len(targets)

    @vf.reward(weight=0.2)
    async def prediction_accuracy(self, trace: vf.Trace) -> float:
        return self._line_accuracy(trace) if self.config.prediction_reward else 0.0

    @vf.reward(weight=0.2)
    async def false_match_penalty(self, trace: vf.Trace) -> float:
        return -self._false_match_rate(trace) if self.config.false_match_penalty else 0.0

    @vf.metric
    async def prediction_line_accuracy(self, trace: vf.Trace) -> float:
        return self._line_accuracy(trace)

    @vf.metric
    async def false_match_rate(self, trace: vf.Trace) -> float:
        return self._false_match_rate(trace)


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
