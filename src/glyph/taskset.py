"""Load Python function tasks and score real test execution."""

from __future__ import annotations

import json
from pathlib import Path

import verifiers.v1 as vf


def _message_value(message, key: str) -> str:
    if isinstance(message, dict):
        value = message.get(key)
    else:
        value = getattr(message, key, None)
    return "" if value is None else str(value)


def message_role(message) -> str:
    return _message_value(message, "role")


def message_content(message) -> str:
    return _message_value(message, "content")


def message_tool_call_id(message) -> str:
    return _message_value(message, "tool_call_id")


def load_rows(data_path: str, max_samples: int | None = None) -> list[dict]:
    rows = [json.loads(line) for line in Path(data_path).read_text(encoding="utf-8").splitlines()]
    return rows[:max_samples]


class GlyphTaskData(vf.TaskData):
    arm: str = "a"
    case_id: str
    trace_prefix: str
    test_code: str


class GlyphTaskConfig(vf.TaskConfig):
    max_trace_tokens: int = 4096
    # Arm B reward terms; off means 0 reward, and the value is still logged as a metric.
    prediction_reward: bool = True
    false_match_penalty: bool = True


PLACEHOLDER = b"# Write your function here.\n"


class GlyphTask(vf.Task[GlyphTaskData, vf.State, GlyphTaskConfig]):
    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        await runtime.write(f"{self.data.trace_prefix}/solution.py", PLACEHOLDER)
        await runtime.write(".glyph/tests.py", self.data.test_code.encode("utf-8"))

    async def finalize(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        if reason := self._truncation_reason(trace):
            raise vf.TaskError(reason)
        payload = await runtime.read(".glyph/trace.json")
        trace.info["glyph"] = json.loads(payload.decode("utf-8"))

    def _truncation_reason(self, trace: vf.Trace) -> str | None:
        if trace.is_truncated:
            return f"truncated rollout: {trace.stop_condition or 'generation length'}"
        longest = max((branch.num_total_tokens for branch in trace.branches), default=0)
        if longest > self.config.max_trace_tokens:
            trace.stop_condition = "max_total_tokens"
            return (
                f"rollout has {longest} tokens, exceeding the training limit "
                f"of {self.config.max_trace_tokens}"
            )
        return None

    def _evaluate(self, trace: vf.Trace) -> tuple[float, bool]:
        cached = trace.info.get("glyph_evaluation")
        if cached is not None:
            return float(cached[0]), bool(cached[1])

        if self._truncation_reason(trace):
            evaluation = (0.0, False)
            trace.info["glyph_evaluation"] = evaluation
            return evaluation

        state = trace.info.get("glyph") or {}
        calls = state.get("calls") or []
        results = state.get("results") or {}
        final_verification = state.get("final_verification") or {}
        successful_call_id = next(
            (
                call.get("id")
                for call in calls
                if call.get("tool") == "python_test"
                and bool((results.get(call.get("id")) or {}).get("success"))
            ),
            None,
        )
        messages = trace.branches[-1].messages if trace.branches else []
        valid = bool(
            successful_call_id
            and final_verification.get("success")
            and not state.get("protocol_errors")
            and messages
            and message_role(messages[-1]) == "assistant"
            and message_content(messages[-1]).strip().startswith("FINAL:")
            and any(
                message_role(message) == "tool"
                and message_tool_call_id(message) == successful_call_id
                for message in messages[:-1]
            )
        )
        evaluation = (float(valid), valid)
        trace.info["glyph_evaluation"] = evaluation
        return evaluation

    @vf.reward(weight=1.0)
    async def mbpp_reward(self, trace: vf.Trace) -> float:
        return self._evaluate(trace)[0]

    @vf.metric
    async def passed(self, trace: vf.Trace) -> float:
        return float(self._evaluate(trace)[1])

    @staticmethod
    def _targets(trace: vf.Trace) -> list[dict]:
        return (trace.info.get("glyph") or {}).get("prediction_targets") or []

    def _line_accuracy(self, trace: vf.Trace) -> float:
        """Fraction of predicted test values that match the executed candidate."""
        lines = [
            (target["predicted"][i] if i < len(target["predicted"]) else None) == actual
            for target in self._targets(trace)
            for i, actual in enumerate(target["actual"])
        ]
        return sum(lines) / len(lines) if lines else 0.0

    def _false_match_rate(self, trace: vf.Trace) -> float:
        """Fraction of predictions that claim every test matches on a failing candidate."""
        targets = self._targets(trace)
        if not targets:
            return 0.0
        return sum(t["claims_all_match"] and not t["passes"] for t in targets) / len(targets)

    @vf.reward(weight=0.2)
    async def prediction_accuracy(self, trace: vf.Trace) -> float:
        """Always 0 for Arm A, which makes no predictions."""
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
    data_path: str | None = None
    max_samples: int | None = None
    task: GlyphTaskConfig = GlyphTaskConfig()


class GlyphTaskset(vf.Taskset[GlyphTask, GlyphTasksetConfig]):
    def load(self) -> list[GlyphTask]:
        if not self.config.data_path:
            raise ValueError("GlyphTaskset requires an explicit data_path")
        rows = load_rows(self.config.data_path, self.config.max_samples)
        return [
            GlyphTask(
                GlyphTaskData(
                    idx=idx,
                    name=row["case_id"],
                    prompt=row["prompt"],
                    arm=row["arm"],
                    case_id=row["case_id"],
                    trace_prefix=row["trace_prefix"],
                    test_code=row["test_code"],
                ),
                self.config.task,
            )
            for idx, row in enumerate(rows)
        ]


__all__ = ["GlyphTaskset"]
