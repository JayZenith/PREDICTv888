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
    path = Path(data_path)
    rows: list[dict] = []
    with path.open(encoding="utf-8") as source:
        for line_no, line in enumerate(source, 1):
            if max_samples is not None and len(rows) >= max_samples:
                break
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc.msg}") from exc
            required = {"case_id", "prompt", "trace_prefix", "test_code"}
            if not isinstance(row, dict) or not required <= row.keys():
                raise ValueError(f"{path}:{line_no}: missing required task fields")
            if not isinstance(row["prompt"], list):
                raise ValueError(f"{path}:{line_no}: prompt must be a message list")
            rows.append(row)
    return rows


class GlyphTaskData(vf.TaskData):
    arm: str = "a"
    case_id: str
    source: str = "mbpp"
    source_task_id: int
    trace_prefix: str
    test_code: str


class GlyphTaskConfig(vf.TaskConfig):
    max_trace_tokens: int = 4096


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

    @vf.metric
    async def prediction_accuracy(self, trace: vf.Trace) -> float:
        """Fraction of predicted test values that match the executed candidate."""
        targets = (trace.info.get("glyph") or {}).get("prediction_targets") or []
        lines = [
            (target["predicted"][i] if i < len(target["predicted"]) else None) == actual
            for target in targets
            for i, actual in enumerate(target["actual"])
        ]
        return sum(lines) / len(lines) if lines else 0.0


class GlyphTasksetConfig(vf.TasksetConfig):
    data_path: str | None = None
    max_samples: int | None = None
    task: GlyphTaskConfig = GlyphTaskConfig()


class GlyphTaskset(vf.Taskset[GlyphTask, GlyphTasksetConfig]):
    def load(self) -> list[GlyphTask]:
        if not self.config.data_path:
            raise ValueError("GlyphTaskset requires an explicit data_path")
        data_path = Path(self.config.data_path).expanduser().resolve(strict=True)
        rows = load_rows(str(data_path), self.config.max_samples)
        tasks: list[GlyphTask] = []
        for idx, row in enumerate(rows):
            tasks.append(
                GlyphTask(
                    GlyphTaskData(
                        idx=idx,
                        name=row["case_id"],
                        prompt=row["prompt"],
                        arm=row.get("arm", "a"),
                        case_id=row["case_id"],
                        source=row.get("source", "mbpp"),
                        source_task_id=int(row.get("task_id", idx)),
                        trace_prefix=row["trace_prefix"],
                        test_code=row["test_code"],
                    ),
                    self.config.task,
                )
            )
        return tasks


__all__ = ["GlyphTaskset"]
