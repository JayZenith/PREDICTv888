# /// script
# requires-python = ">=3.11"
# dependencies = ["openai==2.32.0"]
# ///
"""Sandbox-side PREDICT agent loop for Python function tasks."""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
import re
import secrets
import signal
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from openai import AsyncOpenAI


PASS = "PASS"
ASSERTION_FAILURE = "ASSERTION_FAILURE"
RUNTIME_ERROR = "RUNTIME_ERROR"
SYNTAX_ERROR = "SYNTAX_ERROR"
TIMEOUT = "TIMEOUT"
OTHER = "OTHER"
OUTCOME_CLASSES = frozenset(
    {PASS, ASSERTION_FAILURE, RUNTIME_ERROR, SYNTAX_ERROR, TIMEOUT, OTHER}
)
DECISIONS = frozenset({"KEEP", "REVISE"})
SUPPORTED_TOOLS = frozenset({"read_file", "apply_patch", "python_test"})
IM_END_TOKEN_ID = 151645
TOOL_NAME_RE = re.compile(r"^[A-Za-z_]\w*$")
PREDICTION_RE = re.compile(
    r"<PREDICTION>\s*([A-Z_]+)\s*</PREDICTION>", re.DOTALL
)
DECISION_RE = re.compile(r"<DECISION>\s*([A-Z_]+)\s*</DECISION>", re.DOTALL)


@dataclass(frozen=True)
class Call:
    tool: str
    id: str
    params: dict[str, str]


@dataclass(frozen=True)
class Result:
    success: bool
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool = False
    outcome: str | None = None


def parse_call_line(line: str) -> tuple[Call | None, str | None]:
    stripped = line.strip()
    if not stripped.startswith("CALL "):
        return None, None
    try:
        _, rest = stripped.split(None, 1)
        tool, payload = rest.split(None, 1)
        decoded = json.loads(payload)
    except (ValueError, json.JSONDecodeError) as exc:
        return None, f"malformed CALL: {exc}"
    if not TOOL_NAME_RE.fullmatch(tool) or not isinstance(decoded, dict):
        return None, "malformed CALL"
    params = dict(decoded)
    call_id = params.pop("id", None)
    if not isinstance(call_id, str) or not call_id:
        return None, "CALL requires a string id"
    if any(not isinstance(value, str) for value in params.values()):
        return None, "CALL arguments must be strings"
    return Call(tool, call_id, params), None


def parse_calls(
    text: str, seen_ids: set[str] | None = None
) -> tuple[list[Call], list[str]]:
    calls: list[Call] = []
    errors: list[str] = []
    seen = set(seen_ids or ())
    for line_no, line in enumerate(text.splitlines(), 1):
        call, error = parse_call_line(line)
        if error:
            errors.append(f"line {line_no}: {error}")
        if call is None:
            continue
        if call.id in seen:
            errors.append(f"line {line_no}: duplicate CALL id {call.id}")
            continue
        seen.add(call.id)
        calls.append(call)
    return calls, errors


def parse_prediction_decision(
    text: str,
) -> tuple[str | None, str | None, list[str]]:
    predictions = PREDICTION_RE.findall(text)
    decisions = DECISION_RE.findall(text)
    errors: list[str] = []
    prediction = predictions[0] if len(predictions) == 1 else None
    decision = decisions[0] if len(decisions) == 1 else None
    if len(predictions) != 1:
        errors.append("Arm B requires exactly one PREDICTION")
    elif prediction not in OUTCOME_CLASSES:
        errors.append(f"unknown prediction outcome: {prediction}")
    if len(decisions) != 1:
        errors.append("Arm B requires exactly one DECISION")
    elif decision not in DECISIONS:
        errors.append(f"unknown decision: {decision}")
    return prediction, decision, errors


def validate_turn_shape(
    text: str,
    calls: list[Call],
    *,
    arm: str,
    pending_candidate: bool,
) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not calls:
        if len(lines) == 1 and lines[0].startswith("FINAL:"):
            return []
        return ["assistant turn must contain exactly one CALL or one FINAL"]
    if len(calls) != 1:
        return ["assistant turn requires exactly one CALL"]
    call_lines = [line for line in lines if line.startswith("CALL ")]
    if arm == "b" and pending_candidate:
        if (
            len(lines) != 3
            or len(call_lines) != 1
            or not PREDICTION_RE.fullmatch(lines[0])
            or not DECISION_RE.fullmatch(lines[1])
            or lines[2] != call_lines[0]
        ):
            return [
                "Arm B prediction turn requires PREDICTION, DECISION, and one CALL"
            ]
        return []
    if len(lines) != 1 or len(call_lines) != 1:
        return ["CALL turn cannot contain additional text"]
    return []


def confined_path(value: str, root: Path, *, require_exists: bool = False) -> Path:
    raw = Path(value)
    candidate = raw.resolve(strict=False) if raw.is_absolute() else (Path.cwd() / raw).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        candidate = (root / raw).resolve()
    candidate.relative_to(root)
    if require_exists and not candidate.exists():
        raise FileNotFoundError(candidate)
    return candidate


def _failed_outcome(stderr: str, timed_out: bool = False) -> str:
    if timed_out:
        return TIMEOUT
    if "SyntaxError" in stderr or "IndentationError" in stderr:
        return SYNTAX_ERROR
    if "AssertionError" in stderr:
        return ASSERTION_FAILURE
    if "Traceback (most recent call last)" in stderr:
        return RUNTIME_ERROR
    return OTHER


def run_hidden_tests(project: Path, test_code: str, timeout: int) -> Result:
    solution = project / "solution.py"
    if not solution.is_file():
        return Result(False, "", "solution.py not found", -1, outcome=OTHER)
    source = solution.read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="predict-python-") as temporary:
        root = Path(temporary)
        check = root / "check.py"
        marker = f"PREDICT_TESTS_PASSED_{secrets.token_hex(16)}"
        check.write_text(
            f"{source.rstrip()}\n\n{test_code.rstrip()}\n\n"
            f"import sys as _predict_sys\n"
            f"_predict_sys.__stdout__.write({marker!r} + '\\n')\n",
            encoding="utf-8",
        )
        env = {
            "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
            "LANG": "C.UTF-8",
            "HOME": "/tmp",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        try:
            process = subprocess.Popen(
                ["python3", "-B", str(check)],
                cwd=root,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            try:
                stdout, stderr = process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.communicate()
                return Result(
                    False,
                    "",
                    f"tests timed out after {timeout}s",
                    -1,
                    timed_out=True,
                    outcome=TIMEOUT,
                )
        except OSError as exc:
            return Result(False, "", str(exc), exc.errno or -1, outcome=OTHER)

    if process.returncode == 0 and stdout.strip().splitlines()[-1:] == [marker]:
        return Result(True, "tests passed", "", 0, outcome=PASS)

    outcome = _failed_outcome(stderr)
    detail = {
        ASSERTION_FAILURE: "tests failed",
        RUNTIME_ERROR: "generated solution raised a runtime error",
        SYNTAX_ERROR: "generated solution has a syntax error",
        OTHER: "tests failed",
    }[outcome]
    return Result(False, "", detail, process.returncode, outcome=outcome)


def execute_tool(call: Call, root: Path, test_code: str, timeout: int) -> Result:
    if call.tool not in SUPPORTED_TOOLS:
        return Result(False, "", f"unknown tool: {call.tool}", -1)
    try:
        if call.tool == "read_file":
            path = confined_path(call.params.get("file_path", ""), root, require_exists=True)
            return Result(True, path.read_text(encoding="utf-8")[:8000], "", 0)
        if call.tool == "apply_patch":
            path = confined_path(call.params.get("file_path", ""), root, require_exists=True)
            find = call.params.get("find")
            replace = call.params.get("replace")
            if find is None or replace is None:
                return Result(False, "", "apply_patch needs file_path, find, replace", -1)
            text = path.read_text(encoding="utf-8")
            count = text.count(find)
            if count != 1:
                return Result(False, "", f"find must occur exactly once; found {count}", -1)
            path.write_text(text.replace(find, replace, 1), encoding="utf-8")
            return Result(True, "patch applied", "", 0)
        project = confined_path(
            call.params.get("project_path", "."), root, require_exists=True
        )
        return run_hidden_tests(project, test_code, timeout)
    except (OSError, UnicodeError, ValueError) as exc:
        return Result(False, "", f"tool error: {exc}", -1)


def result_block(call_id: str, result: Result) -> str:
    lines = [f"status: {'success' if result.success else 'failed'}"]
    if result.timed_out:
        lines.append("timed_out: true")
    if result.stdout:
        lines.append(f"stdout:\n{result.stdout.strip()}")
    if result.stderr:
        lines.append(f"stderr:\n{result.stderr.strip()}")
    return f"RESULT {call_id}:\n" + "\n".join(lines)


def _candidate_sha256(root: Path) -> str:
    return hashlib.sha256((root / "solution.py").read_bytes()).hexdigest()


def _prediction_target(
    *,
    candidate_call_id: str,
    context_messages: list[dict],
    sampled_prediction: str | None,
    decision: str | None,
    result: Result,
    shadow: bool,
    root: Path,
) -> dict:
    if result.outcome not in OUTCOME_CLASSES:
        raise RuntimeError("candidate test did not produce an outcome class")
    return {
        "candidate_call_id": candidate_call_id,
        "context_messages": context_messages,
        "sampled_prediction": sampled_prediction,
        "actual": result.outcome,
        "decision": decision,
        "shadow": shadow,
        "candidate_sha256": _candidate_sha256(root),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--trace-prefix", required=True)
    parser.add_argument("--test-file", default=".glyph/tests.py")
    parser.add_argument("--arm", choices=("a", "b"), default="a")
    parser.add_argument("--max-tool-calls", type=int, default=8)
    parser.add_argument("--tool-timeout", type=int, default=30)
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    root = confined_path(args.trace_prefix, Path.cwd().resolve(), require_exists=True)
    messages = json.loads(os.environ["GLYPH_INITIAL_MESSAGES"])
    test_code = Path(args.test_file).read_text(encoding="utf-8")
    client = AsyncOpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=1800.0,
        max_retries=0,
    )
    executed: list[Call] = []
    results: dict[str, Result] = {}
    errors: list[str] = []
    prediction_targets: list[dict] = []
    pending_candidate: str | None = None

    async def complete() -> str:
        completion = await client.chat.completions.create(
            model=args.model,
            messages=messages,
            extra_body={"stop_token_ids": [IM_END_TOKEN_ID]},
        )
        return completion.choices[0].message.content or ""

    def append_result(call: Call, result: Result) -> None:
        executed.append(call)
        results[call.id] = result
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call.id,
                "content": result_block(call.id, result),
            }
        )

    try:
        while True:
            context_messages = copy.deepcopy(messages)
            assistant = await complete()
            messages.append({"role": "assistant", "content": assistant})
            calls, parse_errors = parse_calls(
                assistant, {call.id for call in executed}
            )
            errors.extend(parse_errors)
            if parse_errors:
                break
            shape_errors = validate_turn_shape(
                assistant,
                calls,
                arm=args.arm,
                pending_candidate=pending_candidate is not None,
            )
            errors.extend(shape_errors)
            if shape_errors:
                break

            if args.arm == "b" and pending_candidate is not None:
                prediction, decision, protocol_errors = parse_prediction_decision(
                    assistant
                )
                expected = "python_test" if decision == "KEEP" else "apply_patch"
                if decision in DECISIONS and calls[0].tool != expected:
                    protocol_errors.append(
                        f"{decision} requires CALL {expected}, got {calls[0].tool}"
                    )
                if len(executed) >= args.max_tool_calls:
                    protocol_errors.append("visible tool-call budget exhausted")

                if protocol_errors:
                    errors.extend(protocol_errors)
                    shadow = run_hidden_tests(root, test_code, args.tool_timeout)
                    prediction_targets.append(
                        _prediction_target(
                            candidate_call_id=pending_candidate,
                            context_messages=context_messages,
                            sampled_prediction=prediction,
                            decision=decision,
                            result=shadow,
                            shadow=True,
                            root=root,
                        )
                    )
                    break

                call = calls[0]
                if decision == "REVISE":
                    shadow = run_hidden_tests(root, test_code, args.tool_timeout)
                    prediction_targets.append(
                        _prediction_target(
                            candidate_call_id=pending_candidate,
                            context_messages=context_messages,
                            sampled_prediction=prediction,
                            decision=decision,
                            result=shadow,
                            shadow=True,
                            root=root,
                        )
                    )
                    result = execute_tool(call, root, test_code, args.tool_timeout)
                    append_result(call, result)
                    pending_candidate = call.id if result.success else None
                else:
                    result = execute_tool(call, root, test_code, args.tool_timeout)
                    prediction_targets.append(
                        _prediction_target(
                            candidate_call_id=pending_candidate,
                            context_messages=context_messages,
                            sampled_prediction=prediction,
                            decision=decision,
                            result=result,
                            shadow=False,
                            root=root,
                        )
                    )
                    append_result(call, result)
                    pending_candidate = None
                continue

            if not calls:
                break

            remaining = args.max_tool_calls - len(executed)
            if remaining <= 0:
                errors.append("visible tool-call budget exhausted")
                break
            for call in calls[:remaining]:
                if args.arm == "b" and call.tool == "python_test":
                    errors.append(
                        "Arm B must predict and choose KEEP before python_test"
                    )
                    break
                result = execute_tool(call, root, test_code, args.tool_timeout)
                append_result(call, result)
                if args.arm == "b" and call.tool == "apply_patch" and result.success:
                    pending_candidate = call.id
            if errors:
                break
    finally:
        await client.close()

    final_verification = run_hidden_tests(root, test_code, args.tool_timeout)
    record = {
        "arm": args.arm,
        "calls": [asdict(call) for call in executed],
        "final_verification": asdict(final_verification),
        "results": {
            call_id: asdict(result) for call_id, result in results.items()
        },
        "prediction_targets": prediction_targets,
        "protocol_errors": errors,
    }
    path = Path(".glyph/trace.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
