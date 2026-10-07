# /// script
# requires-python = ">=3.11"
# dependencies = ["openai==2.32.0"]
# ///
"""The agent loop, run in the sandbox for one task.

Each assistant turn is one `CALL tool {json}` line or a `FINAL: ...` line. Tools:
read_file, apply_patch (find/replace in a file) and python_test (run solution.py against the
task's asserts and return the interpreter's output). Arm B must also start the turn after a
successful patch with a PREDICTION block. The loop writes .glyph/trace.json for scoring.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
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

TEST_FILE = Path(".glyph/tests.py")
STOP_TOKEN_ID = 151643  # <|endoftext|>, the turn end in configs/chat_template.jinja
TOOLS = ("read_file", "apply_patch", "python_test")
PREDICTION_TURN_RE = re.compile(r"<PREDICTION>\n(.+)\n</PREDICTION>\n(CALL [^\n]+)", re.DOTALL)
VALUE_TIMEOUT = 5
MAX_FEEDBACK_CHARS = 2000
SANDBOX_ENV = {
    "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
    "LANG": "C.UTF-8",
    "HOME": "/tmp",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",  # set/dict reprs print in a fixed order, so predictions grade the same every run
}


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


# ---------------------------------------------------------------- parsing the assistant turn

def parse_calls(text: str, seen_ids: set[str] = frozenset()) -> tuple[list[Call], list[str]]:
    """Every `CALL tool {json}` line in the turn, and any errors in them."""
    calls, errors, seen = [], [], set(seen_ids)
    for line_no, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line.startswith("CALL "):
            continue
        try:
            _, tool, payload = line.split(None, 2)
            params = json.loads(payload)
        except ValueError as exc:
            errors.append(f"line {line_no}: malformed CALL: {exc}")
            continue
        call_id = params.pop("id", None) if isinstance(params, dict) else None
        if not re.fullmatch(r"[A-Za-z_]\w*", tool) or not isinstance(call_id, str) or not call_id:
            errors.append(f"line {line_no}: malformed CALL")
        elif any(not isinstance(v, str) for v in params.values()):
            errors.append(f"line {line_no}: CALL arguments must be strings")
        elif call_id in seen:
            errors.append(f"line {line_no}: duplicate CALL id {call_id}")
        else:
            seen.add(call_id)
            calls.append(Call(tool, call_id, params))
    return calls, errors


def turn_shape_error(text: str, calls: list[Call], *, arm: str, after_patch: bool) -> str | None:
    """A turn is one CALL or one FINAL; Arm B's turn after a patch is PREDICTION + one CALL."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not calls:
        if len(lines) == 1 and lines[0].startswith("FINAL:"):
            return None
        return "assistant turn must contain exactly one CALL or one FINAL"
    if len(calls) != 1:
        return "assistant turn requires exactly one CALL"
    if arm == "b" and after_patch:
        if PREDICTION_TURN_RE.fullmatch(text.strip()):
            return None
        return "Arm B turn after a patch requires a PREDICTION block, then one CALL"
    return None if len(lines) == 1 else "CALL turn cannot contain additional text"


def prediction_lines(text: str) -> list[str]:
    return PREDICTION_TURN_RE.fullmatch(text.strip()).group(1).splitlines()


def predicted_values(text: str) -> list[str]:
    """The VALUE in each `CALL = VALUE, expected EXPECTED` line."""
    values = []
    for line in prediction_lines(text):
        rest = line.split(" = ", 1)[-1]
        values.append(rest.rpartition(", expected ")[0] or rest)
    return values


def claims_all_match(text: str) -> bool:
    """Every line predicts exactly the expected value."""
    parts = [line.split(" = ", 1)[-1].rpartition(", expected ") for line in prediction_lines(text)]
    return all(value and sep and value == expected for value, sep, expected in parts)


# ---------------------------------------------------------------- tools

def confined_path(value: str, root: Path) -> Path:
    """Resolve a tool path relative to the workspace or to the project, inside the project."""
    path = (Path.cwd() / value).resolve()
    if not path.is_relative_to(root):
        path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"{value} is outside the project")
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def run_hidden_tests(project: Path, test_code: str, timeout: int) -> Result:
    """Run solution.py followed by the asserts and return the interpreter's own output."""
    solution = project / "solution.py"
    if not solution.is_file():
        return Result(False, "", "solution.py not found", -1)
    # A random marker printed after the asserts, so an early sys.exit(0) cannot pass.
    marker = f"PREDICT_TESTS_PASSED_{secrets.token_hex(16)}"
    source = solution.read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="predict-python-") as temporary:
        Path(temporary, "check.py").write_text(
            f"{source.rstrip()}\n\n{test_code.rstrip()}\n\n"
            f"import sys as _predict_sys\n"
            f"_predict_sys.__stdout__.write({marker!r} + '\\n')\n",
            encoding="utf-8",
        )
        try:
            process = subprocess.Popen(
                ["python3", "-B", "check.py"], cwd=temporary, env=SANDBOX_ENV, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
            )
            try:
                stdout, stderr = process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.communicate()
                return Result(False, "", f"tests timed out after {timeout}s", -1, timed_out=True)
        except OSError as exc:
            return Result(False, "", str(exc), exc.errno or -1)
        stderr = stderr.replace(f"{temporary}/", "")  # keep output independent of the temp dir
    passed = process.returncode == 0 and stdout.strip().splitlines()[-1:] == [marker]
    stdout = stdout.replace(marker + "\n", "")
    return Result(passed, stdout[-MAX_FEEDBACK_CHARS:], stderr[-MAX_FEEDBACK_CHARS:], process.returncode)


def execute_tool(call: Call, root: Path, test_code: str, timeout: int) -> Result:
    if call.tool not in TOOLS:
        return Result(False, "", f"unknown tool: {call.tool}", -1)
    try:
        if call.tool == "read_file":
            path = confined_path(call.params.get("file_path", ""), root)
            return Result(True, path.read_text(encoding="utf-8")[:8000], "", 0)
        if call.tool == "apply_patch":
            path = confined_path(call.params.get("file_path", ""), root)
            find, replace = call.params.get("find"), call.params.get("replace")
            if find is None or replace is None:
                return Result(False, "", "apply_patch needs file_path, find, replace", -1)
            text = path.read_text(encoding="utf-8")
            if text.count(find) != 1:
                return Result(False, "", f"find must occur exactly once; found {text.count(find)}", -1)
            path.write_text(text.replace(find, replace, 1), encoding="utf-8")
            return Result(True, "patch applied", "", 0)
        project = confined_path(call.params.get("project_path", "."), root)
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


# ---------------------------------------------------------------- grading Arm B's predictions

# Prints what the solution returns on one assert: repr of the left side of `==` (or of the
# whole assert), or "raises ExceptionName".
VALUE_SCRIPT = r"""
import ast, sys
source = open(sys.argv[1], encoding="utf-8").read()
tests = ast.parse(open(sys.argv[2], encoding="utf-8").read())
index = int(sys.argv[3])
namespace = {"__name__": "solution"}
try:
    exec(compile(source, "solution.py", "exec"), namespace)
    asserts = 0
    for node in tests.body:
        if not isinstance(node, ast.Assert):
            exec(compile(ast.Module(body=[node], type_ignores=[]), "tests", "exec"), namespace)
            continue
        if asserts == index:
            test = node.test
            if isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq):
                value = repr(eval(compile(ast.Expression(test.left), "tests", "eval"), namespace))
            else:
                value = repr(bool(eval(compile(ast.Expression(test), "tests", "eval"), namespace)))
            break
        asserts += 1
except BaseException as exc:
    value = "raises " + type(exc).__name__
sys.__stdout__.write("\n" + value + "\n")
"""


def actual_values(project: Path, test_file: Path) -> list[str]:
    """What the current solution returns on each assert, each in a fresh process."""
    count = sum(isinstance(node, ast.Assert) for node in ast.parse(test_file.read_text()).body)
    solution = (project / "solution.py").resolve()
    values = []
    with tempfile.TemporaryDirectory(prefix="predict-values-") as cwd:
        for index in range(count):
            process = subprocess.Popen(
                ["python3", "-B", "-c", VALUE_SCRIPT, str(solution), str(test_file.resolve()), str(index)],
                cwd=cwd, env=SANDBOX_ENV, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            try:
                stdout, _ = process.communicate(timeout=VALUE_TIMEOUT)
                values.append(stdout.rstrip("\n").rsplit("\n", 1)[-1])
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.communicate()
                values.append("times out")
    return values


# ---------------------------------------------------------------- the loop

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--trace-prefix", required=True)
    parser.add_argument("--arm", choices=("a", "b"), default="a")
    parser.add_argument("--max-tool-calls", type=int, default=8)
    parser.add_argument("--tool-timeout", type=int, default=30)
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    root = Path(args.trace_prefix).resolve(strict=True)
    messages = json.loads(os.environ["GLYPH_INITIAL_MESSAGES"])
    test_code = TEST_FILE.read_text(encoding="utf-8")
    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key, timeout=1800.0, max_retries=0)
    executed: list[Call] = []
    results: dict[str, Result] = {}
    errors: list[str] = []
    prediction_targets: list[dict] = []
    after_patch = False

    try:
        while True:
            completion = await client.chat.completions.create(
                model=args.model, messages=messages, extra_body={"stop_token_ids": [STOP_TOKEN_ID]},
            )
            assistant = completion.choices[0].message.content or ""
            messages.append({"role": "assistant", "content": assistant})

            calls, parse_errors = parse_calls(assistant, {call.id for call in executed})
            if parse_errors:
                errors.extend(parse_errors)
                break
            if shape_error := turn_shape_error(assistant, calls, arm=args.arm, after_patch=after_patch):
                errors.append(shape_error)
                break
            if not calls:  # FINAL
                break

            if args.arm == "b" and after_patch:
                # Grade the prediction against the patched code. Nothing is shown to the agent.
                prediction_targets.append({
                    "predicted": predicted_values(assistant),
                    "actual": actual_values(root, TEST_FILE),
                    "claims_all_match": claims_all_match(assistant),
                    "passes": run_hidden_tests(root, test_code, args.tool_timeout).success,
                })

            if len(executed) >= args.max_tool_calls:
                errors.append("visible tool-call budget exhausted")
                break
            call = calls[0]
            result = execute_tool(call, root, test_code, args.tool_timeout)
            executed.append(call)
            results[call.id] = result
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result_block(call.id, result)})
            after_patch = call.tool == "apply_patch" and result.success
    finally:
        await client.close()

    record = {
        "arm": args.arm,
        "calls": [asdict(call) for call in executed],
        "results": {call_id: asdict(result) for call_id, result in results.items()},
        "final_verification": asdict(run_hidden_tests(root, test_code, args.tool_timeout)),
        "prediction_targets": prediction_targets,
        "protocol_errors": errors,
    }
    Path(".glyph/trace.json").write_text(json.dumps(record), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
