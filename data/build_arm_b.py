"""Build Arm B from Arm A; run with uv run python -m data.build_arm_b.

Arm B uses Arm A's tasks and patches. After every patch it writes a PREDICTION block: for each
assert, the value the patched code really returns (found by running it) next to the expected
value. A wrong patch is followed directly by the next patch, so Arm A's failing tests are
dropped. Writes data/sft/arm_b/train.jsonl and data/arm_b_{train,test}.jsonl.
"""
from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from glyph.program import SANDBOX_ENV, actual_values, parse_calls  # noqa: E402

PLACEHOLDER = "# Write your function here.\n"
SYSTEM = (
    "You are a Python coding agent. After apply_patch succeeds, predict what your code "
    "returns on each test: emit <PREDICTION>, one line per test as `CALL = VALUE, expected "
    "EXPECTED`, then </PREDICTION>. If every line matches, run python_test; otherwise "
    "apply_patch to fix it. After FINAL, stop."
)
MAX_TOKENS = 1792  # SFT seq_len

# Prints repr(expected value) for each assert, one per line.
EXPECTED_SCRIPT = r"""
import ast, sys
namespace = {}
for node in ast.parse(open(sys.argv[1]).read()).body:
    if not isinstance(node, ast.Assert):
        exec(compile(ast.Module([node], []), "tests", "exec"), namespace)
        continue
    test = node.test
    if isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq):
        print(repr(eval(compile(ast.Expression(test.comparators[0]), "tests", "eval"), namespace)))
    else:
        print("True")
"""


def assert_calls(test_code: str) -> list[str]:
    """The source text of each assert's call: the left side of `==`, or the whole assert."""
    calls = []
    for node in ast.parse(test_code).body:
        if isinstance(node, ast.Assert):
            test = node.test
            is_eq = isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq)
            calls.append(ast.get_source_segment(test_code, test.left if is_eq else test))
    return calls


def prediction_block(code: str, test_code: str, expected: list[str]) -> str:
    with tempfile.TemporaryDirectory() as directory:
        Path(directory, "solution.py").write_text(code)
        Path(directory, "tests.py").write_text(test_code)
        values = actual_values(Path(directory), Path(directory, "tests.py"))
    lines = []
    for call, value, exp in zip(assert_calls(test_code), values, expected, strict=True):
        no_value = value.startswith("raises ") or value == "times out"
        lines.append(f"{call} = {value}" if no_value else f"{call} = {value}, expected {exp}")
    return "<PREDICTION>\n" + "\n".join(lines) + "\n</PREDICTION>\n"


def build(row: dict) -> dict:
    with tempfile.TemporaryDirectory() as directory:
        Path(directory, "tests.py").write_text(row["test_code"])
        script = subprocess.run([sys.executable, "-c", EXPECTED_SCRIPT, "tests.py"], cwd=directory,
                                env=SANDBOX_ENV, capture_output=True, text=True, check=True)
    expected = script.stdout.splitlines()

    messages = row["messages"]
    out = [{**messages[0], "content": SYSTEM}, messages[1]]
    code, block, n = PLACEHOLDER, "", 0
    for call_message, result in zip(messages[2:-1:2], messages[3:-1:2]):
        (call,), _ = parse_calls(call_message["content"])
        if call.tool == "python_test" and "\nstatus: failed" in result["content"]:
            continue  # Arm B patches again instead of running a failing test
        n += 1
        new_id = f"c{n}"
        out.append({**call_message, "content": block + re.sub(rf'"id":\s*"{call.id}"', f'"id":"{new_id}"', call_message["content"], count=1)})
        out.append({**result, "content": result["content"].replace(f"RESULT {call.id}:", f"RESULT {new_id}:", 1)})
        block = ""
        if call.tool == "apply_patch":
            code = code.replace(call.params["find"], call.params["replace"], 1)
            block = prediction_block(code, row["test_code"], expected)
    out.append(messages[-1])  # FINAL
    return {"arm": "b", "case_id": row["case_id"], "messages": out}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def main():
    from transformers import AutoTokenizer

    rows = [json.loads(line) for line in (ROOT / "data/sft/arm_a/train.jsonl").read_text().splitlines()]
    out = [build(row) for row in rows]

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B-Base")
    template = (ROOT / "configs/chat_template.jinja").read_text()
    longest = max(len(tokenizer(tokenizer.apply_chat_template(r["messages"], chat_template=template, tokenize=False),
                                add_special_tokens=False)["input_ids"]) for r in out)
    assert longest <= MAX_TOKENS, f"longest trace has {longest} tokens"

    write_jsonl(ROOT / "data/sft/arm_b/train.jsonl", out)
    for split in ("train", "test"):
        tasks = [json.loads(line) for line in (ROOT / f"data/arm_a_{split}.jsonl").read_text().splitlines()]
        for task in tasks:
            task["arm"] = "b"
            task["prompt"][0]["content"] = SYSTEM
        write_jsonl(ROOT / f"data/arm_b_{split}.jsonl", tasks)
    print(f"wrote {len(out)} Arm B traces (longest {longest} tokens) and data/arm_b_{{train,test}}.jsonl")


if __name__ == "__main__":
    main()
