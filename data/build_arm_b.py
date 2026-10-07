"""Rebuild Arm B from Arm A; run with uv run python -m data.build_arm_b."""
from __future__ import annotations

import ast
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile
import tokenize
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
SYSTEM = (
    "You are a Python coding agent. After apply_patch succeeds, predict what your code "
    "returns on each test: emit <PREDICTION>, one line per test as `CALL = VALUE, expected "
    "EXPECTED`, then </PREDICTION>. If every line matches, run python_test; otherwise "
    "apply_patch to fix it. After FINAL, stop."
)
CALL = re.compile(r"CALL (read_file|apply_patch|python_test) (\{.*\})", re.DOTALL)
OLD = re.compile(r"<" + r"DECISION>|<PREDICTION>\s*(?:PASS|ASSERTION_FAILURE|RUNTIME_ERROR|SYNTAX_ERROR|TIMEOUT|OTHER|OUTCOME)\s*</PREDICTION>")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def specs(test_code):
    tree = ast.parse(test_code)
    assertions = [n for n in ast.walk(tree) if isinstance(n, ast.Assert)]
    assertions.sort(key=lambda n: (n.lineno, n.col_offset))
    require(all(n in tree.body for n in assertions), "Nested asserts need explicit execution semantics")
    result = []
    for node in assertions:
        expr = node.test
        equal = isinstance(expr, ast.Compare) and len(expr.ops) == 1 and isinstance(expr.ops[0], ast.Eq)
        expression = ast.get_source_segment(test_code, expr)
        if equal:
            # AST operand spans omit redundant parentheses. Split the original
            # comparison at its top-level == to retain the exact RHS spelling.
            depth = 0
            offsets = [0]
            for line in expression.splitlines(keepends=True):
                offsets.append(offsets[-1] + len(line))
            for token in tokenize.generate_tokens(io.StringIO(expression).readline):
                if token.type != tokenize.OP:
                    continue
                if token.string == "==" and depth == 0:
                    start = offsets[token.start[0] - 1] + token.start[1]
                    end = offsets[token.end[0] - 1] + token.end[1]
                    result.append((expression[:start].strip(), expression[end:].strip(), True))
                    break
                if token.string in "([{":
                    depth += 1
                elif token.string in ")]}":
                    depth -= 1
            else:
                raise RuntimeError("Could not locate equality operator")
        else:
            result.append((expression, "True", False))
    require(result, "No asserts found")
    return result


def worker(mode):
    # Model code is executed only in this disposable child, never in the generator.
    random.seed(0)
    test_code = Path("tests.txt").read_text()
    namespace = {"__name__": "solution", "__file__": str(Path("solution.py").resolve())}
    if mode == "test":
        exec(compile(Path("solution.py").read_text(), "solution.py", "exec"), namespace)
        exec(compile(test_code, "test_code", "exec"), namespace)
        return
    descriptions = specs(test_code)
    selected = int(mode.split(":")[1])
    result_file = Path("predictions.jsonl")

    def emit(value, match, expected=None):
        with result_file.open("a") as stream:
            stream.write(json.dumps([value, match, expected]) + "\n")

    try:
        exec(compile(Path("solution.py").read_text(), "solution.py", "exec"), namespace)
    except BaseException as exc:
        emit("raises " + type(exc).__name__, False)
        return
    index = 0
    for node in ast.parse(test_code).body:
        if not isinstance(node, ast.Assert):
            exec(compile(ast.Module(body=[node], type_ignores=[]), "test_code", "exec"), namespace)
            continue
        lhs, rhs, equal = descriptions[index]
        index += 1
        if index - 1 != selected:
            continue
        try:
            # Expected is evaluated before the candidate call so it is known even if LHS raises.
            expected = eval(compile(rhs, "test_code", "eval"), namespace) if equal else True
            actual = eval(compile(lhs, "test_code", "eval"), namespace)
            if equal:
                value = repr(actual)
                match = bool(actual == expected)
            else:
                match = bool(actual)
                value = repr(match)
            emit(value, match, repr(expected))
        except BaseException as exc:
            emit("raises " + type(exc).__name__, False)
        return


def execute(code, test_code, mode="predict"):
    if mode == "predict":
        descriptions = specs(test_code)
        values = [execute(code, test_code, f"predict:{i}") for i in range(len(descriptions))]
        lines = []
        for (lhs, _, _), (value, _, expected) in zip(descriptions, values, strict=True):
            # Expected is printed as repr so a match always reads as identical text.
            suffix = "" if value.startswith("raises ") or value == "times out" else f", expected {expected}"
            lines.append(f"{lhs} = {value}{suffix}")
        return lines, [match for _, match, _ in values]
    with tempfile.TemporaryDirectory(prefix="arm_b_") as directory:
        directory = Path(directory)
        (directory / "solution.py").write_text(code)
        (directory / "tests.txt").write_text(test_code)
        try:
            process = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--worker", mode],
                cwd=directory, env={**os.environ, "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1"},
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5,
            )
            require(process.returncode == 0,
                    f"Candidate {mode} failed: {process.stderr.decode(errors='replace')}")
            timed_out = False
        except subprocess.TimeoutExpired:
            timed_out = True
        if mode == "test":
            require(not timed_out, "Final test_code timed out")
            return None
        path = directory / "predictions.jsonl"
        values = [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
        if timed_out:
            return ["times out", False, None]
        require(len(values) == 1, "Missing prediction result")
        return values[0]


def parse_call(message):
    match = CALL.fullmatch(message["content"])
    require(message["role"] == "assistant" and match is not None, "Invalid Arm A CALL")
    return match[1], json.loads(match[2])


def build(row):
    messages = row["messages"]
    require(messages[0]["role"] == "system" and messages[1]["role"] == "user", "Invalid prefix")
    out = [{**messages[0], "content": SYSTEM}, copy.deepcopy(messages[1])]
    patches = sum(m["content"].startswith("CALL apply_patch ") for m in messages)
    wrong = patches - 1
    require(wrong in ((0,) if row["trace_type"] == "direct" else (1, 2)), "Wrong candidate count")
    code = None
    pending = ""
    count = 0
    seen_patches = 0
    predictions = []
    for index in range(2, len(messages) - 1, 2):
        message, result = messages[index:index + 2]
        name, args = parse_call(message)
        require(result["role"] == "tool" and result["content"].startswith(f"RESULT {args['id']}:\n"), "Invalid RESULT")
        if name == "python_test" and seen_patches < patches:
            require(result["content"].startswith(f"RESULT {args['id']}:\nstatus: failed"), "Unexpected failing test RESULT")
            continue
        count += 1
        call_id = f"c{count}"
        # Only replace the CALL id; preserve all other original bytes, including patch strings.
        original_id = args["id"]
        new_call, replacements = re.subn(r'"id":\s*"' + re.escape(original_id) + '"', '"id":"' + call_id + '"', message["content"], count=1)
        require(replacements == 1, "CALL id missing")
        out.append({**message, "content": pending + new_call})
        pending = ""
        out.append({**result, "content": result["content"].replace(f"RESULT {original_id}:\n", f"RESULT {call_id}:\n", 1)})
        if name == "read_file":
            require(count == 1, "Unexpected read_file")
            prefix = f"RESULT {original_id}:\nstatus: success\nstdout:\n"
            require(result["content"].startswith(prefix), "Unexpected read_file result")
            code = result["content"][len(prefix):]
            # Tool display strips the trailing newline; recover exact initial bytes from find.
            next_name, next_args = parse_call(messages[index + 2])
            require(next_name == "apply_patch" and next_args["find"].rstrip("\n") == code, "Initial solution differs from read_file")
            code = next_args["find"]
        elif name == "apply_patch":
            require(result["content"] == f"RESULT {original_id}:\nstatus: success\nstdout:\npatch applied", "Patch did not succeed")
            require(code is not None and code.count(args["find"]) == 1, "Patch find is not unique")
            code = code.replace(args["find"], args["replace"], 1)
            seen_patches += 1
            digest_key = "final_code_sha256" if seen_patches == patches else "candidate_code_sha256" if seen_patches == 1 else None
            if digest_key:
                require(hashlib.sha256(code.encode()).hexdigest() == row[digest_key], "Candidate hash differs from Arm A")
            prediction = execute(code, row["test_code"])
            require(prediction == execute(code, row["test_code"]), f"Non-repeatable prediction: {row['case_id']}")
            lines, matches = prediction
            require(all(matches) == (seen_patches == patches), f"Unexpected candidate matches: {row['case_id']}")
            predictions.extend(lines)
            pending = "<PREDICTION>\n" + "\n".join(lines) + "\n</PREDICTION>\n"
        else:
            require(seen_patches == patches, "Premature python_test")
            require(result["content"].startswith(f"RESULT {original_id}:\nstatus: success"), "Nonpassing python_test retained")
            execute(code, row["test_code"], "test")
    require(not pending and out[-2]["content"].split("\n")[-1].startswith("CALL python_test "), "Missing final test")
    require(messages[-1]["content"].startswith("FINAL:"), "Missing FINAL")
    out.append(copy.deepcopy(messages[-1]))
    return {**row, "arm": "b", "messages": out, "recovery_mode": None, "num_wrong_candidates": wrong}, predictions


def encode(rows):
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows).encode()


def main():
    from transformers import AutoTokenizer

    source = ROOT / "data/sft/arm_a/train.jsonl"
    rows = [json.loads(s) for s in source.read_text().splitlines()]
    require(len(rows) == 212 and len({r["case_id"] for r in rows}) == 212, "Expected 212 distinct Arm A cases")
    require(Counter(r["trace_type"] for r in rows) == {"direct": 142, "recovery": 70}, "Unexpected Arm A trace counts")
    output, lines = [], []
    # Each worker launches fresh child processes; map retains Arm A's row order.
    with ThreadPoolExecutor(max_workers=4) as pool:
        for index, (built, predicted) in enumerate(pool.map(build, rows), 1):
            output.append(built)
            lines.extend(predicted)
            if index % 25 == 0:
                print(f"Validated {index}/212 traces", flush=True)
    require([r["case_id"] for r in output] == [r["case_id"] for r in rows], "Case order changed")
    non_equal = sum(not equal for r in rows for _, _, equal in specs(r["test_code"]))
    print(f"Non-== asserts: {non_equal}; raises: {sum(' = raises ' in s for s in lines)}; timeouts: {sum(s.endswith(' = times out') for s in lines)}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B-Base")
    template = (ROOT / "configs/chat_template.jinja").read_text()
    rendered = [tokenizer.apply_chat_template(r["messages"], chat_template=template,
                tokenize=False, add_generation_prompt=False) for r in output]
    lengths = [len(tokenizer(text, add_special_tokens=False)["input_ids"]) for text in rendered]
    maximum = max(lengths)
    print(f"Maximum SFT tokens: {maximum} ({output[lengths.index(maximum)]['case_id']})", flush=True)
    too_long = [(r["case_id"], n) for r, n in zip(output, lengths, strict=True) if n > 1792]
    require(not too_long, f"Rows exceed 1792 tokens; no output written, no truncation: {too_long}")
    changes = {ROOT / "data/sft/arm_b/train.jsonl": encode(output)}
    for split in ("train", "validation", "test"):
        path = ROOT / f"data/arm_b_{split}.jsonl"
        # Replace just the serialized system string, preserving every other byte.
        updated = []
        for line in path.read_text().splitlines(keepends=True):
            old = json.loads(line)
            system = [m for m in old["prompt"] if m["role"] == "system"]
            require(len(system) == 1, "Expected one system message")
            before = json.dumps(system[0]["content"], ensure_ascii=False)
            after = json.dumps(SYSTEM, ensure_ascii=False)
            require(line.count(before) == 1, "System string not unique")
            revised = line.replace(before, after, 1)
            expected = copy.deepcopy(old)
            next(m for m in expected["prompt"] if m["role"] == "system")["content"] = SYSTEM
            require(json.loads(revised) == expected, "Non-system task field changed")
            updated.append(revised)
        changes[path] = "".join(updated).encode()
    for path in (ROOT / "data").rglob("*"):
        if path.is_file() and path.suffix in {".jsonl", ".json", ".txt", ".md", ".py"}:
            content = changes.get(path, path.read_bytes()).decode()
            require(not OLD.search(content), f"Old protocol remains in {path}")
    for path, content in changes.items():
        path.write_bytes(content)
    print("Wrote 212 Arm B traces and updated all three task system prompts.")
    for label, example in [("DIRECT", next(r for r in output if r["num_wrong_candidates"] == 0)),
                           ("TWO WRONG CANDIDATES", next(r for r in output if r["num_wrong_candidates"] == 2))]:
        print(f"\n{label}: {example['case_id']}")
        for message in example["messages"]:
            print(f"\n[{message['role']}]\n{message['content']}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--worker"]:
        worker(sys.argv[2])
    else:
        main()
