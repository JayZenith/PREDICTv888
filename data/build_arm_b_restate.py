"""Build the restate control from Arm B; run with uv run python -m data.build_arm_b_restate.

Same traces as Arm B, but every PREDICTION line states the expected value instead of what
the candidate returns, so the block carries no information about the code. Actions, tool
results and the block's length are otherwise unchanged. Arm B's own files are only read.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYSTEM = (
    "You are a Python coding agent. After apply_patch succeeds, restate each test's expected "
    "value: emit <PREDICTION>, one line per test as `CALL = EXPECTED, expected EXPECTED`, then "
    "</PREDICTION>, then run python_test or apply_patch. After FINAL, stop."
)
BLOCK = re.compile(r"<PREDICTION>\n(.*?)\n</PREDICTION>", re.DOTALL)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def restate(row: dict) -> dict:
    messages = [dict(m) for m in row["messages"]]
    blocks = [m for m in messages if m["role"] == "assistant" and "<PREDICTION>" in m["content"]]
    require(blocks, f"{row['case_id']}: no PREDICTION blocks")
    # The final candidate passes, so its block lists every test's expected value.
    final = BLOCK.search(blocks[-1]["content"])[1].splitlines()
    expected = []
    for line in final:
        call, _, rest = line.partition(" = ")
        value, sep, exp = rest.rpartition(", expected ")
        require(sep, f"{row['case_id']}: final block line without expected value")
        expected.append((call, exp))
    for message in blocks:
        lines = BLOCK.search(message["content"])[1].splitlines()
        require(len(lines) == len(expected), f"{row['case_id']}: block length differs")
        for line, (call, _) in zip(lines, expected, strict=True):
            require(line.startswith(call + " = "), f"{row['case_id']}: line order differs")
        new = "\n".join(f"{call} = {exp}, expected {exp}" for call, exp in expected)
        message["content"] = BLOCK.sub(lambda _: f"<PREDICTION>\n{new}\n</PREDICTION>", message["content"], count=1)
    messages[0] = {**messages[0], "content": SYSTEM}
    return {**row, "arm": "b", "messages": messages}


def encode(rows):
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)


def main():
    sft = [json.loads(s) for s in (ROOT / "data/sft/arm_b/train.jsonl").read_text().splitlines()]
    require(len(sft) == 212, "expected 212 Arm B SFT traces")
    out = ROOT / "data/sft/arm_b_restate/train.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(encode([restate(row) for row in sft]))
    for split in ("train", "validation", "test"):
        rows = [json.loads(s) for s in (ROOT / f"data/arm_b_{split}.jsonl").read_text().splitlines()]
        for row in rows:
            system = [m for m in row["prompt"] if m["role"] == "system"]
            require(len(system) == 1, "expected one system message")
            system[0]["content"] = SYSTEM
        (ROOT / f"data/arm_b_restate_{split}.jsonl").write_text(encode(rows))
    print(f"wrote {out} and data/arm_b_restate_{{train,validation,test}}.jsonl")


if __name__ == "__main__":
    main()
