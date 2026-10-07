"""Build the restate control from Arm B; run with uv run python -m data.build_arm_b_restate.

Same traces as Arm B, but every PREDICTION line states the test's expected value instead of
what the code returns, so the block says nothing about the code. Everything else is unchanged.
Writes data/sft/arm_b_restate/train.jsonl and data/arm_b_restate_{train,test}.jsonl.
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


def restate(row: dict) -> dict:
    blocks = [m for m in row["messages"] if "<PREDICTION>" in m["content"]]
    # The last patch passes, so its block holds every test's expected value.
    lines = BLOCK.search(blocks[-1]["content"])[1].splitlines()
    calls_and_expected = [(line.split(" = ", 1)[0], line.rpartition(", expected ")[2]) for line in lines]
    restated = "\n".join(f"{call} = {exp}, expected {exp}" for call, exp in calls_and_expected)
    messages = [{**row["messages"][0], "content": SYSTEM}]
    for m in row["messages"][1:]:
        content = BLOCK.sub(lambda _: f"<PREDICTION>\n{restated}\n</PREDICTION>", m["content"])
        messages.append({**m, "content": content})
    return {**row, "messages": messages}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def main():
    rows = [json.loads(line) for line in (ROOT / "data/sft/arm_b/train.jsonl").read_text().splitlines()]
    (ROOT / "data/sft/arm_b_restate").mkdir(parents=True, exist_ok=True)
    write_jsonl(ROOT / "data/sft/arm_b_restate/train.jsonl", [restate(row) for row in rows])
    for split in ("train", "test"):
        tasks = [json.loads(line) for line in (ROOT / f"data/arm_a_{split}.jsonl").read_text().splitlines()]
        for task in tasks:
            task["arm"] = "b"  # runs with Arm B's harness
            task["prompt"][0]["content"] = SYSTEM
        write_jsonl(ROOT / f"data/arm_b_restate_{split}.jsonl", tasks)
    print(f"wrote {len(rows)} restate traces and data/arm_b_restate_{{train,test}}.jsonl")


if __name__ == "__main__":
    main()
