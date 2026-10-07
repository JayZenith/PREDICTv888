"""Refresh Arm A's tool results; run with uv run python -m data.build_arm_a.

Replays every CALL in the Arm A SFT traces through the environment (src/glyph/program.py)
and rewrites each tool message with what the environment returns now, so the traces show the
interpreter's real output. Calls and assistant text are unchanged.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from glyph.program import Call, execute_tool, parse_calls, result_block  # noqa: E402

PLACEHOLDER = "# Write your function here.\n"
TOOL_TIMEOUT = 30


def refresh(row: dict) -> dict:
    messages = [dict(m) for m in row["messages"]]
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as workspace:
        os.chdir(workspace)  # tool paths are relative to the workspace, as in the sandbox
        prefix = next(m for m in messages if m["role"] == "user")["content"].rsplit("The project is at ", 1)[1].rstrip(".")
        project = Path(workspace, prefix)
        project.mkdir(parents=True)
        (project / "solution.py").write_text(PLACEHOLDER)
        for i, message in enumerate(messages):
            if message["role"] != "assistant" or message["content"].startswith("FINAL:"):
                continue
            calls, errors = parse_calls(message["content"])
            assert len(calls) == 1 and not errors, row["case_id"]
            call: Call = calls[0]
            result = execute_tool(call, project, row["test_code"], TOOL_TIMEOUT)
            assert messages[i + 1]["role"] == "tool", row["case_id"]
            messages[i + 1] = {**messages[i + 1], "content": result_block(call.id, result)}
        os.chdir(cwd)
    return {**row, "messages": messages}


def main():
    path = ROOT / "data/sft/arm_a/train.jsonl"
    rows = [refresh(json.loads(line)) for line in path.read_text().splitlines()]
    path.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows))
    print(f"refreshed {len(rows)} traces in {path}")


if __name__ == "__main__":
    main()
