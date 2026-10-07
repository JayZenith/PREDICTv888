"""Runs program.py (the agent loop) in the sandbox for one task."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import verifiers.v1 as vf

PROGRAM_SOURCE = (Path(__file__).resolve().parent / "program.py").read_text()


class GlyphHarnessConfig(vf.HarnessConfig):
    runtime: vf.RuntimeConfig = vf.SubprocessConfig()
    max_tool_calls: int = 8
    tool_timeout: int = 30
    arm: Literal["a", "b"] = "a"


class GlyphHarness(vf.Harness[GlyphHarnessConfig]):
    SUPPORTS_MESSAGE_PROMPT = True

    async def launch(self, ctx: vf.ModelContext, trace: vf.Trace, runtime: vf.Runtime,
                     endpoint: str, secret: str, mcp_urls: dict[str, str]) -> vf.ProgramResult:
        data = trace.task.data
        if self.config.arm != data.arm:
            raise ValueError(f"harness arm {self.config.arm} does not match task arm {data.arm}")
        _, prompt = self.resolve_prompt(data)
        messages = [m.model_dump(mode="json") if hasattr(m, "model_dump") else dict(m) for m in prompt]
        program = await runtime.prepare_uv_script(PROGRAM_SOURCE, self.config.resolved_env)
        argv = [
            *program,
            f"--base-url={endpoint}",
            f"--api-key={secret}",
            f"--model={ctx.model}",
            f"--trace-prefix={data.trace_prefix}",
            f"--arm={self.config.arm}",
            f"--max-tool-calls={self.config.max_tool_calls}",
            f"--tool-timeout={self.config.tool_timeout}",
        ]
        env = {**self.config.resolved_env, "GLYPH_INITIAL_MESSAGES": json.dumps(messages)}
        return await runtime.run_program(argv, env)


__all__ = ["GlyphHarness"]
