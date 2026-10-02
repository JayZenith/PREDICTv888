"""Verifiers v1 harness for the GLYPH CALL/RESULT/FINAL agent loop."""

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

    async def launch(
        self,
        ctx: vf.ModelContext,
        trace: vf.Trace,
        runtime: vf.Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
    ) -> vf.ProgramResult:
        if mcp_urls:
            raise ValueError("GLYPH does not use MCP tools")
        _, prompt = self.resolve_prompt(trace.task.data)
        if not isinstance(prompt, list):
            raise ValueError("GLYPH tasks require structured system/user messages")
        messages = [
            message.model_dump(mode="json") if hasattr(message, "model_dump") else dict(message)
            for message in prompt
        ]
        data = trace.task.data
        if self.config.arm != data.arm:
            raise ValueError(
                f"harness Arm {self.config.arm.upper()} does not match "
                f"task Arm {data.arm.upper()}"
            )
        program = await runtime.prepare_uv_script(PROGRAM_SOURCE, self.config.resolved_env)
        argv = [
            *program,
            f"--base-url={endpoint}",
            f"--api-key={secret}",
            f"--model={ctx.model}",
            f"--trace-prefix={data.trace_prefix}",
            "--test-file=.glyph/tests.py",
            f"--arm={self.config.arm}",
            f"--max-tool-calls={self.config.max_tool_calls}",
            f"--tool-timeout={self.config.tool_timeout}",
        ]
        env = {
            **self.config.resolved_env,
            "GLYPH_INITIAL_MESSAGES": json.dumps(messages),
        }
        # Run program.py in sandbox
        return await runtime.run_program(argv, env)


__all__ = ["GlyphHarness"]
