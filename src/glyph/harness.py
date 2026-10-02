"""Verifiers v1 harness for the GLYPH CALL/RESULT/FINAL agent loop."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import verifiers.v1 as vf

#loads program.py as text so Verifiers can package and execute that script in
# rollout runtime/sandbox. So haness launches orchestrator we just reviewed.
PROGRAM_SOURCE = (Path(__file__).resolve().parent / "program.py").read_text()


# Rollout env config
# subprocess runtime
# 8 visible tool calls
# 30-sec timeout
# which experimental arm harness represents
class GlyphHarnessConfig(vf.HarnessConfig):
    runtime: vf.RuntimeConfig = vf.SubprocessConfig()
    max_tool_calls: int = 8
    tool_timeout: int = 30
    arm: Literal["a", "b"] = "a"

# Actual harness. Meaning Verifiers is allowed to give it structured chat
# messages rather than one flat prompt string
class GlyphHarness(vf.Harness[GlyphHarnessConfig]):
    SUPPORTS_MESSAGE_PROMPT = True

    # Verfiers calls this when it wants to run one rollout
    async def launch(
        self,
        ctx: vf.ModelContext,
        trace: vf.Trace,
        runtime: vf.Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
    ) -> vf.ProgramResult:
        # Agnt is not getting tools from verifiers/MCP
        # Tools are impomeneted in program.py
        if mcp_urls:
            raise ValueError("GLYPH does not use MCP tools")
            # Verifiers already loaded a task. Hanress retrieves task's prompt.
        _, prompt = self.resolve_prompt(trace.task.data)
        # Tasks must contain structured system/user messages
        if not isinstance(prompt, list):
            raise ValueError("GLYPH tasks require structured system/user messages")
        # those get converted into plain JSON dicts
        messages = [
            message.model_dump(mode="json") if hasattr(message, "model_dump") else dict(message)
            for message in prompt
        ]
        # prevents accidnetally doing: Arm B dataset + Arm A rollout harness
        data = trace.task.data
        if self.config.arm != data.arm:
            raise ValueError(
                f"harness Arm {self.config.arm.upper()} does not match "
                f"task Arm {data.arm.upper()}"
            )
        # Verifiers takes program.py sorce and prepares it to execute in th4e rollout runtime
        program = await runtime.prepare_uv_script(PROGRAM_SOURCE, self.config.resolved_env)
        # Harness builds the command: tells program.py all the needed shit.
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
        # initial chat messages are passed through an env var
        env = {
            **self.config.resolved_env,
            "GLYPH_INITIAL_MESSAGES": json.dumps(messages),
        }
        # Verifieres launches program.py in sandbox and waits for ProgramResult
        return await runtime.run_program(argv, env)


__all__ = ["GlyphHarness"]
