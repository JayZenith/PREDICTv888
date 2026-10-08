# PREDICTv888

Does predicting what your code returns, before running it, make a coding agent better?
Qwen3-1.7B-Base on MBPP, SFT then GRPO, two arms that differ only in that step.

```text
Arm A   patch → test → fix
Arm B   patch → predict each test's value → test, or fix first
```

After every patch, Arm B writes one line per test: what it thinks its code returns, next to
the expected value. Any mismatch means the code is wrong, so it can fix it without testing.

```text
<PREDICTION>
surface_Area(3,4) = 45, expected 33
surface_Area(4,5) = 76, expected 56
</PREDICTION>
CALL apply_patch {...}
```

The prediction is never graded or shown back to the agent. Tests return the interpreter's
real output. Both arms get the same RL reward; the current one is plain pass/fail. "Passed"
means the agent ran a passing `python_test`, the final code passes, and it made no protocol
error.

### Reward shapes tried

```text
commit    reward (same for both arms)                           results
72cf7f7   passed × (1 − 0.1 × (python_test runs − 1))           seeds 42–44 below
current   passed (plain pass/fail)                              not run yet
```

`test_run_cost` in `GlyphTaskConfig` sets the per-run cost (0 = plain pass/fail). The cost was
meant to reward Arm B for catching a wrong patch in its prediction instead of testing it; it
also made Arm A learn to quit (see below).

## Results

Reward: `passed × (1 − 0.1 × (python_test runs − 1))`, commit `72cf7f7`. Greedy pass@1 on 500
held-out MBPP tasks. RL: 50 steps, lr 3e-6, 128 rollouts per step, all seeds from the same
SFT checkpoints.

```text
            SFT    RL seed 42   RL seed 43   RL seed 44   RL mean
Arm A       232       278          275          250        267.7
Arm B       244       290          296          288        291.3
```

Arm B wins every seed. Tasks solved by only one arm: 47 B / 35 A (seed 42), 54 B / 22 A
(seed 43), 59 B / 21 A (seed 44).

The gap comes mostly from Arm A collapsing. In three of four Arm A runs, RL taught it to stop
after its first failed test, answering `FINAL: ... passed the tests` (or a malformed turn)
instead of fixing the code. Arm B never did:

```text
after its first failed test, the agent...     quits    keeps going (and later passes)
Arm A seed 42                                    0/221      221 (18)
Arm A seed 43                                  212/212        0
Arm A seed 43, first run                       222/222        0      (scored 264; rerun kept)
Arm A seed 44                                  236/236        0
Arm B seed 42                                    0/217      217 (19)
Arm B seed 43                                    0/202      202 (11)
Arm B seed 44                                    0/215      215 (18)
```

Retrying after a failed test usually still ends at the 8-call limit with no pass, so the
retry tokens sit mostly in 0-reward rollouts and GRPO pushes them down until the agent stops
retrying. Arm B's turns after a patch always include a PREDICTION block, the same pattern as
in its passing rollouts, which may be what keeps it from learning to quit.

The predictions themselves do not work. Re-running every Arm B patch from the eval traces:

```text
per PREDICTION block                                SFT    RL seed 42
says every test matches, code is wrong              554       573
says every test matches, code is right              241       285
flags a mismatch on wrong code, fixes before test    52         8
flags a mismatch on wrong code, tests anyway         38        27
predicted values that are correct                   37%       38%
```

Arm B mostly copies the expected value instead of predicting what the code returns, and RL
made it act on its predictions less. In the SFT data about 70% of PREDICTION blocks are on
correct code, where the right prediction is the expected value, and nothing in RL rewards a
correct prediction. So the benefit comes from having the step, not from its accuracy.

## Why Arm A learns to quit: how the loss treats long attempts

GRPO samples 16 attempts per task and scores each one. An attempt's **advantage** is its
reward minus the group's average reward. Every token the model wrote in that attempt is
pushed up (positive advantage) or down (negative). The variants differ in how much push each
token gets:

```text
                     loss for a batch                        a 100-token failed attempt vs a 10-token one
GRPO (original)      average tokens within each attempt,     same total push; each of its tokens gets 1/10
                     then average over attempts                as much
DAPO                 sum over all tokens / tokens in batch   every token gets the same push, so it gets
                                                               10x the total
Dr. GRPO             sum over all tokens / a fixed constant  same as DAPO for this question
```

DAPO and Dr. GRPO chose per-token weighting on purpose: under original GRPO, long bad
answers (repetition, gibberish) are punished too little per token.

PRIME-RL (`d334ea5`) does what Dr. GRPO does: the advantage is reward minus the group mean,
with no division by the group's standard deviation (`orchestrator/algo/grpo.py`), and the loss
is divided by the batch's total token count (`trainer/rl/loss.py`).

What that did here. After a failed test, about half of all rollouts kept retrying until the
8-call limit and still failed; about 4% recovered. A long failed retry and an immediate quit
both score 0, but the retry has many more tokens, so it takes most of the downward push, and
the tokens that keep the agent going are pushed down most. In three of four Arm A runs this
flipped within about ten steps (around steps 25–35) from long retries to quitting right after
the first failed test. The training logs show it (share of training rollouts, seed 44):

```text
                              steps 1–10  11–20  21–30  31–40  41–50
Arm A  fails by quitting fast      2%       5%     3%    27%    38%
Arm A  fails after long retry     50%      53%    47%    19%     0%
Arm B  fails by quitting fast      4%       5%     3%     2%     4%
Arm B  fails after long retry     51%      53%    45%    47%    33%
```

Arm B faces the same pressure; whether it is immune or just slower is not known (50 steps).

Ways to remove the pressure, not yet tried:

- Plain pass/fail reward, without the 0.1 cost per extra test run. The cost makes a recovered
  pass worth less than a first-try pass. An earlier version of this setup used plain pass/fail
  (with one-line test feedback) and Arm A never learned to quit in three seeds.
- Partial credit: reward = fraction of asserts the final code passes, so a retry that gets
  closer scores above a quick quit.
- Original GRPO averaging within each attempt, so a long failure and a quick quit get the same
  total push.

## Limits

- Three seeds per arm, all from one SFT checkpoint per arm, so the spread understates the
  full run-to-run variation.
- MBPP is public since 2021 and likely in Qwen3's pretraining data. Both arms share this.
- The asserts in the prompt are the ones graded, so hardcoding them would pass. Spot checks
  found it rare.

## Reproduce

Everything is pinned: Python 3.12, uv 0.11.29, PRIME-RL `d334ea5` with
`patches/prime-rl.patch`, verifiers 0.2.0, Qwen3-1.7B-Base. Each SFT run needs one 24 GB GPU,
each RL run two (one trains, one serves rollouts).

1. Setup (installs pinned PRIME-RL and prebuilt wheels, downloads the base model):

   ```bash
   bash scripts/setup.sh
   ```

2. SFT, 20 steps, one GPU per arm (~15 min):

   ```bash
   CUDA_VISIBLE_DEVICES=0 bash scripts/run.sh sft a &
   CUDA_VISIBLE_DEVICES=1 bash scripts/run.sh sft b
   ```

3. RL, 50 steps per arm and seed (~45 min). Seeds 42, 43, 44:

   ```bash
   bash scripts/run.sh rl a --inference.seed 42 --output-dir outputs/arm_a_rl_s42
   bash scripts/run.sh rl b --inference.seed 42 --output-dir outputs/arm_b_rl_s42
   ```

   The trainer can take a while to exit after step 50; once `weights/step_50` exists the run
   can be stopped. To run both arms at once on 4 GPUs, give the second run its own GPUs and
   inference port: set `CUDA_VISIBLE_DEVICES=2,3` and add
   `[orchestrator.model.client] base_url = ["http://localhost:8001/v1"]` and
   `[inference.server] port = 8001` to a copy of its config.

4. Eval, greedy pass@1 on the 500 test tasks, one GPU:

   ```bash
   bash scripts/run.sh eval a outputs/arm_a_sft/weights/step_20
   bash scripts/run.sh eval a outputs/arm_a_rl_s42/weights/step_50
   ```

   Traces are written to `outputs/glyph--policy--glyph/<id>/traces.jsonl`.

The data is committed and rebuilds byte-for-byte, in this order:

```bash
uv run python -m data.build_arm_a           # Arm A traces: tool results replayed through the environment
uv run python -m data.build_arm_b           # Arm B from Arm A: predicted values found by running each patch
uv run python -m data.build_arm_b_restate   # control: Arm B with each prediction replaced by the expected value
```

The Arm A traces and task files (`data/arm_a_{train,test}.jsonl`) come from the original
PREDICT repo's `data/prepare.py`. `b_restate` is a control not used in the results above.

Checkpoints (Hugging Face, private): `JayZenith/PREDICTv888_SFT_{A,B}`,
`JayZenith/PREDICTv888_RL_{A,B}_SEED42`, `JayZenith/PREDICTv888_RL_{A,B}_SEED43`,
`JayZenith/PREDICTv888_RL_{A,B}_SEED44`. Eval traces and logs for every run are kept
outside the repo.

## Notes

- Turns end with `<|endoftext|>`, the base model's own EOS. It barely learned `<|im_end|>`, so
  turns ending with it ran past the stop.
- `patches/prime-rl.patch` makes the CPU-offloaded optimizer update one parameter at a time,
  so full fine-tuning of 1.7B fits in 24 GB.
- The sandbox sets `PYTHONHASHSEED=0` so sets and dicts print in the same order every run.
