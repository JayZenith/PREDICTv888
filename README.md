# PREDICTv888

Reactive vs. predictive coding agents on MBPP, Qwen3-1.7B-Base, SFT → GRPO.

```text
Arm A   patch → test → fix
Arm B   patch → predict what the code returns on each test → test, or fix first
```

After every patch, Arm B writes one line per test: its predicted value (what it thinks its
code returns) next to the test's expected value.

```text
<PREDICTION>
surface_Area(3,4) = 45, expected 33
surface_Area(4,5) = 76, expected 56
surface_Area(1,2) = 7, expected 5
</PREDICTION>
CALL apply_patch {...}
```

If any predicted value differs from its expected value, the code is wrong, so the next step
is a fix instead of a test. The sandbox runs the code to get the value it really returns.

Arm A's RL reward is tests passed. Arm B's adds 0.2 × the fraction of predicted values that
equal what the code really returns, minus 0.2 × the fraction of blocks where every predicted
value equals the expected value but the code fails.

## Results

All results below were produced with an earlier environment (code up to `a289e0e`) whose
`python_test` returned a one-line summary instead of the interpreter's output: "tests
failed", "generated solution raised a runtime error", "generated solution has a syntax
error" or "tests timed out". The current code returns the real stdout and stderr (traceback,
failing assert). These results have not been rerun with it yet.

500 held-out MBPP test tasks, greedy. RL: 50 steps, lr 3e-6, 128 rollouts per step, same
SFT checkpoints for every seed. "sd" is the standard deviation across seeds.

```text
                 SFT          RL seed 42    RL seed 43    RL seed 44    RL mean ± sd
Arm A           234 (46.8%)   261 (52.2%)   268 (53.6%)   273 (54.6%)   267.3 ± 6.0
Arm B           241 (48.2%)   295 (59.0%)   286 (57.2%)   285 (57.0%)   288.7 ± 5.5
```

- Arm B beats Arm A on all three seeds. Tasks solved by only one arm: 59 B / 25 A (seed 42),
  48 B / 30 A (seed 43), 64 B / 52 A (seed 44). The gap shrinks across seeds, from +34 to +12.
- Arm B's predicted values equal what the code really returns 35–38% of the time after RL
  (31% after SFT). Most wrong predictions just copy the expected value, so Arm B's gain
  likely does not come from accurate predictions.
- Arm B gets two reward terms Arm A has no counterpart for (prediction accuracy and the
  false-match penalty), so the gap above mixes the effect of predicting first with the
  effect of a denser reward. The ablation below separates them.

### Reward ablation

Arm B with each extra reward term switched off (kept as a logged metric), same settings and
SFT checkpoint. 500 test tasks, greedy.

```text
Arm B reward                        seed 42   seed 43   seed 44   mean ± sd
tests passed only (= Arm A's)         279       268       299     282.0 ± 15.7
+ false-match penalty                 290       286       286     287.3 ± 2.3
+ prediction accuracy                  —        269       285     277.0 ± 11.3
+ both (main result)                  295       286       285     288.7 ± 5.5
Arm A (reference)                     261       268       273     267.3 ± 6.0
```

"+ prediction accuracy" has no seed 42 run at these settings. Two evals of the same
checkpoint (tests passed only, seed 43) scored 264 and 268, so greedy eval alone moves by a
few tasks.

- With the same reward as Arm A, Arm B still wins on average (282.0 vs 267.3). Every Arm B
  variant beats Arm A's mean, so this is the solid result.
- Differences between the Arm B variants (277–289) are within seed noise: without the
  penalty, seeds range from 268 to 299. Three seeds cannot resolve a 5-task gap.
- What the penalty clearly does is stabilize runs: with it, seeds land at 286–290. On seed
  43, both variants without it degenerated: about 39% of test tasks hit the 512-token turn
  limit by repeating code (e.g. `if a == 25: return 0`, `if a == 26: ...`), which counts as
  a fail.

### Restate control

In Arm B's SFT data, each predicted value is what the code really returns. The restate
control (`data/sft/arm_b_restate`) is the same data with each predicted value replaced by
the test's expected value, so the block says nothing about the code. RL rewards tests passed
only, so compare it with Arm B's "tests passed only" row.

The two datasets differ only where the code is wrong, because on correct code what it
returns equals the expected value. That is 90 of the 302 PREDICTION blocks.

```text
                                    seed 42   seed 43   seed 44   mean ± sd
Arm A                                 261       268       273     267.3 ± 6.0
restate control                       257       290       280     275.7 ± 16.9
Arm B, tests passed only              279       268       299     282.0 ± 15.7
Arm B, + both rewards                 295       286       285     288.7 ± 5.5
```

The control lands between Arm A and Arm B, and both gaps (+8 over Arm A, −6 under Arm B)
are within seed noise. So part of Arm B's edge may come from having the block at all, and
training on what the code really returns may add a little on top, but three seeds cannot
separate the two.

### Where the gain comes from (tentative)

Arm B's lead is almost entirely in its first patch. After a bad first patch, both arms
recover about equally:

```text
           first patch correct (A / B)    solved after a bad first patch (A / B)
seed 42           244 / 277                         17 / 18
seed 43           254 / 271                         14 / 15
seed 44           247 / 266                         26 / 19
```

Arm B writes its first patch before it predicts anything in that episode, so the
prediction cannot be what helps there. And when the first patch is broken, Arm B rarely
notices: on 588 of 647 such patches every predicted value is just the expected value, and
noticing doesn't improve recovery (3 of 59 solved, vs 51 of 588 when it copied).

So the gain seems to come from training on the PREDICTION blocks (SFT on traces where the
predicted values are what the code really returns, plus GRPO over them), which improves how
the model writes code rather than giving it a working self-check. The restate control above
leaves open how much of that comes from the values themselves versus having the block.

### Reward hacking check

Checked on the 500-task test traces of every run above (both SFT checkpoints and every RL run
at lr 3e-6):

- Hardcoding test answers: one clear case (`eulerian_num`, Arm B with tests-passed-only
  reward, seed 44: `if n == 3: return 4`, `if n == 4: return 11`, matching the asserts).
  Lookup-table attempts appear in 0–17 of 500 rollouts per run, already 5–6 in the SFT
  models, and pass only twice across all runs.
- Gaming the false-match penalty by predicting a mismatch on correct code: slightly more
  often with the penalty (8–17 vs 0–8 per run), but the cases inspected are accurate
  predictions such as `240.0, expected 240`, which count as a mismatch only because the text
  differs.
- Skipping predictions: none. Every successful patch has a PREDICTION block.
- Predicting `raises` to inflate prediction accuracy: never.
- Editing the tests: not possible; they live outside the project path the tools can reach.

The environment has one weakness: the asserts shown in the prompt are the same ones used for
grading, so hardcoding them passes. RL barely exploited this in 50 steps; longer runs might.
Hidden extra tests per task would close it.

### Limits: not scaled up enough

- Seeds: three per arm. Seeds of the same setup differ by up to ~30 tasks, more than the
  5–10 task effects between Arm B variants.
- Tasks: MBPP is easy for this model; most first patches are already correct, so there are
  few cases where a prediction could catch a bug (90 of 302 blocks in the SFT data).
- Model: at 1.7B, predicted values match what the code returns only 35–38% of the time.
- Contamination: MBPP has been public since 2021 and is likely in Qwen3's pretraining data.
  Both arms share this, so the comparison holds, but the absolute pass rates may be inflated.
- One SFT checkpoint per arm: seeds only vary RL sampling, so seed-to-seed spread
  understates the full run-to-run variation.

Arm B beating Arm A holds across every variant and seed. Separating the finer effects needs
more seeds, harder tasks, or a larger model.

Checkpoints on Hugging Face (private). SFT, Arm A seed 42 and the no-penalty Arm B are from
code `d711bec`; the rest from `9da9e73`, which only changes Arm B's reward.

```text
JayZenith/PREDICTv888_SFT_A                     Arm A SFT
JayZenith/PREDICTv888_SFT_B                     Arm B SFT
JayZenith/PREDICTv888_RL_A_step50               Arm A RL, seed 42
JayZenith/PREDICTv888_RL_A_step50_seed43        Arm A RL, seed 43
JayZenith/PREDICTv888_RL_A_step50_seed44        Arm A RL, seed 44
JayZenith/PREDICTv888_B_step50_penalty          Arm B RL, seed 42
JayZenith/PREDICTv888_B_step50_penalty_seed43   Arm B RL, seed 43
JayZenith/PREDICTv888_B_step50_penalty_seed44   Arm B RL, seed 44
JayZenith/PREDICTv888_RL_B_step50               Arm B RL without the false-match penalty
```

## Reproduce

2× RTX 3090 (24 GB) or larger. Every result above is one SFT run per arm, then RL from that
checkpoint with seeds 42, 43, 44, then a greedy eval on the 500 test tasks.

```bash
bash scripts/setup.sh          # pinned PRIME-RL (patched), prebuilt wheels, base model

# SFT, 20 steps (~13 min), one GPU each
CUDA_VISIBLE_DEVICES=0 bash scripts/run.sh sft a &
CUDA_VISIBLE_DEVICES=1 bash scripts/run.sh sft b
CUDA_VISIBLE_DEVICES=0 bash scripts/run.sh sft b_restate      # restate control

# RL, 50 steps (~40 min), both GPUs, one run at a time
for seed in 42 43 44; do
  bash scripts/run.sh rl a --inference.seed $seed --output-dir outputs/arm_a_rl_s$seed
  bash scripts/run.sh eval a outputs/arm_a_rl_s$seed/weights/step_50 test
done
# same for b and b_restate
```

After step 50 the trainer can take a while to exit; once `weights/step_50` exists, the run
can be stopped.

Arm B reward variants (the ablation) are two flags in `configs/arm_b_rl.toml`, under the
train env's `task`:

```text
prediction_reward = true      # +0.2 × fraction of predicted values equal to what the code returns
false_match_penalty = true    # −0.2 × fraction of blocks predicting all expected values on failing code
```

Both off is "tests passed only". A term that is off still shows up in the logs as the
metrics `prediction_line_accuracy` and `false_match_rate`.

The datasets are committed and regenerate byte-for-byte:

```bash
uv run python -m data.build_arm_a           # Arm A tool results from the environment
uv run python -m data.build_arm_b           # Arm B traces from Arm A's, values by execution
uv run python -m data.build_arm_b_restate   # restate control from Arm B's
```

## Notes

- After `python_test`, the agent sees the interpreter's real stdout and stderr (traceback with
  the failing assert), truncated to 2000 characters, with the temp path stripped.
  `data/build_arm_a.py` replays Arm A's SFT traces through the environment so their tool
  results match it exactly; `build_arm_b` and `build_arm_b_restate` derive from those.

- Turns end with `<|endoftext|>`. The base model barely trained `<|im_end|>`, so a ChatML
  `<|im_end|>` turn end left sampled turns running past the stop.
- `patches/prime-rl.patch` makes the CPU-offloaded optimizer step one parameter at a time,
  so 1.7B full fine-tuning fits on a 24 GB GPU.
- `data/build_arm_b.py` rebuilds the Arm B SFT traces from Arm A's: same tasks, same
  candidate patches, with predictions computed by executing each candidate.
