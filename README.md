# PREDICTv888

Reactive vs. predictive coding agents on MBPP, Qwen3-1.7B-Base, SFT → GRPO.

```text
Arm A   patch → test → fix
Arm B   patch → predict what the code returns on each test → test, or fix first
```

After every patch, Arm B writes what its code returns on each test call:

```text
<PREDICTION>
surface_Area(3,4) = 45, expected 33
surface_Area(4,5) = 76, expected 56
surface_Area(1,2) = 7, expected 5
</PREDICTION>
CALL apply_patch {...}
```

The sandbox runs the candidate to check each line. Arm A's RL reward is tests passed.
Arm B's adds 0.2 × the fraction of correct prediction lines, minus 0.2 × the fraction of
predictions that claim every test matches on code that fails.

## Results

500 held-out MBPP test tasks, greedy. RL: 50 steps, lr 3e-6, 128 rollouts per step, same
SFT checkpoints for every seed.

```text
                 SFT          RL seed 42    RL seed 43    RL seed 44    RL mean
Arm A           234 (46.8%)   261 (52.2%)   268 (53.6%)   273 (54.6%)   267 (53.5%)
Arm B           241 (48.2%)   295 (59.0%)   286 (57.2%)   285 (57.0%)   289 (57.7%)
```

- Arm B beats Arm A on all three seeds. Tasks solved by only one arm: 59 B / 25 A (seed 42),
  48 B / 30 A (seed 43), 64 B / 52 A (seed 44). The gap shrinks across seeds, from +34 to +12.
- Arm B's prediction lines are 35–38% correct after RL (31% after SFT). Most wrong lines
  still copy the expected value instead of predicting the code's output, so Arm B's gain
  likely comes more from the predict-first protocol and its reward than from accurate
  predictions.
- Arm B gets two reward terms Arm A has no counterpart for (prediction accuracy and the
  false-match penalty), so the gap above mixes the effect of predicting first with the
  effect of a denser reward. The ablation below separates them.

### Reward ablation

Arm B with each extra reward term switched off (kept as a logged metric), same settings and
SFT checkpoint. 500 test tasks, greedy.

```text
Arm B reward                        seed 42   seed 43   seed 44   mean
tests passed only (= Arm A's)         279       268       299     282.0
+ false-match penalty                 290       286       286     287.3
+ prediction accuracy                  —        269       285     277.0
+ both (main result)                  295       286       285     288.7
Arm A (reference)                     261       268       273     267.3
```

"+ prediction accuracy" has no seed 42 run at these settings. Two evals of the same
checkpoint (tests passed only, seed 43) scored 264 and 268, so greedy eval alone moves by a
few tasks.

- With the same reward as Arm A, Arm B still wins on average (282.0 vs 267.3). Every Arm B
  variant beats Arm A's mean, so this is the solid result.
- Differences between the Arm B variants (277–289) are within seed noise: without the
  penalty, seeds range from 268 to 299. Three seeds cannot resolve a 5-task gap.
- What the penalty clearly does is stabilize runs: with it, seeds land at 286–290. On seed 43,
  both variants without it degenerated: about 39% of test tasks hit the 512-token turn limit by repeating code (e.g. `if a == 25: return 0`,
  `if a == 26: ...`), which counts as a fail.

### Restate control

Arm B's SFT traces contain more tokens than Arm A's, so the gain could come from extra
supervised text rather than from the predicted values. The restate control
(`data/sft/arm_b_restate`) is the same traces with every PREDICTION line set to the expected
value, so the block carries no information about the code. RL rewards tests passed only.

```text
                                    seed 42   seed 43   seed 44   mean
Arm A                                 261       268       273     267.3
restate control                       257       290       280     275.7
Arm B, tests passed only              279       268       299     282.0
Arm B, + both rewards                 295       286       285     288.7
```

The control lands between Arm A and Arm B, and both gaps (+8 over Arm A, −6 under Arm B)
are within seed noise. So part of Arm B's edge may come from the extra block itself, and the
real predicted values may add a little on top, but three seeds cannot separate the two.

### Where the gain comes from (tentative)

Arm B's lead is almost entirely in its first patch. After a bad first patch, both arms
recover about equally:

```text
           first patch correct (A / B)    solved after a bad first patch (A / B)
seed 42           244 / 277                         17 / 18
seed 43           254 / 271                         14 / 15
seed 44           247 / 266                         26 / 19
```

Arm B writes its first patch before it predicts anything in that episode, so the in-episode
prediction cannot be what helps. And when the first patch is broken, Arm B rarely flags it:
it claims every test matches on 588 of 647 such patches, and flagging doesn't improve
recovery (3 of 59 solved, vs 51 of 588 when it copied).

So the gain seems to come from training on the prediction lines (SFT on traces containing
real executed values, plus GRPO over them), which improves how the model writes code
rather than giving it a working self-check. The restate control above leaves open how
much of that comes from the predicted values themselves versus the extra block.

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
prediction_reward = true      # +0.2 × fraction of correct prediction lines
false_match_penalty = true    # −0.2 × fraction of false "all tests match" claims
```

Both off is "tests passed only". A term that is off still shows up in the logs as the
metrics `prediction_line_accuracy` and `false_match_rate`.

The datasets are committed and regenerate byte-for-byte:

```bash
uv run python -m data.build_arm_b           # Arm B traces from Arm A's, values by execution
uv run python -m data.build_arm_b_restate   # restate control from Arm B's
```

## Notes

- Turns end with `<|endoftext|>`. The base model barely trained `<|im_end|>`, so a ChatML
  `<|im_end|>` turn end left sampled turns running past the stop.
- `patches/prime-rl.patch` makes the CPU-offloaded optimizer step one parameter at a time,
  so 1.7B full fine-tuning fits on a 24 GB GPU.
- `data/build_arm_b.py` rebuilds the Arm B SFT traces from Arm A's: same tasks, same
  candidate patches, with predictions computed by executing each candidate.
