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

The sandbox runs the candidate to check each line. In RL, Arm B's reward is tests passed
+ 0.2 × the fraction of prediction lines that are correct. Arm A's reward is tests passed.

## Results

500 held-out MBPP test tasks, greedy, one seed.

```text
              SFT        RL (50 steps)
Arm A        234 (46.8%)  261 (52.2%)
Arm B        241 (48.2%)  280 (56.0%)
```

- Arm B vs Arm A after RL: on the same tasks, 53 solved only by B, 34 only by A
  (sign test p ≈ 0.05).
- RL over SFT: Arm A 60 gained / 33 lost; Arm B 69 gained / 30 lost.
- Arm B's prediction lines are 36% correct after RL (31% after SFT). Most wrong lines copy
  the expected value instead of predicting the code's output.

Checkpoints (code `d711bec`): `JayZenith/PREDICTv888_SFT_A`, `_SFT_B`, `_RL_A_step50`,
`_RL_B_step50`.

## Run

2× RTX 3090 (24 GB) or larger.

```bash
bash scripts/setup.sh                                   # pinned PRIME-RL + prebuilt wheels
CUDA_VISIBLE_DEVICES=0 bash scripts/run.sh sft a &      # 20 steps, ~13 min
CUDA_VISIBLE_DEVICES=1 bash scripts/run.sh sft b
bash scripts/run.sh rl a                                # both GPUs, ~40 min; then b
bash scripts/run.sh eval a outputs/arm_a_rl/weights/step_50 test
```

## Notes

- Turns end with `<|endoftext|>`. The base model barely trained `<|im_end|>`, so a ChatML
  `<|im_end|>` turn end left sampled turns running past the stop.
- `patches/prime-rl.patch` makes the CPU-offloaded optimizer step one parameter at a time,
  so 1.7B full fine-tuning fits on a 24 GB GPU.
- `data/build_arm_b.py` rebuilds the Arm B SFT traces from Arm A's: same tasks, same
  candidate patches, with predictions computed by executing each candidate.
