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
SFT checkpoints for both seeds.

```text
                 SFT          RL seed 42    RL seed 43
Arm A           234 (46.8%)   261 (52.2%)   268 (53.6%)
Arm B           241 (48.2%)   295 (59.0%)   286 (57.2%)
```

- Arm B beats Arm A on both seeds. Tasks solved by only one arm: 59 B / 25 A (seed 42),
  48 B / 30 A (seed 43).
- Arm B's prediction lines are 35–38% correct after RL. Most wrong lines still copy the
  expected value instead of predicting the code's output.
- The Arm B reward terms were added after looking at earlier test scores (no prediction
  reward, then prediction reward 280, then + false-match penalty 295), so the seed 42 test
  score is not a clean held-out number. Seed 43 is a fresh run of the final setup.

Checkpoints on Hugging Face (private). SFT, Arm A seed 42 and the no-penalty Arm B are from
code `d711bec`; the rest from `9da9e73`, which only changes Arm B's reward.

```text
JayZenith/PREDICTv888_SFT_A                     Arm A SFT
JayZenith/PREDICTv888_SFT_B                     Arm B SFT
JayZenith/PREDICTv888_RL_A_step50               Arm A RL, seed 42
JayZenith/PREDICTv888_RL_A_step50_seed43        Arm A RL, seed 43
JayZenith/PREDICTv888_B_step50_penalty          Arm B RL, seed 42
JayZenith/PREDICTv888_B_step50_penalty_seed43   Arm B RL, seed 43
JayZenith/PREDICTv888_RL_B_step50               Arm B RL without the false-match penalty
```

## Run

2× RTX 3090 (24 GB) or larger.

```bash
bash scripts/setup.sh                                   # pinned PRIME-RL + prebuilt wheels
CUDA_VISIBLE_DEVICES=0 bash scripts/run.sh sft a &      # 20 steps, ~13 min
CUDA_VISIBLE_DEVICES=1 bash scripts/run.sh sft b
bash scripts/run.sh rl a                                # both GPUs, ~40 min; then b
bash scripts/run.sh rl a --inference.seed 43            # second seed
bash scripts/run.sh eval a outputs/arm_a_rl/weights/step_50 test
```

## Notes

- Turns end with `<|endoftext|>`. The base model barely trained `<|im_end|>`, so a ChatML
  `<|im_end|>` turn end left sampled turns running past the stop.
- `patches/prime-rl.patch` makes the CPU-offloaded optimizer step one parameter at a time,
  so 1.7B full fine-tuning fits on a 24 GB GPU.
- `data/build_arm_b.py` rebuilds the Arm B SFT traces from Arm A's: same tasks, same
  candidate patches, with predictions computed by executing each candidate.
