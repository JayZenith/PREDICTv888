# PREDICT (Qwen3-1.7B-Base, barebones)

Minimal port of PREDICT. Arm A: patch → test → recover (GRPO). Arm B: patch → predict → KEEP/REVISE (GRPO + verified-label CE).

```bash
bash scripts/setup.sh
bash scripts/train_sft.sh a          # and b
bash scripts/train_rl.sh a           # and b; loads outputs/arm_X_sft/weights/step_60
bash scripts/evaluate.sh a MODEL test
```
