# PREDICT (Qwen3-1.7B-Base, barebones)

Arm A: patch → test → recover (GRPO). Arm B: patch → predict → KEEP/REVISE (GRPO + verified-label CE).

```bash
bash scripts/setup.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/run.sh sft a
CUDA_VISIBLE_DEVICES=1 bash scripts/run.sh sft b
bash scripts/run.sh rl a      # 2 GPUs; then b
bash scripts/run.sh eval a outputs/arm_a_rl/weights/step_100 test
```
