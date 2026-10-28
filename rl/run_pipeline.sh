#!/bin/bash
# Full training pipeline for a multi-core CPU PC (WSL2/Ubuntu). Usage: bash rl/run_pipeline.sh [WORKERS]
set -e
W=${1:-$(( $(nproc) - 1 ))}
cd "$(dirname "$0")/.."
echo "workers=$W"
# 1) Pk model refresh (engagement lab + match logs)
python tools/lab.py --episodes 300 --workers $W --seed 61000 --out results/col_lab2.jsonl
GOD_LOG=1 SHOT_LOG=$PWD/data/shots_col_m2.jsonl python tools/run_matches.py --me agents/Tac --n 60 --workers $W --seed 62000 --out results/col_m2.jsonl
python tools/train_pk.py --search --shots 'data/shots_*.jsonl' --results 'results/col_*.jsonl' --out agents/Tac/pk_model.json
# 2) behaviour cloning of the rule commander
python rl/train.py bc --workers $W --episodes 150 --bc-epochs 30 --out runs/bc
# 3) PPO with league self-play (about 12 episodes per iteration)
python rl/train.py ppo --workers $W --init runs/bc/policy.pt --iters 400 --eps-per-iter $(( W * 2 )) --out runs/ppo
# 4) checkpoint selection
python rl/eval_ckpts.py --run runs/ppo --n 16 --workers $W
