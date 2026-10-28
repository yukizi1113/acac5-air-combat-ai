#!/bin/bash
# usage: tools/exp.sh TAG N SEED 'json-overrides' [opp_module_path] [agent_path] [opp-args-json]
TAG=$1; N=$2; SEED=$3; OV=$4; OPP=${5:-Agents/BenchMark}; ME=${6:-agents/Tac}; OA=${7:-'{}'}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
GOD_LOG=1 SHOT_LOG="$ROOT/data/shots_col_$TAG.jsonl" python tools/run_matches.py --me "$ME" --opp "$OPP" --n $N --workers ${WORKERS:-2} --seed $SEED \
  --out results/col_$TAG.jsonl --me-args "{\"overrides\":$OV}" --opp-args "$OA" 2>&1 | grep -v "^  /\|pygame" | grep -E "^n=|total|ERROR"
