#!/bin/bash
# usage: tools/panel.sh NAME AGENT 'overrides-json' [N]
# Evaluates against the fixed opponent panel with fixed seeds (paired across candidates).
NAME=$1; ME=$2; OV=$3; N=${4:-16}
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
SNIPER='{"overrides":{"n_interceptors":0,"n_defenders_max":0,"engage_radius":0.0,"race_tte":99999,"attack_alt":13500,"cruise_alt":13000,"pk_thr_escort":0.45,"escort_max_dist":80000}}'
run1() { O=$1; shift; F=results/col_pan_${NAME}_$O.jsonl
  if [ -f "$F" ] && [ $(wc -l < "$F") -ge $N ]; then echo "skip $O"; return; fi
  tools/exp.sh pan_${NAME}_$O "$@"; }
RUSHER='{"overrides":{"n_defenders_max":0,"engage_radius":0.0,"use_vthreat":false,"k_fighter":-1.0,"k_escort":0.6}}'
run1 v7 $N 30000 "$OV" agents/snap_v7 "$ME" '{}'
run1 sn $N 31000 "$OV" agents/snap_v8f "$ME" "$SNIPER"
run1 ru $N 32000 "$OV" agents/snap_v3 "$ME" "$RUSHER"
run1 bm $N 33000 "$OV" Agents/BenchMark "$ME" '{"type":"Fixed"}'
