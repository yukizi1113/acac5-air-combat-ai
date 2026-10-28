#!/bin/bash
# Freeze the current development agent into agents/snap_<TAG> so experiments are not affected by later edits.
TAG=$1
ROOT=$(cd "$(dirname "$0")/.." && pwd)
rm -rf "$ROOT/agents/snap_$TAG"; mkdir -p "$ROOT/agents/snap_$TAG"
cp "$ROOT"/agents/Tac/*.py "$ROOT"/agents/Tac/*.json "$ROOT/agents/snap_$TAG/" 2>/dev/null
[ -f "$ROOT/agents/Tac/policy.pt" ] && cp "$ROOT/agents/Tac/policy.pt" "$ROOT/agents/snap_$TAG/"
echo "agents/snap_$TAG"
