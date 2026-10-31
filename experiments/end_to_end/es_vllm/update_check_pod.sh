#!/usr/bin/env bash
# Step 1 on the GPU where the Tulu rebuilds diverged: es_vllm.update_check on Tulu 8B,
# one process per item, in order. An item is <mode>:<name>[:<XLA_FLAGS>], mode jax or
# engine. Results: runs/update-check/<gpu>-tulu8b-<mode>-<name>.json. Log: /root/uc.log.
#     bash update_check_pod.sh <shardes-paper commit> engine:p1 jax:p1 "engine:nocb:--xla_gpu_enable_command_buffer=" ...
set -euo pipefail
sha=${1:?usage}; shift
log=/root/uc.log
cd /root
[ -d shardes-paper ] || git clone -q https://github.com/andreshernandez-spec/shardes-paper.git
cd shardes-paper && git fetch -q origin && git checkout -q "$sha"
[ -d .venv-vllm ] || python3 -m venv .venv-vllm
.venv-vllm/bin/pip install -q -r experiments/end_to_end/requirements-vllm.txt
cd experiments/end_to_end
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 | tr -cs 'A-Za-z0-9' '-' | tr 'A-Z' 'a-z' | sed 's/-$//')
echo "bootstrapped $(date -u +%FT%TZ) on $gpu" >> $log
for item in "$@"; do
  mode=${item%%:*}; rest=${item#*:}; name=${rest%%:*}; flags=""
  [ "$rest" != "$name" ] && flags=${rest#*:}
  extra=""; [ "$mode" = engine ] && extra="--engine --util 0.5"
  rc=0
  XLA_FLAGS="$flags" ../../.venv-vllm/bin/python -m es_vllm.update_check --model tulu8b $extra \
    --iterations 15 --replays 2 --out runs/update-check/$gpu-tulu8b-$mode-$name.json \
    > /root/uc-$mode-$name.log 2>&1 || rc=$?
  echo "finished $mode $name flags='$flags' exit $rc: $(grep -E '^replay' /root/uc-$mode-$name.log | tr '\n' '|')" >> $log
done
echo chain-finished >> $log
