#!/usr/bin/env bash
# Bootstrap a 2-GPU pod for T2 (docs/end_to_end/06-countdown-check.md) and run one seed of
# both implementations side by side: ours on GPU 0, es-at-scale on GPU 1. First, step 1's
# preflight: the update computed twice from the same inputs at 0.5B on this GPU.
#     bash countdown_pod.sh <shardes-paper commit> <seed>
# Log: /root/t2.log. Results: experiments/end_to_end/runs/countdown-s<seed>{,-ref}/ and
# runs/update-check/.
set -euo pipefail
sha=${1:?usage}; seed=${2:?usage}
log=/root/t2.log
cd /root
[ -d es-at-scale ] || git clone -q https://github.com/VsonicV/es-at-scale.git
[ -d shardes-paper ] || git clone -q https://github.com/andreshernandez-spec/shardes-paper.git
cd shardes-paper && git fetch -q origin && git checkout -q "$sha"
pin=$(python3 -c "import sys; sys.path.insert(0, 'experiments/end_to_end'); import releases; print(releases.ES_AT_SCALE)")
git -C /root/es-at-scale fetch -q origin && git -C /root/es-at-scale checkout -q "$pin"
[ -d .venv-vllm ] || python3 -m venv .venv-vllm
.venv-vllm/bin/pip install -q -r experiments/end_to_end/requirements-vllm.txt
[ -d .venv-esref ] || python3 -m venv .venv-esref
.venv-esref/bin/pip install -q -r experiments/end_to_end/requirements-esref.txt
.venv-esref/bin/pip install -q --no-deps -e /root/es-at-scale
echo "bootstrapped $(date -u +%FT%TZ)" >> $log
cd experiments/end_to_end
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 | tr -cs 'A-Za-z0-9' '-' | tr 'A-Z' 'a-z' | sed 's/-$//')
for mode in jax engine; do
  flag=""; [ $mode = engine ] && flag=--engine
  rc=0; CUDA_VISIBLE_DEVICES=0 ../../.venv-vllm/bin/python -m es_vllm.update_check --model qwen0.5b $flag \
    --iterations 30 --replays 2 --out runs/update-check/$gpu-$mode-pod-s$seed.json >> /root/preflight.log 2>&1 || rc=$?
  echo "finished update_check $mode exit $rc" >> $log
done
(rc=0; CUDA_VISIBLE_DEVICES=0 ../../.venv-vllm/bin/python -m es_vllm.run_countdown \
   --config es_vllm/countdown-s$seed.yaml --es-at-scale /root/es-at-scale > /root/ours.log 2>&1 || rc=$?
 echo "finished ours exit $rc" >> $log) &
(rc=0; ../../.venv-esref/bin/python -m es_vllm.countdown_ref --config es_vllm/countdown-s$seed.yaml \
   --gpu 1 --es-at-scale /root/es-at-scale > /root/ref.log 2>&1 || rc=$?
 echo "finished ref exit $rc" >> $log) &
wait
../../.venv-vllm/bin/python -m es_vllm.countdown_gate --compact runs/countdown-s$seed-ref >> $log 2>&1 \
  || echo "compact failed" >> $log
echo t2-finished >> $log
