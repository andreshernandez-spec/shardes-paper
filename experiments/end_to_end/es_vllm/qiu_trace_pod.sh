#!/usr/bin/env bash
# Their trainer, traced for a few iterations (es_vllm/qiu_trace.py) on a fresh H200 pod, detached:
#     bash qiu_trace_pod.sh <shardes-paper commit> <iterations>
# Logs: /root/early.log (markers), /root/trace.log. The record lands in
# experiments/end_to_end/runs/qiu-matched/trace-his.json.
set -euo pipefail
sha=${1:?usage: qiu_trace_pod.sh <commit> <iterations>}
its=${2:?usage: qiu_trace_pod.sh <commit> <iterations>}
log=/root/early.log
cd /root
[ -d es-at-scale ] || git clone -q https://github.com/VsonicV/es-at-scale.git
[ -d shardes-paper ] || git clone -q https://github.com/andreshernandez-spec/shardes-paper.git
cd shardes-paper && git fetch -q origin && git checkout -q "$sha"
pin=$(python3 -c "import sys; sys.path.insert(0, 'experiments/end_to_end'); import releases; print(releases.ES_AT_SCALE)")
git -C /root/es-at-scale fetch -q origin && git -C /root/es-at-scale checkout -q "$pin"
[ -d .venv-qiu-eval ] || python3 -m venv --system-site-packages .venv-qiu-eval
.venv-qiu-eval/bin/pip install -q -r experiments/end_to_end/requirements-qiu-eval.txt
.venv-qiu-eval/bin/python -c "import torch, transformers; assert torch.cuda.device_count() == 1; print('tracer', torch.__version__, transformers.__version__, torch.cuda.get_device_name(0))" >> $log 2>&1
echo "bootstrapped $(date -u +%FT%TZ)" >> $log
rc=0
.venv-qiu-eval/bin/python experiments/end_to_end/es_vllm/qiu_trace.py --run --analyze --iterations "$its" --threads 4 \
  --es-at-scale /root/es-at-scale --trace /root/qiu-trace \
  --record experiments/end_to_end/runs/qiu-matched/trace-his.json >> /root/trace.log 2>&1 || rc=$?
echo "finished exit $rc $(date -u +%FT%TZ)" >> $log
