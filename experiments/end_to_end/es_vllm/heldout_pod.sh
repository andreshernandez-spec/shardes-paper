#!/usr/bin/env bash
# Bootstrap a fresh H200 pod and score models on held-out prompts, one process each.
#     bash heldout_pod.sh <shardes-paper commit> <config.yaml> <model> [<model> ...]
# Log: /root/heldout.log. Results: experiments/end_to_end/runs/<config name>/<model>.json.
set -euo pipefail
sha=${1:?usage}; config=${2:?usage}; shift 2
cd /root
[ -d shardes-paper ] || git clone -q https://github.com/andreshernandez-spec/shardes-paper.git
cd shardes-paper && git fetch -q origin && git checkout -q "$sha"
[ -d .venv-vllm ] || python3 -m venv .venv-vllm
.venv-vllm/bin/pip install -q -r experiments/end_to_end/requirements-vllm.txt
[ -d .venv-verify ] || python3 -m venv .venv-verify
.venv-verify/bin/pip install -q -r experiments/end_to_end/requirements-verify.txt
cd experiments/end_to_end
{
  echo "#!/usr/bin/env bash"
  echo "cd $(pwd)"
  for m in "$@"; do
    echo "../../.venv-vllm/bin/python -m es_vllm.heldout --config es_vllm/$config --model $m >> /root/heldout.log 2>&1; echo \"finished $m exit \$?\" >> /root/heldout.log"
  done
  echo "echo chain-finished >> /root/heldout.log"
} > /root/heldout-chain.sh
bash -n /root/heldout-chain.sh
setsid nohup bash /root/heldout-chain.sh > /dev/null 2>&1 < /dev/null &
echo launched
exit 0
