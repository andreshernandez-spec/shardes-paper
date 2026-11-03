#!/usr/bin/env bash
# Bootstrap a fresh single-GPU RunPod pod and run ES configs one after another, detached.
# Run on the pod:
#
#     bash pod.sh <shardes-paper commit> <config.yaml> [<config.yaml> ...]
#
# Log: /root/es.log. Runs land in experiments/end_to_end/runs/<config name>/. vLLM 0.30.0's
# torch and JAX's CUDA plugin are built for CUDA 13.0: create the pod with
# minCudaVersion 13.0.
set -euo pipefail
sha=${1:?usage: pod.sh <commit> <config.yaml> ...}
shift
[ $# -ge 1 ] || { echo "usage: pod.sh <commit> <config.yaml> ..."; exit 2; }

cd /root
[ -d shardes-paper ] || git clone -q https://github.com/andreshernandez-spec/shardes-paper.git
cd shardes-paper
git fetch -q origin
git checkout -q "$sha"

# PEP 668 system python: venvs, never --break-system-packages. Two of them: the engine
# with JAX, and the verifiers pinned as the RL run had them (sympy 1.13.1, antlr 4.11).
[ -d .venv-vllm ] || python3 -m venv .venv-vllm
.venv-vllm/bin/pip install -q -r experiments/end_to_end/requirements-vllm.txt
[ -d .venv-verify ] || python3 -m venv .venv-verify
.venv-verify/bin/pip install -q -r experiments/end_to_end/requirements-verify.txt

.venv-vllm/bin/python -c "import torch, jax, vllm; n = torch.cuda.device_count(); assert n == 1, n; print(vllm.__version__, jax.devices(), torch.cuda.get_device_name(0))"

cd experiments/end_to_end
{
  echo "#!/usr/bin/env bash"
  echo "cd $(pwd)"
  for c in "$@"; do
    echo "../../.venv-vllm/bin/python -m es_vllm.run_tulu --config es_vllm/$c >> /root/es.log 2>&1; echo \"finished $c exit \$?\" >> /root/es.log"
  done
  echo "echo chain-finished >> /root/es.log"
} > /root/es-chain.sh
bash -n /root/es-chain.sh
setsid nohup bash /root/es-chain.sh > /dev/null 2>&1 < /dev/null &
echo launched
exit 0
