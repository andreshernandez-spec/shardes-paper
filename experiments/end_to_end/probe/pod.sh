#!/usr/bin/env bash
# Bootstrap a fresh single-GPU RunPod pod (runpod/pytorch image) and launch the
# throughput probe detached. Run on the pod:
#
#     bash pod.sh <shardes-paper commit> <results dir, e.g. results-a100>
#
# The probe log goes to /root/probe.log; results land in
# experiments/end_to_end/probe/<results dir>/, one JSON per cell.
set -euo pipefail
sha=${1:?usage: pod.sh <commit> <results dir>}
out=${2:?usage: pod.sh <commit> <results dir>}

cd /root
[ -d shardes-paper ] || git clone -q https://github.com/andreshernandez-spec/shardes-paper.git
cd shardes-paper
git fetch -q origin
git checkout -q "$sha"

# The system python is PEP 668 managed: a venv, never --break-system-packages.
[ -d .venv-vllm ] || python3 -m venv .venv-vllm
. .venv-vllm/bin/activate
pip install -q -r experiments/end_to_end/requirements-vllm.txt

# One GPU, visible to torch, before anything long starts. vLLM 0.30.0's torch is built
# for CUDA 13.0, so the host driver must support 13.0 (create the pod with
# allowedCudaVersions 13.0 or later); on an older driver torch sees no GPU and this fails.
python -c "import torch, vllm; n = torch.cuda.device_count(); assert n == 1, n; print(vllm.__version__, torch.cuda.get_device_name(0))"

cd experiments/end_to_end/probe
setsid nohup python probe_throughput.py --config probe.yaml --out "$out" \
    > /root/probe.log 2>&1 < /dev/null &
echo launched
exit 0
