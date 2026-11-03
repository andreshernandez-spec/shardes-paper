# Sourced by qiu_match_pod.sh and qiu_grpo_pod.sh with $sha and $log set: his clone at
# the pin, this repository at $sha, the vLLM venv and his evaluator's venv, each checked.
cd /root
[ -d es-at-scale ] || git clone -q https://github.com/VsonicV/es-at-scale.git
[ -d shardes-paper ] || git clone -q https://github.com/andreshernandez-spec/shardes-paper.git
cd shardes-paper && git fetch -q origin && git checkout -q "$sha"
pin=$(python3 -c "import sys; sys.path.insert(0, 'experiments/end_to_end'); import releases; print(releases.ES_AT_SCALE)")
git -C /root/es-at-scale fetch -q origin && git -C /root/es-at-scale checkout -q "$pin"
[ -d .venv-vllm ] || python3 -m venv .venv-vllm
.venv-vllm/bin/pip install -q -r experiments/end_to_end/requirements-vllm.txt
[ -d .venv-qiu-eval ] || python3 -m venv --system-site-packages .venv-qiu-eval
.venv-qiu-eval/bin/pip install -q -r experiments/end_to_end/requirements-qiu-eval.txt
.venv-vllm/bin/python -c "import torch, jax, vllm; n = torch.cuda.device_count(); assert n == 1, n; print(vllm.__version__, jax.devices(), torch.cuda.get_device_name(0))" >> $log 2>&1
.venv-qiu-eval/bin/python -c "import torch, transformers, flash_attn; assert torch.__version__.startswith('2.8'), torch.__version__; print('evaluator', torch.__version__, transformers.__version__, flash_attn.__version__)" >> $log 2>&1
echo "bootstrapped $(date -u +%FT%TZ)" >> $log
