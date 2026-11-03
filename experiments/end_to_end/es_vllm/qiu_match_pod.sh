#!/usr/bin/env bash
# Arm B of docs/end_to_end/12-qiu-matched.md on a fresh H200 pod, detached:
#     bash qiu_match_pod.sh <shardes-paper commit> <seed> [<arm>]
# The driver trains and writes checkpoints after 0, 100, ..., 500 updates; beside it, his
# eval_countdown.py scores each checkpoint as it lands. With an arm (qiu_match.ARMS, e.g.
# decay of docs/end_to_end/13-step-decay.md) the driver branches from arm B's log, the
# decode checks are not repeated (the branch's first checkpoint has to be arm B's, bit for
# bit), and his evaluator scores the arm's own checkpoints.
# Logs: /root/early.log (markers), /root/run.log (driver), /root/eval.log (evaluator).
set -euo pipefail
sha=${1:?usage: qiu_match_pod.sh <commit> <seed> [<arm>]}
seed=${2:?usage: qiu_match_pod.sh <commit> <seed> [<arm>]}
arm=${3:-}
log=/root/early.log
ck=/root/qiu-ckpt
out=runs/qiu-matched/qwen3b-s$seed
ks="0 100 200 300 400 500"
armargs=()
case "$arm" in
  "") ;;
  decay) out=$out-decay; ks="300 350"; armargs=(--arm decay) ;;
  *) echo "unknown arm $arm" >&2; exit 2 ;;
esac
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
E2E=/root/shardes-paper/experiments/end_to_end
for p in fp32 bf16; do                      # float32: the rule alone; bf16: what the run decodes with
  [ -n "$arm" ] && break
  f=""; [ $p = fp32 ] && f=--fp32
  .venv-qiu-eval/bin/python experiments/end_to_end/es_vllm/qiu_decode_check.py --engine hf $f --out /root/dc-hf-$p.json \
    --es-at-scale /root/es-at-scale >> /root/check.log 2>&1
  if ! (cd $E2E && ../../.venv-vllm/bin/python -m es_vllm.qiu_decode_check --engine vllm $f --hf /root/dc-hf-$p.json \
        --record $out/decode-check-$p.json --es-at-scale /root/es-at-scale >> /root/check.log 2>&1); then
    echo "decode check $p did not pass $(date -u +%FT%TZ)" >> $log
    exit 1
  fi
  echo "decode check $p passes $(date -u +%FT%TZ)" >> $log
done
cd $E2E
../../.venv-vllm/bin/python -m es_vllm.qiu_match --out $out --ckpt $ck --seed "$seed" --es-at-scale /root/es-at-scale "${armargs[@]}" \
  >> /root/run.log 2>&1 &
pid=$!
echo "launched driver $pid $(date -u +%FT%TZ)" >> $log
for k in $ks; do
  while [ ! -f $ck/ckpt-$k/saved.json ]; do
    if ! kill -0 $pid 2>/dev/null; then echo "driver gone before checkpoint $k" >> $log; break 2; fi
    sleep 60
  done
  o=$E2E/$out/qiu-eval/ckpt-$k
  if (cd /root/es-at-scale/archive && PYTHONPATH=countdown /root/shardes-paper/.venv-qiu-eval/bin/python eval_countdown.py \
        --model_path $ck/ckpt-$k --base_model_name Qwen/Qwen2.5-3B-Instruct \
        --eval_data_path countdown/data/countdown.json --eval_samples 2000 --eval_offset -2000 \
        --max_new_tokens 1024 --batch_size 500 --save_responses --show_examples 5 --output_dir $o) >> /root/eval.log 2>&1; then
    gzip -f $o/eval_detailed_results.json
    acc=$(python3 -c "import json; print(json.load(open('$o/summary.json'))['eval_stats']['accuracy'])")
    echo "eval ckpt-$k accuracy $acc $(date -u +%FT%TZ)" >> $log
  else
    echo "eval ckpt-$k failed $(date -u +%FT%TZ)" >> $log
  fi
done
rc=0
wait $pid || rc=$?
echo "finished exit $rc $(date -u +%FT%TZ)" >> $log
