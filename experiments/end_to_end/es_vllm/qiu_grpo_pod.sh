#!/usr/bin/env bash
# GRPO on Qiu et al.'s setup (docs/end_to_end/14-grpo-matched.md) on a fresh H200 pod, detached:
#     bash qiu_grpo_pod.sh <shardes-paper commit> <seed>
# Run again on the same pod to resume from the last full checkpoint. When the driver has
# finished, his eval_countdown.py scores the saved checkpoints one after the other (beside
# the training run the GPU has no room for it).
# Logs: /root/early.log (markers), /root/run.log (driver), /root/eval.log (evaluator).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
sha=${1:?usage: qiu_grpo_pod.sh <commit> <seed>}
seed=${2:?usage: qiu_grpo_pod.sh <commit> <seed>}
log=/root/early.log
ck=/root/qiu-ckpt
out=runs/qiu-matched/qwen3b-grpo-s$seed
source "$here/qiu_bootstrap.sh"
E2E=/root/shardes-paper/experiments/end_to_end
cd $E2E
resume=(); [ -f $out/log.jsonl ] && resume=(--resume)
if ! ../../.venv-vllm/bin/python -m es_vllm.qiu_grpo --out $out --ckpt $ck --workdir /root/grpo-state --seed "$seed" \
    --es-at-scale /root/es-at-scale "${resume[@]}" >> /root/run.log 2>&1; then
  echo "driver failed $(date -u +%FT%TZ)" >> $log
  exit 1
fi
echo "driver finished $(date -u +%FT%TZ)" >> $log
for d in $ck/ckpt-*; do
  k=${d##*-}
  o=$E2E/$out/qiu-eval/ckpt-$k
  [ -f $o/summary.json ] && continue
  if (cd /root/es-at-scale/archive && PYTHONPATH=countdown /root/shardes-paper/.venv-qiu-eval/bin/python eval_countdown.py \
        --model_path $d --base_model_name Qwen/Qwen2.5-3B-Instruct \
        --eval_data_path countdown/data/countdown.json --eval_samples 2000 --eval_offset -2000 \
        --max_new_tokens 1024 --batch_size 500 --save_responses --show_examples 5 --output_dir $o) >> /root/eval.log 2>&1; then
    gzip -f $o/eval_detailed_results.json
    acc=$(python3 -c "import json; print(json.load(open('$o/summary.json'))['eval_stats']['accuracy'])")
    echo "eval ckpt-$k accuracy $acc $(date -u +%FT%TZ)" >> $log
  else
    echo "eval ckpt-$k failed $(date -u +%FT%TZ)" >> $log
  fi
done
echo "finished $(date -u +%FT%TZ)" >> $log
