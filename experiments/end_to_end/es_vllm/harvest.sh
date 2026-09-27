#!/usr/bin/env bash
# Pull a pod's ES runs and log into experiments/end_to_end/runs/, as they land.
#     bash es_vllm/harvest.sh <host> <port> <pod label>      # from experiments/end_to_end
set -euo pipefail
host=${1:?host}; port=${2:?port}; label=${3:?label}
ssh_cmd="ssh -i $HOME/.ssh/id_runpod -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o ConnectTimeout=20 -p $port"
rsync -az -e "$ssh_cmd" "root@$host:/root/shardes-paper/experiments/end_to_end/runs/" runs/
$ssh_cmd "root@$host" "cat /root/boot.log /root/es.log 2>/dev/null" | gzip -c > "runs/pod-$label-log.txt.gz"
ls runs
