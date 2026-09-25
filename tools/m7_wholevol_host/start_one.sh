#!/bin/bash
# start one GPU worker: start_one.sh 0|1
. $HOME/m7_wholevol/env.sh
cd $HOME/m7_wholevol
export M7W_COMMIT=$(cat $HOME/m7_wholevol/COMMIT 2>/dev/null || echo unknown)
g=$1
CUDA_VISIBLE_DEVICES=$g setsid nohup $PY $T worker >> logs/worker_gpu$g.log 2>&1 < /dev/null &
echo $! > pids/worker_gpu$g
echo started worker_gpu$g pid $(cat pids/worker_gpu$g)
