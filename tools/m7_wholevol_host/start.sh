#!/bin/bash
# forlindesk2: start the m7 whole-volume run: 2 GPU workers, the finisher (levels 4-7, upload, verify,
# delete) and the status loop. Idempotent w.r.t. finished work (row_*.done markers). Run detached.
. $HOME/m7_wholevol/env.sh
cd $HOME/m7_wholevol
mkdir -p logs pids uploaded
rm -f STOP /vesuvius/m7_wholevol/queue/*.claim
export M7W_COMMIT=$(cat $HOME/m7_wholevol/COMMIT 2>/dev/null || echo unknown)
for g in 0 1; do
  CUDA_VISIBLE_DEVICES=$g setsid nohup $PY $T worker > logs/worker_gpu$g.log 2>&1 < /dev/null &
  echo $! > pids/worker_gpu$g
done
setsid nohup $PY $T finisher > logs/finisher.log 2>&1 < /dev/null &
echo $! > pids/finisher
setsid nohup bash -c "while [ ! -f $HOME/m7_wholevol/STOP ]; do $PY $T status --write > /dev/null 2>&1; sleep 60; done" > /dev/null 2>&1 < /dev/null &
echo $! > pids/status
echo started $(date)
