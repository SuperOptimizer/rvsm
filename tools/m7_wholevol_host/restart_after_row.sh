#!/bin/bash
# restart worker gpu$1 right after its next finished shard row (resume-compatible code updates lose
# only the warm-up window rows). Run detached: setsid nohup ~/m7_wholevol/restart_after_row.sh 0 &
g=$1; L=$HOME/m7_wholevol/logs/worker_gpu$g.log
n0=$(grep -c "\[row\]" $L)
while [ "$(grep -c "\[row\]" $L)" -le "$n0" ]; do sleep 10; done
sleep 5    # the writer thread has logged: the row's done marker is on disk
$HOME/m7_wholevol/stop_one.sh worker_gpu$g
sleep 3
$HOME/m7_wholevol/start_one.sh $g
echo "restarted gpu$g at $(date)"
