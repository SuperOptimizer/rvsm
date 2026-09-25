#!/bin/bash
. $HOME/m7_wholevol/env.sh
cd $HOME/m7_wholevol
export M7W_COMMIT=$(cat $HOME/m7_wholevol/COMMIT 2>/dev/null || echo unknown)
setsid nohup $PY $T finisher >> logs/finisher.log 2>&1 < /dev/null &
echo $! > pids/finisher; echo started finisher $(cat pids/finisher)
