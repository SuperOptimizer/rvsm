#!/bin/bash
# stop one m7_wholevol process by its pid file name (worker_gpu0 | worker_gpu1 | finisher | status);
# a worker's unit claim is released so the other worker (or its restart) can take the unit.
cd $HOME/m7_wholevol
f=pids/$1
[ -f "$f" ] || { echo "no pid file $f"; exit 1; }
p=$(cat $f); kill -0 $p 2>/dev/null && kill $p && echo "stopped $1 pid $p"; rm -f $f
case $1 in worker_gpu*) sleep 2; grep -l "\"${1#worker_}\"" /vesuvius/m7_wholevol/queue/*.claim 2>/dev/null | xargs -r rm -v ;; esac
