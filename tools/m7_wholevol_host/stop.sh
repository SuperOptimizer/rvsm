#!/bin/bash
# forlindesk2: stop every m7_wholevol process by its pid file (no pattern kills). Finished shard rows are
# kept (row_*.done); a restart recomputes only the unfinished rows of the units in flight.
cd $HOME/m7_wholevol
touch STOP
for f in pids/*; do
  [ -f "$f" ] || continue
  p=$(cat $f)
  if kill -0 $p 2>/dev/null; then kill $p; echo "stopped $(basename $f) pid $p"; fi
  rm -f $f
done
sleep 3
rm -f /vesuvius/m7_wholevol/queue/*.claim
