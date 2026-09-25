#!/bin/bash
# one-shot status: per-volume rows/ETA + the last lines of every log + GPUs/RAM/disk
cd $HOME/m7_wholevol
. ./env.sh; $PY $T status 2>/dev/null | $PY -c "import json,sys; d=json.load(sys.stdin); print(d['t'], 'windows/s (both GPUs):', d.get('windows_per_s_total')); [print(' ', v['vol'], v['Gvox'], 'Gvox, stride', v['stride'], 'rows', v['rows_done'], 'ETA', v.get('eta_utc','?')) for v in d['volumes']]"
for f in logs/worker_gpu0.log logs/worker_gpu1.log logs/finisher.log; do echo "== $f"; grep -v "\[TRT\]" $f | grep -v "^\s" | tail -2 | cut -c1-220; done
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,power.draw,temperature.gpu --format=csv,noheader
free -g | sed -n 2p; df -h /vesuvius | tail -1
