#!/bin/bash
# one-shot status: status.json summary + the last lines of every log + GPUs
cd $HOME/m7_wholevol
. ./env.sh; $PY $T status 2>/dev/null | $PY -c "import json,sys; d=json.load(sys.stdin); print(d['t'], 'rate/worker', d.get('rate_Mvox_s_per_worker'), 'Mvox/s'); [print(' ', v['vol'], v['Gvox'], 'Gvox rows', v['rows_done'], 'UPLOADED' if v['uploaded'] else '', v.get('eta_utc','')) for v in d['volumes']]"
for f in logs/worker_gpu0.log logs/worker_gpu1.log logs/finisher.log; do echo "== $f"; grep -v "\[TRT\]" $f | tail -3; done
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,power.draw,temperature.gpu --format=csv,noheader
free -g | head -2 | tail -1; df -h /vesuvius | tail -1
