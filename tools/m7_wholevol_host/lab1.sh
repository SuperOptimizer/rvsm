#!/bin/bash
# engine lab on GPU 1 (worker gpu1 stopped): baseline timing, then builds + timings of the variants
. $HOME/m7_wholevol/env.sh
export CUDA_VISIBLE_DEVICES=1
L=$HOME/m7_wholevol/rvsm/tools/m7_engine_lab.py
BASE=$(ls /vesuvius/tsm/models/trt/m7_p192_b1_fp16_*.plan)
D=/vesuvius/m7_wholevol/lab
$PY $L time --plan $BASE --n 20
$PY $L time --plan $BASE --n 20 --graph
for spec in "192 1 5" "256 1 3" "192 2 3" "224 1 3" "288 1 3" "256 1 5"; do
  set -- $spec
  $PY $L build --window $1 --batch $2 --level $3 2>&1 | grep -v "^\[" | tail -2
  $PY $L time --plan $D/m7_p$1_b$2_L$3.plan --n 12
  $PY $L time --plan $D/m7_p$1_b$2_L$3.plan --n 12 --graph
done
echo LABDONE
