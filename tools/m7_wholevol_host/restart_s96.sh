#!/bin/bash
# one-off: switch production to stride 96 (step 0.5): stop, drop the stride-144 PHerc0343P rows, restart
~/m7_wholevol/stop.sh
sleep 5
rm -rf /vesuvius/m7_wholevol/state/PHerc0343P_20250521134555 /vesuvius/m7_wholevol/out/20250521134555-surface-* /vesuvius/m7_wholevol/stage_*.u8
~/m7_wholevol/start.sh
