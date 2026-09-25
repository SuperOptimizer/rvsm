#!/bin/bash
# one-off: stop, drop the first-version (old grid) PHerc0343P output, restart on the new code
~/m7_wholevol/stop.sh
sleep 5
rm -rf /vesuvius/m7_wholevol/state/PHerc0343P_20250521134555 /vesuvius/m7_wholevol/out/20250521134555-surface-* /vesuvius/m7_wholevol/stage_*.u8
~/m7_wholevol/start.sh
