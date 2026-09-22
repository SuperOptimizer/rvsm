"""The rvsm command line. The subcommands land with the modules they drive; this is the entry point."""
from __future__ import annotations

import sys

USAGE = """rvsm <command> [options]

  run        cfg.toml | --ct URL|PATH --umbilicus PATH|auto --out DIR --gpus 0[,1]
             --mode auto|resident|timeshare [--rounds N] [--steps N] [--size 30m6] [--init ckpt.pt]
  produce    --out DIR (--teacher recto,m7 | --student ckpt [--sign -1]) --region Z Y X
  train      cfg.toml
  eval       --out DIR [--ckpt P] [--round R] [--tifxyz DIR] [--json]
  export     --out DIR --ckpt P --box Z Y X DZ DY DX --dest DIR
  calibrate  --out DIR --ckpt P
  pretrain   cfg.toml [--steps N]
  ladder     cfg.toml --sizes 15m,30m6,60m
  status | stop | ledger --rebuild | umbilicus --ct URL --out umbilicus.json
"""

COMMANDS = ("run", "produce", "train", "eval", "export", "calibrate", "pretrain", "ladder",
            "status", "stop", "ledger", "umbilicus")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    print(USAGE, end="")
    if argv and argv[0] in COMMANDS:
        print(f"\n{argv[0]}: not implemented yet in this build.")
        return 2
    return 0 if not argv else 2


if __name__ == "__main__":
    raise SystemExit(main())
