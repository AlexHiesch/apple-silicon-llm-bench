#!/usr/bin/env python3
"""python -m agent_bench …"""

from __future__ import annotations

import sys


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "detect":
        from .detect import print_report
        print_report()
        return 0
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        from .run_matrix import main as run_main
        return run_main(sys.argv[2:])
    # default → run_matrix (accepts --list etc.)
    from .run_matrix import main as run_main
    return run_main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
