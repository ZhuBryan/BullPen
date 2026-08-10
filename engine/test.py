"""Run every module's self-check. `python test.py`"""

import importlib
import sys
import traceback

MODULES = ["orderbook", "capture", "replay", "research"]

failed = []
for name in MODULES:
    try:
        importlib.import_module(name).demo()
    except Exception:
        failed.append(name)
        print(f"FAIL {name}")
        traceback.print_exc()

print(f"\n{len(MODULES) - len(failed)}/{len(MODULES)} passed")
sys.exit(1 if failed else 0)
