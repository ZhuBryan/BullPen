"""Run every module's self-check. `python test.py`"""

import asyncio
import importlib
import sys
import traceback

MODULES = ["orderbook", "capture", "replay", "research", "risk"]

failed = []
for name in MODULES:
    try:
        importlib.import_module(name).demo()
    except Exception:
        failed.append(name)
        print(f"FAIL {name}")
        traceback.print_exc()

ran = len(MODULES)

# The live-capture test drives a real websocket server, so it needs the
# optional `websockets` package. Skipped rather than failed when absent, since
# the sequencing logic it covers is already tested offline in capture.py.
try:
    import websockets  # noqa: F401
    live = importlib.import_module("test_capture_live")
except ImportError:
    print("skip test_capture_live (pip install websockets to enable)")
else:
    ran += 1
    try:
        asyncio.run(live.main())
    except Exception:
        failed.append("test_capture_live")
        print("FAIL test_capture_live")
        traceback.print_exc()

print(f"\n{ran - len(failed)}/{ran} passed")
sys.exit(1 if failed else 0)
