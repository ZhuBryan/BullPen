"""Run every module's self-check. `python test.py`"""

import asyncio
import importlib
import sys
import traceback

MODULES = ["orderbook", "capture", "replay", "research", "risk",
           "fix", "fix_session", "oms", "execution", "instruments", "metrics",
           "fastbook", "test_pyo3", "study", "leadlag"]

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
except ImportError:
    print("skip live capture tests (pip install websockets to enable)")
else:
    # Both exchanges, each against a local fake server: Binance (sequence-gap
    # resync) and Kraken (CRC32 checksum resync).
    for name in ("test_capture_live", "test_kraken_live"):
        ran += 1
        try:
            asyncio.run(importlib.import_module(name).main())
        except Exception:
            failed.append(name)
            print(f"FAIL {name}")
            traceback.print_exc()

print(f"\n{ran - len(failed)}/{ran} passed")
sys.exit(1 if failed else 0)
