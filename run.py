"""Start the Bullpen server. Run:  python run.py"""
import os

import uvicorn

if __name__ == "__main__":
    # Render (and most cloud hosts) tell us which port to listen on via the
    # $PORT environment variable -- it's assigned dynamically and is NOT
    # always 8000. Locally, $PORT isn't set, so we fall back to 8000 to
    # match "open http://localhost:8000" in the README.
    port = int(os.environ.get("PORT", 8000))

    # Cloud hosts route external traffic to the container on 0.0.0.0 (all
    # network interfaces), not 127.0.0.1 (localhost-only). 0.0.0.0 still
    # works fine for local browsing at http://localhost:<port>.
    host = "0.0.0.0"

    print(f"\n  Bullpen running at  http://localhost:{port}\n")
    uvicorn.run("backend.main:app", host=host, port=port, reload=True)
