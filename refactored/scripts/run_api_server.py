#!/usr/bin/env python3
"""
API Server Launcher

Run the Kabaddi API server in standalone mode (without the processing loop).
This is useful for serving archived videos and events without running live processing.

Usage:
    python scripts/run_api_server.py
    python scripts/run_api_server.py --host 0.0.0.0 --port 8000
"""

import argparse
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def main():
    parser = argparse.ArgumentParser(description="Run Kabaddi API server in standalone mode.")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--log-level", default="info", choices=["critical", "error", "warning", "info", "debug"], 
                        help="Logging level (default: info)")
    args = parser.parse_args()

    import uvicorn
    from kabaddi.api import app

    print(f"Starting Kabaddi API server on {args.host}:{args.port}")
    print(f"API docs available at: http://{args.host}:{args.port}/docs")
    print("Press Ctrl+C to stop")

    uvicorn.run(
        "kabaddi.api.server:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()

