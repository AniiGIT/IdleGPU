# Copyright (C) 2026 AniiGIT
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
broker/__main__.py - Entry point for the IdleGPU broker.

Usage:
  python -m broker [--host HOST] [--port PORT] [--dev]

Options:
  --host HOST   Address to bind on (default: 0.0.0.0)
  --port PORT   Port to listen on (default: 8765)
  --dev         Enable debug logging and auto-reload

Phase 1: plaintext WebSocket only. mTLS is added in Phase 2.
"""

from __future__ import annotations

import argparse
import logging
import sys


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m broker",
        description="IdleGPU Broker - agent registry and job routing.",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Address to bind on (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port to listen on (default: 8765)",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Enable debug logging (Phase 1 plaintext mode only)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    log_level = logging.DEBUG if args.dev else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger(__name__)
    logger.info("starting IdleGPU broker on %s:%d", args.host, args.port)

    if args.dev:
        logger.warning(
            "dev mode: plaintext WebSocket, no authentication. "
            "Do not expose this port to untrusted networks."
        )

    try:
        import uvicorn  # noqa: PLC0415 - runtime dep, imported here to give a clear error
    except ImportError:
        print(
            "error: uvicorn is not installed. Run: pip install 'uvicorn[standard]'",
            file=sys.stderr,
        )
        sys.exit(1)

    from .app import app  # noqa: PLC0415

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="debug" if args.dev else "info",
    )


if __name__ == "__main__":
    main()
