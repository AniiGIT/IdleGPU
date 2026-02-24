# Copyright (C) 2026 AniiGIT
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
broker/__main__.py - Entry point for the IdleGPU broker.

Usage:
  python -m broker [--host HOST] [--port PORT] [--data-dir DIR] [--dev]

Options:
  --host HOST       Address to bind on (overrides broker.toml)
  --port PORT       Port to listen on (overrides broker.toml)
  --data-dir DIR    Data directory for certs and runtime state (overrides broker.toml)
  --dev             Enable debug logging and plaintext mode

Config file:
  Linux:   /etc/idlegpu/broker.toml
  Windows: %PROGRAMDATA%\\idlegpu\\broker.toml

CLI flags always override the config file. The config file is optional;
built-in defaults are used when it is absent.

Phase 1: plaintext WebSocket only. mTLS is added in Phase 2 via `idlegpu-broker setup`.
"""

from __future__ import annotations

import argparse
import logging
import sys

from .config import load_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="idlegpu-broker",
        description="IdleGPU Broker - agent registry and job routing.",
    )
    parser.add_argument(
        "--host",
        default=None,
        metavar="HOST",
        help="Address to bind on (overrides broker.toml server.host).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        metavar="PORT",
        help="Port to listen on (overrides broker.toml server.port).",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        metavar="DIR",
        dest="data_dir",
        help="Data directory for certs and runtime state (overrides broker.toml data.data_dir).",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Enable debug logging (plaintext mode only).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Load config first; CLI flags override individual values below.
    cfg = load_config()

    if args.host is not None:
        cfg.server.host = args.host
    if args.port is not None:
        cfg.server.port = args.port
    if args.data_dir is not None:
        cfg.data.data_dir = args.data_dir

    log_level = logging.DEBUG if args.dev else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger(__name__)
    logger.info(
        "starting IdleGPU broker on %s:%d (data_dir=%s)",
        cfg.server.host, cfg.server.port, cfg.data.data_dir,
    )

    if args.dev:
        logger.warning(
            "dev mode: plaintext WebSocket, no authentication. "
            "Do not expose this port to untrusted networks."
        )

    if cfg.tls.is_configured():
        logger.info(
            "TLS configured: ca=%s cert=%s",
            cfg.tls.ca_cert, cfg.tls.broker_cert,
        )
    else:
        logger.info(
            "TLS not configured -- starting in plaintext mode. "
            "Run `idlegpu-broker setup` to generate certificates."
        )

    try:
        import uvicorn  # noqa: PLC0415
    except ImportError:
        print(
            "error: uvicorn is not installed. Run: pip install 'uvicorn[standard]'",
            file=sys.stderr,
        )
        sys.exit(1)

    from .app import app  # noqa: PLC0415

    # Pass the resolved config to the app so Phase 2 endpoints can access data_dir.
    app.state.cfg = cfg

    uvicorn_kwargs: dict = {
        "host": cfg.server.host,
        "port": cfg.server.port,
        "log_level": "debug" if args.dev else "info",
    }

    # mTLS: when all three cert paths are configured, pass them to uvicorn.
    # ssl_cert_reqs=CERT_REQUIRED enforces mutual TLS (client must present a cert).
    if cfg.tls.is_configured():
        import ssl  # noqa: PLC0415
        uvicorn_kwargs["ssl_certfile"] = cfg.tls.broker_cert
        uvicorn_kwargs["ssl_keyfile"] = cfg.tls.broker_key
        uvicorn_kwargs["ssl_ca_certs"] = cfg.tls.ca_cert
        uvicorn_kwargs["ssl_cert_reqs"] = ssl.CERT_REQUIRED

    uvicorn.run(app, **uvicorn_kwargs)


if __name__ == "__main__":
    main()
