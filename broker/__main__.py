# Copyright (C) 2026 AniiGIT
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
broker/__main__.py - Entry point for the IdleGPU broker.

Usage:
  idlegpu-broker setup  [--hostname HOST] [--data-dir DIR] [--config FILE] [--start]
  idlegpu-broker start  [--host ADDR] [--port PORT] [--data-dir DIR] [--config FILE] [--dev]

Commands:
  setup   Generate the local CA, broker certificate, and one-time enrollment
          token. Writes [tls] paths to broker.toml automatically.
          Pass --start to immediately start the broker when done.

  start   Load config and start the broker WebSocket + REST server.
          Runs until interrupted (Ctrl+C or SIGTERM).
          Uses mTLS automatically when [tls] cert paths are configured.

Config file:
  Linux:   /etc/idlegpu/broker.toml
  Windows: %PROGRAMDATA%\\idlegpu\\broker.toml

CLI flags always override the config file values. The config file is
optional -- built-in defaults are used when it is absent.
"""

from __future__ import annotations

import argparse
import logging
import socket
import sys
import threading
import tomllib
from pathlib import Path

from .config import load_config, system_config_path


# ---------------------------------------------------------------------------
# Config file helpers
# ---------------------------------------------------------------------------


def _patch_toml_tls(tls_values: dict, data_dir: Path) -> Path | None:
    """
    Write or update the [tls] section in broker.toml.

    Tries the system config path first; falls back to data_dir/broker.toml
    on permission errors. Returns the path that was successfully written, or
    None when all candidates fail (prints manual fallback instructions then).

    Reads the existing file (if present), merges tls_values into the parsed
    data, and writes back using tomli_w. All other sections are preserved.
    Note: tomli_w does not round-trip comments -- they are not preserved on
    re-write, but all key/value pairs remain intact.
    """
    try:
        import tomli_w  # noqa: PLC0415
    except ImportError:
        print(
            "error: tomli-w is not installed. Run: pip install 'tomli-w>=1.0'",
            file=sys.stderr,
        )
        sys.exit(1)

    candidates = [system_config_path(), data_dir / "broker.toml"]
    for config_path in candidates:
        data: dict = {}
        if config_path.exists():
            try:
                with open(config_path, "rb") as fh:
                    data = tomllib.load(fh)
            except (OSError, tomllib.TOMLDecodeError):
                pass  # start fresh; existing file could not be parsed

        data["tls"] = tls_values

        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, "wb") as fh:
                tomli_w.dump(data, fh)
            return config_path
        except OSError:
            continue

    # All candidates failed — print manual fallback.
    print(
        "Warning: could not write TLS config to any location.",
        file=sys.stderr,
    )
    print(
        "Add the following to your broker.toml [tls] section manually:",
        file=sys.stderr,
    )
    print("[tls]", file=sys.stderr)
    for key, val in tls_values.items():
        print(f'{key} = "{val}"', file=sys.stderr)
    return None


# ---------------------------------------------------------------------------
# Subcommand: setup
# ---------------------------------------------------------------------------


def cmd_setup(args: argparse.Namespace) -> None:
    """
    Generate CA, broker cert, enrollment token, and write [tls] to broker.toml.

    Pass --start to immediately start the broker after setup completes.
    """
    from .pki import check_cert_expiry, setup  # noqa: PLC0415

    cfg = load_config(path=Path(args.config) if args.config else None)

    if args.data_dir is not None:
        cfg.data.data_dir = args.data_dir

    hostname = args.hostname or socket.gethostname()
    data_dir = Path(cfg.data.data_dir)

    print("IdleGPU broker setup")
    print(f"  Hostname : {hostname}")
    print(f"  Data dir : {data_dir}")
    print()

    ca_cert_path, broker_cert_path, broker_key_path, token, fingerprint, san_entries = setup(
        data_dir, hostname
    )

    cert_days = check_cert_expiry((data_dir / "broker.crt").read_bytes())

    config_path = _patch_toml_tls(
        {
            "ca_cert":     str(ca_cert_path),
            "broker_cert": str(broker_cert_path),
            "broker_key":  str(broker_key_path),
        },
        data_dir,
    )

    print("Certificates generated.")
    print(f"  Broker cert expires in {cert_days} days.")
    print()
    print("Broker cert Subject Alternative Names:")
    for entry in san_entries:
        print(f"  {entry}")
    print("  Agents and sidecars may connect using any of the above.")
    print()
    print("CA fingerprint (SHA-256):")
    print(f"  {fingerprint}")
    print("  Share this with agents for out-of-band verification.")
    print()
    if config_path is not None:
        print(f"TLS configuration written to {config_path}.")
    print()
    print("One-time enrollment token:")
    print(f"  {token}")
    print()
    print("Enroll each agent with:")
    print(f"  idlegpu-agent enroll --broker {hostname} --token {token}")
    print()
    print("The enrollment token is single-use. Run `idlegpu-broker setup` again")
    print("to generate a new token for additional agents.")

    if args.start:
        print()
        print("Starting broker (Ctrl+C to stop)...")
        print()
        cmd_start(argparse.Namespace(host=None, port=None, data_dir=args.data_dir, config=args.config, dev=False))


# ---------------------------------------------------------------------------
# Subcommand: start
# ---------------------------------------------------------------------------


def cmd_start(args: argparse.Namespace) -> None:
    """Load config and run the broker until interrupted."""
    import asyncio  # noqa: PLC0415

    cfg = load_config(path=Path(args.config) if args.config else None)

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

    from .app import app, enroll_app  # noqa: PLC0415

    # Thread config through to both apps for use by endpoints.
    app.state.cfg = cfg
    enroll_app.state.cfg = cfg

    # ------------------------------------------------------------------ #
    # Enrollment server (plain HTTP, daemon thread)                        #
    # Serves /ca.crt and /enroll only.  Runs alongside the main mTLS      #
    # server so agents can bootstrap before they hold client certificates. #
    # ------------------------------------------------------------------ #

    enroll_port = (
        cfg.server.enroll_port
        if cfg.server.enroll_port is not None
        else cfg.server.port + 1
    )

    class _EnrollServer(uvicorn.Server):
        """Uvicorn server that does not install OS signal handlers.

        The main server owns SIGINT/SIGTERM; the enrollment server runs
        in a daemon thread and is killed automatically on process exit.
        """

        def install_signal_handlers(self) -> None:
            pass

    def _run_enroll_server() -> None:
        enroll_cfg = uvicorn.Config(
            enroll_app,
            host=cfg.server.host,
            port=enroll_port,
            log_level="debug" if args.dev else "info",
        )
        asyncio.run(_EnrollServer(enroll_cfg).serve())

    enroll_thread = threading.Thread(target=_run_enroll_server, daemon=True)
    enroll_thread.start()
    logger.info(
        "enrollment server (plain HTTP) listening on %s:%d",
        cfg.server.host, enroll_port,
    )

    # ------------------------------------------------------------------ #
    # Main server (mTLS when [tls] is configured)                          #
    # ------------------------------------------------------------------ #

    uvicorn_kwargs: dict = {
        "host": cfg.server.host,
        "port": cfg.server.port,
        "log_level": "debug" if args.dev else "info",
    }

    # mTLS: when all three cert paths are configured, enable mutual TLS.
    # ssl_cert_reqs=CERT_REQUIRED enforces client certificate presentation.
    if cfg.tls.is_configured():
        import ssl  # noqa: PLC0415
        uvicorn_kwargs["ssl_certfile"] = cfg.tls.broker_cert
        uvicorn_kwargs["ssl_keyfile"] = cfg.tls.broker_key
        uvicorn_kwargs["ssl_ca_certs"] = cfg.tls.ca_cert
        uvicorn_kwargs["ssl_cert_reqs"] = ssl.CERT_REQUIRED

    uvicorn.run(app, **uvicorn_kwargs)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="idlegpu-broker",
        description="IdleGPU Broker - agent registry and job routing.",
    )
    sub = parser.add_subparsers(dest="command", metavar="command")
    sub.required = True

    # setup
    p_setup = sub.add_parser(
        "setup",
        help="Generate CA, broker cert, and enrollment token.",
    )
    p_setup.add_argument(
        "--hostname",
        default=None,
        metavar="HOST",
        help="Hostname for broker cert SAN (default: system hostname).",
    )
    p_setup.add_argument(
        "--data-dir",
        default=None,
        metavar="DIR",
        dest="data_dir",
        help="Data directory for generated files (overrides broker.toml).",
    )
    p_setup.add_argument(
        "--config",
        default=None,
        metavar="FILE",
        help="Explicit path to broker.toml (bypasses auto-discovery).",
    )
    p_setup.add_argument(
        "--start",
        action="store_true",
        help="Start the broker immediately after setup completes.",
    )

    # start
    p_start = sub.add_parser(
        "start",
        help="Start the broker server.",
    )
    p_start.add_argument(
        "--host",
        default=None,
        metavar="ADDR",
        help="Address to bind on (overrides broker.toml server.host).",
    )
    p_start.add_argument(
        "--port",
        type=int,
        default=None,
        metavar="PORT",
        help="Port to listen on (overrides broker.toml server.port).",
    )
    p_start.add_argument(
        "--data-dir",
        default=None,
        metavar="DIR",
        dest="data_dir",
        help="Data directory (overrides broker.toml data.data_dir).",
    )
    p_start.add_argument(
        "--config",
        default=None,
        metavar="FILE",
        help="Explicit path to broker.toml (bypasses auto-discovery).",
    )
    p_start.add_argument(
        "--dev",
        action="store_true",
        help="Enable debug logging (plaintext mode only).",
    )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    dispatch = {
        "setup": cmd_setup,
        "start": cmd_start,
    }

    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    handler(args)


if __name__ == "__main__":
    main()
