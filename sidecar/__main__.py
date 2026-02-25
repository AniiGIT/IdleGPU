# Copyright (C) 2026 AniiGIT
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
sidecar/__main__.py - Entry point for the IdleGPU sidecar process.

The sidecar runs inside a Docker container alongside CUDA applications.
It bridges the Unix socket IPC from libidlegpu-cuda.so to the broker's
WebSocket endpoint via mTLS.

Commands
────────
  idlegpu-sidecar [start]   Run the sidecar bridge (reads config from env vars).
                            "start" is the default when no subcommand is given,
                            so existing Docker CMD entries need no change.
  idlegpu-sidecar enroll    Obtain mTLS certificates from the broker.

Start (env vars, see sidecar/config.py):
  IDLEGPU_BROKER_HOST   broker hostname or IP  (required)
  IDLEGPU_AGENT_ID      agent UUID to pair with  (required)
  IDLEGPU_SOCKET        Unix socket path (default: /var/run/idlegpu/cuda.sock)
  IDLEGPU_BROKER_PORT   broker port (default: 8765)
  IDLEGPU_CA_CERT       path to CA cert PEM; enables one-way TLS (broker verified)
  IDLEGPU_SIDECAR_CERT  path to sidecar client cert PEM; required for mTLS
  IDLEGPU_SIDECAR_KEY   path to sidecar client key PEM; required for mTLS
  IDLEGPU_DEV           set to "1" for plaintext ws:// (development only)

Enroll:
  idlegpu-sidecar enroll --broker HOST --token TOKEN [--port PORT]
                         [--enroll-port PORT] [--data-dir DIR]
                         [--env-cert-dir DIR]

  --data-dir DIR      Host directory to write cert files (default: /etc/idlegpu/tls)
  --env-cert-dir DIR  Path where those files appear inside the container,
                      used in sidecar.env (defaults to --data-dir)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import ssl
import sys
from pathlib import Path

from .config import load_from_env
from .cuda_bridge import CudaBridge
from .ipc_server import IpcServer


def _build_ssl_ctx(cfg_tls) -> ssl.SSLContext | None:  # type: ignore[no-untyped-def]
    """Build a client SSLContext, or None for dev/plaintext mode.

    One-way TLS (IDLEGPU_CA_CERT only): verifies the broker certificate but
    sends no client certificate.
    mTLS (all three vars set): mutual authentication — both sides present certs.
    """
    if not cfg_tls.enabled:
        return None

    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.load_verify_locations(cfg_tls.ca_cert)
    if cfg_tls.mtls_enabled:
        ctx.load_cert_chain(cfg_tls.sidecar_cert, cfg_tls.sidecar_key)
    # Require the broker to present a certificate signed by our CA.
    ctx.verify_mode = ssl.CERT_REQUIRED
    ctx.check_hostname = True
    return ctx


async def _run() -> None:
    try:
        cfg = load_from_env()
    except RuntimeError as exc:
        logging.critical("sidecar: configuration error: %s", exc)
        sys.exit(1)

    if cfg.dev_mode:
        logging.warning(
            "sidecar: running in dev mode — plaintext ws:// connections only. "
            "Never use in production."
        )

    ssl_ctx = None if cfg.dev_mode else _build_ssl_ctx(cfg.tls)
    if ssl_ctx is None and not cfg.dev_mode:
        logging.warning(
            "sidecar: TLS not configured — using plaintext ws://. "
            "Set IDLEGPU_CA_CERT for one-way TLS, or set all three "
            "(IDLEGPU_CA_CERT, IDLEGPU_SIDECAR_CERT, IDLEGPU_SIDECAR_KEY) for mTLS."
        )
    elif ssl_ctx is not None:
        tls_mode = "mTLS" if cfg.tls.mtls_enabled else "one-way TLS"
        logging.info("sidecar: %s enabled", tls_mode)

    bridge = CudaBridge(
        broker_host=cfg.broker_host,
        broker_port=cfg.broker_port,
        agent_id=cfg.agent_id,
        ssl_ctx=ssl_ctx,
    )
    server = IpcServer(socket_path=cfg.socket_path, bridge=bridge)

    logging.info(
        "sidecar: starting (agent_id=%s, broker=%s:%d, socket=%s)",
        cfg.agent_id, cfg.broker_host, cfg.broker_port, cfg.socket_path,
    )

    bridge_task = asyncio.create_task(bridge.run_forever(), name="cuda-bridge")
    server_task = asyncio.create_task(server.run(), name="ipc-server")

    try:
        # Run until one of the tasks fails or the process is interrupted.
        done, pending = await asyncio.wait(
            {bridge_task, server_task},
            return_when=asyncio.FIRST_EXCEPTION,
        )
        for task in done:
            exc = task.exception()
            if exc is not None:
                logging.critical("sidecar: fatal error in %s: %s", task.get_name(), exc)
    except asyncio.CancelledError:
        pass
    finally:
        bridge.stop()
        bridge_task.cancel()
        server_task.cancel()
        await asyncio.gather(bridge_task, server_task, return_exceptions=True)
        logging.info("sidecar: stopped")


# ---------------------------------------------------------------------------
# Subcommand: start
# ---------------------------------------------------------------------------


def cmd_start(_args: argparse.Namespace) -> None:
    """Load config from environment variables and run the sidecar bridge."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass


# ---------------------------------------------------------------------------
# Subcommand: enroll
# ---------------------------------------------------------------------------


def cmd_enroll(args: argparse.Namespace) -> None:
    """
    Enroll this sidecar with the broker to obtain mTLS certificates.

    Fetches the CA cert from the plain HTTP enrollment port, presents its
    SHA-256 fingerprint for TOFU verification, generates an ECDSA P-256
    keypair and CSR, submits it with the one-time token, saves the certificate
    files, and writes sidecar.env for docker-compose env_file:.

    Run this on the host before starting the sidecar container. Volume-mount
    --data-dir into the container at the path given by --env-cert-dir.
    """
    import uuid  # noqa: PLC0415

    from .pki import (  # noqa: PLC0415
        cert_fingerprint,
        default_data_dir,
        enroll_sidecar,
        fetch_ca_cert,
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    data_dir = Path(args.data_dir) if args.data_dir else default_data_dir()
    env_cert_dir = Path(args.env_cert_dir) if args.env_cert_dir else None

    # Reuse an existing sidecar_id so re-enrollment keeps the same identity.
    sidecar_id_file = data_dir / "sidecar_id"
    if sidecar_id_file.exists():
        sidecar_id = sidecar_id_file.read_text(encoding="ascii").strip()
    else:
        sidecar_id = str(uuid.uuid4())

    enroll_port = args.enroll_port if args.enroll_port is not None else args.port + 1

    print(f"Enrolling sidecar {sidecar_id}")
    print(f"  Broker      : {args.broker}:{args.port}")
    print(f"  Enroll port : {enroll_port}")
    print(f"  Data dir    : {data_dir}")
    if env_cert_dir:
        print(f"  Env cert dir: {env_cert_dir}")
    print()

    # Step 1: fetch CA cert from the plain HTTP enrollment port.
    try:
        ca_cert_pem = fetch_ca_cert(args.broker, enroll_port)
    except Exception as exc:
        print(f"error: could not fetch CA cert: {exc}", file=sys.stderr)
        sys.exit(1)

    # Step 2: TOFU fingerprint confirmation.
    fingerprint = cert_fingerprint(ca_cert_pem)
    print("CA certificate fingerprint (SHA-256):")
    print(f"  {fingerprint}")
    print()
    print("Compare this fingerprint with the one printed by `idlegpu-broker setup`.")
    print("If they do not match, abort and investigate before proceeding.")
    print()

    try:
        answer = input("Does this fingerprint match? [y/N] ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nAborted.")
        sys.exit(1)

    if answer.lower() != "y":
        print("Enrollment aborted.")
        sys.exit(1)

    print()

    # Step 3: generate keypair/CSR, POST to broker, save certs, write sidecar.env.
    try:
        env_path = enroll_sidecar(
            broker_host=args.broker,
            enroll_port=enroll_port,
            token=args.token,
            sidecar_id=sidecar_id,
            data_dir=data_dir,
            ca_cert_pem=ca_cert_pem,
            env_cert_dir=env_cert_dir,
        )
    except Exception as exc:
        print(f"Enrollment failed: {exc}", file=sys.stderr)
        sys.exit(1)

    # Persist sidecar_id so re-enrollment reuses the same identity.
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        sidecar_id_file.write_text(sidecar_id + "\n", encoding="ascii")
    except OSError:
        pass

    cert_dir_display = env_cert_dir if env_cert_dir else data_dir
    print("Enrollment successful.")
    print(f"  Certificates : {data_dir}/")
    print(f"  sidecar.env  : {env_path}")
    print()
    print("Add to your docker-compose sidecar service:")
    print(f"  env_file:")
    print(f"    - {env_path}")
    if env_cert_dir and env_cert_dir != data_dir:
        print(f"  volumes:")
        print(f"    - {data_dir}:{env_cert_dir}:ro")
    print()
    print("Run `idlegpu-broker setup` to generate a new token for the next enrollment.")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="idlegpu-sidecar",
        description="IdleGPU Sidecar - CUDA forwarding bridge for Docker containers.",
    )
    sub = parser.add_subparsers(dest="command", metavar="command")
    sub.required = False  # Default to "start" when no subcommand is given.

    # start
    sub.add_parser(
        "start",
        help="Run the sidecar bridge (reads config from environment variables).",
    )

    # enroll
    p_enroll = sub.add_parser(
        "enroll",
        help="Obtain mTLS certificates from the broker.",
    )
    p_enroll.add_argument(
        "--broker",
        required=True,
        metavar="HOST",
        help="Broker hostname or IP.",
    )
    p_enroll.add_argument(
        "--port",
        type=int,
        default=8765,
        metavar="PORT",
        help="Broker main port (default: 8765).",
    )
    p_enroll.add_argument(
        "--enroll-port",
        type=int,
        default=None,
        metavar="PORT",
        dest="enroll_port",
        help=(
            "Broker enrollment port for CA cert fetch and CSR submission "
            "(plain HTTP). Default: main port + 1."
        ),
    )
    p_enroll.add_argument(
        "--token",
        required=True,
        metavar="TOKEN",
        help="One-time enrollment token from `idlegpu-broker setup`.",
    )
    p_enroll.add_argument(
        "--data-dir",
        default=None,
        metavar="DIR",
        dest="data_dir",
        help="Host directory to write certificate files (default: /etc/idlegpu/tls).",
    )
    p_enroll.add_argument(
        "--env-cert-dir",
        default=None,
        metavar="DIR",
        dest="env_cert_dir",
        help=(
            "Path where the cert files will appear inside the container, "
            "written into sidecar.env. Defaults to --data-dir."
        ),
    )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # Default to "start" when invoked without a subcommand (Docker CMD compat).
    if args.command is None or args.command == "start":
        cmd_start(args)
    elif args.command == "enroll":
        cmd_enroll(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
