# Copyright (C) 2026 AniiGIT
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
sidecar/__main__.py - Entry point for the IdleGPU sidecar process.

The sidecar runs inside a Docker container alongside CUDA applications.
It bridges the Unix socket IPC from libidlegpu-cuda.so to the broker's
WebSocket endpoint via mTLS.

Usage
─────
  python -m sidecar                 # reads all config from env vars
  idlegpu-sidecar                   # same, via installed console script

Required environment variables:
  IDLEGPU_BROKER_HOST   broker hostname or IP
  IDLEGPU_AGENT_ID      agent UUID to pair with

Optional environment variables (see sidecar/config.py for full list):
  IDLEGPU_SOCKET        Unix socket path (default: /var/run/idlegpu/cuda.sock)
  IDLEGPU_BROKER_PORT   broker port (default: 8765)
  IDLEGPU_CA_CERT       path to CA cert PEM; enables one-way TLS (broker verified)
  IDLEGPU_SIDECAR_CERT  path to sidecar client cert PEM; required for mTLS
  IDLEGPU_SIDECAR_KEY   path to sidecar client key PEM; required for mTLS
  IDLEGPU_DEV           set to "1" for plaintext ws:// (development only)
"""

from __future__ import annotations

import asyncio
import logging
import ssl
import sys

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


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
