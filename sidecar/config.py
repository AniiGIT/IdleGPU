# Copyright (C) 2026 AniiGIT
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
sidecar/config.py - Sidecar configuration.

All settings can be supplied via environment variables so the sidecar
runs cleanly in Docker without a config file.

Environment variables
─────────────────────
  IDLEGPU_SOCKET        Unix socket path the shim connects to
                        (default: /var/run/idlegpu/cuda.sock)
  IDLEGPU_BROKER_HOST   Broker hostname or IP
  IDLEGPU_BROKER_PORT   Broker mTLS port (default: 8765)
  IDLEGPU_AGENT_ID      Agent UUID to pair with
  IDLEGPU_CA_CERT       Path to CA certificate PEM.
                        If set alone: one-way TLS (broker cert verified, no client cert).
                        If set with the two keys below: mutual TLS (mTLS).
                        If absent: plaintext ws:// (dev only).
  IDLEGPU_SIDECAR_CERT  Path to sidecar client certificate PEM (mTLS only)
  IDLEGPU_SIDECAR_KEY   Path to sidecar client key PEM (mTLS only)
  IDLEGPU_DEV           Set to "1" to use ws:// instead of wss://
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


_SOCKET_DEFAULT = "/var/run/idlegpu/cuda.sock"
_BROKER_PORT_DEFAULT = 8765


@dataclass
class TlsConfig:
    ca_cert: str | None
    sidecar_cert: str | None
    sidecar_key: str | None

    @property
    def enabled(self) -> bool:
        """True when at least a CA cert is set (one-way TLS or mTLS)."""
        return self.ca_cert is not None

    @property
    def mtls_enabled(self) -> bool:
        """True when all three cert paths are set (mutual TLS)."""
        return all(p is not None for p in (self.ca_cert, self.sidecar_cert, self.sidecar_key))


@dataclass
class SidecarConfig:
    socket_path: str
    broker_host: str
    broker_port: int
    agent_id: str
    dev_mode: bool
    tls: TlsConfig


def load_from_env() -> SidecarConfig:
    """Load configuration from environment variables.

    Raises RuntimeError if required variables (IDLEGPU_BROKER_HOST,
    IDLEGPU_AGENT_ID) are missing.
    """
    broker_host = os.environ.get("IDLEGPU_BROKER_HOST", "")
    agent_id = os.environ.get("IDLEGPU_AGENT_ID", "")

    missing = []
    if not broker_host:
        missing.append("IDLEGPU_BROKER_HOST")
    if not agent_id:
        missing.append("IDLEGPU_AGENT_ID")
    if missing:
        raise RuntimeError(
            f"missing required environment variables: {', '.join(missing)}"
        )

    tls = TlsConfig(
        ca_cert=os.environ.get("IDLEGPU_CA_CERT") or None,
        sidecar_cert=os.environ.get("IDLEGPU_SIDECAR_CERT") or None,
        sidecar_key=os.environ.get("IDLEGPU_SIDECAR_KEY") or None,
    )

    return SidecarConfig(
        socket_path=os.environ.get("IDLEGPU_SOCKET", _SOCKET_DEFAULT),
        broker_host=broker_host,
        broker_port=int(os.environ.get("IDLEGPU_BROKER_PORT", str(_BROKER_PORT_DEFAULT))),
        agent_id=agent_id,
        dev_mode=os.environ.get("IDLEGPU_DEV", "0") == "1",
        tls=tls,
    )
