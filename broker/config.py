# Copyright (C) 2026 AniiGIT
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
broker/config.py - Configuration loading for the IdleGPU broker.

Reads broker.toml from the system-wide config location:
  Linux:   /etc/idlegpu/broker.toml
  Windows: %PROGRAMDATA%\\idlegpu\\broker.toml

On first run (file absent), the bundled default is copied to the system
location and the user is told where it landed.

Config is read once at startup; changes require a broker restart.
TLS private keys are never stored in this file — they live in the data
directory with restricted permissions.
"""

from __future__ import annotations

import logging
import os
import tomllib
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path

# Plain str prevents static analysers from narrowing platform branches.
_OS_NAME: str = os.name

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Typed config sections
# ---------------------------------------------------------------------------


@dataclass
class ServerSection:
    """Bind address and port for the broker WebSocket + REST server."""

    host: str = "0.0.0.0"
    """Address to bind on. Use 0.0.0.0 for LAN access."""

    port: int = 8765
    """TCP port the broker listens on (mTLS when configured). Range: 1-65535."""

    enroll_port: int | None = None
    """
    Plain HTTP port for agent enrollment (/ca.crt and /enroll only).

    Agents fetch the CA cert and submit their CSR here before they have
    credentials for the mTLS main port -- solving the chicken-and-egg problem.
    None means use port + 1 at runtime (default: 8766 when port is 8765).
    Range: 1-65535.
    """


@dataclass
class TlsSection:
    """Mutual TLS certificate paths. All None = plaintext mode."""

    ca_cert: str | None = None
    """Path to the CA certificate PEM file shared with agents."""

    broker_cert: str | None = None
    """Path to the broker's TLS certificate PEM file."""

    broker_key: str | None = None
    """Path to the broker's TLS private key PEM file (mode 600 on Linux)."""

    def is_configured(self) -> bool:
        """Return True when all three TLS paths are set."""
        return (
            self.ca_cert is not None
            and self.broker_cert is not None
            and self.broker_key is not None
        )


def _default_data_dir() -> str:
    """Return the platform-appropriate data directory path."""
    if _OS_NAME == "nt":
        data = os.environ.get("PROGRAMDATA", r"C:\ProgramData")
        return str(Path(data) / "idlegpu")
    else:
        return "/var/lib/idlegpu"


@dataclass
class DataSection:
    """Runtime data directory for certs, tokens, and other broker state."""

    data_dir: str = field(default_factory=_default_data_dir)
    """Absolute path to the broker data directory."""


@dataclass
class BrokerConfig:
    """Complete broker configuration. All fields have safe defaults."""

    server: ServerSection = field(default_factory=ServerSection)
    tls: TlsSection = field(default_factory=TlsSection)
    data: DataSection = field(default_factory=DataSection)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _parse_int(value: object, key: str, default: int, lo: int, hi: int) -> int:
    """Parse and range-check an integer config value; warn and return default on failure."""
    try:
        v = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        logger.warning("config: %s must be an integer, using default %d", key, default)
        return default
    if not lo <= v <= hi:
        logger.warning(
            "config: %s=%d is outside valid range [%d, %d], using default %d",
            key, v, lo, hi, default,
        )
        return default
    return v


def _parse_nonempty_str(value: object, key: str, default: str) -> str:
    """Parse a non-empty string config value; warn and return default on failure."""
    if not isinstance(value, str) or not value.strip():
        logger.warning(
            "config: %s must be a non-empty string, using default %r", key, default
        )
        return default
    return value


def _parse_optional_path(value: object, key: str) -> str | None:
    """Parse an optional file path string. Returns None if absent or empty."""
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        logger.warning("config: %s must be a non-empty string; ignoring", key)
        return None
    return value


# ---------------------------------------------------------------------------
# Section parsers
# ---------------------------------------------------------------------------


def _parse_server(raw: dict) -> ServerSection:  # type: ignore[type-arg]
    s = ServerSection()
    s.host = _parse_nonempty_str(
        raw.get("host", s.host), "server.host", "0.0.0.0"
    )
    s.port = _parse_int(
        raw.get("port", s.port), "server.port", 8765, 1, 65535
    )
    raw_enroll = raw.get("enroll_port")
    if raw_enroll is not None:
        s.enroll_port = _parse_int(raw_enroll, "server.enroll_port", s.port + 1, 1, 65535)
    return s


def _parse_tls(raw: dict) -> TlsSection:  # type: ignore[type-arg]
    s = TlsSection()
    s.ca_cert = _parse_optional_path(raw.get("ca_cert"), "tls.ca_cert")
    s.broker_cert = _parse_optional_path(raw.get("broker_cert"), "tls.broker_cert")
    s.broker_key = _parse_optional_path(raw.get("broker_key"), "tls.broker_key")
    return s


def _parse_data(raw: dict) -> DataSection:  # type: ignore[type-arg]
    s = DataSection()  # initialises data_dir via _default_data_dir()
    data_dir = raw.get("data_dir")
    if data_dir is not None:
        s.data_dir = _parse_nonempty_str(data_dir, "data.data_dir", s.data_dir)
    return s


# ---------------------------------------------------------------------------
# System config location
# ---------------------------------------------------------------------------


def system_config_path() -> Path:
    """Return the system-wide broker config path for the current platform."""
    if _OS_NAME == "nt":
        data = os.environ.get("PROGRAMDATA", r"C:\ProgramData")
        return Path(data) / "idlegpu" / "broker.toml"
    else:
        return Path("/etc/idlegpu/broker.toml")


# ---------------------------------------------------------------------------
# First-run seeding
# ---------------------------------------------------------------------------


def _seed_default_config(dest: Path) -> None:
    """
    Copy the bundled default broker.toml to *dest* using importlib.resources.

    Prints a user-facing message so the operator knows where the file landed.
    Silently logs and continues if the write fails (e.g. insufficient permissions).
    """
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        content = files("broker.defaults").joinpath("broker.toml").read_bytes()
        dest.write_bytes(content)
        print(f"IdleGPU: no broker config found. Created default config at:\n  {dest}")
        print("Edit it to customise bind address, port, and TLS paths, then restart.")
    except OSError as exc:
        logger.warning(
            "could not write default broker config to %s: %s; using built-in defaults",
            dest,
            exc,
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def load_config(path: Path | None = None) -> BrokerConfig:
    """
    Load BrokerConfig from broker.toml.

    Behaviour:
    - If *path* is None, the system-wide location is used.
    - If the file does not exist, the bundled default is seeded there and
      built-in defaults are returned (no parse needed; bundled == defaults).
    - Missing TOML keys use their dataclass defaults.
    - Out-of-range values log a warning and fall back to the default.
    - Config is read once; changes require a broker restart.
    """
    if path is None:
        path = system_config_path()

    if not path.exists():
        _seed_default_config(path)
        return BrokerConfig()  # bundled default == dataclass defaults; no parse needed

    with open(path, "rb") as fh:
        raw = tomllib.load(fh)

    return BrokerConfig(
        server=_parse_server(raw.get("server", {})),
        tls=_parse_tls(raw.get("tls", {})),
        data=_parse_data(raw.get("data", {})),
    )
