# Copyright (C) 2026 AniiGIT
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
agent/config.py - Configuration loading for the IdleGPU agent.

Reads agent.toml from the system-wide config location:
  Linux:   /etc/idlegpu/agent.toml
  Windows: %PROGRAMDATA%\\idlegpu\\agent.toml

On first run (file absent), the bundled default is copied to the system
location using importlib.resources and the user is told where it landed.

Config is read once at startup; changes require an agent restart.
Secrets (TLS certificates, HMAC keys) are NEVER stored in this file.
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
class IdleSection:
    """Idle-detection thresholds and poll intervals."""

    gpu_threshold: int = 10
    """% GPU utilization; strictly below this → GPU idle. Range: 0–100."""

    cpu_threshold: int = 20
    """% CPU utilization (all-core average); below this → CPU idle. Range: 0–100."""

    input_idle_seconds: int = 300
    """Seconds since last input event to consider the machine idle. Range: 1–86400."""

    poll_idle_seconds: float = 10.0
    """Check interval (seconds) while waiting for the machine to go idle."""

    poll_active_seconds: float = 3.0
    """Check interval (seconds) while the GPU is offered; faster for quick reclaim."""


@dataclass
class BrokerSection:
    """Broker connection parameters."""

    host: str = "broker.local"
    """Hostname or IP of the IdleGPU broker. Agent always dials out."""

    port: int = 443
    """TCP port the broker listens on. Range: 1–65535."""


def _default_transparency_log() -> str:
    """Return the platform-appropriate transparency log path."""
    if _OS_NAME == "nt":
        data = os.environ.get("PROGRAMDATA", r"C:\ProgramData")
        return str(Path(data) / "idlegpu" / "transparency.log")
    else:
        return "/var/lib/idlegpu/transparency.log"


@dataclass
class LoggingSection:
    """Transparency log settings."""

    transparency_log: str = field(default_factory=_default_transparency_log)
    """Absolute path to the append-only transparency log file."""


@dataclass
class TlsSection:
    """Mutual TLS certificate paths. All None = plaintext mode (dev only)."""

    ca_cert: str | None = None
    """Path to the CA certificate PEM used to verify the broker's identity."""

    agent_cert: str | None = None
    """Path to this agent's TLS certificate PEM file."""

    agent_key: str | None = None
    """Path to this agent's TLS private key PEM file (mode 600 on Linux)."""

    def is_configured(self) -> bool:
        """Return True when all three TLS paths are set."""
        return (
            self.ca_cert is not None
            and self.agent_cert is not None
            and self.agent_key is not None
        )


@dataclass
class AgentConfig:
    """Complete agent configuration. All fields have safe defaults."""

    idle: IdleSection = field(default_factory=IdleSection)
    broker: BrokerSection = field(default_factory=BrokerSection)
    logging: LoggingSection = field(default_factory=LoggingSection)
    tls: TlsSection = field(default_factory=TlsSection)


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
            key,
            v,
            lo,
            hi,
            default,
        )
        return default
    return v


def _parse_pos_float(value: object, key: str, default: float) -> float:
    """Parse a positive float config value; warn and return default on failure."""
    try:
        v = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        logger.warning("config: %s must be a number, using default %.1f", key, default)
        return default
    if v <= 0:
        logger.warning(
            "config: %s=%.1f must be positive, using default %.1f", key, v, default
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


def _parse_idle(raw: dict) -> IdleSection:  # type: ignore[type-arg]
    s = IdleSection()
    s.gpu_threshold = _parse_int(
        raw.get("gpu_threshold", s.gpu_threshold),
        "idle.gpu_threshold",
        10,
        0,
        100,
    )
    s.cpu_threshold = _parse_int(
        raw.get("cpu_threshold", s.cpu_threshold),
        "idle.cpu_threshold",
        20,
        0,
        100,
    )
    s.input_idle_seconds = _parse_int(
        raw.get("input_idle_seconds", s.input_idle_seconds),
        "idle.input_idle_seconds",
        300,
        1,
        86400,
    )
    s.poll_idle_seconds = _parse_pos_float(
        raw.get("poll_idle_seconds", s.poll_idle_seconds),
        "idle.poll_idle_seconds",
        10.0,
    )
    s.poll_active_seconds = _parse_pos_float(
        raw.get("poll_active_seconds", s.poll_active_seconds),
        "idle.poll_active_seconds",
        3.0,
    )
    return s


def _parse_broker(raw: dict) -> BrokerSection:  # type: ignore[type-arg]
    s = BrokerSection()
    s.host = _parse_nonempty_str(
        raw.get("host", s.host),
        "broker.host",
        "broker.local",
    )
    s.port = _parse_int(
        raw.get("port", s.port),
        "broker.port",
        443,
        1,
        65535,
    )
    return s


def _parse_logging(raw: dict) -> LoggingSection:  # type: ignore[type-arg]
    s = LoggingSection()  # initialises transparency_log via _default_transparency_log()
    tlog = raw.get("transparency_log")
    if tlog is not None:
        s.transparency_log = _parse_nonempty_str(
            tlog, "logging.transparency_log", s.transparency_log
        )
    return s


def _parse_tls(raw: dict) -> TlsSection:  # type: ignore[type-arg]
    s = TlsSection()
    s.ca_cert = _parse_optional_path(raw.get("ca_cert"), "tls.ca_cert")
    s.agent_cert = _parse_optional_path(raw.get("agent_cert"), "tls.agent_cert")
    s.agent_key = _parse_optional_path(raw.get("agent_key"), "tls.agent_key")
    return s


# ---------------------------------------------------------------------------
# System config location
# ---------------------------------------------------------------------------


def system_config_path() -> Path:
    """Return the system-wide config path for the current platform."""
    if _OS_NAME == "nt":
        data = os.environ.get("PROGRAMDATA", r"C:\ProgramData")
        return Path(data) / "idlegpu" / "agent.toml"
    else:
        return Path("/etc/idlegpu/agent.toml")


# ---------------------------------------------------------------------------
# First-run seeding
# ---------------------------------------------------------------------------


def _seed_default_config(dest: Path) -> None:
    """
    Copy the bundled default config to *dest* using importlib.resources.

    Prints a user-facing message so the operator knows where the file landed.
    Silently logs and continues if the write fails (e.g. insufficient permissions).
    """
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        content = files("agent.defaults").joinpath("agent.toml").read_bytes()
        dest.write_bytes(content)
        print(f"IdleGPU: no config found. Created default config at:\n  {dest}")
        print("Edit it to customise thresholds and broker settings, then restart.")
    except OSError as exc:
        logger.warning(
            "could not write default config to %s: %s; using built-in defaults",
            dest,
            exc,
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def load_config(path: Path | None = None) -> AgentConfig:
    """
    Load AgentConfig from agent.toml.

    Behaviour:
    - If *path* is None, the system-wide location is used.
    - If the file does not exist, the bundled default is seeded there and
      built-in defaults are returned (no parse needed; bundled == defaults).
    - Missing TOML keys use their dataclass defaults.
    - Out-of-range values log a warning and fall back to the default.
    - Config is read once; changes require an agent restart.
    """
    if path is None:
        path = system_config_path()

    if not path.exists():
        _seed_default_config(path)
        return AgentConfig()  # bundled default == dataclass defaults; no parse needed

    with open(path, "rb") as fh:
        raw = tomllib.load(fh)

    return AgentConfig(
        idle=_parse_idle(raw.get("idle", {})),
        broker=_parse_broker(raw.get("broker", {})),
        logging=_parse_logging(raw.get("logging", {})),
        tls=_parse_tls(raw.get("tls", {})),
    )
