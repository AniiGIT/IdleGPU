# Copyright (C) 2026 AniiGIT
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
agent/transparency_log.py - Append-only local transparency log.

Every significant agent event is written here so the operator always knows
what their GPU was used for and when. The log is append-only and never
truncated by the agent.

Format (one event per line):
  [2026-01-01 03:14:22] CONNECTED    broker=192.168.1.10
  [2026-01-01 03:14:25] JOB_START    backend=cuda-server
  [2026-01-01 03:28:55] JOB_END      duration=00:14:30
  [2026-01-01 03:28:56] DISCONNECTED reason=host_active

The log path comes from LoggingSection (loaded from agent.toml) so no
paths are hardcoded here.

Security notes:
- Only the broker address is written as an identifier; no other IPs.
- No credentials, keys, certificates, or file paths beyond config values.
- The caller is responsible for not passing sensitive data as kwargs.
"""

from __future__ import annotations

import enum
import sys
import threading
from datetime import datetime
from pathlib import Path

from .config import LoggingSection


# ---------------------------------------------------------------------------
# Event type enum
# ---------------------------------------------------------------------------


class Event(enum.Enum):
    CONNECTED = "CONNECTED"
    DISCONNECTED = "DISCONNECTED"
    JOB_START = "JOB_START"
    JOB_END = "JOB_END"
    IDLE = "IDLE"
    ACTIVE = "ACTIVE"


# Width of the widest event name, used to align the details column.
_EVENT_WIDTH = max(len(e.value) for e in Event)


# ---------------------------------------------------------------------------
# TransparencyLog
# ---------------------------------------------------------------------------


class TransparencyLog:
    """Append-only, thread-safe log of agent events.

    Usage::

        log = TransparencyLog(cfg.logging)
        log.write(Event.CONNECTED, broker="192.168.1.10")
        log.write(Event.JOB_START, backend="cuda-server")
        log.write(Event.JOB_END, duration="00:14:30")
        log.write(Event.DISCONNECTED, reason="host_active")

    A failed write prints a one-line warning to stderr and returns normally.
    The agent is never crashed by a log failure.
    """

    def __init__(self, cfg: LoggingSection) -> None:
        self._path = Path(cfg.transparency_log)
        self._lock = threading.Lock()
        self._ensure_dir()

    def _ensure_dir(self) -> None:
        """Create the log directory if it does not exist."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            print(
                f"idlegpu: could not create log directory"
                f" {self._path.parent}: {exc}",
                file=sys.stderr,
            )

    def write(self, event: Event, **details: str) -> None:
        """Append one event line to the log.

        Args:
            event:      The event type (from the Event enum).
            **details:  Optional key=value pairs written after the event name.
                        Values must be plain strings. Do not pass secrets.

        Example::

            log.write(Event.JOB_END, duration="00:14:30")
            # writes: [2026-01-01 03:28:55] JOB_END      duration=00:14:30
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        event_col = event.value.ljust(_EVENT_WIDTH)
        detail_str = " ".join(f"{k}={v}" for k, v in details.items())

        if detail_str:
            line = f"[{timestamp}] {event_col} {detail_str}\n"
        else:
            line = f"[{timestamp}] {event_col}\n"

        with self._lock:
            try:
                with open(self._path, "a", encoding="utf-8") as fh:
                    fh.write(line)
            except OSError as exc:
                print(
                    f"idlegpu: transparency log write failed"
                    f" ({self._path}): {exc}"
                    f" -- if the log directory was deleted, restart the agent to recreate it.",
                    file=sys.stderr,
                )
