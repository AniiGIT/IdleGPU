# Copyright (C) 2026 AniiGIT
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
broker/registry.py - In-memory registry of connected agents.

Tracks every agent that currently has an open WebSocket connection to the
broker: its identity, current idle/active state, and latest metrics.

Thread-safe: the registry is written from the WebSocket handler coroutines
(which run in the asyncio event loop) and read by the /status HTTP endpoint.
Both paths hold the lock for only the minimum time needed.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class AgentRecord:
    """Snapshot of one connected agent."""

    agent_id: str
    """Stable identifier supplied by the agent in the hello message."""

    hostname: str
    """Self-reported hostname of the gaming PC."""

    remote_addr: str
    """IP address of the incoming WebSocket connection."""

    connected_at: datetime
    """Timestamp when the agent completed its hello handshake."""

    state: str = "unknown"
    """Current idle/active state: 'idle', 'active', or 'unknown' before first status."""

    last_seen: datetime = field(default_factory=datetime.now)
    """Timestamp of the most recent message received from this agent."""

    gpu_pct: int | None = None
    """Most recent GPU utilization percent, or None if not yet reported."""

    cpu_pct: float | None = None
    """Most recent CPU utilization percent, or None if not yet reported."""

    input_secs: int | None = None
    """Seconds since last user input, or None if not yet reported."""


class AgentRegistry:
    """Thread-safe in-memory store of currently connected agents.

    Agents are keyed by agent_id. An agent is present in the registry only
    while its WebSocket connection is open; it is removed on disconnect.
    """

    def __init__(self) -> None:
        self._agents: dict[str, AgentRecord] = {}
        self._lock = threading.Lock()

    def add(self, record: AgentRecord) -> None:
        """Register a newly connected agent. Replaces any stale record with the same id."""
        with self._lock:
            self._agents[record.agent_id] = record

    def remove(self, agent_id: str) -> None:
        """Remove an agent on disconnect. Silent if the id is not present."""
        with self._lock:
            self._agents.pop(agent_id, None)

    def update(
        self,
        agent_id: str,
        state: str | None = None,
        gpu_pct: int | None = None,
        cpu_pct: float | None = None,
        input_secs: int | None = None,
    ) -> None:
        """Apply a status update from an agent and refresh its last_seen timestamp."""
        with self._lock:
            rec = self._agents.get(agent_id)
            if rec is None:
                return
            if state is not None:
                rec.state = state
            if gpu_pct is not None:
                rec.gpu_pct = gpu_pct
            if cpu_pct is not None:
                rec.cpu_pct = cpu_pct
            if input_secs is not None:
                rec.input_secs = input_secs
            rec.last_seen = datetime.now()

    def all(self) -> list[AgentRecord]:
        """Return a snapshot of all currently connected agents."""
        with self._lock:
            return list(self._agents.values())

    def count(self) -> int:
        """Return the number of currently connected agents."""
        with self._lock:
            return len(self._agents)
