# Copyright (C) 2026 AniiGIT
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
agent/connection.py - Outbound WebSocket connections from agent to broker.

The agent always dials out; it never opens an inbound port. This module
owns the connection lifecycle for both channels:

  Control channel  (JSON,   /ws/agent)
    hello handshake, idle status updates, transparency logging, reconnect.

  CUDA channel     (binary, /ws/cuda/{agent_id})
    Opened when the system first goes idle; closed when it goes active.
    Carries binary CUDA API call frames from the broker/sidecar.
    See agent/cuda_channel.py and agent/frame.py for the frame format.

Both channels connect over wss:// (mTLS) when an ssl.SSLContext is
supplied, or ws:// in dev mode / when TLS is unconfigured.

Control channel message protocol (JSON):

  Agent -> Broker:
    {"type": "hello",  "agent_id": "<uuid>", "hostname": "<name>"}
    {"type": "status", "state": "idle|active",
                       "input_secs": N, "gpu_pct": N|null, "cpu_pct": N.N}

  Broker -> Agent:
    {"type": "welcome"}
    {"type": "pong"}    (automatic; websockets library handles ping/pong)
"""

from __future__ import annotations

import asyncio
import json
import logging
import socket
import ssl
import uuid
from pathlib import Path

import websockets
import websockets.exceptions

from .config import AgentConfig
from .cuda_channel import CudaChannel
from .idle_monitor import is_system_idle, warmup_cpu
from .transparency_log import Event, TransparencyLog

logger = logging.getLogger(__name__)

# Reconnect backoff bounds (seconds).
_RECONNECT_MIN = 2.0
_RECONNECT_MAX = 60.0

# Seconds to wait for the broker welcome message after sending hello.
_WELCOME_TIMEOUT = 10.0

# WebSocket keep-alive: send a ping every N seconds, close if no pong within N seconds.
_PING_INTERVAL = 20
_PING_TIMEOUT = 10


# ---------------------------------------------------------------------------
# Agent identity
# ---------------------------------------------------------------------------


def _load_or_create_agent_id(data_dir: Path) -> str:
    """Return a stable UUID for this agent, creating and persisting it on first run.

    The id is stored as plain text in data_dir/agent_id. If the file cannot
    be read or written, an ephemeral id is returned and a warning is logged.
    """
    id_file = data_dir / "agent_id"

    try:
        if id_file.exists():
            candidate = id_file.read_text(encoding="ascii").strip()
            if candidate:
                return candidate
    except OSError as exc:
        logger.warning("could not read agent_id from %s: %s", id_file, exc)

    new_id = str(uuid.uuid4())

    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        id_file.write_text(new_id + "\n", encoding="ascii")
        logger.info("created new agent_id %s (stored in %s)", new_id, id_file)
    except OSError as exc:
        logger.warning(
            "could not persist agent_id to %s: %s; using ephemeral id %s",
            id_file, exc, new_id,
        )

    return new_id


# ---------------------------------------------------------------------------
# BrokerConnection
# ---------------------------------------------------------------------------


class BrokerConnection:
    """Manages the outbound WebSocket connection from agent to broker.

    Call run() from the asyncio event loop. It connects, registers, runs
    the idle detection loop, and reconnects automatically on failure.
    """

    def __init__(
        self,
        cfg: AgentConfig,
        tlog: TransparencyLog,
        ssl_ctx: ssl.SSLContext | None = None,
    ) -> None:
        data_dir = Path(cfg.logging.transparency_log).parent
        self._cfg = cfg
        self._tlog = tlog
        self._ssl_ctx = ssl_ctx
        self._agent_id = _load_or_create_agent_id(data_dir)
        self._hostname = socket.gethostname()
        self._cuda = CudaChannel(cfg, self._agent_id, ssl_ctx)

    async def run(self) -> None:
        """Connect to the broker and keep reconnecting on failure.

        Reconnect delay starts at _RECONNECT_MIN seconds and doubles on each
        consecutive failure up to _RECONNECT_MAX. A successful session resets
        the delay.
        """
        delay = _RECONNECT_MIN

        while True:
            try:
                await self._session()
                # Clean disconnect (e.g. broker sent close frame); reset delay.
                delay = _RECONNECT_MIN
            except Exception as exc:
                logger.warning(
                    "broker connection failed: %s; retrying in %.0fs", exc, delay
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, _RECONNECT_MAX)

    async def _session(self) -> None:
        """Run one connected session until the WebSocket closes."""
        scheme = "wss" if self._ssl_ctx is not None else "ws"
        uri = f"{scheme}://{self._cfg.broker.host}:{self._cfg.broker.port}/ws/agent"
        logger.info("connecting to broker at %s", uri)

        connect_kwargs: dict = {
            "ping_interval": _PING_INTERVAL,
            "ping_timeout": _PING_TIMEOUT,
        }
        if self._ssl_ctx is not None:
            connect_kwargs["ssl"] = self._ssl_ctx

        async with websockets.connect(  # type: ignore[attr-defined]
            uri,
            **connect_kwargs,
        ) as ws:
            await self._handshake(ws)

            self._tlog.write(Event.CONNECTED, broker=self._cfg.broker.host)
            logger.info("connected to broker (agent_id=%s)", self._agent_id)

            try:
                await self._idle_loop(ws)
            finally:
                # Ensure the CUDA channel is closed whenever the control
                # channel drops, regardless of the reason.
                await self._cuda.close()
                self._tlog.write(Event.DISCONNECTED, reason="connection_lost")

    async def _handshake(self, ws: websockets.WebSocketClientProtocol) -> None:  # type: ignore[name-defined]
        """Send hello and wait for welcome. Raises on failure."""
        await ws.send(json.dumps({
            "type": "hello",
            "agent_id": self._agent_id,
            "hostname": self._hostname,
        }))

        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=_WELCOME_TIMEOUT)
        except asyncio.TimeoutError:
            raise ConnectionError(
                f"broker did not send welcome within {_WELCOME_TIMEOUT:.0f}s"
            )

        msg = json.loads(raw)
        if msg.get("type") != "welcome":
            raise ConnectionError(
                f"expected welcome from broker, got {msg.get('type')!r}"
            )

    async def _idle_loop(self, ws: websockets.WebSocketClientProtocol) -> None:  # type: ignore[name-defined]
        """Poll idle state, stream status updates, and manage the CUDA channel."""
        warmup_cpu()
        was_idle = False
        cfg = self._cfg.idle

        while True:
            idle, input_secs, gpu_pct, cpu_pct = is_system_idle(cfg)

            # Log transitions and open/close the CUDA channel accordingly.
            if idle and not was_idle:
                self._tlog.write(Event.IDLE, input=f"{input_secs}s")
                await self._cuda.open()

            elif not idle and was_idle:
                self._tlog.write(Event.ACTIVE)
                await self._cuda.close()

            was_idle = idle

            await ws.send(json.dumps({
                "type": "status",
                "state": "idle" if idle else "active",
                "input_secs": input_secs,
                "gpu_pct": gpu_pct,        # may be null (NVML unavailable)
                "cpu_pct": round(cpu_pct, 1),
            }))

            interval = cfg.poll_active_seconds if idle else cfg.poll_idle_seconds
            await asyncio.sleep(interval)
