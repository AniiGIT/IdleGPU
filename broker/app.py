# Copyright (C) 2026 AniiGIT
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
broker/app.py - FastAPI application: agent WebSocket endpoint and /status API.

Phase 1 note: connections are plaintext WebSocket with no authentication.
mTLS and signed job payloads are added in Phase 2. Do not deploy Phase 1
on an untrusted network.

Agent WebSocket protocol (JSON messages):

  Agent -> Broker:
    {"type": "hello",  "agent_id": "<uuid>", "hostname": "<name>"}
    {"type": "status", "state": "idle|active", "input_secs": N,
                       "gpu_pct": N, "cpu_pct": N.N}
    {"type": "ping"}

  Broker -> Agent:
    {"type": "welcome"}
    {"type": "pong"}

The agent must send hello as its very first message. The connection is
closed with code 1002 if any other message arrives first or if hello is
missing required fields.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from .registry import AgentRecord, AgentRegistry

logger = logging.getLogger(__name__)

# Module-level registry shared between the WebSocket handler and /status.
registry = AgentRegistry()

app = FastAPI(
    title="IdleGPU Broker",
    description="Agent registry and job routing for IdleGPU.",
    version="0.1.0",
)

# Seconds to wait for the hello message after a WebSocket connection opens.
_HELLO_TIMEOUT = 10.0


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------


@app.get("/status")
async def status() -> dict:
    """Return the list of currently connected agents and their state."""
    agents = [
        {
            "agent_id": r.agent_id,
            "hostname": r.hostname,
            "remote_addr": r.remote_addr,
            "connected_at": r.connected_at.isoformat(),
            "state": r.state,
            "last_seen": r.last_seen.isoformat(),
            "gpu_pct": r.gpu_pct,
            "cpu_pct": r.cpu_pct,
            "input_secs": r.input_secs,
        }
        for r in registry.all()
    ]
    return {"agent_count": len(agents), "agents": agents}


# ---------------------------------------------------------------------------
# Agent WebSocket endpoint
# ---------------------------------------------------------------------------


@app.websocket("/ws/agent")
async def agent_websocket(websocket: WebSocket) -> None:
    """Handle one agent WebSocket session from connect through disconnect."""
    await _handle_agent_session(websocket)


async def _handle_agent_session(websocket: WebSocket) -> None:
    """
    Manage the lifecycle of a single agent connection.

    Separated from the route handler so it can be unit-tested without
    a running ASGI server.
    """
    remote = websocket.client.host if websocket.client else "unknown"
    agent_id: str | None = None

    try:
        await websocket.accept()

        # -- hello handshake -------------------------------------------------
        try:
            raw = await asyncio.wait_for(
                websocket.receive_json(), timeout=_HELLO_TIMEOUT
            )
        except asyncio.TimeoutError:
            logger.warning("agent %s did not send hello within %.0fs; closing", remote, _HELLO_TIMEOUT)
            await websocket.close(code=1001, reason="hello timeout")
            return

        if raw.get("type") != "hello":
            logger.warning("agent %s sent '%s' before hello; closing", remote, raw.get("type"))
            await websocket.close(code=1002, reason="expected hello")
            return

        agent_id = str(raw.get("agent_id", "")).strip()
        hostname = str(raw.get("hostname", "unknown")).strip()

        if not agent_id:
            logger.warning("agent %s sent hello with empty agent_id; closing", remote)
            await websocket.close(code=1002, reason="agent_id required")
            return

        registry.add(AgentRecord(
            agent_id=agent_id,
            hostname=hostname,
            remote_addr=remote,
            connected_at=datetime.now(),
        ))
        logger.info("agent connected: id=%s hostname=%s addr=%s", agent_id, hostname, remote)
        await websocket.send_json({"type": "welcome"})

        # -- message loop ----------------------------------------------------
        while True:
            msg = await websocket.receive_json()
            msg_type = msg.get("type", "")

            if msg_type == "status":
                _apply_status(agent_id, msg)

            elif msg_type == "ping":
                await websocket.send_json({"type": "pong"})

            else:
                logger.warning("agent %s sent unknown message type %r; ignoring", agent_id, msg_type)

    except WebSocketDisconnect:
        logger.info("agent disconnected: id=%s addr=%s", agent_id or remote, remote)

    except Exception as exc:
        logger.warning("agent session error (id=%s addr=%s): %s", agent_id or remote, remote, exc)

    finally:
        if agent_id:
            registry.remove(agent_id)


def _apply_status(agent_id: str, msg: dict) -> None:  # type: ignore[type-arg]
    """Parse a status message and push the update into the registry."""
    state = str(msg.get("state", "unknown"))

    raw_gpu = msg.get("gpu_pct")
    gpu_pct = int(raw_gpu) if raw_gpu is not None else None

    raw_cpu = msg.get("cpu_pct")
    cpu_pct = float(raw_cpu) if raw_cpu is not None else None

    raw_input = msg.get("input_secs")
    input_secs = int(raw_input) if raw_input is not None else None

    registry.update(
        agent_id,
        state=state,
        gpu_pct=gpu_pct,
        cpu_pct=cpu_pct,
        input_secs=input_secs,
    )
