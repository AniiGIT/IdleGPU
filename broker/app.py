# Copyright (C) 2026 AniiGIT
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
broker/app.py - Two FastAPI applications: enrollment server and main broker.

Two separate ASGI apps run on separate ports:

  enroll_app  (plain HTTP, enroll_port):
    GET  /ca.crt          - Serve CA certificate PEM (unauthenticated; public artifact)
    POST /enroll          - Accept {token, agent_id, csr_pem}; return signed agent cert
    POST /enroll/sidecar  - Accept {token, sidecar_id, csr_pem}; return signed sidecar cert
                            Both endpoints consume the same single-use token.

    Runs on a dedicated plain HTTP port so agents and sidecars can bootstrap
    credentials before they have a client certificate for the mTLS main port.

  app  (mTLS when [tls] is configured, server.port):
    GET  /status            - Connected agent list and metrics
    WS   /ws/agent          - Agent control channel (JSON, idle status)
    WS   /ws/cuda/{id}           - Agent CUDA channel (binary frames, msgpack)
    WS   /ws/sidecar/{id}        - Sidecar CUDA channel; pairs immediately or queues
    WS   /ws/sidecar/wait/{id}   - Sidecar CUDA channel; waits for named agent (preferred)

CUDA channel binary frame format (see broker/frame.py):

  [4 bytes big-endian: msg_type] [4 bytes: call_id] [4 bytes: payload_len]
  [payload_len bytes: msgpack dict]

  The broker routes binary frames verbatim — it never decodes payload.
  Frames flow: sidecar → broker → agent (calls), agent → broker → sidecar (returns).

Agent control channel protocol (JSON messages):

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
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from .cuda_router import CudaRouter
from .registry import AgentRecord, AgentRegistry

logger = logging.getLogger(__name__)

# Module-level singletons shared between route handlers.
registry = AgentRegistry()
cuda_router = CudaRouter()

app = FastAPI(
    title="IdleGPU Broker",
    description="Agent registry and job routing for IdleGPU.",
    version="0.1.0",
)

# Enrollment server -- plain HTTP on enroll_port (see broker/__main__.py).
# Only serves /ca.crt and /enroll so agents can bootstrap before they hold a
# client certificate for the mTLS main port.
enroll_app = FastAPI(
    title="IdleGPU Enrollment",
    description="Agent enrollment bootstrap (plain HTTP, separate port).",
    version="0.1.0",
)

# Seconds to wait for the hello message after a WebSocket connection opens.
_HELLO_TIMEOUT = 10.0


# ---------------------------------------------------------------------------
# Enrollment helpers
# ---------------------------------------------------------------------------


def _cfg_data_dir(request: Request) -> Path:
    """Return the data directory from the app-state config set by cmd_start."""
    cfg = getattr(request.app.state, "cfg", None)
    if cfg is None:
        raise HTTPException(status_code=503, detail="Broker not fully initialised.")
    return Path(cfg.data.data_dir)


# ---------------------------------------------------------------------------
# Enrollment REST endpoints (HTTP plaintext -- bootstrap only)
# ---------------------------------------------------------------------------


class EnrollRequest(BaseModel):
    token: str
    agent_id: str
    csr_pem: str


class EnrollResponse(BaseModel):
    ca_cert_pem: str
    agent_cert_pem: str


class EnrollSidecarRequest(BaseModel):
    token: str
    sidecar_id: str
    csr_pem: str


class EnrollSidecarResponse(BaseModel):
    ca_cert_pem: str
    sidecar_cert_pem: str


@enroll_app.get("/ca.crt")
async def get_ca_cert(request: Request) -> Response:
    """
    Serve the broker CA certificate PEM.

    Unauthenticated -- agents fetch this before they have any credentials.
    Operators should verify the CA fingerprint printed by `idlegpu-broker setup`
    out of band before trusting a new broker.
    """
    data_dir = _cfg_data_dir(request)
    ca_path = data_dir / "ca.crt"

    if not ca_path.exists():
        raise HTTPException(
            status_code=404,
            detail="CA certificate not found. Run `idlegpu-broker setup` first.",
        )

    try:
        pem = ca_path.read_text(encoding="ascii")
    except OSError as exc:
        logger.error("could not read CA cert %s: %s", ca_path, exc)
        raise HTTPException(status_code=500, detail="Could not read CA certificate.")

    return PlainTextResponse(content=pem, media_type="application/x-pem-file")


@enroll_app.post("/enroll", response_model=EnrollResponse)
async def enroll(body: EnrollRequest, request: Request) -> EnrollResponse:
    """
    Enroll an agent: validate the one-time token and sign the submitted CSR.

    The token is consumed on first successful use. Any subsequent call with
    the same token is rejected with 403. Run `idlegpu-broker setup` to
    generate a fresh token for additional agents.
    """
    from .pki import consume_enrollment_token, sign_csr  # noqa: PLC0415

    data_dir = _cfg_data_dir(request)
    ca_path = data_dir / "ca.crt"
    ca_key_path = data_dir / "ca.key"

    if not ca_path.exists() or not ca_key_path.exists():
        raise HTTPException(
            status_code=503,
            detail="PKI not initialised. Run `idlegpu-broker setup` first.",
        )

    # Validate and consume the token atomically -- returns False on mismatch.
    if not consume_enrollment_token(data_dir, body.token):
        logger.warning(
            "enrollment rejected for agent %s: invalid or expired token",
            body.agent_id,
        )
        raise HTTPException(
            status_code=403,
            detail="Invalid or expired enrollment token.",
        )

    try:
        ca_cert_pem = ca_path.read_bytes()
        ca_key_pem = ca_key_path.read_bytes()
    except OSError as exc:
        logger.error("could not read CA credentials: %s", exc)
        raise HTTPException(status_code=500, detail="Could not read CA credentials.")

    try:
        agent_cert_pem = sign_csr(ca_cert_pem, ca_key_pem, body.csr_pem.encode())
    except ValueError as exc:
        logger.warning("CSR signing failed for agent %s: %s", body.agent_id, exc)
        raise HTTPException(status_code=400, detail=f"CSR error: {exc}")

    logger.info("enrolled agent %s", body.agent_id)
    return EnrollResponse(
        ca_cert_pem=ca_cert_pem.decode("ascii"),
        agent_cert_pem=agent_cert_pem.decode("ascii"),
    )


@enroll_app.post("/enroll/sidecar", response_model=EnrollSidecarResponse)
async def enroll_sidecar(body: EnrollSidecarRequest, request: Request) -> EnrollSidecarResponse:
    """
    Enroll a sidecar: validate the one-time token and sign the submitted CSR.

    Identical to POST /enroll except the request body carries a sidecar_id and
    the response field is sidecar_cert_pem.  Uses the same single-use token
    mechanism -- run `idlegpu-broker setup` to generate a fresh token for each
    enrollment (agents and sidecars share the same token pool).
    """
    from .pki import consume_enrollment_token, sign_csr  # noqa: PLC0415

    data_dir = _cfg_data_dir(request)
    ca_path = data_dir / "ca.crt"
    ca_key_path = data_dir / "ca.key"

    if not ca_path.exists() or not ca_key_path.exists():
        raise HTTPException(
            status_code=503,
            detail="PKI not initialised. Run `idlegpu-broker setup` first.",
        )

    if not consume_enrollment_token(data_dir, body.token):
        logger.warning(
            "enrollment rejected for sidecar %s: invalid or expired token",
            body.sidecar_id,
        )
        raise HTTPException(
            status_code=403,
            detail="Invalid or expired enrollment token.",
        )

    try:
        ca_cert_pem = ca_path.read_bytes()
        ca_key_pem = ca_key_path.read_bytes()
    except OSError as exc:
        logger.error("could not read CA credentials: %s", exc)
        raise HTTPException(status_code=500, detail="Could not read CA credentials.")

    try:
        sidecar_cert_pem = sign_csr(ca_cert_pem, ca_key_pem, body.csr_pem.encode())
    except ValueError as exc:
        logger.warning("CSR signing failed for sidecar %s: %s", body.sidecar_id, exc)
        raise HTTPException(status_code=400, detail=f"CSR error: {exc}")

    logger.info("enrolled sidecar %s", body.sidecar_id)
    return EnrollSidecarResponse(
        ca_cert_pem=ca_cert_pem.decode("ascii"),
        sidecar_cert_pem=sidecar_cert_pem.decode("ascii"),
    )


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


# ---------------------------------------------------------------------------
# Agent CUDA channel WebSocket endpoint
# ---------------------------------------------------------------------------


@app.websocket("/ws/cuda/{agent_id}")
async def agent_cuda_websocket(websocket: WebSocket, agent_id: str) -> None:
    """
    Accept the agent's binary CUDA channel.

    The agent opens this connection when it transitions to idle and holds it
    open until it goes active again. While open, the broker pairs incoming
    sidecar connections with this channel and relays binary CUDA frames
    between them.

    Binary frame format: see broker/frame.py.
    """
    await websocket.accept()
    await cuda_router.handle_agent(agent_id, websocket)


# ---------------------------------------------------------------------------
# Sidecar CUDA channel WebSocket endpoint
# ---------------------------------------------------------------------------


@app.websocket("/ws/sidecar/{agent_id}")
async def sidecar_cuda_websocket(websocket: WebSocket, agent_id: str) -> None:
    """
    Accept a sidecar connection targeting a specific agent.

    If the named agent has an idle CUDA channel, pairs and relays immediately.
    If not, queues the sidecar and pairs the moment the agent becomes idle.
    The connection stays open until paired and the relay completes.

    Prefer /ws/sidecar/wait/{agent_id} for new sidecar implementations —
    it is semantically identical but makes the waiting intent explicit.
    """
    await websocket.accept()
    await cuda_router.handle_sidecar(agent_id, websocket)


@app.websocket("/ws/sidecar/wait/{agent_id}")
async def sidecar_wait_websocket(websocket: WebSocket, agent_id: str) -> None:
    """
    Accept a sidecar connection that waits for the named agent to become idle.

    Identical to /ws/sidecar/{agent_id} — both endpoints use the same push
    pairing model. This path exists as the canonical endpoint for sidecar
    implementations; the plain /ws/sidecar/{agent_id} path is retained for
    backwards compatibility.
    """
    await websocket.accept()
    await cuda_router.handle_sidecar(agent_id, websocket)
