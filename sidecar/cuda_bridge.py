# Copyright (C) 2026 AniiGIT
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
sidecar/cuda_bridge.py - WebSocket connection from the sidecar to the broker.

The sidecar connects to:
    wss://broker/ws/sidecar/wait/{agent_id}

The broker holds this connection open until the named agent registers an
idle CUDA channel, then pairs them and forwards CUDA frames bidirectionally.
No polling or reconnect backoff is needed while waiting for the agent —
the connection simply stays open.

Call flow for a single CUDA API call:
  1. IpcServer receives an IPC call from the shim and decodes it to a dict.
  2. IpcServer calls CudaBridge.call(call_id, func_id, payload_dict).
  3. CudaBridge sends a CUDA_CALL frame (msgpack) to the broker WebSocket.
  4. The broker forwards it to the agent; the agent executes and responds.
  5. The broker forwards the CUDA_RETURN frame back to the sidecar.
  6. CudaBridge._recv_loop receives the response frame and resolves the
     asyncio.Future that call() is waiting on.
  7. call() returns the response dict to IpcServer.
  8. IpcServer encodes the dict to packed-C bytes and sends back to shim.

Multiple shim connections can be active simultaneously. All share this
single WebSocket. Calls are matched to responses by call_id.
"""

from __future__ import annotations

import asyncio
import logging
import ssl

import websockets
import websockets.exceptions

from .frame import CUDA_CALL, CUDA_RETURN, DISCONNECT, Frame, NVENC_CALL, NVENC_RETURN, pack, unpack

logger = logging.getLogger(__name__)

_RECONNECT_DELAY = 5.0
_PING_INTERVAL = 20
_PING_TIMEOUT = 10


class CudaBridge:
    """
    Manages the outbound WebSocket connection to the broker's sidecar endpoint.

    Start the bridge with run_forever() as a background asyncio task.
    Use call() to forward individual IPC requests and await responses.
    """

    def __init__(self, broker_host: str, broker_port: int, agent_id: str,
                 ssl_ctx: ssl.SSLContext | None = None) -> None:
        self._broker_host = broker_host
        self._broker_port = broker_port
        self._agent_id = agent_id
        self._ssl_ctx = ssl_ctx

        # Pending calls: call_id → Future that resolves to the response dict.
        self._pending: dict[int, asyncio.Future] = {}  # type: ignore[type-arg]

        # Serialise sends so concurrent IPC handlers don't interleave frames.
        self._send_lock = asyncio.Lock()

        # Set when the WebSocket is connected; cleared on disconnect.
        self._connected = asyncio.Event()

        # Active WebSocket — written only by the run_forever task.
        self._ws: websockets.WebSocketClientProtocol | None = None  # type: ignore[name-defined]

        # Set to True to stop run_forever cleanly.
        self._stopping = False

    async def run_forever(self) -> None:
        """Connect to broker and reconnect automatically on failure.

        Call this as an asyncio background task. It exits cleanly when
        _stopping is True or the task is cancelled.
        """
        scheme = "wss" if self._ssl_ctx is not None else "ws"
        uri = (
            f"{scheme}://{self._broker_host}:{self._broker_port}"
            f"/ws/sidecar/wait/{self._agent_id}"
        )

        connect_kwargs: dict = {
            "ping_interval": _PING_INTERVAL,
            "ping_timeout": _PING_TIMEOUT,
        }
        if self._ssl_ctx is not None:
            connect_kwargs["ssl"] = self._ssl_ctx

        try:
            while not self._stopping:
                try:
                    logger.info("cuda bridge: connecting to %s", uri)
                    async with websockets.connect(  # type: ignore[attr-defined]
                        uri,
                        **connect_kwargs,
                    ) as ws:
                        self._ws = ws
                        self._connected.set()
                        logger.info("cuda bridge: connected (agent_id=%s)", self._agent_id)
                        try:
                            await self._recv_loop(ws)
                        finally:
                            self._ws = None
                            self._connected.clear()
                            self._cancel_pending("broker disconnected")
                        logger.info("cuda bridge: connection closed cleanly")

                except websockets.exceptions.ConnectionClosed as exc:
                    self._ws = None
                    self._connected.clear()
                    self._cancel_pending(f"connection closed: {exc}")
                    if not self._stopping:
                        logger.warning(
                            "cuda bridge: disconnected (%s); retrying in %.0fs",
                            exc, _RECONNECT_DELAY,
                        )
                        await asyncio.sleep(_RECONNECT_DELAY)

                except OSError as exc:
                    self._cancel_pending(f"connection failed: {exc}")
                    if not self._stopping:
                        logger.warning(
                            "cuda bridge: could not connect (%s); retrying in %.0fs",
                            exc, _RECONNECT_DELAY,
                        )
                        await asyncio.sleep(_RECONNECT_DELAY)

        except asyncio.CancelledError:
            self._cancel_pending("bridge task cancelled")

    async def call(self, call_id: int, func_id: int, payload: dict) -> dict:  # type: ignore[type-arg]
        """Send a CUDA_CALL frame and await the CUDA_RETURN response.

        Raises ConnectionError if the broker connection is not available.
        """
        await self._connected.wait()

        if self._ws is None:
            raise ConnectionError("no active broker connection")

        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()  # type: ignore[type-arg]
        self._pending[call_id] = fut

        try:
            frame = Frame(msg_type=CUDA_CALL, call_id=call_id, payload=payload)
            async with self._send_lock:
                await self._ws.send(pack(frame))

            return await fut  # type: ignore[return-value]

        except Exception:
            self._pending.pop(call_id, None)
            raise

    async def send_disconnect(self, reason: str = "sidecar_stopping") -> None:
        """Send a DISCONNECT frame to the broker (best-effort)."""
        if self._ws is None:
            return
        try:
            frame = Frame(msg_type=DISCONNECT, call_id=0, payload={"reason": reason})
            async with self._send_lock:
                await self._ws.send(pack(frame))
        except Exception as exc:
            logger.debug("cuda bridge: send_disconnect failed: %s", exc)

    def stop(self) -> None:
        """Signal the bridge to stop reconnecting."""
        self._stopping = True
        self._connected.set()  # unblock any waiting call()

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _recv_loop(
        self,
        ws: websockets.WebSocketClientProtocol,  # type: ignore[name-defined]
    ) -> None:
        """Receive frames from broker and resolve pending call() futures."""
        async for raw in ws:
            if not isinstance(raw, bytes):
                continue
            try:
                frame = unpack(raw)
            except (ValueError, Exception) as exc:
                logger.warning("cuda bridge: malformed frame: %s", exc)
                continue

            if frame.msg_type in (CUDA_RETURN, NVENC_RETURN):
                fut = self._pending.pop(frame.call_id, None)
                if fut is not None and not fut.done():
                    fut.set_result(frame.payload)
                else:
                    logger.warning(
                        "cuda bridge: unexpected call_id %d in response (already resolved or unknown)",
                        frame.call_id,
                    )

            elif frame.msg_type == DISCONNECT:
                logger.info("cuda bridge: received DISCONNECT from broker")
                break

            else:
                logger.warning(
                    "cuda bridge: unexpected msg_type=%d; ignoring", frame.msg_type
                )

    def _cancel_pending(self, reason: str) -> None:
        """Cancel all pending call() futures with a ConnectionError."""
        if not self._pending:
            return
        exc = ConnectionError(f"cuda bridge: {reason}")
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(exc)
        self._pending.clear()
