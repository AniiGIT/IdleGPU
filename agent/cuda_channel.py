# Copyright (C) 2026 AniiGIT
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
agent/cuda_channel.py - Second outbound WebSocket from agent to broker.

While the control channel (connection.py) carries JSON idle-status messages,
the CUDA channel carries binary CUDA API call frames from the broker/sidecar
to the agent, and return-value frames back.

Lifecycle (managed by BrokerConnection._idle_loop):

  System goes idle   → CudaChannel.open()  called
  CUDA calls arrive  → _dispatch_frame() executes on real GPU via CudaExecutor
  System goes active → CudaChannel.close(graceful=True) sends DISCONNECT first

Frame format: see agent/frame.py.
"""

from __future__ import annotations

import asyncio
import logging
import ssl

import websockets
import websockets.exceptions

from .config import AgentConfig
from .cuda_executor import CudaExecutor
from .frame import (
    CUDA_CALL, CUDA_RETURN, DISCONNECT, NVENC_CALL, NVENC_RETURN,
    Frame, type_name, pack, unpack,
)

logger = logging.getLogger(__name__)

# Seconds to wait before retrying a failed CUDA channel connection.
_RECONNECT_DELAY = 5.0

# WebSocket keep-alive settings.
_PING_INTERVAL = 20
_PING_TIMEOUT = 10


class CudaChannel:
    """
    Second outbound WebSocket from the agent to the broker.

    Open while the system is idle; closed when the system goes active.
    Receives binary CUDA frames from the broker (forwarded from the sidecar)
    and will dispatch them to the local CUDA executor in Phase 2D.

    Usage:
        channel = CudaChannel(cfg, agent_id, ssl_ctx)
        await channel.open()   # call when system becomes idle
        ...
        await channel.close()  # call when system becomes active
    """

    def __init__(
        self,
        cfg: AgentConfig,
        agent_id: str,
        ssl_ctx: ssl.SSLContext | None = None,
    ) -> None:
        self._cfg = cfg
        self._agent_id = agent_id
        self._ssl_ctx = ssl_ctx
        self._task: asyncio.Task | None = None  # type: ignore[type-arg]
        self._executor = CudaExecutor()
        # Active WebSocket reference for graceful close; written by _run task.
        self._active_ws: websockets.WebSocketClientProtocol | None = None  # type: ignore[name-defined]

    @property
    def is_open(self) -> bool:
        """True while the channel task is running."""
        return self._task is not None and not self._task.done()

    async def open(self) -> None:
        """Open the CUDA channel in a background task.

        No-op if already open.
        """
        if self.is_open:
            return
        self._task = asyncio.create_task(self._run(), name="cuda-channel")
        logger.info("cuda channel opening for agent %s", self._agent_id)

    async def close(self, graceful: bool = False) -> None:
        """Close the CUDA channel and wait for the background task to finish.

        When graceful=True, sends a DISCONNECT frame to the broker before
        cancelling the task so the sidecar can drain in-flight calls cleanly.

        No-op if already closed.
        """
        if not self.is_open:
            return
        assert self._task is not None

        if graceful and self._active_ws is not None:
            try:
                disconnect = pack(Frame(
                    msg_type=DISCONNECT,
                    call_id=0,
                    payload={"reason": "host_active"},
                ))
                await asyncio.wait_for(self._active_ws.send(disconnect), timeout=1.0)
                logger.debug("cuda channel: sent graceful DISCONNECT to broker")
            except Exception as exc:
                logger.debug("cuda channel: graceful DISCONNECT failed: %s", exc)

        self._task.cancel()
        await asyncio.gather(self._task, return_exceptions=True)
        self._task = None
        self._active_ws = None
        logger.info("cuda channel closed for agent %s", self._agent_id)

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    async def _run(self) -> None:
        """Connect to the broker's CUDA channel endpoint and receive frames."""
        scheme = "wss" if self._ssl_ctx is not None else "ws"
        uri = (
            f"{scheme}://{self._cfg.broker.host}:{self._cfg.broker.port}"
            f"/ws/cuda/{self._agent_id}"
        )

        connect_kwargs: dict = {
            "ping_interval": _PING_INTERVAL,
            "ping_timeout": _PING_TIMEOUT,
        }
        if self._ssl_ctx is not None:
            connect_kwargs["ssl"] = self._ssl_ctx

        try:
            while True:
                try:
                    async with websockets.connect(  # type: ignore[attr-defined]
                        uri,
                        **connect_kwargs,
                    ) as ws:
                        self._active_ws = ws
                        logger.info("cuda channel connected to broker (%s)", uri)
                        try:
                            await self._recv_loop(ws)
                        finally:
                            self._active_ws = None
                        # Clean close by broker — do not retry.
                        logger.info("cuda channel: broker closed connection cleanly")
                        return

                except websockets.exceptions.ConnectionClosed as exc:
                    logger.warning(
                        "cuda channel disconnected (%s); retrying in %.0fs",
                        exc, _RECONNECT_DELAY,
                    )
                    await asyncio.sleep(_RECONNECT_DELAY)

                except OSError as exc:
                    logger.warning(
                        "cuda channel: could not connect to %s (%s); retrying in %.0fs",
                        uri, exc, _RECONNECT_DELAY,
                    )
                    await asyncio.sleep(_RECONNECT_DELAY)

        except asyncio.CancelledError:
            logger.debug("cuda channel task cancelled (going active)")

    async def _recv_loop(
        self,
        ws: websockets.WebSocketClientProtocol,  # type: ignore[name-defined]
    ) -> None:
        """Receive binary frames and dispatch them."""
        while True:
            raw = await ws.recv()

            if not isinstance(raw, bytes):
                logger.warning(
                    "cuda channel: unexpected text frame; ignoring"
                )
                continue

            try:
                frame = unpack(raw)
            except (ValueError, Exception) as exc:
                logger.warning("cuda channel: malformed frame: %s", exc)
                continue

            await self._dispatch_frame(frame, ws)

    async def _dispatch_frame(
        self,
        frame: Frame,
        ws: websockets.WebSocketClientProtocol,  # type: ignore[name-defined]
    ) -> None:
        """Execute an incoming CUDA or NVENC call and send the return frame."""
        func_name: str = frame.payload.get("func", "<unknown>")

        if frame.msg_type == CUDA_CALL:
            logger.debug(
                "cuda channel: CUDA_CALL call_id=%d func=%s",
                frame.call_id, func_name,
            )
            # Execute on the local GPU (synchronous ctypes call — fast enough
            # that running in the event loop thread is acceptable for Tier 1).
            resp = self._executor.dispatch(frame.payload)
            return_frame = Frame(
                msg_type=CUDA_RETURN,
                call_id=frame.call_id,
                payload=resp,
            )
            await ws.send(pack(return_frame))

        elif frame.msg_type == NVENC_CALL:
            logger.debug(
                "cuda channel: NVENC_CALL call_id=%d func=%s",
                frame.call_id, func_name,
            )
            resp = self._executor.dispatch_nvenc(frame.payload)
            return_frame = Frame(
                msg_type=NVENC_RETURN,
                call_id=frame.call_id,
                payload=resp,
            )
            await ws.send(pack(return_frame))

        elif frame.msg_type == DISCONNECT:
            logger.info(
                "cuda channel: received DISCONNECT from sidecar (call_id=%d)",
                frame.call_id,
            )
            # Broker will close the WebSocket; recv_loop will terminate.

        else:
            logger.warning(
                "cuda channel: unexpected msg_type=%s call_id=%d; ignoring",
                type_name(frame.msg_type), frame.call_id,
            )
