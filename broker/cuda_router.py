# Copyright (C) 2026 AniiGIT
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
broker/cuda_router.py - Routes binary CUDA frames between agents and sidecars.

Lifecycle
─────────
  1. Agent connects to /ws/cuda/{agent_id}
       → router.handle_agent(agent_id, ws) is called
       → agent is stored in _idle[agent_id] and waits

  2. Sidecar connects to /ws/sidecar/{agent_id}
       → router.handle_sidecar(agent_id, ws) is called
       → broker pops the idle agent, assigns a job_id
       → bidirectional relay starts: frames flow broker-transparently
       → relay ends when either side disconnects
       → agent's waiting coroutine is unblocked and cleans up

  3. At any point before pairing, agent can disconnect
       → relay_done event is set, agent handler cleans up normally

The broker never inspects frame payload — it reads only the 12-byte header
to know how many bytes to forward (see frame.py).

Thread safety
─────────────
All operations run on the same asyncio event loop. No locking required.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field

from starlette.websockets import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


@dataclass
class _AgentEntry:
    """State for one idle agent CUDA channel waiting to be paired."""
    ws: WebSocket
    # Set by handle_sidecar when a relay is assigned, or by _monitor_task
    # when the agent disconnects before pairing.
    relay_done: asyncio.Event = field(default_factory=asyncio.Event)
    # Background task that drains the agent WebSocket while idle.
    # Cancelled by handle_sidecar when relay takes over.
    monitor_task: asyncio.Task | None = field(default=None)  # type: ignore[type-arg]


class CudaRouter:
    """Maintains idle CUDA channels and relays frames for active jobs.

    Instantiated once at module level in broker/app.py and shared between
    the two WebSocket route handlers.
    """

    def __init__(self) -> None:
        # Idle agent CUDA channels awaiting a sidecar connection.
        self._idle: dict[str, _AgentEntry] = {}

    # ------------------------------------------------------------------ #
    # Public helpers (read-only queries)                                   #
    # ------------------------------------------------------------------ #

    def idle_agent_ids(self) -> list[str]:
        """Return agent_ids of agents currently waiting for a CUDA job."""
        return list(self._idle)

    # ------------------------------------------------------------------ #
    # Agent handler                                                        #
    # ------------------------------------------------------------------ #

    async def handle_agent(self, agent_id: str, ws: WebSocket) -> None:
        """
        Called by the /ws/cuda/{agent_id} WebSocket handler.

        Registers the agent's CUDA channel as idle and blocks until:
          - a sidecar connects and the relay completes, OR
          - the agent disconnects.
        """
        entry = _AgentEntry(ws=ws)

        async def _monitor() -> None:
            """Drain the WebSocket while idle so disconnects are detected."""
            try:
                while True:
                    await ws.receive_bytes()
                    # Receiving pre-relay data is unexpected; discard silently.
            except WebSocketDisconnect:
                entry.relay_done.set()
            except asyncio.CancelledError:
                pass  # relay task is taking over

        entry.monitor_task = asyncio.create_task(_monitor())
        self._idle[agent_id] = entry
        logger.info("cuda channel: agent %s registered (idle)", agent_id)

        try:
            await entry.relay_done.wait()
        finally:
            if entry.monitor_task and not entry.monitor_task.done():
                entry.monitor_task.cancel()
                await asyncio.gather(entry.monitor_task, return_exceptions=True)
            self._idle.pop(agent_id, None)
            logger.info("cuda channel: agent %s unregistered", agent_id)

    # ------------------------------------------------------------------ #
    # Sidecar handler                                                      #
    # ------------------------------------------------------------------ #

    async def handle_sidecar(self, agent_id: str, sidecar_ws: WebSocket) -> None:
        """
        Called by the /ws/sidecar/{agent_id} WebSocket handler.

        Pops an idle agent, assigns a job_id, and relays binary frames
        bidirectionally until either side disconnects.
        """
        entry = self._idle.pop(agent_id, None)
        if entry is None:
            logger.warning(
                "sidecar requested agent %s but no idle CUDA channel found",
                agent_id,
            )
            await sidecar_ws.close(code=4001, reason=f"agent {agent_id} not available")
            return

        # Cancel the idle monitor — relay takes exclusive ownership of agent_ws.
        if entry.monitor_task and not entry.monitor_task.done():
            entry.monitor_task.cancel()
            await asyncio.gather(entry.monitor_task, return_exceptions=True)

        job_id = str(uuid.uuid4())
        logger.info(
            "job %s: paired sidecar with agent %s — relay starting",
            job_id, agent_id,
        )

        try:
            await self._relay(job_id, entry.ws, sidecar_ws)
        finally:
            # Unblock the agent handler (in handle_agent).
            entry.relay_done.set()
            logger.info("job %s: relay ended", job_id)

    # ------------------------------------------------------------------ #
    # Bidirectional frame relay                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    async def _relay(
        job_id: str,
        agent_ws: WebSocket,
        sidecar_ws: WebSocket,
    ) -> None:
        """
        Relay raw binary frames between agent and sidecar.

        The broker never inspects payload — it forwards every binary frame
        verbatim. The relay ends as soon as one side disconnects; the other
        side is then closed.
        """

        async def _forward(
            src: WebSocket,
            dst: WebSocket,
            direction: str,
        ) -> None:
            try:
                while True:
                    data = await src.receive_bytes()
                    await dst.send_bytes(data)
            except WebSocketDisconnect:
                logger.info(
                    "job %s: %s disconnected", job_id, direction
                )
            except RuntimeError as exc:
                # Starlette raises RuntimeError when sending to a closed WS.
                logger.info(
                    "job %s: %s closed mid-send (%s)", job_id, direction, exc
                )

        t_agent_to_sidecar = asyncio.create_task(
            _forward(agent_ws, sidecar_ws, "agent→sidecar")
        )
        t_sidecar_to_agent = asyncio.create_task(
            _forward(sidecar_ws, agent_ws, "sidecar→agent")
        )

        # Run until one direction finishes, then cancel the other.
        done, pending = await asyncio.wait(
            {t_agent_to_sidecar, t_sidecar_to_agent},
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
