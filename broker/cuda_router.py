# Copyright (C) 2026 AniiGIT
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
broker/cuda_router.py - Routes binary CUDA frames between agents and sidecars.

Lifecycle
─────────
  1. Agent connects to /ws/cuda/{agent_id}
       → router.handle_agent(agent_id, ws) is called
       → if a sidecar is already queued for this agent, pair immediately
       → otherwise store in _idle and wait

  2. Sidecar connects to /ws/sidecar/{agent_id} or /ws/sidecar/wait/{agent_id}
       → router.handle_sidecar(agent_id, ws) is called
       → if agent is idle: pop from _idle, relay immediately
       → if agent is not idle: add to _waiting[agent_id] queue and block

  3. Pairing occurs — relay runs bidirectionally until either side disconnects.

  4. At any point before pairing, either side may disconnect cleanly:
       → agent disconnect: relay_done event is set; handle_agent cleans up.
       → sidecar disconnect: sidecar monitor wakes handle_sidecar; exits
         without starting a relay.

Push model
──────────
handle_agent checks _waiting before registering in _idle.  The moment an
agent connects, any queued sidecar is paired with zero delay — no polling,
no reconnect backoff on the sidecar side.

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
    # Set by _do_relay when the relay completes, or by _monitor when the agent
    # disconnects before pairing.
    relay_done: asyncio.Event = field(default_factory=asyncio.Event)
    # Background task that drains the agent WebSocket while idle.
    # Cancelled by _do_relay when relay takes exclusive ownership.
    monitor_task: asyncio.Task | None = field(default=None)  # type: ignore[type-arg]


@dataclass
class _WaitingEntry:
    """State for one sidecar queued while waiting for an agent to become idle."""
    sidecar_ws: WebSocket
    # Set by handle_agent when it claims this sidecar, or by the sidecar
    # monitor task when the sidecar disconnects before pairing.
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    # Filled in by handle_agent before setting ready.
    # None if the sidecar disconnected before an agent became available.
    agent_entry: _AgentEntry | None = field(default=None)


class CudaRouter:
    """Maintains idle CUDA channels and relays frames for active jobs.

    Instantiated once at module level in broker/app.py and shared between
    the WebSocket route handlers.
    """

    def __init__(self) -> None:
        # Idle agent CUDA channels awaiting a sidecar connection.
        self._idle: dict[str, _AgentEntry] = {}
        # Sidecars queued per agent_id waiting for that agent to become idle.
        self._waiting: dict[str, list[_WaitingEntry]] = {}

    # ------------------------------------------------------------------ #
    # Public helpers (read-only queries)                                   #
    # ------------------------------------------------------------------ #

    def idle_agent_ids(self) -> list[str]:
        """Return agent_ids of agents currently waiting for a CUDA job."""
        return list(self._idle)

    def waiting_count(self, agent_id: str) -> int:
        """Return the number of sidecars currently queued for agent_id."""
        return len(self._waiting.get(agent_id, []))

    # ------------------------------------------------------------------ #
    # Agent handler                                                        #
    # ------------------------------------------------------------------ #

    async def handle_agent(self, agent_id: str, ws: WebSocket) -> None:
        """
        Called by the /ws/cuda/{agent_id} WebSocket handler.

        If a sidecar is already queued for this agent, pairs immediately.
        Otherwise registers the agent's CUDA channel as idle and blocks until:
          - a sidecar connects and the relay completes, OR
          - the agent disconnects.
        """
        entry = _AgentEntry(ws=ws)

        async def _monitor() -> None:
            """Drain the WebSocket while idle so disconnects are detected."""
            try:
                while True:
                    await ws.receive_bytes()
                    # Pre-relay data is unexpected; discard silently.
            except WebSocketDisconnect:
                entry.relay_done.set()
            except asyncio.CancelledError:
                pass  # relay is taking over

        entry.monitor_task = asyncio.create_task(_monitor())

        # Pair immediately with a waiting sidecar if one is already queued.
        waiting_list = self._waiting.get(agent_id)
        if waiting_list:
            waiting = waiting_list.pop(0)
            if not waiting_list:
                del self._waiting[agent_id]
            waiting.agent_entry = entry
            waiting.ready.set()
            logger.info(
                "cuda channel: agent %s paired immediately with waiting sidecar",
                agent_id,
            )
        else:
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
        Called by /ws/sidecar/{agent_id} and /ws/sidecar/wait/{agent_id}.

        If the named agent is idle, pairs and relays immediately.
        Otherwise adds the sidecar to the waiting queue for agent_id and
        blocks until the agent becomes idle or the sidecar disconnects.
        """
        entry = self._idle.pop(agent_id, None)
        if entry is not None:
            # Agent already idle — pair and relay immediately.
            await self._do_relay(entry, agent_id, sidecar_ws)
            return

        # Agent not yet available — queue and wait.
        waiting = _WaitingEntry(sidecar_ws=sidecar_ws)
        self._waiting.setdefault(agent_id, []).append(waiting)
        logger.info(
            "sidecar queued: waiting for agent %s to become idle", agent_id
        )

        # Monitor the sidecar socket while waiting so disconnects are detected.
        async def _sidecar_monitor() -> None:
            try:
                while True:
                    await sidecar_ws.receive_bytes()
                    # Unexpected data while waiting; discard.
            except (WebSocketDisconnect, RuntimeError):
                # Sidecar disconnected — wake the waiter.
                waiting.ready.set()
            except asyncio.CancelledError:
                pass

        monitor = asyncio.create_task(_sidecar_monitor())
        try:
            await waiting.ready.wait()
        finally:
            monitor.cancel()
            await asyncio.gather(monitor, return_exceptions=True)
            # Remove from waiting list if not already popped by handle_agent.
            waiting_list = self._waiting.get(agent_id, [])
            try:
                waiting_list.remove(waiting)
            except ValueError:
                pass
            if not waiting_list:
                self._waiting.pop(agent_id, None)

        if waiting.agent_entry is None:
            # Sidecar disconnected before an agent became available.
            logger.info(
                "sidecar disconnected while waiting for agent %s", agent_id
            )
            return

        await self._do_relay(waiting.agent_entry, agent_id, sidecar_ws)

    # ------------------------------------------------------------------ #
    # Relay helpers                                                        #
    # ------------------------------------------------------------------ #

    async def _do_relay(
        self,
        entry: _AgentEntry,
        agent_id: str,
        sidecar_ws: WebSocket,
    ) -> None:
        """Cancel the idle monitor and run the bidirectional relay."""
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
            entry.relay_done.set()
            logger.info("job %s: relay ended", job_id)

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
                logger.info("job %s: %s disconnected", job_id, direction)
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
