# Copyright (C) 2026 AniiGIT
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
sidecar/ipc_server.py - Unix socket IPC server that bridges libidlegpu-cuda.so
                        to the broker WebSocket via CudaBridge.

Listens on IDLEGPU_SOCKET (default /var/run/idlegpu/cuda.sock) for
connections from processes that have the shim loaded via LD_PRELOAD.

For each connected shim process, the server:
  1. Reads IpcReqHeader (16 bytes): magic, func_id, call_id, payload_len.
  2. Reads payload_len bytes of packed-C payload.
  3. Decodes the payload via ipc_codec.decode_req().
  4. Forwards to the agent via CudaBridge.call().
  5. Encodes the agent's response via ipc_codec.encode_resp().
  6. Writes IpcRespHeader (12 bytes) + response payload back to the shim.

cuModuleLoad special case: the shim sends a filename.  The IPC server reads
the file on the sidecar's filesystem and converts the call to cuModuleLoadData
so the agent receives image bytes rather than a path it cannot access.
"""

from __future__ import annotations

import asyncio
import logging
import os
import struct

from .cuda_bridge import CudaBridge
from .ipc_codec import (
    REQ_HDR, RESP_HDR, IPC_MAGIC,
    FN_cuModuleLoad, FN_cuModuleLoadData,
    decode_req, encode_resp,
)

logger = logging.getLogger(__name__)

# Maximum IPC payload size: 16 MiB (matches IPC_MAX_PAYLOAD in the shim).
_MAX_PAYLOAD = 16 * 1024 * 1024


class IpcServer:
    """
    Asyncio Unix socket server.  Accepts shim connections and serialises
    each IPC call to the broker via the shared CudaBridge.
    """

    def __init__(self, socket_path: str, bridge: CudaBridge) -> None:
        self._socket_path = socket_path
        self._bridge = bridge

    async def run(self) -> None:
        """Start listening and serve connections until cancelled."""
        # Remove a stale socket file if present.
        try:
            os.unlink(self._socket_path)
        except FileNotFoundError:
            pass

        os.makedirs(os.path.dirname(self._socket_path), exist_ok=True)

        server = await asyncio.start_unix_server(
            self._handle_connection,
            path=self._socket_path,
        )
        # Make the socket world-writable so processes running as a different
        # UID (e.g. the Unmanic container) can connect.  The sidecar still
        # validates all calls through the IPC protocol; this is not a security
        # boundary — both containers share the same Docker network volume.
        os.chmod(self._socket_path, 0o666)
        logger.info("ipc_server: listening on %s", self._socket_path)

        async with server:
            try:
                await server.serve_forever()
            except asyncio.CancelledError:
                logger.info("ipc_server: shutting down")

    # ── Per-connection handler ─────────────────────────────────────────────────

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle one shim connection until it closes or errors."""
        peer = writer.get_extra_info("peername", "<unknown>")
        logger.debug("ipc_server: shim connected (%s)", peer)

        try:
            while True:
                # ── Read request header ──────────────────────────────────────
                try:
                    hdr_bytes = await reader.readexactly(REQ_HDR.size)
                except asyncio.IncompleteReadError:
                    break  # shim disconnected cleanly

                magic, func_id, call_id, payload_len = REQ_HDR.unpack(hdr_bytes)

                if magic != IPC_MAGIC:
                    logger.error(
                        "ipc_server: bad magic 0x%08x from %s; closing", magic, peer
                    )
                    break

                if payload_len > _MAX_PAYLOAD:
                    logger.error(
                        "ipc_server: payload too large (%d bytes); closing", payload_len
                    )
                    break

                # ── Read request payload ─────────────────────────────────────
                if payload_len > 0:
                    try:
                        payload = await reader.readexactly(payload_len)
                    except asyncio.IncompleteReadError:
                        break
                else:
                    payload = b""

                # ── Decode IPC → dict ────────────────────────────────────────
                try:
                    req_dict = decode_req(func_id, payload)
                except Exception as exc:
                    logger.error(
                        "ipc_server: decode_req failed for func_id=%d: %s", func_id, exc
                    )
                    await self._send_error(writer, call_id)
                    continue

                # ── cuModuleLoad: read file locally, upgrade to LoadData ──────
                actual_func_id = func_id
                if func_id == FN_cuModuleLoad:
                    fname: str = req_dict.get("fname", "")
                    try:
                        with open(fname, "rb") as f:
                            image_bytes = f.read()
                        req_dict = {"func": "cuModuleLoadData", "image": image_bytes}
                        actual_func_id = FN_cuModuleLoadData
                    except OSError as exc:
                        logger.error(
                            "ipc_server: cuModuleLoad: cannot read %r: %s", fname, exc
                        )
                        await self._send_error(writer, call_id,
                                               cuda_result=1)  # CUDA_ERROR_FILE_NOT_FOUND
                        continue

                # ── Forward to broker via WebSocket ──────────────────────────
                try:
                    resp_dict = await self._bridge.call(call_id, actual_func_id, req_dict)
                except ConnectionError as exc:
                    logger.warning("ipc_server: bridge call failed: %s", exc)
                    await self._send_error(writer, call_id)
                    continue

                cuda_result: int = resp_dict.get("result", 0)

                # ── Encode response dict → IPC bytes ─────────────────────────
                try:
                    resp_payload = encode_resp(actual_func_id, cuda_result, resp_dict, req_dict)
                except Exception as exc:
                    logger.error(
                        "ipc_server: encode_resp failed for func_id=%d: %s",
                        actual_func_id, exc,
                    )
                    await self._send_error(writer, call_id)
                    continue

                # ── Send IpcRespHeader + payload to shim ─────────────────────
                resp_hdr = RESP_HDR.pack(call_id, cuda_result, len(resp_payload))
                writer.write(resp_hdr + resp_payload)
                await writer.drain()

        except asyncio.CancelledError:
            # Task was cancelled (sidecar shutting down or broker disconnected).
            # Log cleanly and re-raise so asyncio can finish cleanup — swallowing
            # CancelledError causes "Task was destroyed but it is pending" warnings.
            logger.warning("ipc_server: connection task cancelled (%s)", peer)
            raise
        except (ConnectionResetError, BrokenPipeError):
            # Shim process exited mid-call (e.g. application crash or IPC timeout
            # on the shim side).  Expected; no stack trace needed.
            logger.debug("ipc_server: shim disconnected (%s)", peer)
        except Exception as exc:
            logger.error("ipc_server: unexpected error for %s: %s", peer, exc)
        finally:
            writer.close()
            logger.debug("ipc_server: closed connection (%s)", peer)

    async def _send_error(
        self,
        writer: asyncio.StreamWriter,
        call_id: int,
        cuda_result: int = 999,  # CUDA_ERROR_UNKNOWN
    ) -> None:
        """Send an error IpcRespHeader with no payload."""
        try:
            hdr = RESP_HDR.pack(call_id, cuda_result, 0)
            writer.write(hdr)
            await writer.drain()
        except Exception:
            pass
