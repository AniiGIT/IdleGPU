# Copyright (C) 2026 AniiGIT
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
broker/frame.py - Binary CUDA channel frame format.

Wire layout (12-byte header, big-endian):

  Offset  Size  Field
  ──────  ────  ─────────────────────────────────────────────────────────
  0       4     msg_type     - message type constant (see MSG_* below)
  4       4     call_id      - caller-assigned id matching calls to returns
  8       4     payload_len  - byte length of the msgpack payload that follows
  12      N     payload      - msgpack-encoded dict

The broker is a stateless frame router: it reads the 12-byte header to know
how many payload bytes to forward, but never inspects or modifies payload
contents. Payload decoding only happens at the agent (executing CUDA calls)
and at the sidecar (intercepting CUDA calls).

Message type constants
──────────────────────
  CUDA_CALL    (1)  sidecar  → agent   - CUDA runtime API invocation
  CUDA_RETURN  (2)  agent    → sidecar - CUDA runtime API return value
  NVENC_CALL   (3)  sidecar  → agent   - NvEncodeAPI invocation
  NVENC_RETURN (4)  agent    → sidecar - NvEncodeAPI return value
  ERROR        (5)  either   → either  - unrecoverable protocol error
  DISCONNECT   (6)  either   → either  - graceful shutdown notification
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import msgpack  # type: ignore[import-untyped]

# ── message type constants ───────────────────────────────────────────────────

CUDA_CALL    = 1
CUDA_RETURN  = 2
NVENC_CALL   = 3
NVENC_RETURN = 4
ERROR        = 5
DISCONNECT   = 6

# Human-readable names used in log messages.
_TYPE_NAMES: dict[int, str] = {
    CUDA_CALL:    "CUDA_CALL",
    CUDA_RETURN:  "CUDA_RETURN",
    NVENC_CALL:   "NVENC_CALL",
    NVENC_RETURN: "NVENC_RETURN",
    ERROR:        "ERROR",
    DISCONNECT:   "DISCONNECT",
}


def type_name(msg_type: int) -> str:
    """Return a human-readable name for a message type constant."""
    return _TYPE_NAMES.get(msg_type, f"UNKNOWN({msg_type})")


# ── header struct ─────────────────────────────────────────────────────────────

# Three big-endian unsigned 32-bit integers: msg_type, call_id, payload_len.
_HEADER = struct.Struct("!III")
HEADER_SIZE: int = _HEADER.size  # 12


# ── Frame dataclass ───────────────────────────────────────────────────────────

@dataclass
class Frame:
    """A single CUDA channel frame (decoded)."""
    msg_type: int
    call_id: int
    payload: dict  # type: ignore[type-arg]

    def type_name(self) -> str:
        return type_name(self.msg_type)


# ── encode / decode ───────────────────────────────────────────────────────────

def pack(frame: Frame) -> bytes:
    """Serialise a Frame to wire bytes.

    Raises TypeError if frame.payload is not msgpack-serialisable.
    """
    payload_bytes: bytes = msgpack.packb(frame.payload, use_bin_type=True)
    header = _HEADER.pack(frame.msg_type, frame.call_id, len(payload_bytes))
    return header + payload_bytes


def unpack(data: bytes) -> Frame:
    """Deserialise a Frame from wire bytes.

    Raises ValueError on truncated data or payload length mismatch.
    Raises msgpack.UnpackException on malformed payload.
    """
    if len(data) < HEADER_SIZE:
        raise ValueError(
            f"frame too short: need {HEADER_SIZE} header bytes, got {len(data)}"
        )

    msg_type, call_id, payload_len = _HEADER.unpack(data[:HEADER_SIZE])
    payload_bytes = data[HEADER_SIZE:]

    if len(payload_bytes) != payload_len:
        raise ValueError(
            f"payload length mismatch: header says {payload_len}, "
            f"got {len(payload_bytes)}"
        )

    payload: dict = msgpack.unpackb(payload_bytes, raw=False)
    return Frame(msg_type=msg_type, call_id=call_id, payload=payload)
