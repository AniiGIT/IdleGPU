# Copyright (C) 2026 AniiGIT
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
sidecar/frame.py - Binary CUDA channel frame format.

Wire layout (12-byte big-endian header):

  Offset  Size  Field
  ──────  ────  ─────────────────────────────────────────────────────────
  0       4     msg_type     - message type constant (see MSG_* below)
  4       4     call_id      - caller-assigned id matching calls to returns
  8       4     payload_len  - byte length of the msgpack payload that follows
  12      N     payload      - msgpack-encoded dict

This file is a mirror of broker/frame.py and agent/frame.py. The sidecar
is deployed separately (Docker container) so it carries its own copy.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import msgpack  # type: ignore[import-untyped]

# ── message type constants ────────────────────────────────────────────────────

CUDA_CALL    = 1
CUDA_RETURN  = 2
NVENC_CALL   = 3
NVENC_RETURN = 4
ERROR        = 5
DISCONNECT   = 6

_TYPE_NAMES: dict[int, str] = {
    CUDA_CALL:    "CUDA_CALL",
    CUDA_RETURN:  "CUDA_RETURN",
    NVENC_CALL:   "NVENC_CALL",
    NVENC_RETURN: "NVENC_RETURN",
    ERROR:        "ERROR",
    DISCONNECT:   "DISCONNECT",
}


def type_name(msg_type: int) -> str:
    return _TYPE_NAMES.get(msg_type, f"UNKNOWN({msg_type})")


# ── header struct ─────────────────────────────────────────────────────────────

_HEADER = struct.Struct("!III")
HEADER_SIZE: int = _HEADER.size  # 12


# ── Frame dataclass ───────────────────────────────────────────────────────────

@dataclass
class Frame:
    msg_type: int
    call_id: int
    payload: dict  # type: ignore[type-arg]

    def type_name(self) -> str:
        return type_name(self.msg_type)


# ── encode / decode ───────────────────────────────────────────────────────────

def pack(frame: Frame) -> bytes:
    payload_bytes: bytes = msgpack.packb(frame.payload, use_bin_type=True)
    header = _HEADER.pack(frame.msg_type, frame.call_id, len(payload_bytes))
    return header + payload_bytes


def unpack(data: bytes) -> Frame:
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
