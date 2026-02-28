# Copyright (C) 2026 AniiGIT
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
sidecar/ipc_codec.py - Translate between the shim's IPC binary format and
                       msgpack dicts for the WebSocket CUDA channel.

Two public functions:

  decode_req(func_id, payload_bytes) -> dict
      Parse the packed-C IPC request payload and return a dict suitable
      for msgpack serialisation on the WebSocket channel.  The dict always
      includes a "func" key with the CUDA function name.

  encode_resp(func_id, cuda_result, resp_dict, req_dict) -> bytes
      Encode the agent's response dict back to packed-C bytes for the IPC
      response payload.  req_dict is the decoded request (needed for
      variable-length responses like cuDeviceGetName).

IPC wire headers (defined in idlegpu_shim.h; reproduced here for reference):

  IpcReqHeader  16 bytes  <IIII  magic, func_id, call_id, payload_len
  IpcRespHeader 12 bytes  <III   call_id, cuda_result, payload_len

This file only covers the payload that follows each header.
All C structs are __attribute__((packed)) on little-endian x86-64.
"""

from __future__ import annotations

import struct

# ── IPC frame headers ─────────────────────────────────────────────────────────

REQ_HDR  = struct.Struct("<IIII")  # magic, func_id, call_id, payload_len  (16 B)
RESP_HDR = struct.Struct("<III")   # call_id, cuda_result, payload_len     (12 B)
IPC_MAGIC = 0x49475055             # "IGPU"

# ── Function ID constants (mirror idlegpu_shim.h) ─────────────────────────────

FN_cuInit                  =  1
FN_cuDeviceGet             =  2
FN_cuDeviceGetCount        =  3
FN_cuDeviceGetName         =  4
FN_cuDeviceGetAttribute    =  5
FN_cuDeviceTotalMem        =  6
FN_cuCtxCreate             =  7
FN_cuCtxDestroy            =  8
FN_cuCtxSetCurrent         =  9
FN_cuCtxGetCurrent         = 10
FN_cuMemAlloc              = 11
FN_cuMemFree               = 12
FN_cuMemcpyHtoD            = 13
FN_cuMemcpyDtoH            = 14
FN_cuMemcpyDtoD            = 15
FN_cuMemsetD8              = 16
FN_cuMemsetD16             = 17
FN_cuMemsetD32             = 18
FN_cuMemGetInfo            = 19
FN_cuDriverGetVersion      = 20
FN_cuModuleLoad            = 21
FN_cuModuleLoadData        = 22
FN_cuModuleGetFunction     = 23
FN_cuLaunchKernel          = 24
FN_cuModuleUnload          = 25
FN_cuStreamCreate          = 31
FN_cuStreamDestroy         = 32
FN_cuStreamSynchronize     = 33
FN_cuStreamWaitEvent       = 34
FN_cuEventCreate           = 35
FN_cuEventDestroy          = 36
FN_cuEventRecord           = 37
FN_cuEventSynchronize      = 38
FN_NvEncOpenEncodeSession       = 41
FN_NvEncInitializeEncoder       = 42
FN_NvEncCreateInputBuffer       = 43
FN_NvEncCreateBitstreamBuffer   = 44
FN_NvEncEncodePicture           = 45
FN_NvEncLockBitstream           = 46
FN_NvEncUnlockBitstream         = 47
FN_NvEncDestroyInputBuffer      = 48
FN_NvEncDestroyBitstreamBuffer  = 49
FN_NvEncDestroyEncoder          = 50
FN_cuGetExportTable             = 51  # runtime internals
# Extended device / context — Phase 2E promotions
FN_cuDeviceComputeCapability    = 52
FN_cuDeviceGetUuid              = 53
FN_cuCtxPushCurrent             = 54
FN_cuCtxPopCurrent              = 55
FN_cuDeviceGetLuid              = 56
FN_cuCtxSetLimit                = 57
FN_cuCtxGetLimit                = 58
# Extended memory — pitched 2D and managed allocations
FN_cuMemAllocPitch              = 59
FN_cuMemAllocManaged            = 60
# Async memset (stream-ordered)
FN_cuMemsetD8Async              = 61
FN_cuMemsetD16Async             = 62
FN_cuMemsetD32Async             = 63
# Async copies
FN_cuMemcpyDtoDAsync            = 64
# Stream creation with priority
FN_cuStreamCreateWithPriority   = 65
# Context introspection
FN_cuCtxSynchronize             = 66
FN_cuCtxGetDevice               = 67
FN_cuCtxGetFlags                = 68
FN_cuCtxGetApiVersion           = 69
FN_cuCtxGetCacheConfig          = 70
FN_cuCtxSetCacheConfig          = 71
FN_cuCtxGetSharedMemConfig      = 72
FN_cuCtxSetSharedMemConfig      = 73
# Kernel / function attributes
FN_cuFuncGetAttribute           = 74
FN_cuFuncSetAttribute           = 75
FN_cuFuncSetCacheConfig         = 76
FN_cuFuncSetSharedMemConfig     = 77
# Occupancy
FN_cuOccupancyMaxActiveBlocksPerMultiprocessor = 78
# Async host↔device copies (variable-length; handled in SPECIAL section)
FN_cuMemcpyHtoDAsync            = 79
FN_cuMemcpyDtoHAsync            = 80

# ── Simple codec table ────────────────────────────────────────────────────────
#
# Each entry: (func_name, req_fmt, req_fields, resp_fmt, resp_fields)
#
#   req_fmt / resp_fmt  : struct format string, or None for no payload.
#                         Prefix '<' = little-endian (host byte order on x86-64).
#   req_fields          : tuple of field names extracted by req_fmt.
#   resp_fields         : tuple of field names packed by resp_fmt.
#
# Functions with variable-length tails or binary response payloads are handled
# by the _SPECIAL decode/encode tables below.

_Simple = tuple[str, str | None, tuple[str, ...], str | None, tuple[str, ...]]

_SIMPLE: dict[int, _Simple] = {
    FN_cuInit:             ("cuInit",             "<I",      ("flags",),                               None,   ()),
    FN_cuDeviceGet:        ("cuDeviceGet",         "<i",      ("ordinal",),                             "<i",   ("device",)),
    FN_cuDeviceGetCount:   ("cuDeviceGetCount",    None,      (),                                       "<i",   ("count",)),
    FN_cuDeviceGetAttribute:("cuDeviceGetAttribute","<ii",    ("attrib", "device"),                     "<i",   ("value",)),
    FN_cuDeviceTotalMem:   ("cuDeviceTotalMem",    "<i",      ("device",),                              "<Q",   ("bytes",)),
    FN_cuCtxCreate:        ("cuCtxCreate",         "<Ii",     ("flags", "device"),                      "<Q",   ("ctx_handle",)),
    FN_cuCtxDestroy:       ("cuCtxDestroy",        "<Q",      ("ctx_handle",),                          None,   ()),
    FN_cuCtxSetCurrent:    ("cuCtxSetCurrent",     "<Q",      ("ctx_handle",),                          None,   ()),
    FN_cuCtxGetCurrent:    ("cuCtxGetCurrent",     None,      (),                                       "<Q",   ("ctx_handle",)),
    FN_cuMemAlloc:         ("cuMemAlloc",          "<Q",      ("bytesize",),                            "<Q",   ("dptr",)),
    FN_cuMemFree:          ("cuMemFree",           "<Q",      ("dptr",),                                None,   ()),
    FN_cuMemcpyDtoD:       ("cuMemcpyDtoD",        "<QQQ",    ("dst", "src", "byte_count"),             None,   ()),
    # Req_cuMemsetD8:  {uint64 dst, uint8 value, uint8[7] _pad, uint64 count} packed = 24 B
    FN_cuMemsetD8:         ("cuMemsetD8",          "<QB7xQ",  ("dst", "value", "count"),                None,   ()),
    # Req_cuMemsetD16: {uint64 dst, uint16 value, uint8[6] _pad, uint64 count} packed = 24 B
    FN_cuMemsetD16:        ("cuMemsetD16",         "<QH6xQ",  ("dst", "value", "count"),                None,   ()),
    # Req_cuMemsetD32: {uint64 dst, uint32 value, uint8[4] _pad, uint64 count} packed = 24 B
    FN_cuMemsetD32:        ("cuMemsetD32",         "<QI4xQ",  ("dst", "value", "count"),                None,   ()),
    FN_cuMemGetInfo:       ("cuMemGetInfo",        None,      (),                                       "<QQ",  ("free", "total")),
    FN_cuDriverGetVersion:        ("cuDriverGetVersion",        None,  (),              "<i",  ("version",)),
    # Phase 2E extended device / context
    FN_cuDeviceComputeCapability: ("cuDeviceComputeCapability", "<i",  ("device",),     "<ii", ("major", "minor")),
    FN_cuCtxPushCurrent:          ("cuCtxPushCurrent",          "<Q",  ("ctx_handle",), None,  ()),
    FN_cuCtxPopCurrent:           ("cuCtxPopCurrent",           None,  (),              "<Q",  ("ctx_handle",)),
    # Req_cuCtxSetLimit: {int32 limit, uint64 value} = 12 B packed
    FN_cuCtxSetLimit:             ("cuCtxSetLimit",             "<iQ", ("limit", "value"), None, ()),
    # Req_cuCtxGetLimit: {int32 limit} = 4 B; Resp: {uint64 value} = 8 B
    FN_cuCtxGetLimit:             ("cuCtxGetLimit",             "<i",  ("limit",),      "<Q",  ("value",)),
    # Req_cuMemAllocPitch: {uint64 width_bytes, uint64 height, uint32 element_size} = 20 B
    # Resp_cuMemAllocPitch: {uint64 dptr, uint64 pitch} = 16 B
    FN_cuMemAllocPitch:           ("cuMemAllocPitch",           "<QQI", ("width_bytes", "height", "element_size"), "<QQ", ("dptr", "pitch")),
    # Req_cuMemAllocManaged: {uint64 bytesize, uint32 flags} = 12 B
    # Resp_cuMemAllocManaged: {uint64 dptr} = 8 B
    FN_cuMemAllocManaged:         ("cuMemAllocManaged",         "<QI",  ("bytesize", "flags"),                     "<Q",  ("dptr",)),
    # Async memset: Req = {uint64 dst, uint8/uint16/uint32 value, pad, uint64 count, uint64 stream}
    FN_cuMemsetD8Async:   ("cuMemsetD8Async",   "<QB7xQQ", ("dst", "value", "count", "stream_handle"), None, ()),
    FN_cuMemsetD16Async:  ("cuMemsetD16Async",  "<QH6xQQ", ("dst", "value", "count", "stream_handle"), None, ()),
    FN_cuMemsetD32Async:  ("cuMemsetD32Async",  "<QI4xQQ", ("dst", "value", "count", "stream_handle"), None, ()),
    # Async D-to-D: Req = {uint64 dst, uint64 src, uint64 byte_count, uint64 stream}
    FN_cuMemcpyDtoDAsync: ("cuMemcpyDtoDAsync", "<QQQQ",   ("dst", "src", "byte_count", "stream_handle"), None, ()),
    # Stream creation with priority: Req = {uint32 flags, int32 priority}; Resp = {uint64 stream_handle}
    FN_cuStreamCreateWithPriority: ("cuStreamCreateWithPriority", "<Ii", ("flags", "priority"), "<Q", ("stream_handle",)),
    # Context introspection
    FN_cuCtxSynchronize:       ("cuCtxSynchronize",       None,  (),                    None,  ()),
    FN_cuCtxGetDevice:         ("cuCtxGetDevice",         None,  (),                    "<i",  ("device",)),
    FN_cuCtxGetFlags:          ("cuCtxGetFlags",          None,  (),                    "<I",  ("flags",)),
    FN_cuCtxGetApiVersion:     ("cuCtxGetApiVersion",     "<Q",  ("ctx_handle",),        "<I",  ("version",)),
    FN_cuCtxGetCacheConfig:    ("cuCtxGetCacheConfig",    None,  (),                    "<i",  ("config",)),
    FN_cuCtxSetCacheConfig:    ("cuCtxSetCacheConfig",    "<i",  ("config",),            None,  ()),
    FN_cuCtxGetSharedMemConfig:("cuCtxGetSharedMemConfig",None,  (),                    "<i",  ("config",)),
    FN_cuCtxSetSharedMemConfig:("cuCtxSetSharedMemConfig","<i",  ("config",),            None,  ()),
    # Kernel / function attributes
    # Req_cuFuncGetAttribute: {int32 attrib, uint64 func_handle} = 12 B
    FN_cuFuncGetAttribute:     ("cuFuncGetAttribute",     "<iQ", ("attrib", "func_handle"), "<i", ("value",)),
    # Req_cuFuncSetAttribute: {uint64 func_handle, int32 attrib, int32 value} = 16 B
    FN_cuFuncSetAttribute:     ("cuFuncSetAttribute",     "<Qii", ("func_handle", "attrib", "value"), None, ()),
    # Req_cuFuncSetCacheConfig: {uint64 func_handle, int32 config} = 12 B
    FN_cuFuncSetCacheConfig:   ("cuFuncSetCacheConfig",   "<Qi", ("func_handle", "config"), None, ()),
    FN_cuFuncSetSharedMemConfig:("cuFuncSetSharedMemConfig","<Qi",("func_handle", "config"), None, ()),
    # Req_cuOccupancyMaxActiveBlocksPerMultiprocessor: {uint64 func, int32 blockSize, uint64 dynSmem} = 20 B
    FN_cuOccupancyMaxActiveBlocksPerMultiprocessor: (
        "cuOccupancyMaxActiveBlocksPerMultiprocessor",
        "<QiQ", ("func_handle", "block_size", "dynamic_smem_size"),
        "<i",   ("num_blocks",),
    ),
    FN_cuModuleUnload:     ("cuModuleUnload",      "<Q",      ("mod_handle",),                          None,   ()),
    FN_cuStreamCreate:     ("cuStreamCreate",      "<I",      ("flags",),                               "<Q",   ("stream_handle",)),
    FN_cuStreamDestroy:    ("cuStreamDestroy",     "<Q",      ("stream_handle",),                       None,   ()),
    FN_cuStreamSynchronize:("cuStreamSynchronize", "<Q",      ("stream_handle",),                       None,   ()),
    FN_cuStreamWaitEvent:  ("cuStreamWaitEvent",   "<QQI",    ("stream_handle", "event_handle", "flags"),None,  ()),
    FN_cuEventCreate:      ("cuEventCreate",       "<I",      ("flags",),                               "<Q",   ("event_handle",)),
    FN_cuEventDestroy:     ("cuEventDestroy",      "<Q",      ("event_handle",),                        None,   ()),
    FN_cuEventRecord:      ("cuEventRecord",       "<QQ",     ("event_handle", "stream_handle"),        None,   ()),
    FN_cuEventSynchronize: ("cuEventSynchronize",  "<Q",      ("event_handle",),                        None,   ()),
}

# Pre-compile the struct objects for the simple table.
_SIMPLE_STRUCTS: dict[int, tuple[struct.Struct | None, struct.Struct | None]] = {
    fid: (
        struct.Struct(entry[1]) if entry[1] else None,
        struct.Struct(entry[3]) if entry[3] else None,
    )
    for fid, entry in _SIMPLE.items()
}

# ── Special handlers ──────────────────────────────────────────────────────────

# Req_cuLaunchKernel layout (48 bytes fixed):
# {uint64 func_handle, uint32 grid_x,y,z, uint32 block_x,y,z, uint32 shared_mem,
#  uint64 stream_handle, uint32 num_params}
# followed by num_params * 8 bytes of parameter values.
_LAUNCH_HDR = struct.Struct("<QIIIIIIIQI")  # 48 B

# Req_cuModuleLoad: {uint32 fname_len} followed by fname_len bytes of filename.
_MODULE_LOAD_HDR = struct.Struct("<I")  # 4 B

# Req_cuModuleLoadData: {uint64 image_len} followed by image_len bytes.
_MODULE_LOAD_DATA_HDR = struct.Struct("<Q")  # 8 B

# Req_cuModuleGetFunction: {uint64 mod_handle, uint32 name_len} + name bytes.
_MODULE_GET_FN_HDR = struct.Struct("<QI")  # 12 B

# Req_cuDeviceGetName: {uint32 len, int32 device}  (response is raw name bytes)
_DEVICE_GET_NAME_HDR = struct.Struct("<Ii")  # 8 B

# Req_cuMemcpyHtoD: {uint64 dst, uint64 byte_count} + byte_count bytes of data.
_MEMCPY_HTOD_HDR = struct.Struct("<QQ")  # 16 B

# Req_cuMemcpyDtoH: {uint64 src, uint64 byte_count}  (response is raw bytes)
_MEMCPY_DTOH_HDR = struct.Struct("<QQ")  # 16 B

# _Req_HtoDAsync_Hdr: {uint64 dst, uint64 byte_count, uint64 stream_handle} + data bytes
_MEMCPY_HTOD_ASYNC_HDR = struct.Struct("<QQQ")  # 24 B

# _Req_DtoHAsync_Hdr: {uint64 src, uint64 byte_count, uint64 stream_handle}  (response: raw bytes)
_MEMCPY_DTOH_ASYNC_HDR = struct.Struct("<QQQ")  # 24 B


# ── Public API ────────────────────────────────────────────────────────────────

def decode_req(func_id: int, payload: bytes) -> dict:  # type: ignore[type-arg]
    """Decode an IPC request payload into a msgpack-friendly dict.

    The returned dict always contains "func": "<function name>" plus any
    function-specific fields.  Binary data (e.g. cuMemcpyHtoD host bytes)
    is included as a "data" key with a bytes value.
    """

    # ── Simple table lookup ───────────────────────────────────────────────────
    if func_id in _SIMPLE:
        entry = _SIMPLE[func_id]
        name, req_fmt, req_fields, _, _ = entry
        d: dict = {"func": name}  # type: ignore[type-arg]
        if req_fmt is not None:
            s = _SIMPLE_STRUCTS[func_id][0]
            assert s is not None
            values = s.unpack_from(payload)
            for k, v in zip(req_fields, values):
                d[k] = v
        return d

    # ── Special cases ─────────────────────────────────────────────────────────

    if func_id == FN_cuDeviceGetName:
        name_len, device = _DEVICE_GET_NAME_HDR.unpack_from(payload)
        return {"func": "cuDeviceGetName", "len": name_len, "device": device}

    if func_id == FN_cuMemcpyHtoD:
        dst, byte_count = _MEMCPY_HTOD_HDR.unpack_from(payload)
        data = payload[_MEMCPY_HTOD_HDR.size:]
        return {"func": "cuMemcpyHtoD", "dst": dst, "byte_count": byte_count, "data": data}

    if func_id == FN_cuMemcpyDtoH:
        src, byte_count = _MEMCPY_DTOH_HDR.unpack_from(payload)
        return {"func": "cuMemcpyDtoH", "src": src, "byte_count": byte_count}

    if func_id == FN_cuMemcpyHtoDAsync:
        dst, byte_count, stream_handle = _MEMCPY_HTOD_ASYNC_HDR.unpack_from(payload)
        data = payload[_MEMCPY_HTOD_ASYNC_HDR.size:]
        return {"func": "cuMemcpyHtoDAsync", "dst": dst,
                "byte_count": byte_count, "stream_handle": stream_handle, "data": data}

    if func_id == FN_cuMemcpyDtoHAsync:
        src, byte_count, stream_handle = _MEMCPY_DTOH_ASYNC_HDR.unpack_from(payload)
        return {"func": "cuMemcpyDtoHAsync", "src": src,
                "byte_count": byte_count, "stream_handle": stream_handle}

    if func_id == FN_cuModuleLoad:
        # The shim reads the PTX/binary file; we send its contents so the
        # agent can call cuModuleLoadData without filesystem access.
        # NOTE: file reading happens in ipc_server.py before calling decode_req;
        # if this path is reached without substitution, forward as-is.
        (fname_len,) = _MODULE_LOAD_HDR.unpack_from(payload)
        fname = payload[_MODULE_LOAD_HDR.size:_MODULE_LOAD_HDR.size + fname_len]
        return {"func": "cuModuleLoad", "fname": fname.rstrip(b"\x00").decode("utf-8", errors="replace")}

    if func_id == FN_cuModuleLoadData:
        (image_len,) = _MODULE_LOAD_DATA_HDR.unpack_from(payload)
        image = payload[_MODULE_LOAD_DATA_HDR.size:_MODULE_LOAD_DATA_HDR.size + int(image_len)]
        return {"func": "cuModuleLoadData", "image": image}

    if func_id == FN_cuModuleGetFunction:
        mod_handle, name_len = _MODULE_GET_FN_HDR.unpack_from(payload)
        name = payload[_MODULE_GET_FN_HDR.size:_MODULE_GET_FN_HDR.size + name_len]
        return {
            "func": "cuModuleGetFunction",
            "mod_handle": mod_handle,
            "name": name.rstrip(b"\x00").decode("utf-8", errors="replace"),
        }

    if func_id == FN_cuLaunchKernel:
        (func_handle, gx, gy, gz, bx, by, bz, shared_mem,
         stream_handle, num_params) = _LAUNCH_HDR.unpack_from(payload)
        params_offset = _LAUNCH_HDR.size
        params = []
        for i in range(num_params):
            (val,) = struct.unpack_from("<Q", payload, params_offset + i * 8)
            params.append(val)
        return {
            "func": "cuLaunchKernel",
            "func_handle": func_handle,
            "grid": [gx, gy, gz],
            "block": [bx, by, bz],
            "shared_mem": shared_mem,
            "stream_handle": stream_handle,
            "params": params,
        }

    # NVENC calls: pass through opaquely (Phase 2F will add full codec)
    if FN_NvEncOpenEncodeSession <= func_id <= FN_NvEncDestroyEncoder:
        return {"func": f"nvenc_{func_id}", "raw": payload}

    if func_id == FN_cuGetExportTable:
        # Request payload: 16-byte CUuuid identifying the capability table.
        uuid_hex = payload[:16].hex() if len(payload) >= 16 else "00" * 16
        return {"func": "cuGetExportTable", "export_table_id": uuid_hex}

    if func_id == FN_cuDeviceGetUuid:
        (device,) = struct.unpack_from("<i", payload)
        return {"func": "cuDeviceGetUuid", "device": device}

    if func_id == FN_cuDeviceGetLuid:
        (device,) = struct.unpack_from("<i", payload)
        return {"func": "cuDeviceGetLuid", "device": device}

    raise ValueError(f"unknown func_id {func_id}")


def encode_resp(func_id: int, cuda_result: int, resp: dict, req: dict) -> bytes:  # type: ignore[type-arg]
    """Encode an agent response dict to IPC response payload bytes.

    Returns the raw bytes that follow IpcRespHeader in the IPC response.
    Returns b"" when the function has no output payload.
    """

    # ── Simple table lookup ───────────────────────────────────────────────────
    if func_id in _SIMPLE:
        _, _, _, resp_fmt, resp_fields = _SIMPLE[func_id]
        if resp_fmt is None or not resp_fields:
            return b""
        s = _SIMPLE_STRUCTS[func_id][1]
        assert s is not None
        values = tuple(resp.get(f, 0) for f in resp_fields)
        return s.pack(*values)

    # ── Special cases ─────────────────────────────────────────────────────────

    if func_id == FN_cuDeviceGetName:
        # Response payload is the null-terminated name, up to req["len"] bytes.
        max_len: int = req.get("len", 256)
        name_str: str = resp.get("name", "")
        name_bytes = name_str.encode("ascii", errors="replace")
        # Truncate to (max_len - 1) to always include a null terminator.
        name_bytes = name_bytes[: max_len - 1] + b"\x00"
        # Pad to exactly max_len bytes so the shim receives the expected length.
        return name_bytes.ljust(max_len, b"\x00")

    if func_id == FN_cuMemcpyHtoD:
        return b""  # no output payload

    if func_id == FN_cuMemcpyHtoDAsync:
        return b""  # no output payload

    if func_id == FN_cuMemcpyDtoHAsync:
        return resp.get("data", b"")

    if func_id == FN_cuMemcpyDtoH:
        # Response payload is the raw device memory bytes.
        data: bytes = resp.get("data", b"")
        return data

    if func_id == FN_cuModuleLoad:
        mod_handle: int = resp.get("mod_handle", 0)
        return struct.pack("<Q", mod_handle)

    if func_id == FN_cuModuleLoadData:
        mod_handle = resp.get("mod_handle", 0)
        return struct.pack("<Q", mod_handle)

    if func_id == FN_cuModuleGetFunction:
        func_handle: int = resp.get("func_handle", 0)
        return struct.pack("<Q", func_handle)

    if func_id == FN_cuLaunchKernel:
        return b""

    # NVENC: pass through opaque response bytes
    if FN_NvEncOpenEncodeSession <= func_id <= FN_NvEncDestroyEncoder:
        raw: bytes = resp.get("raw", b"")
        return raw

    if func_id == FN_cuGetExportTable:
        # Response payload: uint32_t entry_count followed by entry_count * 8
        # bytes of raw function pointer values (little-endian uint64).
        table_bytes: bytes = resp.get("export_table", b"")
        entry_count: int = len(table_bytes) // 8  # whole entries only
        return struct.pack("<I", entry_count) + table_bytes[: entry_count * 8]

    if func_id == FN_cuDeviceGetUuid:
        # Response payload: Resp_cuDeviceGetUuid — 16 bytes of UUID.
        uuid_bytes: bytes = resp.get("uuid", b"")
        return uuid_bytes[:16].ljust(16, b"\x00")

    if func_id == FN_cuDeviceGetLuid:
        # Response payload: Resp_cuDeviceGetLuid — 8-byte LUID + uint32 mask.
        luid_bytes: bytes = resp.get("luid", b"")
        device_node_mask: int = resp.get("device_node_mask", 0)
        return luid_bytes[:8].ljust(8, b"\x00") + struct.pack("<I", device_node_mask)

    raise ValueError(f"unknown func_id {func_id} in encode_resp")
