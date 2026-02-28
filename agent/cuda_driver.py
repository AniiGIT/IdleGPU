# Copyright (C) 2026 AniiGIT
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
agent/cuda_driver.py - ctypes wrapper for the CUDA Driver API.

Loads libcuda.so.1 (Linux) or nvcuda.dll (Windows) and exposes thin Python
wrappers for all Tier 1 CUDA functions.  Each wrapper returns a (CUresult, ...)
tuple where the first element is the integer CUDA error code (0 = success).

Opaque handles (CUcontext, CUmodule, CUfunction, CUstream, CUevent) are
represented as plain Python ints (uint64).  They are the actual pointer values
from the driver and remain valid as long as the driver resource exists.

The driver is loaded lazily on first use.  If loading fails, CudaDriverError
is raised with a descriptive message.

Thread safety: ctypes is thread-safe for individual calls; no additional
locking is applied here.  The caller (CudaExecutor) serialises if needed.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import struct
import sys
from typing import Any

# ── CUresult constants (commonly used here for clarity) ───────────────────────

CUDA_SUCCESS                  =   0
CUDA_ERROR_INVALID_VALUE      = 100  # also used as CUDA_ERROR_NO_DEVICE for cuInit
CUDA_ERROR_OUT_OF_MEMORY      = 200
CUDA_ERROR_NOT_INITIALIZED    = 300
CUDA_ERROR_DEINITIALIZED      = 301
CUDA_ERROR_NOT_SUPPORTED      = 801
CUDA_ERROR_UNKNOWN            = 999


class CudaDriverError(RuntimeError):
    """Raised when the CUDA driver library cannot be loaded."""


# ── Library loader ────────────────────────────────────────────────────────────

_lib: Any = None  # ctypes handle; set by _ensure_loaded()


def _ensure_loaded() -> Any:
    global _lib
    if _lib is not None:
        return _lib

    if sys.platform == "win32":
        names = ["nvcuda.dll"]
    else:
        names = ["libcuda.so.1", "libcuda.so"]

    for name in names:
        try:
            _lib = ctypes.CDLL(name)
            return _lib
        except OSError:
            continue

    raise CudaDriverError(
        f"could not load CUDA driver library (tried: {names}); "
        "is the NVIDIA driver installed?"
    )


def _h(c_void_p_value: int | None) -> int:
    """Convert a ctypes c_void_p value to int (0 for NULL/None)."""
    return c_void_p_value if c_void_p_value is not None else 0


# ── Tier 1 wrappers ───────────────────────────────────────────────────────────

def cuInit(flags: int) -> int:
    lib = _ensure_loaded()
    return int(lib.cuInit(ctypes.c_uint(flags)))


def cuDeviceGetCount() -> tuple[int, int]:
    lib = _ensure_loaded()
    count = ctypes.c_int(0)
    r = int(lib.cuDeviceGetCount(ctypes.byref(count)))
    return r, int(count.value)


def cuDeviceGet(ordinal: int) -> tuple[int, int]:
    lib = _ensure_loaded()
    dev = ctypes.c_int(0)
    r = int(lib.cuDeviceGet(ctypes.byref(dev), ctypes.c_int(ordinal)))
    return r, int(dev.value)


def cuDeviceGetName(length: int, device: int) -> tuple[int, str]:
    lib = _ensure_loaded()
    buf = ctypes.create_string_buffer(length)
    r = int(lib.cuDeviceGetName(buf, ctypes.c_int(length), ctypes.c_int(device)))
    name = buf.value.decode("ascii", errors="replace") if r == 0 else ""
    return r, name


def cuDeviceGetAttribute(attrib: int, device: int) -> tuple[int, int]:
    lib = _ensure_loaded()
    value = ctypes.c_int(0)
    r = int(lib.cuDeviceGetAttribute(
        ctypes.byref(value), ctypes.c_int(attrib), ctypes.c_int(device)
    ))
    return r, int(value.value)


def cuDeviceTotalMem(device: int) -> tuple[int, int]:
    lib = _ensure_loaded()
    # cuDeviceTotalMem_v2 takes size_t*; use c_size_t for portability.
    mem = ctypes.c_size_t(0)
    # Try the versioned symbol first; fall back to unversioned.
    fn = getattr(lib, "cuDeviceTotalMem_v2", None) or lib.cuDeviceTotalMem
    r = int(fn(ctypes.byref(mem), ctypes.c_int(device)))
    return r, int(mem.value)


def cuCtxCreate(flags: int, device: int) -> tuple[int, int]:
    lib = _ensure_loaded()
    ctx = ctypes.c_void_p(0)
    fn = getattr(lib, "cuCtxCreate_v2", None) or lib.cuCtxCreate
    r = int(fn(ctypes.byref(ctx), ctypes.c_uint(flags), ctypes.c_int(device)))
    return r, _h(ctx.value)


def cuCtxDestroy(ctx_handle: int) -> int:
    lib = _ensure_loaded()
    fn = getattr(lib, "cuCtxDestroy_v2", None) or lib.cuCtxDestroy
    return int(fn(ctypes.c_void_p(ctx_handle)))


def cuCtxSetCurrent(ctx_handle: int) -> int:
    lib = _ensure_loaded()
    return int(lib.cuCtxSetCurrent(ctypes.c_void_p(ctx_handle)))


def cuCtxGetCurrent() -> tuple[int, int]:
    lib = _ensure_loaded()
    ctx = ctypes.c_void_p(0)
    r = int(lib.cuCtxGetCurrent(ctypes.byref(ctx)))
    return r, _h(ctx.value)


def cuMemAlloc(bytesize: int) -> tuple[int, int]:
    lib = _ensure_loaded()
    dptr = ctypes.c_uint64(0)
    fn = getattr(lib, "cuMemAlloc_v2", None) or lib.cuMemAlloc
    r = int(fn(ctypes.byref(dptr), ctypes.c_size_t(bytesize)))
    return r, int(dptr.value)


def cuMemFree(dptr: int) -> int:
    lib = _ensure_loaded()
    fn = getattr(lib, "cuMemFree_v2", None) or lib.cuMemFree
    return int(fn(ctypes.c_uint64(dptr)))


def cuMemcpyHtoD(dst: int, data: bytes) -> int:
    lib = _ensure_loaded()
    fn = getattr(lib, "cuMemcpyHtoD_v2", None) or lib.cuMemcpyHtoD
    return int(fn(ctypes.c_uint64(dst), data, ctypes.c_size_t(len(data))))


def cuMemcpyDtoH(src: int, byte_count: int) -> tuple[int, bytes]:
    lib = _ensure_loaded()
    buf = ctypes.create_string_buffer(byte_count)
    fn = getattr(lib, "cuMemcpyDtoH_v2", None) or lib.cuMemcpyDtoH
    r = int(fn(buf, ctypes.c_uint64(src), ctypes.c_size_t(byte_count)))
    return r, bytes(buf.raw) if r == CUDA_SUCCESS else b""


def cuMemcpyDtoD(dst: int, src: int, byte_count: int) -> int:
    lib = _ensure_loaded()
    fn = getattr(lib, "cuMemcpyDtoD_v2", None) or lib.cuMemcpyDtoD
    return int(fn(ctypes.c_uint64(dst), ctypes.c_uint64(src), ctypes.c_size_t(byte_count)))


def cuMemsetD8(dst: int, value: int, count: int) -> int:
    lib = _ensure_loaded()
    fn = getattr(lib, "cuMemsetD8_v2", None) or lib.cuMemsetD8
    return int(fn(ctypes.c_uint64(dst), ctypes.c_ubyte(value), ctypes.c_size_t(count)))


def cuMemsetD16(dst: int, value: int, count: int) -> int:
    lib = _ensure_loaded()
    fn = getattr(lib, "cuMemsetD16_v2", None) or lib.cuMemsetD16
    return int(fn(ctypes.c_uint64(dst), ctypes.c_ushort(value), ctypes.c_size_t(count)))


def cuMemsetD32(dst: int, value: int, count: int) -> int:
    lib = _ensure_loaded()
    fn = getattr(lib, "cuMemsetD32_v2", None) or lib.cuMemsetD32
    return int(fn(ctypes.c_uint64(dst), ctypes.c_uint(value), ctypes.c_size_t(count)))


def cuMemGetInfo() -> tuple[int, int, int]:
    """Returns (CUresult, free_bytes, total_bytes)."""
    lib = _ensure_loaded()
    free = ctypes.c_size_t(0)
    total = ctypes.c_size_t(0)
    fn = getattr(lib, "cuMemGetInfo_v2", None) or lib.cuMemGetInfo
    r = int(fn(ctypes.byref(free), ctypes.byref(total)))
    return r, int(free.value), int(total.value)


def cuDriverGetVersion() -> tuple[int, int]:
    """Returns (CUresult, driver_version_int).

    driver_version_int encodes the installed driver version as
    major * 1000 + minor * 10 (e.g. 12050 for driver 520.xx supporting
    CUDA 12.5).  Required by ffmpeg's NVENC encoder initialisation.
    """
    lib = _ensure_loaded()
    version = ctypes.c_int(0)
    r = int(lib.cuDriverGetVersion(ctypes.byref(version)))
    return r, int(version.value)


def cuModuleLoadData(image: bytes) -> tuple[int, int]:
    lib = _ensure_loaded()
    mod = ctypes.c_void_p(0)
    r = int(lib.cuModuleLoadData(ctypes.byref(mod), image))
    return r, _h(mod.value)


def cuModuleGetFunction(mod_handle: int, name: str) -> tuple[int, int]:
    lib = _ensure_loaded()
    func = ctypes.c_void_p(0)
    name_bytes = name.encode("ascii") + b"\x00"
    r = int(lib.cuModuleGetFunction(
        ctypes.byref(func),
        ctypes.c_void_p(mod_handle),
        name_bytes,
    ))
    return r, _h(func.value)


def cuLaunchKernel(
    func_handle: int,
    grid: list[int],
    block: list[int],
    shared_mem: int,
    stream_handle: int,
    params: list[int],
) -> int:
    lib = _ensure_loaded()

    # Build void** kernelParams: array of void* pointers to per-argument storage.
    n = len(params)
    if n > 0:
        param_storage = [ctypes.c_uint64(p) for p in params]
        PTR_ARRAY = ctypes.c_void_p * n
        ptr_array = PTR_ARRAY(*[ctypes.addressof(v) for v in param_storage])
        kernel_params = ctypes.cast(ptr_array, ctypes.POINTER(ctypes.c_void_p))
    else:
        kernel_params = None  # type: ignore[assignment]

    r = int(lib.cuLaunchKernel(
        ctypes.c_void_p(func_handle),
        ctypes.c_uint(grid[0]), ctypes.c_uint(grid[1]), ctypes.c_uint(grid[2]),
        ctypes.c_uint(block[0]), ctypes.c_uint(block[1]), ctypes.c_uint(block[2]),
        ctypes.c_uint(shared_mem),
        ctypes.c_void_p(stream_handle),
        kernel_params,
        None,  # extra = NULL
    ))
    return r


def cuModuleUnload(mod_handle: int) -> int:
    lib = _ensure_loaded()
    return int(lib.cuModuleUnload(ctypes.c_void_p(mod_handle)))


def cuStreamCreate(flags: int) -> tuple[int, int]:
    lib = _ensure_loaded()
    stream = ctypes.c_void_p(0)
    r = int(lib.cuStreamCreate(ctypes.byref(stream), ctypes.c_uint(flags)))
    return r, _h(stream.value)


def cuStreamDestroy(stream_handle: int) -> int:
    lib = _ensure_loaded()
    fn = getattr(lib, "cuStreamDestroy_v2", None) or lib.cuStreamDestroy
    return int(fn(ctypes.c_void_p(stream_handle)))


def cuStreamSynchronize(stream_handle: int) -> int:
    lib = _ensure_loaded()
    return int(lib.cuStreamSynchronize(ctypes.c_void_p(stream_handle)))


def cuStreamWaitEvent(stream_handle: int, event_handle: int, flags: int) -> int:
    lib = _ensure_loaded()
    return int(lib.cuStreamWaitEvent(
        ctypes.c_void_p(stream_handle),
        ctypes.c_void_p(event_handle),
        ctypes.c_uint(flags),
    ))


def cuEventCreate(flags: int) -> tuple[int, int]:
    lib = _ensure_loaded()
    event = ctypes.c_void_p(0)
    r = int(lib.cuEventCreate(ctypes.byref(event), ctypes.c_uint(flags)))
    return r, _h(event.value)


def cuEventDestroy(event_handle: int) -> int:
    lib = _ensure_loaded()
    fn = getattr(lib, "cuEventDestroy_v2", None) or lib.cuEventDestroy
    return int(fn(ctypes.c_void_p(event_handle)))


def cuEventRecord(event_handle: int, stream_handle: int) -> int:
    lib = _ensure_loaded()
    return int(lib.cuEventRecord(
        ctypes.c_void_p(event_handle),
        ctypes.c_void_p(stream_handle),
    ))


def cuEventSynchronize(event_handle: int) -> int:
    lib = _ensure_loaded()
    return int(lib.cuEventSynchronize(ctypes.c_void_p(event_handle)))


def cuGetExportTable(uuid_bytes: bytes) -> tuple[int, bytes]:
    """Call the real cuGetExportTable and serialise the table as raw bytes.

    Returns (CUresult, table_bytes) where table_bytes contains the raw
    64-bit function pointer values from the driver's export table, packed
    as little-endian uint64.  Entries are read until a NULL pointer is
    found or EXPORT_TABLE_MAX_ENTRIES (256) entries have been consumed.

    The pointer values are agent-side virtual addresses — they are NOT
    callable in the sidecar process.  The sidecar reconstructs a heap
    copy so that the CUDA runtime sees a non-NULL table and passes its
    capability presence checks.
    """
    _EXPORT_TABLE_MAX_ENTRIES = 256

    lib = _ensure_loaded()

    # cuGetExportTable signature:
    #   CUresult cuGetExportTable(const void **ppExportTable,
    #                             const CUuuid *pExportTableId)
    table_ptr = ctypes.c_void_p(0)
    # Pack the 16-byte UUID into a ctypes byte array.
    if len(uuid_bytes) < 16:
        uuid_bytes = uuid_bytes.ljust(16, b"\x00")
    uuid_buf = (ctypes.c_uint8 * 16)(*uuid_bytes[:16])

    r = int(lib.cuGetExportTable(ctypes.byref(table_ptr), ctypes.byref(uuid_buf)))
    if r != CUDA_SUCCESS:
        return r, b""

    ptr_val: int | None = table_ptr.value
    if ptr_val is None or ptr_val == 0:
        # Driver returned success but NULL table — return empty bytes; the
        # shim will substitute a non-NULL sentinel.
        return CUDA_SUCCESS, b""

    # Read the table as an array of uint64 (void*) values.
    ptr_array_type = ctypes.c_uint64 * _EXPORT_TABLE_MAX_ENTRIES
    try:
        ptr_array = ptr_array_type.from_address(ptr_val)
        entries: list[int] = []
        for i in range(_EXPORT_TABLE_MAX_ENTRIES):
            val = int(ptr_array[i])
            if val == 0:
                break
            entries.append(val)
    except Exception as exc:
        # from_address can raise if the pointer is somehow invalid.
        import logging
        logging.getLogger(__name__).warning(
            "cuda_driver: cuGetExportTable: failed to read table at 0x%x: %s",
            ptr_val, exc,
        )
        return CUDA_SUCCESS, b""

    table_bytes = struct.pack(f"<{len(entries)}Q", *entries) if entries else b""
    return CUDA_SUCCESS, table_bytes
