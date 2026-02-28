# Copyright (C) 2026 AniiGIT
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
agent/cuda_executor.py - Dispatch incoming CUDA call frames to the local GPU.

CudaExecutor receives a decoded call dict (e.g. {"func": "cuMemAlloc", ...})
from the CUDA channel and calls the corresponding real CUDA Driver API function
via agent.cuda_driver (ctypes).  It returns a response dict suitable for
packing back into a CUDA_RETURN frame.

Response dict convention
─────────────────────────
  Every response dict contains "result": int (CUresult; 0 = success).
  Output fields (handles, values, bytes) are present only when result == 0
  and the function produces output.

Handle transparency
───────────────────
  Opaque CUDA handles (CUcontext, CUmodule, CUfunction, CUstream, CUevent)
  are real pointer values within the agent process's address space.  They are
  returned to the sidecar as plain ints and passed back unchanged for
  subsequent calls.  No local handle mapping table is needed.

NVENC
─────
  NVENC calls arrive as NVENC_CALL frames.  The executor delegates to
  agent.nvenc_executor.dispatch() when present; otherwise returns
  CUDA_ERROR_NOT_SUPPORTED.  nvenc_executor is imported lazily to avoid
  errors on systems without NVENC.
"""

from __future__ import annotations

import logging
from typing import Any

from . import cuda_driver as drv
from .cuda_driver import CudaDriverError, CUDA_ERROR_NOT_SUPPORTED, CUDA_ERROR_UNKNOWN

logger = logging.getLogger(__name__)


class CudaExecutor:
    """
    Executes CUDA API calls on the local physical GPU.

    Instantiate once and reuse across many frames.  The underlying CUDA
    driver is loaded lazily on first call.
    """

    def __init__(self) -> None:
        self._nvenc: Any = None          # lazy-loaded nvenc_executor module
        self._nvenc_checked = False

    def dispatch(self, req: dict) -> dict:  # type: ignore[type-arg]
        """Execute a CUDA call and return a response dict.

        Never raises — all errors are captured as {"result": error_code}.
        """
        func: str = req.get("func", "")

        handler = _HANDLERS.get(func)
        if handler is None:
            logger.warning("cuda_executor: unimplemented function %r; returning NOT_SUPPORTED", func)
            return {"result": CUDA_ERROR_NOT_SUPPORTED}

        try:
            return handler(req)
        except CudaDriverError as exc:
            logger.error("cuda_executor: driver load error in %s: %s", func, exc)
            return {"result": CUDA_ERROR_UNKNOWN}
        except Exception as exc:
            logger.error("cuda_executor: unexpected error in %s: %s", func, exc)
            return {"result": CUDA_ERROR_UNKNOWN}

    def dispatch_nvenc(self, req: dict) -> dict:  # type: ignore[type-arg]
        """Execute an NVENC call, delegating to nvenc_executor if available."""
        if not self._nvenc_checked:
            self._nvenc_checked = True
            try:
                from . import nvenc_executor
                self._nvenc = nvenc_executor
            except ImportError:
                self._nvenc = None

        if self._nvenc is not None:
            try:
                return self._nvenc.dispatch(req)
            except Exception as exc:
                logger.error("cuda_executor: nvenc error: %s", exc)
                return {"result": CUDA_ERROR_UNKNOWN}

        logger.warning("cuda_executor: nvenc_executor not available; returning NOT_SUPPORTED")
        return {"result": CUDA_ERROR_NOT_SUPPORTED}


# ── Per-function handlers ─────────────────────────────────────────────────────
# Each handler: (req: dict) -> dict

def _cuInit(req: dict) -> dict:  # type: ignore[type-arg]
    r = drv.cuInit(req["flags"])
    return {"result": r}

def _cuDeviceGetCount(req: dict) -> dict:  # type: ignore[type-arg]
    r, count = drv.cuDeviceGetCount()
    return {"result": r, "count": count}

def _cuDeviceGet(req: dict) -> dict:  # type: ignore[type-arg]
    r, device = drv.cuDeviceGet(req["ordinal"])
    return {"result": r, "device": device}

def _cuDeviceGetName(req: dict) -> dict:  # type: ignore[type-arg]
    r, name = drv.cuDeviceGetName(req["len"], req["device"])
    return {"result": r, "name": name}

def _cuDeviceGetAttribute(req: dict) -> dict:  # type: ignore[type-arg]
    r, value = drv.cuDeviceGetAttribute(req["attrib"], req["device"])
    return {"result": r, "value": value}

def _cuDeviceTotalMem(req: dict) -> dict:  # type: ignore[type-arg]
    r, mem = drv.cuDeviceTotalMem(req["device"])
    return {"result": r, "bytes": mem}

def _cuCtxCreate(req: dict) -> dict:  # type: ignore[type-arg]
    r, ctx = drv.cuCtxCreate(req["flags"], req["device"])
    return {"result": r, "ctx_handle": ctx}

def _cuCtxDestroy(req: dict) -> dict:  # type: ignore[type-arg]
    r = drv.cuCtxDestroy(req["ctx_handle"])
    return {"result": r}

def _cuCtxSetCurrent(req: dict) -> dict:  # type: ignore[type-arg]
    r = drv.cuCtxSetCurrent(req["ctx_handle"])
    return {"result": r}

def _cuCtxGetCurrent(req: dict) -> dict:  # type: ignore[type-arg]
    r, ctx = drv.cuCtxGetCurrent()
    return {"result": r, "ctx_handle": ctx}

def _cuMemAlloc(req: dict) -> dict:  # type: ignore[type-arg]
    r, dptr = drv.cuMemAlloc(req["bytesize"])
    return {"result": r, "dptr": dptr}

def _cuMemFree(req: dict) -> dict:  # type: ignore[type-arg]
    r = drv.cuMemFree(req["dptr"])
    return {"result": r}

def _cuMemcpyHtoD(req: dict) -> dict:  # type: ignore[type-arg]
    data: bytes = req["data"]
    r = drv.cuMemcpyHtoD(req["dst"], data)
    return {"result": r}

def _cuMemcpyDtoH(req: dict) -> dict:  # type: ignore[type-arg]
    r, data = drv.cuMemcpyDtoH(req["src"], req["byte_count"])
    return {"result": r, "data": data}

def _cuMemcpyDtoD(req: dict) -> dict:  # type: ignore[type-arg]
    r = drv.cuMemcpyDtoD(req["dst"], req["src"], req["byte_count"])
    return {"result": r}

def _cuMemsetD8(req: dict) -> dict:  # type: ignore[type-arg]
    r = drv.cuMemsetD8(req["dst"], req["value"], req["count"])
    return {"result": r}

def _cuMemsetD16(req: dict) -> dict:  # type: ignore[type-arg]
    r = drv.cuMemsetD16(req["dst"], req["value"], req["count"])
    return {"result": r}

def _cuMemsetD32(req: dict) -> dict:  # type: ignore[type-arg]
    r = drv.cuMemsetD32(req["dst"], req["value"], req["count"])
    return {"result": r}

def _cuMemGetInfo(req: dict) -> dict:  # type: ignore[type-arg]
    r, free, total = drv.cuMemGetInfo()
    return {"result": r, "free": free, "total": total}

def _cuModuleLoadData(req: dict) -> dict:  # type: ignore[type-arg]
    image: bytes = req["image"]
    r, mod = drv.cuModuleLoadData(image)
    return {"result": r, "mod_handle": mod}

# cuModuleLoad is converted to cuModuleLoadData by the sidecar ipc_server.
# The agent only ever receives cuModuleLoadData.
_cuModuleLoad = _cuModuleLoadData

def _cuModuleGetFunction(req: dict) -> dict:  # type: ignore[type-arg]
    r, func = drv.cuModuleGetFunction(req["mod_handle"], req["name"])
    return {"result": r, "func_handle": func}

def _cuLaunchKernel(req: dict) -> dict:  # type: ignore[type-arg]
    r = drv.cuLaunchKernel(
        req["func_handle"],
        req["grid"],
        req["block"],
        req["shared_mem"],
        req["stream_handle"],
        req["params"],
    )
    return {"result": r}

def _cuModuleUnload(req: dict) -> dict:  # type: ignore[type-arg]
    r = drv.cuModuleUnload(req["mod_handle"])
    return {"result": r}

def _cuStreamCreate(req: dict) -> dict:  # type: ignore[type-arg]
    r, stream = drv.cuStreamCreate(req["flags"])
    return {"result": r, "stream_handle": stream}

def _cuStreamDestroy(req: dict) -> dict:  # type: ignore[type-arg]
    r = drv.cuStreamDestroy(req["stream_handle"])
    return {"result": r}

def _cuStreamSynchronize(req: dict) -> dict:  # type: ignore[type-arg]
    r = drv.cuStreamSynchronize(req["stream_handle"])
    return {"result": r}

def _cuStreamWaitEvent(req: dict) -> dict:  # type: ignore[type-arg]
    r = drv.cuStreamWaitEvent(req["stream_handle"], req["event_handle"], req["flags"])
    return {"result": r}

def _cuEventCreate(req: dict) -> dict:  # type: ignore[type-arg]
    r, event = drv.cuEventCreate(req["flags"])
    return {"result": r, "event_handle": event}

def _cuEventDestroy(req: dict) -> dict:  # type: ignore[type-arg]
    r = drv.cuEventDestroy(req["event_handle"])
    return {"result": r}

def _cuEventRecord(req: dict) -> dict:  # type: ignore[type-arg]
    r = drv.cuEventRecord(req["event_handle"], req["stream_handle"])
    return {"result": r}

def _cuEventSynchronize(req: dict) -> dict:  # type: ignore[type-arg]
    r = drv.cuEventSynchronize(req["event_handle"])
    return {"result": r}


def _cuDriverGetVersion(req: dict) -> dict:  # type: ignore[type-arg]
    r, version = drv.cuDriverGetVersion()
    return {"result": r, "version": version}

def _cuDeviceComputeCapability(req: dict) -> dict:  # type: ignore[type-arg]
    r, major, minor = drv.cuDeviceComputeCapability(req["device"])
    return {"result": r, "major": major, "minor": minor}

def _cuDeviceGetUuid(req: dict) -> dict:  # type: ignore[type-arg]
    r, uuid_bytes = drv.cuDeviceGetUuid(req["device"])
    return {"result": r, "uuid": uuid_bytes}

def _cuDeviceGetLuid(req: dict) -> dict:  # type: ignore[type-arg]
    r, luid, mask = drv.cuDeviceGetLuid(req["device"])
    return {"result": r, "luid": luid, "device_node_mask": mask}

def _cuCtxPushCurrent(req: dict) -> dict:  # type: ignore[type-arg]
    r = drv.cuCtxPushCurrent(req["ctx_handle"])
    return {"result": r}

def _cuCtxPopCurrent(req: dict) -> dict:  # type: ignore[type-arg]
    r, ctx = drv.cuCtxPopCurrent()
    return {"result": r, "ctx_handle": ctx}

def _cuMemAllocPitch(req: dict) -> dict:  # type: ignore[type-arg]
    r, dptr, pitch = drv.cuMemAllocPitch(
        req["width_bytes"], req["height"], req["element_size"]
    )
    return {"result": r, "dptr": dptr, "pitch": pitch}

def _cuMemAllocManaged(req: dict) -> dict:  # type: ignore[type-arg]
    r, dptr = drv.cuMemAllocManaged(req["bytesize"], req["flags"])
    return {"result": r, "dptr": dptr}

def _cuCtxSetLimit(req: dict) -> dict:  # type: ignore[type-arg]
    r = drv.cuCtxSetLimit(req["limit"], req["value"])
    return {"result": r}

def _cuCtxGetLimit(req: dict) -> dict:  # type: ignore[type-arg]
    r, value = drv.cuCtxGetLimit(req["limit"])
    return {"result": r, "value": value}


def _cuMemsetD8Async(req: dict) -> dict:  # type: ignore[type-arg]
    r = drv.cuMemsetD8Async(req["dst"], req["value"], req["count"], req["stream_handle"])
    return {"result": r}

def _cuMemsetD16Async(req: dict) -> dict:  # type: ignore[type-arg]
    r = drv.cuMemsetD16Async(req["dst"], req["value"], req["count"], req["stream_handle"])
    return {"result": r}

def _cuMemsetD32Async(req: dict) -> dict:  # type: ignore[type-arg]
    r = drv.cuMemsetD32Async(req["dst"], req["value"], req["count"], req["stream_handle"])
    return {"result": r}

def _cuMemcpy(req: dict) -> dict:  # type: ignore[type-arg]
    r = drv.cuMemcpy(req["dst"], req["src"], req["byte_count"])
    return {"result": r}

def _cuMemcpyAsync(req: dict) -> dict:  # type: ignore[type-arg]
    r = drv.cuMemcpyAsync(req["dst"], req["src"], req["byte_count"], req["stream_handle"])
    return {"result": r}

def _cuMemcpyDtoDAsync(req: dict) -> dict:  # type: ignore[type-arg]
    r = drv.cuMemcpyDtoDAsync(req["dst"], req["src"], req["byte_count"], req["stream_handle"])
    return {"result": r}

def _cuMemcpyHtoDAsync(req: dict) -> dict:  # type: ignore[type-arg]
    r = drv.cuMemcpyHtoDAsync(req["dst"], req["data"], req["stream_handle"])
    return {"result": r}

def _cuMemcpyDtoHAsync(req: dict) -> dict:  # type: ignore[type-arg]
    r, data = drv.cuMemcpyDtoHAsync(req["src"], req["byte_count"], req["stream_handle"])
    return {"result": r, "data": data}

def _cuStreamCreateWithPriority(req: dict) -> dict:  # type: ignore[type-arg]
    r, stream = drv.cuStreamCreateWithPriority(req["flags"], req["priority"])
    return {"result": r, "stream_handle": stream}

def _cuCtxSynchronize(req: dict) -> dict:  # type: ignore[type-arg]
    r = drv.cuCtxSynchronize()
    return {"result": r}

def _cuCtxGetDevice(req: dict) -> dict:  # type: ignore[type-arg]
    r, device = drv.cuCtxGetDevice()
    return {"result": r, "device": device}

def _cuCtxGetFlags(req: dict) -> dict:  # type: ignore[type-arg]
    r, flags = drv.cuCtxGetFlags()
    return {"result": r, "flags": flags}

def _cuCtxGetApiVersion(req: dict) -> dict:  # type: ignore[type-arg]
    r, version = drv.cuCtxGetApiVersion(req["ctx_handle"])
    return {"result": r, "version": version}

def _cuCtxGetCacheConfig(req: dict) -> dict:  # type: ignore[type-arg]
    r, config = drv.cuCtxGetCacheConfig()
    return {"result": r, "config": config}

def _cuCtxSetCacheConfig(req: dict) -> dict:  # type: ignore[type-arg]
    r = drv.cuCtxSetCacheConfig(req["config"])
    return {"result": r}

def _cuCtxGetSharedMemConfig(req: dict) -> dict:  # type: ignore[type-arg]
    r, config = drv.cuCtxGetSharedMemConfig()
    return {"result": r, "config": config}

def _cuCtxSetSharedMemConfig(req: dict) -> dict:  # type: ignore[type-arg]
    r = drv.cuCtxSetSharedMemConfig(req["config"])
    return {"result": r}

def _cuFuncGetAttribute(req: dict) -> dict:  # type: ignore[type-arg]
    r, value = drv.cuFuncGetAttribute(req["attrib"], req["func_handle"])
    return {"result": r, "value": value}

def _cuFuncSetAttribute(req: dict) -> dict:  # type: ignore[type-arg]
    r = drv.cuFuncSetAttribute(req["func_handle"], req["attrib"], req["value"])
    return {"result": r}

def _cuFuncSetCacheConfig(req: dict) -> dict:  # type: ignore[type-arg]
    r = drv.cuFuncSetCacheConfig(req["func_handle"], req["config"])
    return {"result": r}

def _cuFuncSetSharedMemConfig(req: dict) -> dict:  # type: ignore[type-arg]
    r = drv.cuFuncSetSharedMemConfig(req["func_handle"], req["config"])
    return {"result": r}

def _cuOccupancyMaxActiveBlocksPerMultiprocessor(req: dict) -> dict:  # type: ignore[type-arg]
    r, num_blocks = drv.cuOccupancyMaxActiveBlocksPerMultiprocessor(
        req["func_handle"], req["block_size"], req["dynamic_smem_size"]
    )
    return {"result": r, "num_blocks": num_blocks}


def _cuGetExportTable(req: dict) -> dict:  # type: ignore[type-arg]
    uuid_hex: str = req.get("export_table_id", "00" * 16)
    try:
        uuid_bytes = bytes.fromhex(uuid_hex)
    except ValueError:
        return {"result": CUDA_ERROR_NOT_SUPPORTED}
    r, table_bytes = drv.cuGetExportTable(uuid_bytes)
    return {"result": r, "export_table": table_bytes}


# ── Dispatch table ────────────────────────────────────────────────────────────

_HANDLERS = {
    "cuInit":               _cuInit,
    "cuDeviceGetCount":     _cuDeviceGetCount,
    "cuDeviceGet":          _cuDeviceGet,
    "cuDeviceGetName":      _cuDeviceGetName,
    "cuDeviceGetAttribute": _cuDeviceGetAttribute,
    "cuDeviceTotalMem":     _cuDeviceTotalMem,
    "cuCtxCreate":          _cuCtxCreate,
    "cuCtxDestroy":         _cuCtxDestroy,
    "cuCtxSetCurrent":      _cuCtxSetCurrent,
    "cuCtxGetCurrent":      _cuCtxGetCurrent,
    "cuMemAlloc":           _cuMemAlloc,
    "cuMemFree":            _cuMemFree,
    "cuMemcpyHtoD":         _cuMemcpyHtoD,
    "cuMemcpyDtoH":         _cuMemcpyDtoH,
    "cuMemcpyDtoD":         _cuMemcpyDtoD,
    "cuMemsetD8":           _cuMemsetD8,
    "cuMemsetD16":          _cuMemsetD16,
    "cuMemsetD32":          _cuMemsetD32,
    "cuMemGetInfo":         _cuMemGetInfo,
    "cuModuleLoad":         _cuModuleLoad,      # sidecar rewrites to cuModuleLoadData
    "cuModuleLoadData":     _cuModuleLoadData,
    "cuModuleGetFunction":  _cuModuleGetFunction,
    "cuLaunchKernel":       _cuLaunchKernel,
    "cuModuleUnload":       _cuModuleUnload,
    "cuStreamCreate":       _cuStreamCreate,
    "cuStreamDestroy":      _cuStreamDestroy,
    "cuStreamSynchronize":  _cuStreamSynchronize,
    "cuStreamWaitEvent":    _cuStreamWaitEvent,
    "cuEventCreate":        _cuEventCreate,
    "cuEventDestroy":       _cuEventDestroy,
    "cuEventRecord":        _cuEventRecord,
    "cuEventSynchronize":   _cuEventSynchronize,
    "cuDriverGetVersion":          _cuDriverGetVersion,
    "cuDeviceComputeCapability":   _cuDeviceComputeCapability,
    "cuDeviceGetUuid":             _cuDeviceGetUuid,
    "cuDeviceGetLuid":             _cuDeviceGetLuid,
    "cuCtxPushCurrent":            _cuCtxPushCurrent,
    "cuCtxPopCurrent":             _cuCtxPopCurrent,
    "cuMemAllocPitch":             _cuMemAllocPitch,
    "cuMemAllocManaged":           _cuMemAllocManaged,
    "cuCtxSetLimit":               _cuCtxSetLimit,
    "cuCtxGetLimit":               _cuCtxGetLimit,
    "cuGetExportTable":            _cuGetExportTable,
    # Async memset
    "cuMemsetD8Async":             _cuMemsetD8Async,
    "cuMemsetD16Async":            _cuMemsetD16Async,
    "cuMemsetD32Async":            _cuMemsetD32Async,
    # Generic memcpy (D-to-D semantics over IPC)
    "cuMemcpy":                    _cuMemcpy,
    "cuMemcpyAsync":               _cuMemcpyAsync,
    # Async copies
    "cuMemcpyDtoDAsync":           _cuMemcpyDtoDAsync,
    "cuMemcpyHtoDAsync":           _cuMemcpyHtoDAsync,
    "cuMemcpyDtoHAsync":           _cuMemcpyDtoHAsync,
    # Stream
    "cuStreamCreateWithPriority":  _cuStreamCreateWithPriority,
    # Context introspection
    "cuCtxSynchronize":            _cuCtxSynchronize,
    "cuCtxGetDevice":              _cuCtxGetDevice,
    "cuCtxGetFlags":               _cuCtxGetFlags,
    "cuCtxGetApiVersion":          _cuCtxGetApiVersion,
    "cuCtxGetCacheConfig":         _cuCtxGetCacheConfig,
    "cuCtxSetCacheConfig":         _cuCtxSetCacheConfig,
    "cuCtxGetSharedMemConfig":     _cuCtxGetSharedMemConfig,
    "cuCtxSetSharedMemConfig":     _cuCtxSetSharedMemConfig,
    # Function attributes
    "cuFuncGetAttribute":          _cuFuncGetAttribute,
    "cuFuncSetAttribute":          _cuFuncSetAttribute,
    "cuFuncSetCacheConfig":        _cuFuncSetCacheConfig,
    "cuFuncSetSharedMemConfig":    _cuFuncSetSharedMemConfig,
    # Occupancy
    "cuOccupancyMaxActiveBlocksPerMultiprocessor":
                                   _cuOccupancyMaxActiveBlocksPerMultiprocessor,
}
