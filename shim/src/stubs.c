// Copyright (C) 2026 AniiGIT
// SPDX-License-Identifier: AGPL-3.0-or-later

/*
 * stubs.c - Stubs for all CUDA Driver API functions NOT in the Tier 1 set.
 *
 * Every stub:
 *   1. Logs the function name at WARN level (once per call, always — so
 *      production workloads that hit unimplemented paths are visible).
 *   2. Returns CUDA_ERROR_NOT_SUPPORTED.
 *
 * This ensures that applications which call an unimplemented function get
 * a deterministic, visible error instead of falling through to the real
 * CUDA driver (which would execute on the local GPU, defeating the purpose
 * of the shim).
 *
 * Tier 1 functions (implemented in cuda_init.c, cuda_mem.c,
 * cuda_module.c, cuda_stream.c) are NOT listed here.
 *
 * Organisation: grouped loosely by CUDA Driver API section.  New stubs
 * can be added as Phase 3 Tier 2 functions are promoted to full
 * implementations.
 */

#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "idlegpu_shim.h"

// ── Fixed-header layout for variable-length async copies ─────────────────────
// These structs describe the *fixed* header that precedes the data payload
// when forwarding cuMemcpyHtoDAsync over IPC.  They are local to stubs.c and
// mirror the sidecar's ipc_codec.py SPECIAL decode paths.
typedef struct __attribute__((packed)) {
    uint64_t dst;
    uint64_t byte_count;
    uint64_t stream_handle;
} _Req_HtoDAsync_Hdr;   // 24 bytes

typedef struct __attribute__((packed)) {
    uint64_t src;
    uint64_t byte_count;
    uint64_t stream_handle;
} _Req_DtoHAsync_Hdr;   // 24 bytes

// ── Context management (non-Tier-1) ──────────────────────────────────────────

// cuCtxPushCurrent / cuCtxPushCurrent_v2 — push a context onto the calling
// thread's CUDA context stack.  Local driver is tried first so local contexts
// are handled without an IPC round-trip.  When only a remote GPU is present
// (no local libcuda.so.1) the context was created via IPC and the push must
// go to the agent.
__attribute__((visibility("default")))
CUresult cuCtxPushCurrent_v2(CUcontext ctx) {
    // Local driver path.
    if (g_real.cuCtxPushCurrent_v2 != NULL) {
        return g_real.cuCtxPushCurrent_v2(ctx);
    }

    // Remote context via IPC.
    if (g_ipc_connected) {
        Req_cuCtxPushCurrent req = { .ctx_handle = (uint64_t)(uintptr_t)ctx };
        return ipc_call(FN_cuCtxPushCurrent,
                        &req, (uint32_t)sizeof(req),
                        NULL, 0, NULL);
    }

    return CUDA_ERROR_NOT_SUPPORTED;
}

__attribute__((visibility("default")))
CUresult cuCtxPushCurrent(CUcontext ctx) {
    // cuCtxPushCurrent is the deprecated alias for cuCtxPushCurrent_v2.
    return cuCtxPushCurrent_v2(ctx);
}

__attribute__((visibility("default")))
CUresult cuCtxPopCurrent_v2(CUcontext *pctx) {
    if (pctx == NULL) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    // Local driver path.
    if (g_real.cuCtxPopCurrent_v2 != NULL) {
        return g_real.cuCtxPopCurrent_v2(pctx);
    }

    // Remote context via IPC.
    if (g_ipc_connected) {
        Resp_cuCtxPopCurrent resp = { 0 };
        CUresult r = ipc_call(FN_cuCtxPopCurrent,
                              NULL, 0,
                              &resp, (uint32_t)sizeof(resp), NULL);
        if (r == CUDA_SUCCESS) {
            *pctx = (CUcontext)(uintptr_t)resp.ctx_handle;
        }
        return r;
    }

    return CUDA_ERROR_NOT_SUPPORTED;
}

__attribute__((visibility("default")))
CUresult cuCtxPopCurrent(CUcontext *pctx) {
    // cuCtxPopCurrent is the deprecated alias for cuCtxPopCurrent_v2.
    return cuCtxPopCurrent_v2(pctx);
}

// cuCtxSynchronize — block until all work in the current context is done.
// Local driver first; IPC fallback blocks the agent's context.
__attribute__((visibility("default")))
CUresult cuCtxSynchronize(void) {
    if (g_real.cuCtxSynchronize != NULL)
        return g_real.cuCtxSynchronize();
    if (g_ipc_connected)
        return ipc_call(FN_cuCtxSynchronize, NULL, 0, NULL, 0, NULL);
    return CUDA_ERROR_NOT_SUPPORTED;
}

// cuCtxGetDevice — return the device ordinal for the calling thread's context.
__attribute__((visibility("default")))
CUresult cuCtxGetDevice(int *device) {
    if (device == NULL) return CUDA_ERROR_INVALID_VALUE;
    if (g_real.cuCtxGetDevice != NULL)
        return g_real.cuCtxGetDevice(device);
    if (g_ipc_connected) {
        Resp_cuCtxGetDevice resp = { 0 };
        CUresult r = ipc_call(FN_cuCtxGetDevice, NULL, 0,
                              &resp, (uint32_t)sizeof(resp), NULL);
        if (r == CUDA_SUCCESS) *device = resp.device;
        return r;
    }
    return CUDA_ERROR_NOT_SUPPORTED;
}

// cuCtxGetApiVersion — query the driver API version for a context.
__attribute__((visibility("default")))
CUresult cuCtxGetApiVersion(CUcontext ctx, unsigned int *version) {
    if (version == NULL) return CUDA_ERROR_INVALID_VALUE;
    if (g_real.cuCtxGetApiVersion != NULL)
        return g_real.cuCtxGetApiVersion(ctx, version);
    if (g_ipc_connected) {
        Req_cuCtxGetApiVersion req = { .ctx_handle = (uint64_t)(uintptr_t)ctx };
        Resp_cuCtxGetApiVersion resp = { 0 };
        CUresult r = ipc_call(FN_cuCtxGetApiVersion,
                              &req, (uint32_t)sizeof(req),
                              &resp, (uint32_t)sizeof(resp), NULL);
        if (r == CUDA_SUCCESS) *version = resp.version;
        return r;
    }
    return CUDA_ERROR_NOT_SUPPORTED;
}

// cuCtxGetFlags — return flags the calling thread's context was created with.
__attribute__((visibility("default")))
CUresult cuCtxGetFlags(unsigned int *flags) {
    if (flags == NULL) return CUDA_ERROR_INVALID_VALUE;
    if (g_real.cuCtxGetFlags != NULL)
        return g_real.cuCtxGetFlags(flags);
    if (g_ipc_connected) {
        Resp_cuCtxGetFlags resp = { 0 };
        CUresult r = ipc_call(FN_cuCtxGetFlags, NULL, 0,
                              &resp, (uint32_t)sizeof(resp), NULL);
        if (r == CUDA_SUCCESS) *flags = resp.flags;
        return r;
    }
    return CUDA_ERROR_NOT_SUPPORTED;
}

// cuCtxSetLimit / cuCtxGetLimit — apply to the current CUDA context.
// No device parameter; local driver is tried first (covers local contexts
// created by the real driver).  IPC fallback forwards to the agent's
// current context when only a remote GPU is present.
__attribute__((visibility("default")))
CUresult cuCtxSetLimit(CUlimit limit, size_t value) {
    if (g_real.cuCtxSetLimit != NULL) {
        return g_real.cuCtxSetLimit(limit, value);
    }

    if (g_ipc_connected) {
        Req_cuCtxSetLimit req = {
            .limit = (int32_t)limit,
            .value = (uint64_t)value,
        };
        return ipc_call(FN_cuCtxSetLimit,
                        &req, (uint32_t)sizeof(req),
                        NULL, 0, NULL);
    }

    return CUDA_ERROR_NOT_SUPPORTED;
}

__attribute__((visibility("default")))
CUresult cuCtxGetLimit(size_t *pvalue, CUlimit limit) {
    if (pvalue == NULL) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    if (g_real.cuCtxGetLimit != NULL) {
        return g_real.cuCtxGetLimit(pvalue, limit);
    }

    if (g_ipc_connected) {
        Req_cuCtxGetLimit req = { .limit = (int32_t)limit };
        Resp_cuCtxGetLimit resp = { 0 };
        CUresult r = ipc_call(FN_cuCtxGetLimit,
                              &req, (uint32_t)sizeof(req),
                              &resp, (uint32_t)sizeof(resp), NULL);
        if (r == CUDA_SUCCESS) {
            *pvalue = (size_t)resp.value;
        }
        return r;
    }

    return CUDA_ERROR_NOT_SUPPORTED;
}

__attribute__((visibility("default")))
CUresult cuCtxGetCacheConfig(int *pconfig) {
    if (pconfig == NULL) return CUDA_ERROR_INVALID_VALUE;
    if (g_real.cuCtxGetCacheConfig != NULL)
        return g_real.cuCtxGetCacheConfig(pconfig);
    if (g_ipc_connected) {
        Resp_cuCtxGetCacheConfig resp = { 0 };
        CUresult r = ipc_call(FN_cuCtxGetCacheConfig, NULL, 0,
                              &resp, (uint32_t)sizeof(resp), NULL);
        if (r == CUDA_SUCCESS) *pconfig = resp.config;
        return r;
    }
    return CUDA_ERROR_NOT_SUPPORTED;
}

__attribute__((visibility("default")))
CUresult cuCtxSetCacheConfig(int config) {
    if (g_real.cuCtxSetCacheConfig != NULL)
        return g_real.cuCtxSetCacheConfig(config);
    if (g_ipc_connected) {
        Req_cuCtxSetCacheConfig req = { .config = (int32_t)config };
        return ipc_call(FN_cuCtxSetCacheConfig,
                        &req, (uint32_t)sizeof(req), NULL, 0, NULL);
    }
    return CUDA_ERROR_NOT_SUPPORTED;
}

__attribute__((visibility("default")))
CUresult cuCtxGetSharedMemConfig(int *pConfig) {
    if (pConfig == NULL) return CUDA_ERROR_INVALID_VALUE;
    if (g_real.cuCtxGetSharedMemConfig != NULL)
        return g_real.cuCtxGetSharedMemConfig(pConfig);
    if (g_ipc_connected) {
        Resp_cuCtxGetSharedMemConfig resp = { 0 };
        CUresult r = ipc_call(FN_cuCtxGetSharedMemConfig, NULL, 0,
                              &resp, (uint32_t)sizeof(resp), NULL);
        if (r == CUDA_SUCCESS) *pConfig = resp.config;
        return r;
    }
    return CUDA_ERROR_NOT_SUPPORTED;
}

__attribute__((visibility("default")))
CUresult cuCtxSetSharedMemConfig(int config) {
    if (g_real.cuCtxSetSharedMemConfig != NULL)
        return g_real.cuCtxSetSharedMemConfig(config);
    if (g_ipc_connected) {
        Req_cuCtxSetSharedMemConfig req = { .config = (int32_t)config };
        return ipc_call(FN_cuCtxSetSharedMemConfig,
                        &req, (uint32_t)sizeof(req), NULL, 0, NULL);
    }
    return CUDA_ERROR_NOT_SUPPORTED;
}

__attribute__((visibility("default")))
CUresult cuCtxAttach(CUcontext *pctx, unsigned int flags) {
    (void)pctx; (void)flags;
    SHIM_UNIMPLEMENTED("cuCtxAttach");
}

__attribute__((visibility("default")))
CUresult cuCtxDetach(CUcontext ctx) {
    (void)ctx;
    SHIM_UNIMPLEMENTED("cuCtxDetach");
}

// ── Device management (non-Tier-1) ────────────────────────────────────────────

// cuDriverGetVersion — local driver first, then IPC fallback.
//
// ffmpeg's NVENC encoder calls cuDriverGetVersion at startup to confirm
// the driver supports NVENC.  Local-driver path avoids an IPC round-trip
// for the common case where a real libcuda.so.1 is loaded.
__attribute__((visibility("default")))
CUresult cuDriverGetVersion(int *driverVersion) {
    if (driverVersion == NULL) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    // 1. Local driver — fastest path, no IPC.
    if (g_real.cuDriverGetVersion != NULL) {
        return g_real.cuDriverGetVersion(driverVersion);
    }

    // 2. IPC to agent — remote driver version.
    if (g_ipc_connected) {
        Resp_cuDriverGetVersion resp = { 0 };
        CUresult r = ipc_call(FN_cuDriverGetVersion,
                              NULL, 0,
                              &resp, (uint32_t)sizeof(resp), NULL);
        if (r == CUDA_SUCCESS) {
            *driverVersion = resp.version;
        }
        return r;
    }

    return CUDA_ERROR_NOT_SUPPORTED;
}

// cuDeviceGetUuid — device-routed: local → real driver, remote → IPC.
// Returns the 16-byte UUID identifying the physical device.
__attribute__((visibility("default")))
CUresult cuDeviceGetUuid(CUuuid *uuid, CUdevice dev) {
    if (uuid == NULL) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    // Local device path.
    if ((int)(dev) < g_local_device_count) {
        if (g_real.cuDeviceGetUuid != NULL) {
            return g_real.cuDeviceGetUuid(uuid, dev);
        }
        return CUDA_ERROR_NOT_SUPPORTED;
    }

    // Remote device via IPC.
    if (g_ipc_connected) {
        int remote_ord = (int)(dev) - g_local_device_count;
        Req_cuDeviceGetUuid req = { .device = (int32_t)remote_ord };
        Resp_cuDeviceGetUuid resp = { 0 };
        CUresult r = ipc_call(FN_cuDeviceGetUuid,
                              &req, (uint32_t)sizeof(req),
                              &resp, (uint32_t)sizeof(resp), NULL);
        if (r == CUDA_SUCCESS) {
            memcpy(uuid->bytes, resp.bytes, 16);
        }
        return r;
    }

    return CUDA_ERROR_NOT_SUPPORTED;
}

// cuDeviceGetLuid — Windows DXGI interop identifier.
// Local driver first; IPC fallback sends the LUID from the agent's OS.
// On Linux both local driver and agent return NOT_SUPPORTED.
__attribute__((visibility("default")))
CUresult cuDeviceGetLuid(char *luid, unsigned int *deviceNodeMask, CUdevice dev) {
    if (luid == NULL || deviceNodeMask == NULL) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    // Local device path.
    if ((int)(dev) < g_local_device_count) {
        if (g_real.cuDeviceGetLuid != NULL) {
            return g_real.cuDeviceGetLuid(luid, deviceNodeMask, dev);
        }
        return CUDA_ERROR_NOT_SUPPORTED;
    }

    // Remote device via IPC.
    if (g_ipc_connected) {
        int remote_ord = (int)(dev) - g_local_device_count;
        Req_cuDeviceGetLuid req = { .device = (int32_t)remote_ord };
        Resp_cuDeviceGetLuid resp = { 0 };
        CUresult r = ipc_call(FN_cuDeviceGetLuid,
                              &req, (uint32_t)sizeof(req),
                              &resp, (uint32_t)sizeof(resp), NULL);
        if (r == CUDA_SUCCESS) {
            memcpy(luid, resp.luid, 8);
            *deviceNodeMask = resp.device_node_mask;
        }
        return r;
    }

    return CUDA_ERROR_NOT_SUPPORTED;
}

__attribute__((visibility("default")))
CUresult cuDeviceGetByPCIBusId(int *dev, const char *pciBusId) {
    (void)dev; (void)pciBusId;
    SHIM_UNIMPLEMENTED("cuDeviceGetByPCIBusId");
}

__attribute__((visibility("default")))
CUresult cuDeviceGetPCIBusId(char *pciBusId, int len, int dev) {
    (void)pciBusId; (void)len; (void)dev;
    SHIM_UNIMPLEMENTED("cuDeviceGetPCIBusId");
}

__attribute__((visibility("default")))
CUresult cuDeviceCanAccessPeer(int *canAccessPeer, int dev, int peerDev) {
    (void)canAccessPeer; (void)dev; (void)peerDev;
    SHIM_UNIMPLEMENTED("cuDeviceCanAccessPeer");
}

__attribute__((visibility("default")))
CUresult cuDeviceGetP2PAttribute(int *value, int attrib, int srcDevice, int dstDevice) {
    (void)value; (void)attrib; (void)srcDevice; (void)dstDevice;
    SHIM_UNIMPLEMENTED("cuDeviceGetP2PAttribute");
}

// cuDeviceComputeCapability — device-routed: local devices use the real driver,
// remote devices go via IPC.  Deprecated in CUDA 5.0 but still called by ffmpeg.
__attribute__((visibility("default")))
CUresult cuDeviceComputeCapability(int *major, int *minor, CUdevice dev) {
    if (major == NULL || minor == NULL) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    // Local device path.
    if ((int)(dev) < g_local_device_count) {
        if (g_real.cuDeviceComputeCapability != NULL) {
            return g_real.cuDeviceComputeCapability(major, minor, dev);
        }
        return CUDA_ERROR_NOT_SUPPORTED;
    }

    // Remote device via IPC.
    if (g_ipc_connected) {
        int remote_ord = (int)(dev) - g_local_device_count;
        Req_cuDeviceComputeCapability req = { .device = (int32_t)remote_ord };
        Resp_cuDeviceComputeCapability resp = { 0 };
        CUresult r = ipc_call(FN_cuDeviceComputeCapability,
                              &req, (uint32_t)sizeof(req),
                              &resp, (uint32_t)sizeof(resp), NULL);
        if (r == CUDA_SUCCESS) {
            *major = resp.major;
            *minor = resp.minor;
        }
        return r;
    }

    return CUDA_ERROR_NOT_SUPPORTED;
}

__attribute__((visibility("default")))
CUresult cuDeviceGetProperties(void *prop, int dev) {
    (void)prop; (void)dev;
    SHIM_UNIMPLEMENTED("cuDeviceGetProperties");
}

// ── Memory management (non-Tier-1) ────────────────────────────────────────────

// cuMemAllocPitch_v2 — allocate pitched 2D device memory.
// Local driver first (avoids IPC overhead for local contexts); IPC fallback
// when only a remote GPU is present.  ElementSizeBytes controls the pitch
// alignment guarantee: must be 1, 2, 4, 8, or 16.
__attribute__((visibility("default")))
CUresult cuMemAllocPitch_v2(CUdeviceptr *dptr, size_t *pPitch,
                             size_t WidthInBytes, size_t Height,
                             unsigned int ElementSizeBytes) {
    if (dptr == NULL || pPitch == NULL) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    // Local driver path.
    if (g_real.cuMemAllocPitch_v2 != NULL) {
        return g_real.cuMemAllocPitch_v2(dptr, pPitch,
                                         WidthInBytes, Height, ElementSizeBytes);
    }

    // Remote allocation via IPC.
    if (g_ipc_connected) {
        Req_cuMemAllocPitch req = {
            .width_bytes  = (uint64_t)WidthInBytes,
            .height       = (uint64_t)Height,
            .element_size = (uint32_t)ElementSizeBytes,
        };
        Resp_cuMemAllocPitch resp = { 0 };
        CUresult r = ipc_call(FN_cuMemAllocPitch,
                              &req, (uint32_t)sizeof(req),
                              &resp, (uint32_t)sizeof(resp), NULL);
        if (r == CUDA_SUCCESS) {
            *dptr   = (CUdeviceptr)resp.dptr;
            *pPitch = (size_t)resp.pitch;
        }
        return r;
    }

    return CUDA_ERROR_NOT_SUPPORTED;
}

__attribute__((visibility("default")))
CUresult cuMemAllocPitch(CUdeviceptr *dptr, size_t *pPitch,
                         size_t WidthInBytes, size_t Height, unsigned int ElementSizeBytes) {
    // cuMemAllocPitch is the deprecated alias for cuMemAllocPitch_v2.
    return cuMemAllocPitch_v2(dptr, pPitch, WidthInBytes, Height, ElementSizeBytes);
}

__attribute__((visibility("default")))
CUresult cuMemAllocHost(void **pp, size_t bytesize) {
    (void)pp; (void)bytesize;
    SHIM_UNIMPLEMENTED("cuMemAllocHost");
}

__attribute__((visibility("default")))
CUresult cuMemAllocHost_v2(void **pp, size_t bytesize) {
    (void)pp; (void)bytesize;
    SHIM_UNIMPLEMENTED("cuMemAllocHost_v2");
}

__attribute__((visibility("default")))
CUresult cuMemFreeHost(void *p) {
    (void)p;
    SHIM_UNIMPLEMENTED("cuMemFreeHost");
}

// cuMemAllocManaged — allocate unified managed memory.
// Local driver first; IPC fallback when only a remote GPU is present.
// Common flags: CU_MEM_ATTACH_GLOBAL (0x1) — accessible from any stream;
//               CU_MEM_ATTACH_HOST   (0x2) — initially only CPU-accessible.
__attribute__((visibility("default")))
CUresult cuMemAllocManaged(CUdeviceptr *dptr, size_t bytesize, unsigned int flags) {
    if (dptr == NULL) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    // Local driver path.
    if (g_real.cuMemAllocManaged != NULL) {
        return g_real.cuMemAllocManaged(dptr, bytesize, flags);
    }

    // Remote allocation via IPC.
    if (g_ipc_connected) {
        Req_cuMemAllocManaged req = {
            .bytesize = (uint64_t)bytesize,
            .flags    = (uint32_t)flags,
        };
        Resp_cuMemAllocManaged resp = { 0 };
        CUresult r = ipc_call(FN_cuMemAllocManaged,
                              &req, (uint32_t)sizeof(req),
                              &resp, (uint32_t)sizeof(resp), NULL);
        if (r == CUDA_SUCCESS) {
            *dptr = (CUdeviceptr)resp.dptr;
        }
        return r;
    }

    return CUDA_ERROR_NOT_SUPPORTED;
}

__attribute__((visibility("default")))
CUresult cuMemPrefetchAsync(CUdeviceptr devPtr, size_t count,
                            int dstDevice, CUstream hStream) {
    (void)devPtr; (void)count; (void)dstDevice; (void)hStream;
    SHIM_UNIMPLEMENTED("cuMemPrefetchAsync");
}

__attribute__((visibility("default")))
CUresult cuMemAdvise(CUdeviceptr devPtr, size_t count, int advice, int device) {
    (void)devPtr; (void)count; (void)advice; (void)device;
    SHIM_UNIMPLEMENTED("cuMemAdvise");
}

__attribute__((visibility("default")))
CUresult cuMemHostAlloc(void **pp, size_t bytesize, unsigned int Flags) {
    (void)pp; (void)bytesize; (void)Flags;
    SHIM_UNIMPLEMENTED("cuMemHostAlloc");
}

__attribute__((visibility("default")))
CUresult cuMemHostRegister(void *p, size_t bytesize, unsigned int Flags) {
    (void)p; (void)bytesize; (void)Flags;
    SHIM_UNIMPLEMENTED("cuMemHostRegister");
}

__attribute__((visibility("default")))
CUresult cuMemHostRegister_v2(void *p, size_t bytesize, unsigned int Flags) {
    (void)p; (void)bytesize; (void)Flags;
    SHIM_UNIMPLEMENTED("cuMemHostRegister_v2");
}

__attribute__((visibility("default")))
CUresult cuMemHostUnregister(void *p) {
    (void)p;
    SHIM_UNIMPLEMENTED("cuMemHostUnregister");
}

__attribute__((visibility("default")))
CUresult cuMemHostGetDevicePointer(CUdeviceptr *pdptr, void *p, unsigned int Flags) {
    (void)pdptr; (void)p; (void)Flags;
    SHIM_UNIMPLEMENTED("cuMemHostGetDevicePointer");
}

__attribute__((visibility("default")))
CUresult cuMemHostGetDevicePointer_v2(CUdeviceptr *pdptr, void *p, unsigned int Flags) {
    (void)pdptr; (void)p; (void)Flags;
    SHIM_UNIMPLEMENTED("cuMemHostGetDevicePointer_v2");
}

__attribute__((visibility("default")))
CUresult cuMemHostGetFlags(unsigned int *pFlags, void *p) {
    (void)pFlags; (void)p;
    SHIM_UNIMPLEMENTED("cuMemHostGetFlags");
}

__attribute__((visibility("default")))
CUresult cuIpcGetMemHandle(void *pHandle, CUdeviceptr dptr) {
    (void)pHandle; (void)dptr;
    SHIM_UNIMPLEMENTED("cuIpcGetMemHandle");
}

__attribute__((visibility("default")))
CUresult cuIpcOpenMemHandle(CUdeviceptr *pdptr, void *handle, unsigned int Flags) {
    (void)pdptr; (void)handle; (void)Flags;
    SHIM_UNIMPLEMENTED("cuIpcOpenMemHandle");
}

__attribute__((visibility("default")))
CUresult cuIpcCloseMemHandle(CUdeviceptr dptr) {
    (void)dptr;
    SHIM_UNIMPLEMENTED("cuIpcCloseMemHandle");
}

__attribute__((visibility("default")))
CUresult cuMemGetAddressRange(CUdeviceptr *pbase, size_t *psize, CUdeviceptr dptr) {
    (void)pbase; (void)psize; (void)dptr;
    SHIM_UNIMPLEMENTED("cuMemGetAddressRange");
}

__attribute__((visibility("default")))
CUresult cuMemGetAddressRange_v2(CUdeviceptr *pbase, size_t *psize, CUdeviceptr dptr) {
    (void)pbase; (void)psize; (void)dptr;
    SHIM_UNIMPLEMENTED("cuMemGetAddressRange_v2");
}

// cuMemcpyHtoDAsync_v2 — async host→device copy on a stream.
// Local driver: falls back to synchronous cuMemcpyHtoD (g_real field is a
// proxy for local-driver presence; correct async is implicit via CUDA stream
// ordering).  IPC: serialised round-trip (awaits response before return).
__attribute__((visibility("default")))
CUresult cuMemcpyHtoDAsync_v2(CUdeviceptr dstDevice, const void *srcHost,
                               size_t ByteCount, CUstream hStream) {
    if (srcHost == NULL) return CUDA_ERROR_INVALID_VALUE;
    // Local-driver fallback: reuse synchronous HtoD (safe — all in-process).
    if (g_real.cuMemcpyHtoD != NULL)
        return g_real.cuMemcpyHtoD(dstDevice, srcHost, ByteCount);
    // IPC path: allocate combined header + data buffer.
    if (g_ipc_connected && ByteCount <= IPC_MAX_PAYLOAD - sizeof(_Req_HtoDAsync_Hdr)) {
        uint32_t total = (uint32_t)(sizeof(_Req_HtoDAsync_Hdr) + ByteCount);
        uint8_t *buf = (uint8_t *)malloc(total);
        if (buf == NULL) return CUDA_ERROR_OUT_OF_MEMORY;
        _Req_HtoDAsync_Hdr *hdr = (_Req_HtoDAsync_Hdr *)buf;
        hdr->dst           = (uint64_t)dstDevice;
        hdr->byte_count    = (uint64_t)ByteCount;
        hdr->stream_handle = (uint64_t)(uintptr_t)hStream;
        memcpy(buf + sizeof(_Req_HtoDAsync_Hdr), srcHost, ByteCount);
        CUresult r = ipc_call(FN_cuMemcpyHtoDAsync,
                              buf, total, NULL, 0, NULL);
        free(buf);
        return r;
    }
    return CUDA_ERROR_NOT_SUPPORTED;
}

__attribute__((visibility("default")))
CUresult cuMemcpyHtoDAsync(CUdeviceptr dstDevice, const void *srcHost,
                           size_t ByteCount, CUstream hStream) {
    return cuMemcpyHtoDAsync_v2(dstDevice, srcHost, ByteCount, hStream);
}

// cuMemcpyDtoHAsync_v2 — async device→host copy on a stream.
__attribute__((visibility("default")))
CUresult cuMemcpyDtoHAsync_v2(void *dstHost, CUdeviceptr srcDevice,
                               size_t ByteCount, CUstream hStream) {
    if (dstHost == NULL) return CUDA_ERROR_INVALID_VALUE;
    if (g_real.cuMemcpyDtoH != NULL) {
        // Synchronous fallback via the local driver (safe; driver DMA is async).
        return g_real.cuMemcpyDtoH(dstHost, srcDevice, ByteCount);
    }
    if (g_ipc_connected) {
        _Req_DtoHAsync_Hdr req = {
            .src           = (uint64_t)srcDevice,
            .byte_count    = (uint64_t)ByteCount,
            .stream_handle = (uint64_t)(uintptr_t)hStream,
        };
        uint32_t resp_len = 0;
        CUresult r = ipc_call(FN_cuMemcpyDtoHAsync,
                              &req, (uint32_t)sizeof(req),
                              dstHost, (uint32_t)ByteCount, &resp_len);
        return r;
    }
    return CUDA_ERROR_NOT_SUPPORTED;
}

__attribute__((visibility("default")))
CUresult cuMemcpyDtoHAsync(void *dstHost, CUdeviceptr srcDevice,
                           size_t ByteCount, CUstream hStream) {
    return cuMemcpyDtoHAsync_v2(dstHost, srcDevice, ByteCount, hStream);
}

// cuMemcpyDtoDAsync_v2 — async device-to-device copy on a stream.
// Local driver first; IPC fallback is serialized (awaits IPC response).
__attribute__((visibility("default")))
CUresult cuMemcpyDtoDAsync_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice,
                               size_t ByteCount, CUstream hStream) {
    if (g_real.cuMemcpyDtoDAsync_v2 != NULL)
        return g_real.cuMemcpyDtoDAsync_v2(dstDevice, srcDevice,
                                            ByteCount, hStream);
    if (g_ipc_connected) {
        Req_cuMemcpyDtoDAsync req = {
            .dst           = (uint64_t)dstDevice,
            .src           = (uint64_t)srcDevice,
            .byte_count    = (uint64_t)ByteCount,
            .stream_handle = (uint64_t)(uintptr_t)hStream,
        };
        return ipc_call(FN_cuMemcpyDtoDAsync,
                        &req, (uint32_t)sizeof(req), NULL, 0, NULL);
    }
    return CUDA_ERROR_NOT_SUPPORTED;
}

__attribute__((visibility("default")))
CUresult cuMemcpyDtoDAsync(CUdeviceptr dstDevice, CUdeviceptr srcDevice,
                           size_t ByteCount, CUstream hStream) {
    return cuMemcpyDtoDAsync_v2(dstDevice, srcDevice, ByteCount, hStream);
}

__attribute__((visibility("default")))
CUresult cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount) {
    (void)dst; (void)src; (void)ByteCount;
    SHIM_UNIMPLEMENTED("cuMemcpy");
}

__attribute__((visibility("default")))
CUresult cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src,
                       size_t ByteCount, CUstream hStream) {
    (void)dst; (void)src; (void)ByteCount; (void)hStream;
    SHIM_UNIMPLEMENTED("cuMemcpyAsync");
}

__attribute__((visibility("default")))
CUresult cuMemcpyPeer(CUdeviceptr dstDevice, CUcontext dstContext,
                      CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount) {
    (void)dstDevice; (void)dstContext; (void)srcDevice; (void)srcContext; (void)ByteCount;
    SHIM_UNIMPLEMENTED("cuMemcpyPeer");
}

__attribute__((visibility("default")))
CUresult cuMemcpyPeerAsync(CUdeviceptr dstDevice, CUcontext dstContext,
                           CUdeviceptr srcDevice, CUcontext srcContext,
                           size_t ByteCount, CUstream hStream) {
    (void)dstDevice; (void)dstContext; (void)srcDevice; (void)srcContext;
    (void)ByteCount; (void)hStream;
    SHIM_UNIMPLEMENTED("cuMemcpyPeerAsync");
}

// cuMemsetD8/D16/D32Async — stream-ordered memset variants.
// Local driver first; IPC fallback when only a remote GPU is present.
__attribute__((visibility("default")))
CUresult cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc,
                         size_t N, CUstream hStream) {
    if (g_real.cuMemsetD8Async != NULL)
        return g_real.cuMemsetD8Async(dstDevice, uc, N, hStream);
    if (g_ipc_connected) {
        Req_cuMemsetD8Async req = {
            .dst           = (uint64_t)dstDevice,
            .value         = uc,
            .count         = (uint64_t)N,
            .stream_handle = (uint64_t)(uintptr_t)hStream,
        };
        return ipc_call(FN_cuMemsetD8Async,
                        &req, (uint32_t)sizeof(req), NULL, 0, NULL);
    }
    return CUDA_ERROR_NOT_SUPPORTED;
}

__attribute__((visibility("default")))
CUresult cuMemsetD16Async(CUdeviceptr dstDevice, unsigned short us,
                          size_t N, CUstream hStream) {
    if (g_real.cuMemsetD16Async != NULL)
        return g_real.cuMemsetD16Async(dstDevice, us, N, hStream);
    if (g_ipc_connected) {
        Req_cuMemsetD16Async req = {
            .dst           = (uint64_t)dstDevice,
            .value         = us,
            .count         = (uint64_t)N,
            .stream_handle = (uint64_t)(uintptr_t)hStream,
        };
        return ipc_call(FN_cuMemsetD16Async,
                        &req, (uint32_t)sizeof(req), NULL, 0, NULL);
    }
    return CUDA_ERROR_NOT_SUPPORTED;
}

__attribute__((visibility("default")))
CUresult cuMemsetD32Async(CUdeviceptr dstDevice, unsigned int ui,
                          size_t N, CUstream hStream) {
    if (g_real.cuMemsetD32Async != NULL)
        return g_real.cuMemsetD32Async(dstDevice, ui, N, hStream);
    if (g_ipc_connected) {
        Req_cuMemsetD32Async req = {
            .dst           = (uint64_t)dstDevice,
            .value         = ui,
            .count         = (uint64_t)N,
            .stream_handle = (uint64_t)(uintptr_t)hStream,
        };
        return ipc_call(FN_cuMemsetD32Async,
                        &req, (uint32_t)sizeof(req), NULL, 0, NULL);
    }
    return CUDA_ERROR_NOT_SUPPORTED;
}

__attribute__((visibility("default")))
CUresult cuMemset2D(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui,
                    size_t Width, size_t Height) {
    (void)dstDevice; (void)dstPitch; (void)ui; (void)Width; (void)Height;
    SHIM_UNIMPLEMENTED("cuMemset2D");
}

__attribute__((visibility("default")))
CUresult cuMemset2DAsync(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui,
                         size_t Width, size_t Height, CUstream hStream) {
    (void)dstDevice; (void)dstPitch; (void)ui; (void)Width; (void)Height; (void)hStream;
    SHIM_UNIMPLEMENTED("cuMemset2DAsync");
}

// ── Module management (non-Tier-1) ────────────────────────────────────────────

__attribute__((visibility("default")))
CUresult cuModuleGetGlobal(CUdeviceptr *dptr, size_t *bytes,
                           CUmodule hmod, const char *name) {
    (void)dptr; (void)bytes; (void)hmod; (void)name;
    SHIM_UNIMPLEMENTED("cuModuleGetGlobal");
}

__attribute__((visibility("default")))
CUresult cuModuleGetGlobal_v2(CUdeviceptr *dptr, size_t *bytes,
                               CUmodule hmod, const char *name) {
    (void)dptr; (void)bytes; (void)hmod; (void)name;
    SHIM_UNIMPLEMENTED("cuModuleGetGlobal_v2");
}

__attribute__((visibility("default")))
CUresult cuModuleGetTexRef(void **pTexRef, CUmodule hmod, const char *name) {
    (void)pTexRef; (void)hmod; (void)name;
    SHIM_UNIMPLEMENTED("cuModuleGetTexRef");
}

__attribute__((visibility("default")))
CUresult cuModuleGetSurfRef(void **pSurfRef, CUmodule hmod, const char *name) {
    (void)pSurfRef; (void)hmod; (void)name;
    SHIM_UNIMPLEMENTED("cuModuleGetSurfRef");
}

__attribute__((visibility("default")))
CUresult cuModuleLoadDataEx(CUmodule *module, const void *image,
                             unsigned int numOptions, int *options,
                             void **optionValues) {
    (void)module; (void)image; (void)numOptions; (void)options; (void)optionValues;
    SHIM_UNIMPLEMENTED("cuModuleLoadDataEx");
}

__attribute__((visibility("default")))
CUresult cuModuleLoadFatBinary(CUmodule *module, const void *fatCubin) {
    (void)module; (void)fatCubin;
    SHIM_UNIMPLEMENTED("cuModuleLoadFatBinary");
}

__attribute__((visibility("default")))
CUresult cuLinkCreate(unsigned int numOptions, int *options,
                      void **optionValues, void **stateOut) {
    (void)numOptions; (void)options; (void)optionValues; (void)stateOut;
    SHIM_UNIMPLEMENTED("cuLinkCreate");
}

__attribute__((visibility("default")))
CUresult cuLinkAddData(void *state, int type, void *data, size_t size,
                       const char *name, unsigned int numOptions,
                       int *options, void **optionValues) {
    (void)state; (void)type; (void)data; (void)size; (void)name;
    (void)numOptions; (void)options; (void)optionValues;
    SHIM_UNIMPLEMENTED("cuLinkAddData");
}

__attribute__((visibility("default")))
CUresult cuLinkAddFile(void *state, int type, const char *path,
                       unsigned int numOptions, int *options, void **optionValues) {
    (void)state; (void)type; (void)path; (void)numOptions; (void)options; (void)optionValues;
    SHIM_UNIMPLEMENTED("cuLinkAddFile");
}

__attribute__((visibility("default")))
CUresult cuLinkComplete(void *state, void **cubinOut, size_t *sizeOut) {
    (void)state; (void)cubinOut; (void)sizeOut;
    SHIM_UNIMPLEMENTED("cuLinkComplete");
}

__attribute__((visibility("default")))
CUresult cuLinkDestroy(void *state) {
    (void)state;
    SHIM_UNIMPLEMENTED("cuLinkDestroy");
}

// ── Kernel / function attributes (non-Tier-1) ─────────────────────────────────

// cuFuncGetAttribute — query a scalar attribute of a kernel function.
// The function handle may be a local or remote (IPC) pointer; we rely on
// the invariant that if g_real.cuFuncGetAttribute is available, all
// function handles in this process were created via the local driver.
__attribute__((visibility("default")))
CUresult cuFuncGetAttribute(int *pi, int attrib, CUfunction hfunc) {
    if (pi == NULL) return CUDA_ERROR_INVALID_VALUE;
    if (g_real.cuFuncGetAttribute != NULL)
        return g_real.cuFuncGetAttribute(pi, attrib, hfunc);
    if (g_ipc_connected) {
        Req_cuFuncGetAttribute req = {
            .attrib      = (int32_t)attrib,
            .func_handle = (uint64_t)(uintptr_t)hfunc,
        };
        Resp_cuFuncGetAttribute resp = { 0 };
        CUresult r = ipc_call(FN_cuFuncGetAttribute,
                              &req, (uint32_t)sizeof(req),
                              &resp, (uint32_t)sizeof(resp), NULL);
        if (r == CUDA_SUCCESS) *pi = resp.value;
        return r;
    }
    return CUDA_ERROR_NOT_SUPPORTED;
}

__attribute__((visibility("default")))
CUresult cuFuncSetAttribute(CUfunction hfunc, int attrib, int value) {
    if (g_real.cuFuncSetAttribute != NULL)
        return g_real.cuFuncSetAttribute(hfunc, attrib, value);
    if (g_ipc_connected) {
        Req_cuFuncSetAttribute req = {
            .func_handle = (uint64_t)(uintptr_t)hfunc,
            .attrib      = (int32_t)attrib,
            .value       = (int32_t)value,
        };
        return ipc_call(FN_cuFuncSetAttribute,
                        &req, (uint32_t)sizeof(req), NULL, 0, NULL);
    }
    return CUDA_ERROR_NOT_SUPPORTED;
}

__attribute__((visibility("default")))
CUresult cuFuncSetCacheConfig(CUfunction hfunc, int config) {
    if (g_real.cuFuncSetCacheConfig != NULL)
        return g_real.cuFuncSetCacheConfig(hfunc, config);
    if (g_ipc_connected) {
        Req_cuFuncSetCacheConfig req = {
            .func_handle = (uint64_t)(uintptr_t)hfunc,
            .config      = (int32_t)config,
        };
        return ipc_call(FN_cuFuncSetCacheConfig,
                        &req, (uint32_t)sizeof(req), NULL, 0, NULL);
    }
    return CUDA_ERROR_NOT_SUPPORTED;
}

__attribute__((visibility("default")))
CUresult cuFuncSetSharedMemConfig(CUfunction hfunc, int config) {
    if (g_real.cuFuncSetSharedMemConfig != NULL)
        return g_real.cuFuncSetSharedMemConfig(hfunc, config);
    if (g_ipc_connected) {
        Req_cuFuncSetSharedMemConfig req = {
            .func_handle = (uint64_t)(uintptr_t)hfunc,
            .config      = (int32_t)config,
        };
        return ipc_call(FN_cuFuncSetSharedMemConfig,
                        &req, (uint32_t)sizeof(req), NULL, 0, NULL);
    }
    return CUDA_ERROR_NOT_SUPPORTED;
}

__attribute__((visibility("default")))
CUresult cuParamSetSize(CUfunction hfunc, unsigned int numbytes) {
    (void)hfunc; (void)numbytes;
    SHIM_UNIMPLEMENTED("cuParamSetSize");
}

__attribute__((visibility("default")))
CUresult cuParamSetv(CUfunction hfunc, int offset, void *ptr, unsigned int numbytes) {
    (void)hfunc; (void)offset; (void)ptr; (void)numbytes;
    SHIM_UNIMPLEMENTED("cuParamSetv");
}

__attribute__((visibility("default")))
CUresult cuLaunch(CUfunction f) {
    (void)f;
    SHIM_UNIMPLEMENTED("cuLaunch");
}

__attribute__((visibility("default")))
CUresult cuLaunchGrid(CUfunction f, int grid_width, int grid_height) {
    (void)f; (void)grid_width; (void)grid_height;
    SHIM_UNIMPLEMENTED("cuLaunchGrid");
}

__attribute__((visibility("default")))
CUresult cuLaunchCooperativeKernel(CUfunction f,
                                   unsigned int gridDimX, unsigned int gridDimY,
                                   unsigned int gridDimZ,
                                   unsigned int blockDimX, unsigned int blockDimY,
                                   unsigned int blockDimZ,
                                   unsigned int sharedMemBytes, CUstream hStream,
                                   void **kernelParams) {
    (void)f; (void)gridDimX; (void)gridDimY; (void)gridDimZ;
    (void)blockDimX; (void)blockDimY; (void)blockDimZ;
    (void)sharedMemBytes; (void)hStream; (void)kernelParams;
    SHIM_UNIMPLEMENTED("cuLaunchCooperativeKernel");
}

__attribute__((visibility("default")))
CUresult cuLaunchHostFunc(CUstream hStream, void (*fn)(void *), void *userData) {
    (void)hStream; (void)fn; (void)userData;
    SHIM_UNIMPLEMENTED("cuLaunchHostFunc");
}

// cuOccupancyMaxActiveBlocksPerMultiprocessor — query theoretical occupancy.
// Local driver first; IPC fallback sends function handle + parameters.
__attribute__((visibility("default")))
CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, CUfunction func,
                                                      int blockSize, size_t dynamicSMemSize) {
    if (numBlocks == NULL) return CUDA_ERROR_INVALID_VALUE;
    if (g_real.cuOccupancyMaxActiveBlocksPerMultiprocessor != NULL)
        return g_real.cuOccupancyMaxActiveBlocksPerMultiprocessor(
                   numBlocks, func, blockSize, dynamicSMemSize);
    if (g_ipc_connected) {
        Req_cuOccupancyMaxActiveBlocksPerMultiprocessor req = {
            .func_handle      = (uint64_t)(uintptr_t)func,
            .block_size       = (int32_t)blockSize,
            .dynamic_smem_size = (uint64_t)dynamicSMemSize,
        };
        Resp_cuOccupancyMaxActiveBlocksPerMultiprocessor resp = { 0 };
        CUresult r = ipc_call(FN_cuOccupancyMaxActiveBlocksPerMultiprocessor,
                              &req, (uint32_t)sizeof(req),
                              &resp, (uint32_t)sizeof(resp), NULL);
        if (r == CUDA_SUCCESS) *numBlocks = resp.num_blocks;
        return r;
    }
    return CUDA_ERROR_NOT_SUPPORTED;
}

// cuOccupancyMaxPotentialBlockSize — callback not serialisable over IPC;
// local driver only.
__attribute__((visibility("default")))
CUresult cuOccupancyMaxPotentialBlockSize(int *minGridSize, int *blockSize,
                                          CUfunction func, void *blockSizeToDynamicSMemSize,
                                          size_t dynamicSMemSize, int blockSizeLimit) {
    if (g_real.cuOccupancyMaxPotentialBlockSize != NULL)
        return g_real.cuOccupancyMaxPotentialBlockSize(
                   minGridSize, blockSize, func,
                   blockSizeToDynamicSMemSize, dynamicSMemSize, blockSizeLimit);
    return CUDA_ERROR_NOT_SUPPORTED;
}

// ── Streams (non-Tier-1) ──────────────────────────────────────────────────────

// cuStreamAddCallback — function pointer not serialisable over IPC; local only.
__attribute__((visibility("default")))
CUresult cuStreamAddCallback(CUstream hStream, void *callback,
                             void *userData, unsigned int flags) {
    if (g_real.cuStreamAddCallback != NULL)
        return g_real.cuStreamAddCallback(hStream, callback, userData, flags);
    return CUDA_ERROR_NOT_SUPPORTED;
}

__attribute__((visibility("default")))
CUresult cuStreamAttachMemAsync(CUstream hStream, CUdeviceptr dptr,
                                size_t length, unsigned int flags) {
    (void)hStream; (void)dptr; (void)length; (void)flags;
    SHIM_UNIMPLEMENTED("cuStreamAttachMemAsync");
}

// cuStreamQuery — non-blocking poll; local driver only (remote polling is
// meaningless without extra latency).
__attribute__((visibility("default")))
CUresult cuStreamQuery(CUstream hStream) {
    if (g_real.cuStreamQuery != NULL)
        return g_real.cuStreamQuery(hStream);
    return CUDA_ERROR_NOT_SUPPORTED;
}

// cuStreamGetCtx/Flags/Priority — inspect stream state; local driver only.
__attribute__((visibility("default")))
CUresult cuStreamGetCtx(CUstream hStream, CUcontext *pctx) {
    if (pctx == NULL) return CUDA_ERROR_INVALID_VALUE;
    if (g_real.cuStreamGetCtx != NULL)
        return g_real.cuStreamGetCtx(hStream, pctx);
    return CUDA_ERROR_NOT_SUPPORTED;
}

__attribute__((visibility("default")))
CUresult cuStreamGetFlags(CUstream hStream, unsigned int *flags) {
    if (flags == NULL) return CUDA_ERROR_INVALID_VALUE;
    if (g_real.cuStreamGetFlags != NULL)
        return g_real.cuStreamGetFlags(hStream, flags);
    return CUDA_ERROR_NOT_SUPPORTED;
}

__attribute__((visibility("default")))
CUresult cuStreamGetPriority(CUstream hStream, int *priority) {
    if (priority == NULL) return CUDA_ERROR_INVALID_VALUE;
    if (g_real.cuStreamGetPriority != NULL)
        return g_real.cuStreamGetPriority(hStream, priority);
    return CUDA_ERROR_NOT_SUPPORTED;
}

// cuStreamCreateWithPriority — creates a stream with a scheduling priority.
// Priority is a hint to the driver; local driver first; IPC fallback creates
// the stream on the remote GPU.
__attribute__((visibility("default")))
CUresult cuStreamCreateWithPriority(CUstream *phStream, unsigned int flags, int priority) {
    if (phStream == NULL) return CUDA_ERROR_INVALID_VALUE;
    if (g_real.cuStreamCreateWithPriority != NULL)
        return g_real.cuStreamCreateWithPriority(phStream, flags, priority);
    if (g_ipc_connected) {
        Req_cuStreamCreateWithPriority req = {
            .flags    = flags,
            .priority = (int32_t)priority,
        };
        Resp_cuStreamCreateWithPriority resp = { 0 };
        CUresult r = ipc_call(FN_cuStreamCreateWithPriority,
                              &req, (uint32_t)sizeof(req),
                              &resp, (uint32_t)sizeof(resp), NULL);
        if (r == CUDA_SUCCESS)
            *phStream = (CUstream)(uintptr_t)resp.stream_handle;
        return r;
    }
    return CUDA_ERROR_NOT_SUPPORTED;
}

// ── Events (non-Tier-1) ───────────────────────────────────────────────────────

// cuEventQuery — non-blocking event status check; local driver only.
__attribute__((visibility("default")))
CUresult cuEventQuery(CUevent hEvent) {
    if (g_real.cuEventQuery != NULL)
        return g_real.cuEventQuery(hEvent);
    return CUDA_ERROR_NOT_SUPPORTED;
}

// cuEventElapsedTime — timing between two events; local driver only.
__attribute__((visibility("default")))
CUresult cuEventElapsedTime(float *pMilliseconds, CUevent hStart, CUevent hEnd) {
    if (pMilliseconds == NULL) return CUDA_ERROR_INVALID_VALUE;
    if (g_real.cuEventElapsedTime != NULL)
        return g_real.cuEventElapsedTime(pMilliseconds, hStart, hEnd);
    return CUDA_ERROR_NOT_SUPPORTED;
}

__attribute__((visibility("default")))
CUresult cuIpcGetEventHandle(void *pHandle, CUevent event) {
    (void)pHandle; (void)event;
    SHIM_UNIMPLEMENTED("cuIpcGetEventHandle");
}

__attribute__((visibility("default")))
CUresult cuIpcOpenEventHandle(CUevent *phEvent, void *handle) {
    (void)phEvent; (void)handle;
    SHIM_UNIMPLEMENTED("cuIpcOpenEventHandle");
}

// ── Peer access ───────────────────────────────────────────────────────────────

__attribute__((visibility("default")))
CUresult cuCtxEnablePeerAccess(CUcontext peerContext, unsigned int Flags) {
    (void)peerContext; (void)Flags;
    SHIM_UNIMPLEMENTED("cuCtxEnablePeerAccess");
}

__attribute__((visibility("default")))
CUresult cuCtxDisablePeerAccess(CUcontext peerContext) {
    (void)peerContext;
    SHIM_UNIMPLEMENTED("cuCtxDisablePeerAccess");
}

// ── CUDA Graph API ────────────────────────────────────────────────────────────

__attribute__((visibility("default")))
CUresult cuGraphCreate(void **phGraph, unsigned int flags) {
    (void)phGraph; (void)flags;
    SHIM_UNIMPLEMENTED("cuGraphCreate");
}

__attribute__((visibility("default")))
CUresult cuGraphLaunch(void *hGraphExec, CUstream hStream) {
    (void)hGraphExec; (void)hStream;
    SHIM_UNIMPLEMENTED("cuGraphLaunch");
}

__attribute__((visibility("default")))
CUresult cuGraphExecDestroy(void *hGraphExec) {
    (void)hGraphExec;
    SHIM_UNIMPLEMENTED("cuGraphExecDestroy");
}

__attribute__((visibility("default")))
CUresult cuGraphDestroy(void *hGraph) {
    (void)hGraph;
    SHIM_UNIMPLEMENTED("cuGraphDestroy");
}

__attribute__((visibility("default")))
CUresult cuGraphInstantiate(void **phGraphExec, void *hGraph,
                             void **phErrorNode, char *logBuffer, size_t bufferSize) {
    (void)phGraphExec; (void)hGraph; (void)phErrorNode; (void)logBuffer; (void)bufferSize;
    SHIM_UNIMPLEMENTED("cuGraphInstantiate");
}

__attribute__((visibility("default")))
CUresult cuGraphAddKernelNode(void **phGraphNode, void *hGraph,
                               const void *const *dependencies,
                               size_t numDependencies, const void *nodeParams) {
    (void)phGraphNode; (void)hGraph; (void)dependencies; (void)numDependencies; (void)nodeParams;
    SHIM_UNIMPLEMENTED("cuGraphAddKernelNode");
}

__attribute__((visibility("default")))
CUresult cuGraphAddMemcpyNode(void **phGraphNode, void *hGraph,
                               const void *const *dependencies,
                               size_t numDependencies, const void *copyParams,
                               CUcontext ctx) {
    (void)phGraphNode; (void)hGraph; (void)dependencies; (void)numDependencies;
    (void)copyParams; (void)ctx;
    SHIM_UNIMPLEMENTED("cuGraphAddMemcpyNode");
}

__attribute__((visibility("default")))
CUresult cuGraphAddMemsetNode(void **phGraphNode, void *hGraph,
                               const void *const *dependencies,
                               size_t numDependencies, const void *memsetParams,
                               CUcontext ctx) {
    (void)phGraphNode; (void)hGraph; (void)dependencies; (void)numDependencies;
    (void)memsetParams; (void)ctx;
    SHIM_UNIMPLEMENTED("cuGraphAddMemsetNode");
}

// ── Error reporting ───────────────────────────────────────────────────────────

__attribute__((visibility("default")))
CUresult cuGetErrorName(CUresult error, const char **pStr) {
    (void)error; (void)pStr;
    SHIM_UNIMPLEMENTED("cuGetErrorName");
}

__attribute__((visibility("default")))
CUresult cuGetErrorString(CUresult error, const char **pStr) {
    (void)error; (void)pStr;
    SHIM_UNIMPLEMENTED("cuGetErrorString");
}

// ── Runtime internals ─────────────────────────────────────────────────────────

// cuGetExportTable — local driver only.
//
// Export tables contain driver-internal function pointers that are only
// meaningful as virtual addresses within the process that owns them.
// Forwarding via IPC would return agent-side VAs which are uncallable
// in the sidecar/shim process, and there is no safe way to remap them.
//
// For local devices (index < g_local_device_count): delegate to the real
// libcuda.so.1 directly — no IPC involved, no startup timeout.
//
// For remote-only contexts (no local GPU): return CUDA_ERROR_NOT_SUPPORTED.
// Applications relying on cuGetExportTable use CUDA runtime internals that
// are incompatible with remote forwarding and will need the local driver path.
__attribute__((visibility("default")))
CUresult cuGetExportTable(const void **ppExportTable, const CUuuid *pExportTableId) {
    if (ppExportTable == NULL || pExportTableId == NULL) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    if (g_real.cuGetExportTable != NULL) {
        return g_real.cuGetExportTable(ppExportTable, pExportTableId);
    }

    return CUDA_ERROR_NOT_SUPPORTED;
}

// ── Profiler ──────────────────────────────────────────────────────────────────

__attribute__((visibility("default")))
CUresult cuProfilerInitialize(const char *configFile, const char *outputFile, int outputMode) {
    (void)configFile; (void)outputFile; (void)outputMode;
    SHIM_UNIMPLEMENTED("cuProfilerInitialize");
}

__attribute__((visibility("default")))
CUresult cuProfilerStart(void) {
    SHIM_UNIMPLEMENTED("cuProfilerStart");
}

__attribute__((visibility("default")))
CUresult cuProfilerStop(void) {
    SHIM_UNIMPLEMENTED("cuProfilerStop");
}
