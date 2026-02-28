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

#include "idlegpu_shim.h"

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

__attribute__((visibility("default")))
CUresult cuCtxSynchronize(void) {
    SHIM_UNIMPLEMENTED("cuCtxSynchronize");
}

__attribute__((visibility("default")))
CUresult cuCtxGetDevice(int *device) {
    (void)device;
    SHIM_UNIMPLEMENTED("cuCtxGetDevice");
}

__attribute__((visibility("default")))
CUresult cuCtxGetApiVersion(CUcontext ctx, unsigned int *version) {
    (void)ctx; (void)version;
    SHIM_UNIMPLEMENTED("cuCtxGetApiVersion");
}

__attribute__((visibility("default")))
CUresult cuCtxGetFlags(unsigned int *flags) {
    (void)flags;
    SHIM_UNIMPLEMENTED("cuCtxGetFlags");
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
    (void)pconfig;
    SHIM_UNIMPLEMENTED("cuCtxGetCacheConfig");
}

__attribute__((visibility("default")))
CUresult cuCtxSetCacheConfig(int config) {
    (void)config;
    SHIM_UNIMPLEMENTED("cuCtxSetCacheConfig");
}

__attribute__((visibility("default")))
CUresult cuCtxGetSharedMemConfig(int *pConfig) {
    (void)pConfig;
    SHIM_UNIMPLEMENTED("cuCtxGetSharedMemConfig");
}

__attribute__((visibility("default")))
CUresult cuCtxSetSharedMemConfig(int config) {
    (void)config;
    SHIM_UNIMPLEMENTED("cuCtxSetSharedMemConfig");
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

__attribute__((visibility("default")))
CUresult cuMemAllocPitch(CUdeviceptr *dptr, size_t *pPitch,
                         size_t WidthInBytes, size_t Height, unsigned int ElementSizeBytes) {
    (void)dptr; (void)pPitch; (void)WidthInBytes; (void)Height; (void)ElementSizeBytes;
    SHIM_UNIMPLEMENTED("cuMemAllocPitch");
}

__attribute__((visibility("default")))
CUresult cuMemAllocPitch_v2(CUdeviceptr *dptr, size_t *pPitch,
                             size_t WidthInBytes, size_t Height, unsigned int ElementSizeBytes) {
    (void)dptr; (void)pPitch; (void)WidthInBytes; (void)Height; (void)ElementSizeBytes;
    SHIM_UNIMPLEMENTED("cuMemAllocPitch_v2");
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

__attribute__((visibility("default")))
CUresult cuMemAllocManaged(CUdeviceptr *dptr, size_t bytesize, unsigned int flags) {
    (void)dptr; (void)bytesize; (void)flags;
    SHIM_UNIMPLEMENTED("cuMemAllocManaged");
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

__attribute__((visibility("default")))
CUresult cuMemcpyHtoDAsync(CUdeviceptr dstDevice, const void *srcHost,
                           size_t ByteCount, CUstream hStream) {
    (void)dstDevice; (void)srcHost; (void)ByteCount; (void)hStream;
    SHIM_UNIMPLEMENTED("cuMemcpyHtoDAsync");
}

__attribute__((visibility("default")))
CUresult cuMemcpyHtoDAsync_v2(CUdeviceptr dstDevice, const void *srcHost,
                               size_t ByteCount, CUstream hStream) {
    (void)dstDevice; (void)srcHost; (void)ByteCount; (void)hStream;
    SHIM_UNIMPLEMENTED("cuMemcpyHtoDAsync_v2");
}

__attribute__((visibility("default")))
CUresult cuMemcpyDtoHAsync(void *dstHost, CUdeviceptr srcDevice,
                           size_t ByteCount, CUstream hStream) {
    (void)dstHost; (void)srcDevice; (void)ByteCount; (void)hStream;
    SHIM_UNIMPLEMENTED("cuMemcpyDtoHAsync");
}

__attribute__((visibility("default")))
CUresult cuMemcpyDtoHAsync_v2(void *dstHost, CUdeviceptr srcDevice,
                               size_t ByteCount, CUstream hStream) {
    (void)dstHost; (void)srcDevice; (void)ByteCount; (void)hStream;
    SHIM_UNIMPLEMENTED("cuMemcpyDtoHAsync_v2");
}

__attribute__((visibility("default")))
CUresult cuMemcpyDtoDAsync(CUdeviceptr dstDevice, CUdeviceptr srcDevice,
                           size_t ByteCount, CUstream hStream) {
    (void)dstDevice; (void)srcDevice; (void)ByteCount; (void)hStream;
    SHIM_UNIMPLEMENTED("cuMemcpyDtoDAsync");
}

__attribute__((visibility("default")))
CUresult cuMemcpyDtoDAsync_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice,
                               size_t ByteCount, CUstream hStream) {
    (void)dstDevice; (void)srcDevice; (void)ByteCount; (void)hStream;
    SHIM_UNIMPLEMENTED("cuMemcpyDtoDAsync_v2");
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

__attribute__((visibility("default")))
CUresult cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc,
                         size_t N, CUstream hStream) {
    (void)dstDevice; (void)uc; (void)N; (void)hStream;
    SHIM_UNIMPLEMENTED("cuMemsetD8Async");
}

__attribute__((visibility("default")))
CUresult cuMemsetD16Async(CUdeviceptr dstDevice, unsigned short us,
                          size_t N, CUstream hStream) {
    (void)dstDevice; (void)us; (void)N; (void)hStream;
    SHIM_UNIMPLEMENTED("cuMemsetD16Async");
}

__attribute__((visibility("default")))
CUresult cuMemsetD32Async(CUdeviceptr dstDevice, unsigned int ui,
                          size_t N, CUstream hStream) {
    (void)dstDevice; (void)ui; (void)N; (void)hStream;
    SHIM_UNIMPLEMENTED("cuMemsetD32Async");
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

__attribute__((visibility("default")))
CUresult cuFuncGetAttribute(int *pi, int attrib, CUfunction hfunc) {
    (void)pi; (void)attrib; (void)hfunc;
    SHIM_UNIMPLEMENTED("cuFuncGetAttribute");
}

__attribute__((visibility("default")))
CUresult cuFuncSetAttribute(CUfunction hfunc, int attrib, int value) {
    (void)hfunc; (void)attrib; (void)value;
    SHIM_UNIMPLEMENTED("cuFuncSetAttribute");
}

__attribute__((visibility("default")))
CUresult cuFuncSetCacheConfig(CUfunction hfunc, int config) {
    (void)hfunc; (void)config;
    SHIM_UNIMPLEMENTED("cuFuncSetCacheConfig");
}

__attribute__((visibility("default")))
CUresult cuFuncSetSharedMemConfig(CUfunction hfunc, int config) {
    (void)hfunc; (void)config;
    SHIM_UNIMPLEMENTED("cuFuncSetSharedMemConfig");
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

__attribute__((visibility("default")))
CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, CUfunction func,
                                                      int blockSize, size_t dynamicSMemSize) {
    (void)numBlocks; (void)func; (void)blockSize; (void)dynamicSMemSize;
    SHIM_UNIMPLEMENTED("cuOccupancyMaxActiveBlocksPerMultiprocessor");
}

__attribute__((visibility("default")))
CUresult cuOccupancyMaxPotentialBlockSize(int *minGridSize, int *blockSize,
                                          CUfunction func, void *blockSizeToDynamicSMemSize,
                                          size_t dynamicSMemSize, int blockSizeLimit) {
    (void)minGridSize; (void)blockSize; (void)func;
    (void)blockSizeToDynamicSMemSize; (void)dynamicSMemSize; (void)blockSizeLimit;
    SHIM_UNIMPLEMENTED("cuOccupancyMaxPotentialBlockSize");
}

// ── Streams (non-Tier-1) ──────────────────────────────────────────────────────

__attribute__((visibility("default")))
CUresult cuStreamAddCallback(CUstream hStream, void *callback,
                             void *userData, unsigned int flags) {
    (void)hStream; (void)callback; (void)userData; (void)flags;
    SHIM_UNIMPLEMENTED("cuStreamAddCallback");
}

__attribute__((visibility("default")))
CUresult cuStreamAttachMemAsync(CUstream hStream, CUdeviceptr dptr,
                                size_t length, unsigned int flags) {
    (void)hStream; (void)dptr; (void)length; (void)flags;
    SHIM_UNIMPLEMENTED("cuStreamAttachMemAsync");
}

__attribute__((visibility("default")))
CUresult cuStreamQuery(CUstream hStream) {
    (void)hStream;
    SHIM_UNIMPLEMENTED("cuStreamQuery");
}

__attribute__((visibility("default")))
CUresult cuStreamGetCtx(CUstream hStream, CUcontext *pctx) {
    (void)hStream; (void)pctx;
    SHIM_UNIMPLEMENTED("cuStreamGetCtx");
}

__attribute__((visibility("default")))
CUresult cuStreamGetFlags(CUstream hStream, unsigned int *flags) {
    (void)hStream; (void)flags;
    SHIM_UNIMPLEMENTED("cuStreamGetFlags");
}

__attribute__((visibility("default")))
CUresult cuStreamGetPriority(CUstream hStream, int *priority) {
    (void)hStream; (void)priority;
    SHIM_UNIMPLEMENTED("cuStreamGetPriority");
}

__attribute__((visibility("default")))
CUresult cuStreamCreateWithPriority(CUstream *phStream, unsigned int flags, int priority) {
    (void)phStream; (void)flags; (void)priority;
    SHIM_UNIMPLEMENTED("cuStreamCreateWithPriority");
}

// ── Events (non-Tier-1) ───────────────────────────────────────────────────────

__attribute__((visibility("default")))
CUresult cuEventQuery(CUevent hEvent) {
    (void)hEvent;
    SHIM_UNIMPLEMENTED("cuEventQuery");
}

__attribute__((visibility("default")))
CUresult cuEventElapsedTime(float *pMilliseconds, CUevent hStart, CUevent hEnd) {
    (void)pMilliseconds; (void)hStart; (void)hEnd;
    SHIM_UNIMPLEMENTED("cuEventElapsedTime");
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
