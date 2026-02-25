// Copyright (C) 2026 AniiGIT
// SPDX-License-Identifier: AGPL-3.0-or-later

/*
 * cuda_init.c - Tier 1: Initialisation and device management.
 *
 * Intercepted functions (10):
 *   cuInit, cuDeviceGet, cuDeviceGetCount, cuDeviceGetName,
 *   cuDeviceGetAttribute, cuDeviceTotalMem,
 *   cuCtxCreate, cuCtxDestroy, cuCtxSetCurrent, cuCtxGetCurrent
 *
 * Each function serialises its arguments into the matching Req_* struct
 * (defined in idlegpu_shim.h), calls ipc_call(), and deserialises the
 * Resp_* struct back into the caller's output pointers.
 *
 * _v2 versioned aliases are defined alongside each function because CUDA
 * applications link against the versioned symbols.
 *
 * Handle remapping note:
 *   CUcontext handles returned by the agent (uint64_t) are stored in the
 *   caller's CUcontext pointer as opaque 64-bit values.  The app treats
 *   them as opaque; the shim passes them back unchanged on subsequent calls.
 *   This works because the agent has the real context pointer.
 */

#include <string.h>

#include "idlegpu_shim.h"

// Helper: cast a uint64_t handle to an opaque pointer type.
// We use this so the compiler doesn't warn about pointer-integer casts.
#define HANDLE_TO_PTR(h)   ((void *)(uintptr_t)(h))
#define PTR_TO_HANDLE(p)   ((uint64_t)(uintptr_t)(p))

// ── cuInit ────────────────────────────────────────────────────────────────────

__attribute__((visibility("default")))
CUresult cuInit(unsigned int Flags) {
    if (!g_ipc_connected) {
        // Prefer local GPU; CUDA_ERROR_NO_DEVICE if no real driver available.
        if (g_real.cuInit != NULL) return g_real.cuInit(Flags);
        return CUDA_ERROR_NO_DEVICE;
    }

    Req_cuInit req = { .flags = (uint32_t)Flags };
    return ipc_call(FN_cuInit, &req, sizeof(req), NULL, 0, NULL);
}

// ── cuDeviceGet ───────────────────────────────────────────────────────────────

__attribute__((visibility("default")))
CUresult cuDeviceGet(CUdevice *device, int ordinal) {
    SHIM_REQUIRE_IPC(g_real.cuDeviceGet, device, ordinal);

    Req_cuDeviceGet  req  = { .ordinal = ordinal };
    Resp_cuDeviceGet resp = { 0 };

    CUresult r = ipc_call(FN_cuDeviceGet, &req, sizeof(req),
                          &resp, sizeof(resp), NULL);
    if (r == CUDA_SUCCESS && device != NULL) {
        *device = (CUdevice)resp.device;
    }
    return r;
}

// ── cuDeviceGetCount ──────────────────────────────────────────────────────────

__attribute__((visibility("default")))
CUresult cuDeviceGetCount(int *count) {
    SHIM_REQUIRE_IPC(g_real.cuDeviceGetCount, count);

    // Request payload: empty.
    int32_t resp_count = 0;
    CUresult r = ipc_call(FN_cuDeviceGetCount, NULL, 0,
                          &resp_count, sizeof(resp_count), NULL);
    if (r == CUDA_SUCCESS && count != NULL) {
        *count = (int)resp_count;
    }
    return r;
}

// ── cuDeviceGetName ───────────────────────────────────────────────────────────

__attribute__((visibility("default")))
CUresult cuDeviceGetName(char *name, int len, CUdevice dev) {
    SHIM_REQUIRE_IPC(g_real.cuDeviceGetName, name, len, dev);

    if (name == NULL || len <= 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    Req_cuDeviceGetName req = {
        .len    = (uint32_t)len,
        .device = (int32_t)dev,
    };

    // Response payload: up to `len` bytes of device name.
    CUresult r = ipc_call(FN_cuDeviceGetName, &req, sizeof(req),
                          name, (uint32_t)len, NULL);
    if (r == CUDA_SUCCESS) {
        // Ensure null termination in case the agent fills the buffer exactly.
        name[len - 1] = '\0';
    }
    return r;
}

// ── cuDeviceGetAttribute ──────────────────────────────────────────────────────

__attribute__((visibility("default")))
CUresult cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev) {
    SHIM_REQUIRE_IPC(g_real.cuDeviceGetAttribute, pi, attrib, dev);

    Req_cuDeviceGetAttribute  req  = {
        .attrib = (int32_t)attrib,
        .device = (int32_t)dev,
    };
    Resp_cuDeviceGetAttribute resp = { 0 };

    CUresult r = ipc_call(FN_cuDeviceGetAttribute, &req, sizeof(req),
                          &resp, sizeof(resp), NULL);
    if (r == CUDA_SUCCESS && pi != NULL) {
        *pi = (int)resp.value;
    }
    return r;
}

// ── cuDeviceTotalMem ──────────────────────────────────────────────────────────

__attribute__((visibility("default")))
CUresult cuDeviceTotalMem(size_t *bytes, CUdevice dev) {
    SHIM_REQUIRE_IPC(g_real.cuDeviceTotalMem, bytes, dev);

    Req_cuDeviceTotalMem  req  = { .device = (int32_t)dev };
    Resp_cuDeviceTotalMem resp = { 0 };

    CUresult r = ipc_call(FN_cuDeviceTotalMem, &req, sizeof(req),
                          &resp, sizeof(resp), NULL);
    if (r == CUDA_SUCCESS && bytes != NULL) {
        *bytes = (size_t)resp.bytes;
    }
    return r;
}

// _v2 alias (newer CUDA apps link cuDeviceTotalMem_v2).
__attribute__((visibility("default")))
CUresult cuDeviceTotalMem_v2(size_t *bytes, CUdevice dev) {
    return cuDeviceTotalMem(bytes, dev);
}

// ── cuCtxCreate ───────────────────────────────────────────────────────────────

__attribute__((visibility("default")))
CUresult cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev) {
    SHIM_REQUIRE_IPC(g_real.cuCtxCreate, pctx, flags, dev);

    Req_cuCtxCreate  req  = {
        .flags  = (uint32_t)flags,
        .device = (int32_t)dev,
    };
    Resp_cuCtxCreate resp = { 0 };

    CUresult r = ipc_call(FN_cuCtxCreate, &req, sizeof(req),
                          &resp, sizeof(resp), NULL);
    if (r == CUDA_SUCCESS && pctx != NULL) {
        *pctx = (CUcontext)HANDLE_TO_PTR(resp.ctx_handle);
    }
    return r;
}

// _v2 alias.
__attribute__((visibility("default")))
CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev) {
    return cuCtxCreate(pctx, flags, dev);
}

// ── cuCtxDestroy ──────────────────────────────────────────────────────────────

__attribute__((visibility("default")))
CUresult cuCtxDestroy(CUcontext ctx) {
    SHIM_REQUIRE_IPC(g_real.cuCtxDestroy, ctx);

    Req_cuCtxDestroy req = { .ctx_handle = PTR_TO_HANDLE(ctx) };
    return ipc_call(FN_cuCtxDestroy, &req, sizeof(req), NULL, 0, NULL);
}

__attribute__((visibility("default")))
CUresult cuCtxDestroy_v2(CUcontext ctx) {
    return cuCtxDestroy(ctx);
}

// ── cuCtxSetCurrent ───────────────────────────────────────────────────────────

__attribute__((visibility("default")))
CUresult cuCtxSetCurrent(CUcontext ctx) {
    SHIM_REQUIRE_IPC(g_real.cuCtxSetCurrent, ctx);

    Req_cuCtxSetCurrent req = { .ctx_handle = PTR_TO_HANDLE(ctx) };
    return ipc_call(FN_cuCtxSetCurrent, &req, sizeof(req), NULL, 0, NULL);
}

// ── cuCtxGetCurrent ───────────────────────────────────────────────────────────

__attribute__((visibility("default")))
CUresult cuCtxGetCurrent(CUcontext *pctx) {
    SHIM_REQUIRE_IPC(g_real.cuCtxGetCurrent, pctx);

    Resp_cuCtxGetCurrent resp = { 0 };
    CUresult r = ipc_call(FN_cuCtxGetCurrent, NULL, 0,
                          &resp, sizeof(resp), NULL);
    if (r == CUDA_SUCCESS && pctx != NULL) {
        *pctx = (CUcontext)HANDLE_TO_PTR(resp.ctx_handle);
    }
    return r;
}
