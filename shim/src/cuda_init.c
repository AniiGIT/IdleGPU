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
 * Multi-GPU device routing
 * ─────────────────────────
 * Virtual device index space exposed to the application:
 *
 *   [0 .. g_local_device_count - 1]  → real local CUDA driver
 *   [g_local_device_count .. N]      → remote agent GPU via IPC
 *
 * cuDeviceGetCount returns the sum of local + remote (0 or 1) counts.
 * Device-indexed functions (cuDeviceGet, cuDeviceGetName, etc.) inspect
 * the ordinal / CUdevice value and dispatch accordingly.
 *
 * For context lifecycle (cuCtxDestroy, cuCtxSetCurrent, cuCtxGetCurrent)
 * the device is unknown, so they use SHIM_REQUIRE_IPC: IPC if available,
 * local otherwise.  SHIM_FALLBACK_IF_DEAD provides a post-timeout fallback
 * for read-only operations where falling back to local is harmless.
 *
 * IPC timeout handling
 * ─────────────────────
 * If an IPC call times out, ipc_call() marks the connection dead
 * (g_ipc_connected = 0) and returns CUDA_ERROR_NOT_INITIALIZED.
 * Callers that can safely fall back to a local driver call use
 * SHIM_FALLBACK_IF_DEAD immediately after ipc_call().
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

// ── Routing helpers ───────────────────────────────────────────────────────────

// True if a CUdevice value represents a local (real driver) device.
// CUdevice is effectively an int ordinal in all NVIDIA CUDA implementations.
#define DEV_IS_LOCAL(dev)  ((int)(dev) < g_local_device_count)

// Adjusted ordinal to send to the agent: strips the local device offset.
#define DEV_REMOTE_ORD(dev) ((int)(dev) - g_local_device_count)

// ── cuInit ────────────────────────────────────────────────────────────────────

__attribute__((visibility("default")))
CUresult cuInit(unsigned int Flags) {
    if (!g_ipc_connected) {
        // Prefer local GPU; CUDA_ERROR_NO_DEVICE if no real driver available.
        if (g_real.cuInit != NULL) return g_real.cuInit(Flags);
        return CUDA_ERROR_NO_DEVICE;
    }

    Req_cuInit req = { .flags = (uint32_t)Flags };
    CUresult r = ipc_call(FN_cuInit, &req, sizeof(req), NULL, 0, NULL);
    // If IPC died mid-call (timeout), retry via local driver.
    SHIM_FALLBACK_IF_DEAD(g_real.cuInit, Flags);
    return r;
}

// ── cuDeviceGetCount ──────────────────────────────────────────────────────────
//
// Returns local_count + remote_count without an IPC round-trip.
// remote_count is 1 if IPC is live, 0 otherwise.

__attribute__((visibility("default")))
CUresult cuDeviceGetCount(int *count) {
    if (count != NULL) {
        *count = g_local_device_count + (g_ipc_connected ? 1 : 0);
    }
    return CUDA_SUCCESS;
}

// ── cuDeviceGet ───────────────────────────────────────────────────────────────

__attribute__((visibility("default")))
CUresult cuDeviceGet(CUdevice *device, int ordinal) {
    if (ordinal < g_local_device_count) {
        // Local device: delegate directly to the real driver.
        if (g_real.cuDeviceGet != NULL) return g_real.cuDeviceGet(device, ordinal);
        return CUDA_ERROR_NOT_INITIALIZED;
    }

    // Remote device.
    if (!g_ipc_connected) return CUDA_ERROR_INVALID_VALUE;

    Req_cuDeviceGet  req  = { .ordinal = ordinal - g_local_device_count };
    Resp_cuDeviceGet resp = { 0 };

    CUresult r = ipc_call(FN_cuDeviceGet, &req, sizeof(req),
                          &resp, sizeof(resp), NULL);
    if (r == CUDA_SUCCESS && device != NULL) {
        // Return the virtual ordinal (not the agent's internal device number)
        // so subsequent routing comparisons against g_local_device_count work.
        *device = (CUdevice)ordinal;
    }
    return r;
}

// ── cuDeviceGetName ───────────────────────────────────────────────────────────

__attribute__((visibility("default")))
CUresult cuDeviceGetName(char *name, int len, CUdevice dev) {
    if (name == NULL || len <= 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    if (DEV_IS_LOCAL(dev)) {
        if (g_real.cuDeviceGetName != NULL)
            return g_real.cuDeviceGetName(name, len, dev);
        return CUDA_ERROR_NOT_INITIALIZED;
    }

    if (!g_ipc_connected) return CUDA_ERROR_INVALID_VALUE;

    Req_cuDeviceGetName req = {
        .len    = (uint32_t)len,
        .device = (int32_t)DEV_REMOTE_ORD(dev),
    };

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
    if (DEV_IS_LOCAL(dev)) {
        if (g_real.cuDeviceGetAttribute != NULL)
            return g_real.cuDeviceGetAttribute(pi, attrib, dev);
        return CUDA_ERROR_NOT_INITIALIZED;
    }

    if (!g_ipc_connected) return CUDA_ERROR_INVALID_VALUE;

    Req_cuDeviceGetAttribute  req  = {
        .attrib = (int32_t)attrib,
        .device = (int32_t)DEV_REMOTE_ORD(dev),
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
    if (DEV_IS_LOCAL(dev)) {
        if (g_real.cuDeviceTotalMem != NULL)
            return g_real.cuDeviceTotalMem(bytes, dev);
        return CUDA_ERROR_NOT_INITIALIZED;
    }

    if (!g_ipc_connected) return CUDA_ERROR_INVALID_VALUE;

    Req_cuDeviceTotalMem  req  = { .device = (int32_t)DEV_REMOTE_ORD(dev) };
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
//
// Routes by device: local devices use the real driver, remote devices go via IPC.

__attribute__((visibility("default")))
CUresult cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev) {
    if (DEV_IS_LOCAL(dev)) {
        if (g_real.cuCtxCreate != NULL) return g_real.cuCtxCreate(pctx, flags, dev);
        return CUDA_ERROR_NOT_INITIALIZED;
    }

    if (!g_ipc_connected) return CUDA_ERROR_INVALID_VALUE;

    Req_cuCtxCreate  req  = {
        .flags  = (uint32_t)flags,
        .device = (int32_t)DEV_REMOTE_ORD(dev),
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
//
// No device argument — we don't know which GPU the context belongs to.
// Route via IPC if connected; fall back to local otherwise.  No post-timeout
// fallback: destroying a remote context handle on the local driver would crash.

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
    // Post-timeout fallback: reading the current context from the local driver
    // is safe (worst case returns NULL context).
    SHIM_FALLBACK_IF_DEAD(g_real.cuCtxGetCurrent, pctx);
    if (r == CUDA_SUCCESS && pctx != NULL) {
        *pctx = (CUcontext)HANDLE_TO_PTR(resp.ctx_handle);
    }
    return r;
}
