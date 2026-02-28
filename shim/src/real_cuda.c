// Copyright (C) 2026 AniiGIT
// SPDX-License-Identifier: AGPL-3.0-or-later

/*
 * real_cuda.c - Real CUDA function pointer table and dlopen/dlsym intercept.
 *
 * Provides two features:
 *
 *   1. RealCuda g_real: typed function pointers loaded from the real libcuda.so.1
 *      via RTLD_NEXT.  Used as local-GPU fallback when the IPC connection to the
 *      sidecar is not available.
 *
 *   2. dlopen / dlsym / dlclose overrides: intercept applications that discover
 *      CUDA functions at runtime via dlopen("libcuda.so.1") + dlsym().  We return
 *      FAKE_CUDA_HANDLE from dlopen, then route dlsym lookups on that handle to
 *      our own shim functions — exactly as if the app had linked against libcuda.so.1.
 *
 * Bootstrap note:
 *   real_cuda_init() is called from shim_init() before ipc_connect().  It receives
 *   a "bootstrap_dlsym" pointer obtained via dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5")
 *   so that we can load our own real-function pointers without hitting our dlsym
 *   override (which would recurse back into itself).
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <string.h>

#include "idlegpu_shim.h"

// Sentinel handle returned from our dlopen override for libcuda.so* requests.
// Spells "IGPU" in ASCII; unlikely to alias any real mmap address.
#define FAKE_CUDA_HANDLE ((void *)(uintptr_t)0x49475055UL)

// ── Real CUDA function pointer table ─────────────────────────────────────────

RealCuda g_real = { 0 };

// ── Real dl* function pointers (bootstrapped in real_cuda_init) ───────────────

static void *(*s_real_dlopen)(const char *, int)       = NULL;
static void *(*s_real_dlsym)(void *, const char *)     = NULL;
static int   (*s_real_dlclose)(void *)                 = NULL;

// ── Shim symbol dispatch table (for dlsym interception) ──────────────────────
//
// Maps CUDA function name → our shim's implementation.  All symbols listed here
// are defined in cuda_init.c, cuda_mem.c, cuda_module.c, and cuda_stream.c with
// __attribute__((visibility("default"))).

// Forward declarations so the table can reference all exported shim functions
// without depending on include order.
CUresult cuInit(unsigned int);
CUresult cuDeviceGet(CUdevice *, int);
CUresult cuDeviceGetCount(int *);
CUresult cuDeviceGetName(char *, int, CUdevice);
CUresult cuDeviceGetAttribute(int *, CUdevice_attribute, CUdevice);
CUresult cuDeviceTotalMem(size_t *, CUdevice);
CUresult cuDeviceTotalMem_v2(size_t *, CUdevice);
CUresult cuCtxCreate(CUcontext *, unsigned int, CUdevice);
CUresult cuCtxCreate_v2(CUcontext *, unsigned int, CUdevice);
CUresult cuCtxDestroy(CUcontext);
CUresult cuCtxDestroy_v2(CUcontext);
CUresult cuCtxSetCurrent(CUcontext);
CUresult cuCtxGetCurrent(CUcontext *);
CUresult cuMemAlloc(CUdeviceptr *, size_t);
CUresult cuMemAlloc_v2(CUdeviceptr *, size_t);
CUresult cuMemFree(CUdeviceptr);
CUresult cuMemFree_v2(CUdeviceptr);
CUresult cuMemcpyHtoD(CUdeviceptr, const void *, size_t);
CUresult cuMemcpyHtoD_v2(CUdeviceptr, const void *, size_t);
CUresult cuMemcpyDtoH(void *, CUdeviceptr, size_t);
CUresult cuMemcpyDtoH_v2(void *, CUdeviceptr, size_t);
CUresult cuMemcpyDtoD(CUdeviceptr, CUdeviceptr, size_t);
CUresult cuMemcpyDtoD_v2(CUdeviceptr, CUdeviceptr, size_t);
CUresult cuMemsetD8(CUdeviceptr, unsigned char, size_t);
CUresult cuMemsetD8_v2(CUdeviceptr, unsigned char, size_t);
CUresult cuMemsetD16(CUdeviceptr, unsigned short, size_t);
CUresult cuMemsetD16_v2(CUdeviceptr, unsigned short, size_t);
CUresult cuMemsetD32(CUdeviceptr, unsigned int, size_t);
CUresult cuMemsetD32_v2(CUdeviceptr, unsigned int, size_t);
CUresult cuMemGetInfo(size_t *, size_t *);
CUresult cuMemGetInfo_v2(size_t *, size_t *);
CUresult cuModuleLoad(CUmodule *, const char *);
CUresult cuModuleLoadData(CUmodule *, const void *);
CUresult cuModuleGetFunction(CUfunction *, CUmodule, const char *);
CUresult cuLaunchKernel(CUfunction,
                        unsigned int, unsigned int, unsigned int,
                        unsigned int, unsigned int, unsigned int,
                        unsigned int, CUstream, void **, void **);
CUresult cuModuleUnload(CUmodule);
CUresult cuStreamCreate(CUstream *, unsigned int);
CUresult cuStreamDestroy(CUstream);
CUresult cuStreamDestroy_v2(CUstream);
CUresult cuStreamSynchronize(CUstream);
CUresult cuStreamWaitEvent(CUstream, CUevent, unsigned int);
CUresult cuEventCreate(CUevent *, unsigned int);
CUresult cuEventDestroy(CUevent);
CUresult cuEventDestroy_v2(CUevent);
CUresult cuEventRecord(CUevent, CUstream);
CUresult cuEventSynchronize(CUevent);
// Error reporting / runtime internals (stubs.c)
CUresult cuGetErrorName(CUresult, const char **);
CUresult cuGetErrorString(CUresult, const char **);
CUresult cuGetExportTable(const void **, const CUuuid *);
CUresult cuDriverGetVersion(int *);
// Extended device / context — Phase 2E promotions (stubs.c)
CUresult cuDeviceComputeCapability(int *, int *, CUdevice);
CUresult cuDeviceGetUuid(CUuuid *, CUdevice);
CUresult cuDeviceGetLuid(char *, unsigned int *, CUdevice);
CUresult cuCtxPushCurrent(CUcontext);
CUresult cuCtxPushCurrent_v2(CUcontext);
CUresult cuCtxPopCurrent(CUcontext *);
CUresult cuCtxPopCurrent_v2(CUcontext *);
CUresult cuCtxSetLimit(CUlimit, size_t);
CUresult cuCtxGetLimit(size_t *, CUlimit);
CUresult cuMemAllocPitch(CUdeviceptr *, size_t *, size_t, size_t, unsigned int);
CUresult cuMemAllocPitch_v2(CUdeviceptr *, size_t *, size_t, size_t, unsigned int);
CUresult cuMemAllocManaged(CUdeviceptr *, size_t, unsigned int);

typedef struct { const char *name; void *fn; } ShimSym;

static const ShimSym s_shim_syms[] = {
    { "cuInit",                   (void *)cuInit },
    { "cuDeviceGet",              (void *)cuDeviceGet },
    { "cuDeviceGetCount",         (void *)cuDeviceGetCount },
    { "cuDeviceGetName",          (void *)cuDeviceGetName },
    { "cuDeviceGetAttribute",     (void *)cuDeviceGetAttribute },
    { "cuDeviceTotalMem",         (void *)cuDeviceTotalMem },
    { "cuDeviceTotalMem_v2",      (void *)cuDeviceTotalMem_v2 },
    { "cuCtxCreate",              (void *)cuCtxCreate },
    { "cuCtxCreate_v2",           (void *)cuCtxCreate_v2 },
    { "cuCtxDestroy",             (void *)cuCtxDestroy },
    { "cuCtxDestroy_v2",          (void *)cuCtxDestroy_v2 },
    { "cuCtxSetCurrent",          (void *)cuCtxSetCurrent },
    { "cuCtxGetCurrent",          (void *)cuCtxGetCurrent },
    { "cuMemAlloc",               (void *)cuMemAlloc },
    { "cuMemAlloc_v2",            (void *)cuMemAlloc_v2 },
    { "cuMemFree",                (void *)cuMemFree },
    { "cuMemFree_v2",             (void *)cuMemFree_v2 },
    { "cuMemcpyHtoD",             (void *)cuMemcpyHtoD },
    { "cuMemcpyHtoD_v2",          (void *)cuMemcpyHtoD_v2 },
    { "cuMemcpyDtoH",             (void *)cuMemcpyDtoH },
    { "cuMemcpyDtoH_v2",          (void *)cuMemcpyDtoH_v2 },
    { "cuMemcpyDtoD",             (void *)cuMemcpyDtoD },
    { "cuMemcpyDtoD_v2",          (void *)cuMemcpyDtoD_v2 },
    { "cuMemsetD8",               (void *)cuMemsetD8 },
    { "cuMemsetD8_v2",            (void *)cuMemsetD8_v2 },
    { "cuMemsetD16",              (void *)cuMemsetD16 },
    { "cuMemsetD16_v2",           (void *)cuMemsetD16_v2 },
    { "cuMemsetD32",              (void *)cuMemsetD32 },
    { "cuMemsetD32_v2",           (void *)cuMemsetD32_v2 },
    { "cuMemGetInfo",             (void *)cuMemGetInfo },
    { "cuMemGetInfo_v2",          (void *)cuMemGetInfo_v2 },
    { "cuModuleLoad",             (void *)cuModuleLoad },
    { "cuModuleLoadData",         (void *)cuModuleLoadData },
    { "cuModuleGetFunction",      (void *)cuModuleGetFunction },
    { "cuLaunchKernel",           (void *)cuLaunchKernel },
    { "cuModuleUnload",           (void *)cuModuleUnload },
    { "cuStreamCreate",           (void *)cuStreamCreate },
    { "cuStreamDestroy",          (void *)cuStreamDestroy },
    { "cuStreamDestroy_v2",       (void *)cuStreamDestroy_v2 },
    { "cuStreamSynchronize",      (void *)cuStreamSynchronize },
    { "cuStreamWaitEvent",        (void *)cuStreamWaitEvent },
    { "cuEventCreate",            (void *)cuEventCreate },
    { "cuEventDestroy",           (void *)cuEventDestroy },
    { "cuEventDestroy_v2",        (void *)cuEventDestroy_v2 },
    { "cuEventRecord",            (void *)cuEventRecord },
    { "cuEventSynchronize",       (void *)cuEventSynchronize },
    // Error reporting / runtime internals
    { "cuGetErrorName",               (void *)cuGetErrorName },
    { "cuGetErrorString",             (void *)cuGetErrorString },
    { "cuGetExportTable",             (void *)cuGetExportTable },
    { "cuDriverGetVersion",           (void *)cuDriverGetVersion },
    // Extended device / context — Phase 2E
    { "cuDeviceComputeCapability",    (void *)cuDeviceComputeCapability },
    { "cuDeviceGetUuid",              (void *)cuDeviceGetUuid },
    { "cuDeviceGetLuid",              (void *)cuDeviceGetLuid },
    { "cuCtxPushCurrent",             (void *)cuCtxPushCurrent },
    { "cuCtxPushCurrent_v2",          (void *)cuCtxPushCurrent_v2 },
    { "cuCtxPopCurrent",              (void *)cuCtxPopCurrent },
    { "cuCtxPopCurrent_v2",           (void *)cuCtxPopCurrent_v2 },
    { "cuCtxSetLimit",                (void *)cuCtxSetLimit },
    { "cuCtxGetLimit",                (void *)cuCtxGetLimit },
    // Extended memory — pitched 2D and managed allocations
    { "cuMemAllocPitch",              (void *)cuMemAllocPitch },
    { "cuMemAllocPitch_v2",           (void *)cuMemAllocPitch_v2 },
    { "cuMemAllocManaged",            (void *)cuMemAllocManaged },
    { NULL, NULL },
};

// ── real_cuda_init ─────────────────────────────────────────────────────────────

void real_cuda_init(void *(*bootstrap_dlsym)(void *, const char *)) {
    if (bootstrap_dlsym == NULL) {
        SHIM_WARN("real_cuda_init: no bootstrap dlsym; real-CUDA fallback disabled");
        return;
    }

    // Save the bootstrap as our real dlsym pointer.
    s_real_dlsym = bootstrap_dlsym;

    // Load real dlopen and dlclose via RTLD_NEXT, skipping past our overrides.
    // __typeof__ is a GCC built-in extension valid in all -std=cN modes.
    s_real_dlopen  = (__typeof__(s_real_dlopen)) bootstrap_dlsym(RTLD_NEXT, "dlopen");
    s_real_dlclose = (__typeof__(s_real_dlclose))bootstrap_dlsym(RTLD_NEXT, "dlclose");

    // Load all Tier 1 real CUDA function pointers from the real libcuda.so.1.
    // RTLD_NEXT starts the search after libidlegpu-cuda.so in the link chain,
    // so it finds the real driver if it is already loaded.
    // Pointers remain NULL if libcuda.so.1 is not present at construction time.
#define LOAD(fn) g_real.fn = (__typeof__(g_real.fn))bootstrap_dlsym(RTLD_NEXT, #fn)
    LOAD(cuInit);
    LOAD(cuDeviceGet);
    LOAD(cuDeviceGetCount);
    LOAD(cuDeviceGetName);
    LOAD(cuDeviceGetAttribute);
    LOAD(cuDeviceTotalMem);
    LOAD(cuCtxCreate);
    LOAD(cuCtxDestroy);
    LOAD(cuCtxSetCurrent);
    LOAD(cuCtxGetCurrent);
    LOAD(cuMemAlloc);
    LOAD(cuMemFree);
    LOAD(cuMemcpyHtoD);
    LOAD(cuMemcpyDtoH);
    LOAD(cuMemcpyDtoD);
    LOAD(cuMemsetD8);
    LOAD(cuMemsetD16);
    LOAD(cuMemsetD32);
    LOAD(cuMemGetInfo);
    LOAD(cuModuleLoad);
    LOAD(cuModuleLoadData);
    LOAD(cuModuleGetFunction);
    LOAD(cuLaunchKernel);
    LOAD(cuModuleUnload);
    LOAD(cuStreamCreate);
    LOAD(cuStreamDestroy);
    LOAD(cuStreamSynchronize);
    LOAD(cuStreamWaitEvent);
    LOAD(cuEventCreate);
    LOAD(cuEventDestroy);
    LOAD(cuEventRecord);
    LOAD(cuEventSynchronize);
    // Runtime internals — used as local-driver fallbacks.
    LOAD(cuGetExportTable);
    LOAD(cuDriverGetVersion);
    // Extended device / context — Phase 2E promotions.
    // cuDeviceGetLuid may be NULL on Linux (Windows-specific DXGI function).
    LOAD(cuDeviceComputeCapability);
    LOAD(cuDeviceGetUuid);
    LOAD(cuDeviceGetLuid);
    LOAD(cuCtxPushCurrent_v2);
    LOAD(cuCtxPopCurrent_v2);
    LOAD(cuCtxSetLimit);
    LOAD(cuCtxGetLimit);
    LOAD(cuMemAllocPitch_v2);
    LOAD(cuMemAllocManaged);
#undef LOAD

    // ── Explicit dlopen fallback ───────────────────────────────────────────────
    //
    // RTLD_NEXT only searches libraries that were already in the dynamic linker's
    // link chain when our constructor ran.  In NVIDIA Docker containers the driver
    // stub (libcuda.so.1) is injected via LD_LIBRARY_PATH by the NVIDIA container
    // runtime — but it may not have been dlopen'd yet when LD_PRELOAD fires.
    // If RTLD_NEXT found nothing, fall back to an explicit dlopen so that the
    // local-GPU fallback and g_local_device_count probe work correctly.
    if (g_real.cuInit == NULL && s_real_dlopen != NULL && s_real_dlsym != NULL) {
        // Prefer RTLD_NOLOAD: reuse an existing mapping without adding a new one.
        void *cuda_h = s_real_dlopen("libcuda.so.1", RTLD_LAZY | RTLD_NOLOAD);
        if (cuda_h == NULL) {
            // Not yet loaded — load it from LD_LIBRARY_PATH.
            cuda_h = s_real_dlopen("libcuda.so.1", RTLD_LAZY);
        }
        if (cuda_h != NULL) {
            SHIM_DEBUG("real_cuda_init: RTLD_NEXT missed libcuda.so.1; "
                       "reloading via explicit dlopen");
            // Fill in only the pointers that are still NULL (RTLD_NEXT may have
            // found a subset on some driver configurations).
#define RELOAD(fn) \
            if (g_real.fn == NULL) \
                g_real.fn = (__typeof__(g_real.fn))s_real_dlsym(cuda_h, #fn)
            RELOAD(cuInit);
            RELOAD(cuDeviceGet);
            RELOAD(cuDeviceGetCount);
            RELOAD(cuDeviceGetName);
            RELOAD(cuDeviceGetAttribute);
            RELOAD(cuDeviceTotalMem);
            RELOAD(cuCtxCreate);
            RELOAD(cuCtxDestroy);
            RELOAD(cuCtxSetCurrent);
            RELOAD(cuCtxGetCurrent);
            RELOAD(cuMemAlloc);
            RELOAD(cuMemFree);
            RELOAD(cuMemcpyHtoD);
            RELOAD(cuMemcpyDtoH);
            RELOAD(cuMemcpyDtoD);
            RELOAD(cuMemsetD8);
            RELOAD(cuMemsetD16);
            RELOAD(cuMemsetD32);
            RELOAD(cuMemGetInfo);
            RELOAD(cuModuleLoad);
            RELOAD(cuModuleLoadData);
            RELOAD(cuModuleGetFunction);
            RELOAD(cuLaunchKernel);
            RELOAD(cuModuleUnload);
            RELOAD(cuStreamCreate);
            RELOAD(cuStreamDestroy);
            RELOAD(cuStreamSynchronize);
            RELOAD(cuStreamWaitEvent);
            RELOAD(cuEventCreate);
            RELOAD(cuEventDestroy);
            RELOAD(cuEventRecord);
            RELOAD(cuEventSynchronize);
            RELOAD(cuGetExportTable);
            RELOAD(cuDriverGetVersion);
            RELOAD(cuDeviceComputeCapability);
            RELOAD(cuDeviceGetUuid);
            RELOAD(cuDeviceGetLuid);
            RELOAD(cuCtxPushCurrent_v2);
            RELOAD(cuCtxPopCurrent_v2);
            RELOAD(cuCtxSetLimit);
            RELOAD(cuCtxGetLimit);
            RELOAD(cuMemAllocPitch_v2);
            RELOAD(cuMemAllocManaged);
#undef RELOAD
        } else {
            SHIM_DEBUG("real_cuda_init: libcuda.so.1 not found via explicit dlopen; "
                       "no local GPU fallback");
        }
    }

    SHIM_DEBUG("real_cuda_init: real CUDA fallback %s",
               g_real.cuInit != NULL ? "ready" : "unavailable (no libcuda.so.1)");
}

// ── dlopen override ────────────────────────────────────────────────────────────

__attribute__((visibility("default")))
void *dlopen(const char *filename, int flags) {
    // Intercept libcuda.so* — return our sentinel so any dlsym() on this handle
    // is routed to the shim functions rather than the real driver.
    if (filename != NULL && strncmp(filename, "libcuda.so", 10) == 0) {
        SHIM_DEBUG("dlopen: intercepted %s → FAKE_CUDA_HANDLE", filename);
        return FAKE_CUDA_HANDLE;
    }

    if (s_real_dlopen == NULL) {
        SHIM_WARN("dlopen: real dlopen not available, returning NULL for %s",
                  filename ? filename : "(null)");
        return NULL;
    }
    return s_real_dlopen(filename, flags);
}

// ── dlsym override ─────────────────────────────────────────────────────────────

__attribute__((visibility("default")))
void *dlsym(void *handle, const char *symbol) {
    if (handle == FAKE_CUDA_HANDLE && symbol != NULL) {
        // Look up in our dispatch table.
        for (const ShimSym *s = s_shim_syms; s->name != NULL; ++s) {
            if (strcmp(s->name, symbol) == 0) {
                SHIM_DEBUG("dlsym(FAKE_CUDA_HANDLE, %s) → shim", symbol);
                return s->fn;
            }
        }
        // Not in Tier 1 — return NULL so the caller can detect absence gracefully.
        SHIM_WARN("dlsym(FAKE_CUDA_HANDLE, %s): not in Tier 1 shim; returning NULL",
                  symbol);
        return NULL;
    }

    if (s_real_dlsym == NULL) {
        return NULL;
    }
    return s_real_dlsym(handle, symbol);
}

// ── dlclose override ───────────────────────────────────────────────────────────

__attribute__((visibility("default")))
int dlclose(void *handle) {
    if (handle == FAKE_CUDA_HANDLE) {
        return 0;  // No-op — no real reference held for this sentinel.
    }

    if (s_real_dlclose == NULL) {
        return 0;
    }
    return s_real_dlclose(handle);
}
