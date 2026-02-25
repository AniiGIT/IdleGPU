// Copyright (C) 2026 AniiGIT
// SPDX-License-Identifier: AGPL-3.0-or-later

/*
 * cuda_mem.c - Tier 1: Memory management.
 *
 * Intercepted functions (9):
 *   cuMemAlloc, cuMemFree,
 *   cuMemcpyHtoD, cuMemcpyDtoH, cuMemcpyDtoD,
 *   cuMemsetD8, cuMemsetD16, cuMemsetD32,
 *   cuMemGetInfo
 *
 * Host-to-device transfers (HtoD): the source bytes are appended to the
 * fixed-size request struct and sent in one ipc_call().  The payload can
 * be up to IPC_MAX_PAYLOAD bytes.
 *
 * Device-to-host transfers (DtoH): the fixed request is sent; the response
 * payload contains the host bytes, which are copied into the caller's buffer.
 *
 * Device-to-device transfers (DtoD): only handles are sent; no byte copying
 * on the shim side.
 *
 * All _v2 aliases are defined for applications linked against versioned
 * CUDA driver symbols.
 */

#include <stdlib.h>
#include <string.h>

#include "idlegpu_shim.h"

#define HANDLE_TO_PTR(h)  ((void *)(uintptr_t)(h))
#define PTR_TO_HANDLE(p)  ((uint64_t)(uintptr_t)(p))

// ── cuMemAlloc ────────────────────────────────────────────────────────────────

__attribute__((visibility("default")))
CUresult cuMemAlloc(CUdeviceptr *dptr, size_t bytesize) {
    SHIM_REQUIRE_IPC(g_real.cuMemAlloc, dptr, bytesize);

    Req_cuMemAlloc  req  = { .bytesize = (uint64_t)bytesize };
    Resp_cuMemAlloc resp = { 0 };

    CUresult r = ipc_call(FN_cuMemAlloc, &req, sizeof(req),
                          &resp, sizeof(resp), NULL);
    if (r == CUDA_SUCCESS && dptr != NULL) {
        *dptr = (CUdeviceptr)resp.dptr;
    }
    return r;
}

__attribute__((visibility("default")))
CUresult cuMemAlloc_v2(CUdeviceptr *dptr, size_t bytesize) {
    return cuMemAlloc(dptr, bytesize);
}

// ── cuMemFree ─────────────────────────────────────────────────────────────────

__attribute__((visibility("default")))
CUresult cuMemFree(CUdeviceptr dptr) {
    SHIM_REQUIRE_IPC(g_real.cuMemFree, dptr);

    Req_cuMemFree req = { .dptr = (uint64_t)dptr };
    return ipc_call(FN_cuMemFree, &req, sizeof(req), NULL, 0, NULL);
}

__attribute__((visibility("default")))
CUresult cuMemFree_v2(CUdeviceptr dptr) {
    return cuMemFree(dptr);
}

// ── cuMemcpyHtoD ─────────────────────────────────────────────────────────────

__attribute__((visibility("default")))
CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount) {
    SHIM_REQUIRE_IPC(g_real.cuMemcpyHtoD, dstDevice, srcHost, ByteCount);

    if (ByteCount > IPC_MAX_PAYLOAD) {
        SHIM_WARN("cuMemcpyHtoD: ByteCount %zu exceeds IPC_MAX_PAYLOAD; failing",
                  ByteCount);
        return CUDA_ERROR_INVALID_VALUE;
    }

    // Build a single contiguous buffer: [Req_cuMemcpyHtoD][host bytes]
    uint32_t payload_len = (uint32_t)(sizeof(Req_cuMemcpyHtoD) + ByteCount);
    uint8_t *payload = malloc(payload_len);
    if (payload == NULL) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }

    Req_cuMemcpyHtoD req = {
        .dst        = (uint64_t)dstDevice,
        .byte_count = (uint64_t)ByteCount,
    };
    memcpy(payload, &req, sizeof(req));
    if (ByteCount > 0 && srcHost != NULL) {
        memcpy(payload + sizeof(req), srcHost, ByteCount);
    }

    CUresult r = ipc_call(FN_cuMemcpyHtoD, payload, payload_len, NULL, 0, NULL);
    free(payload);
    return r;
}

__attribute__((visibility("default")))
CUresult cuMemcpyHtoD_v2(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount) {
    return cuMemcpyHtoD(dstDevice, srcHost, ByteCount);
}

// ── cuMemcpyDtoH ─────────────────────────────────────────────────────────────

__attribute__((visibility("default")))
CUresult cuMemcpyDtoH(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount) {
    SHIM_REQUIRE_IPC(g_real.cuMemcpyDtoH, dstHost, srcDevice, ByteCount);

    if (ByteCount > IPC_MAX_PAYLOAD) {
        SHIM_WARN("cuMemcpyDtoH: ByteCount %zu exceeds IPC_MAX_PAYLOAD; failing",
                  ByteCount);
        return CUDA_ERROR_INVALID_VALUE;
    }

    Req_cuMemcpyDtoH req = {
        .src        = (uint64_t)srcDevice,
        .byte_count = (uint64_t)ByteCount,
    };

    // Response payload: ByteCount bytes of host data.
    CUresult r = ipc_call(FN_cuMemcpyDtoH, &req, sizeof(req),
                          dstHost, (uint32_t)ByteCount, NULL);
    return r;
}

__attribute__((visibility("default")))
CUresult cuMemcpyDtoH_v2(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount) {
    return cuMemcpyDtoH(dstHost, srcDevice, ByteCount);
}

// ── cuMemcpyDtoD ─────────────────────────────────────────────────────────────

__attribute__((visibility("default")))
CUresult cuMemcpyDtoD(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount) {
    SHIM_REQUIRE_IPC(g_real.cuMemcpyDtoD, dstDevice, srcDevice, ByteCount);

    Req_cuMemcpyDtoD req = {
        .dst        = (uint64_t)dstDevice,
        .src        = (uint64_t)srcDevice,
        .byte_count = (uint64_t)ByteCount,
    };
    return ipc_call(FN_cuMemcpyDtoD, &req, sizeof(req), NULL, 0, NULL);
}

__attribute__((visibility("default")))
CUresult cuMemcpyDtoD_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount) {
    return cuMemcpyDtoD(dstDevice, srcDevice, ByteCount);
}

// ── cuMemsetD8 ────────────────────────────────────────────────────────────────

__attribute__((visibility("default")))
CUresult cuMemsetD8(CUdeviceptr dstDevice, unsigned char uc, size_t N) {
    SHIM_REQUIRE_IPC(g_real.cuMemsetD8, dstDevice, uc, N);

    Req_cuMemsetD8 req = {
        .dst   = (uint64_t)dstDevice,
        .value = uc,
        .count = (uint64_t)N,
    };
    return ipc_call(FN_cuMemsetD8, &req, sizeof(req), NULL, 0, NULL);
}

__attribute__((visibility("default")))
CUresult cuMemsetD8_v2(CUdeviceptr dstDevice, unsigned char uc, size_t N) {
    return cuMemsetD8(dstDevice, uc, N);
}

// ── cuMemsetD16 ───────────────────────────────────────────────────────────────

__attribute__((visibility("default")))
CUresult cuMemsetD16(CUdeviceptr dstDevice, unsigned short us, size_t N) {
    SHIM_REQUIRE_IPC(g_real.cuMemsetD16, dstDevice, us, N);

    Req_cuMemsetD16 req = {
        .dst   = (uint64_t)dstDevice,
        .value = us,
        .count = (uint64_t)N,
    };
    return ipc_call(FN_cuMemsetD16, &req, sizeof(req), NULL, 0, NULL);
}

__attribute__((visibility("default")))
CUresult cuMemsetD16_v2(CUdeviceptr dstDevice, unsigned short us, size_t N) {
    return cuMemsetD16(dstDevice, us, N);
}

// ── cuMemsetD32 ───────────────────────────────────────────────────────────────

__attribute__((visibility("default")))
CUresult cuMemsetD32(CUdeviceptr dstDevice, unsigned int ui, size_t N) {
    SHIM_REQUIRE_IPC(g_real.cuMemsetD32, dstDevice, ui, N);

    Req_cuMemsetD32 req = {
        .dst   = (uint64_t)dstDevice,
        .value = (uint32_t)ui,
        .count = (uint64_t)N,
    };
    return ipc_call(FN_cuMemsetD32, &req, sizeof(req), NULL, 0, NULL);
}

__attribute__((visibility("default")))
CUresult cuMemsetD32_v2(CUdeviceptr dstDevice, unsigned int ui, size_t N) {
    return cuMemsetD32(dstDevice, ui, N);
}

// ── cuMemGetInfo ──────────────────────────────────────────────────────────────

__attribute__((visibility("default")))
CUresult cuMemGetInfo(size_t *free, size_t *total) {
    SHIM_REQUIRE_IPC(g_real.cuMemGetInfo, free, total);

    Resp_cuMemGetInfo resp = { 0 };
    CUresult r = ipc_call(FN_cuMemGetInfo, NULL, 0,
                          &resp, sizeof(resp), NULL);
    if (r == CUDA_SUCCESS) {
        if (free  != NULL) *free  = (size_t)resp.free;
        if (total != NULL) *total = (size_t)resp.total;
    }
    return r;
}

__attribute__((visibility("default")))
CUresult cuMemGetInfo_v2(size_t *free, size_t *total) {
    return cuMemGetInfo(free, total);
}
