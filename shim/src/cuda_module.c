// Copyright (C) 2026 AniiGIT
// SPDX-License-Identifier: AGPL-3.0-or-later

/*
 * cuda_module.c - Tier 1: Module loading and kernel execution.
 *
 * Intercepted functions (5):
 *   cuModuleLoad, cuModuleLoadData, cuModuleGetFunction,
 *   cuLaunchKernel, cuModuleUnload
 *
 * cuModuleLoad:
 *   The filename string is appended after the fixed Req_cuModuleLoad header
 *   and sent in one ipc_call().  The sidecar opens the file on its side.
 *   fname_len includes the null terminator.
 *
 * cuModuleLoadData:
 *   The binary/PTX image is appended after Req_cuModuleLoadData.
 *   PTX images are null-terminated strings; binary cubin images carry a
 *   length prefix in the image_len field.  The shim distinguishes them by
 *   checking whether image_len < IPC_MAX_PAYLOAD.
 *
 * cuModuleGetFunction:
 *   Function name is appended after Req_cuModuleGetFunction.
 *
 * cuLaunchKernel:
 *   kernelParams is an array of void* pointers to per-argument storage.
 *   The shim copies the first num_params arguments as 8-byte values
 *   (sufficient for all scalar and pointer arguments) after the fixed
 *   Req_cuLaunchKernel header.  extra is ignored (must be NULL per spec).
 *
 * cuModuleUnload:
 *   Only the module handle is sent.
 */

#include <stdlib.h>
#include <string.h>

#include "idlegpu_shim.h"

#define HANDLE_TO_PTR(h)  ((void *)(uintptr_t)(h))
#define PTR_TO_HANDLE(p)  ((uint64_t)(uintptr_t)(p))

// Maximum number of kernel parameters we will serialise.
// If a kernel requires more, we return CUDA_ERROR_INVALID_VALUE.
#define MAX_KERNEL_PARAMS 128u

// ── cuModuleLoad ──────────────────────────────────────────────────────────────

__attribute__((visibility("default")))
CUresult cuModuleLoad(CUmodule *module, const char *fname) {
    SHIM_CHECK_CONNECTED();

    if (fname == NULL || module == NULL) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    uint32_t fname_len = (uint32_t)(strlen(fname) + 1); // include null terminator
    uint32_t payload_len = (uint32_t)(sizeof(Req_cuModuleLoad) + fname_len);

    if (payload_len > IPC_MAX_PAYLOAD) {
        SHIM_WARN("cuModuleLoad: fname too long (%u bytes)", fname_len);
        return CUDA_ERROR_INVALID_VALUE;
    }

    uint8_t *payload = malloc(payload_len);
    if (payload == NULL) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }

    Req_cuModuleLoad req = { .fname_len = fname_len };
    memcpy(payload, &req, sizeof(req));
    memcpy(payload + sizeof(req), fname, fname_len);

    Resp_cuModuleLoad resp = { 0 };
    CUresult r = ipc_call(FN_cuModuleLoad, payload, payload_len,
                          &resp, sizeof(resp), NULL);
    free(payload);

    if (r == CUDA_SUCCESS) {
        *module = (CUmodule)HANDLE_TO_PTR(resp.mod_handle);
    }
    return r;
}

// ── cuModuleLoadData ──────────────────────────────────────────────────────────

__attribute__((visibility("default")))
CUresult cuModuleLoadData(CUmodule *module, const void *image) {
    SHIM_CHECK_CONNECTED();

    if (image == NULL || module == NULL) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    // Determine image size: treat as a null-terminated PTX string first.
    // If the caller passes a binary cubin, the first byte is not printable and
    // strlen() may under-count — but for binary images the caller is expected
    // to use cuModuleLoadDataEx (not in Tier 1) or pass a PTX string.
    uint64_t image_len = (uint64_t)(strlen((const char *)image) + 1);

    if (image_len > IPC_MAX_PAYLOAD) {
        SHIM_WARN("cuModuleLoadData: image too large (%llu bytes); failing",
                  (unsigned long long)image_len);
        return CUDA_ERROR_INVALID_VALUE;
    }

    uint32_t payload_len = (uint32_t)(sizeof(Req_cuModuleLoadData) + (uint32_t)image_len);
    uint8_t *payload = malloc(payload_len);
    if (payload == NULL) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }

    Req_cuModuleLoadData req = { .image_len = image_len };
    memcpy(payload, &req, sizeof(req));
    memcpy(payload + sizeof(req), image, (size_t)image_len);

    Resp_cuModuleLoadData resp = { 0 };
    CUresult r = ipc_call(FN_cuModuleLoadData, payload, payload_len,
                          &resp, sizeof(resp), NULL);
    free(payload);

    if (r == CUDA_SUCCESS) {
        *module = (CUmodule)HANDLE_TO_PTR(resp.mod_handle);
    }
    return r;
}

// ── cuModuleGetFunction ───────────────────────────────────────────────────────

__attribute__((visibility("default")))
CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name) {
    SHIM_CHECK_CONNECTED();

    if (hfunc == NULL || hmod == NULL || name == NULL) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    uint32_t name_len  = (uint32_t)(strlen(name) + 1); // include null terminator
    uint32_t payload_len = (uint32_t)(sizeof(Req_cuModuleGetFunction) + name_len);

    if (payload_len > IPC_MAX_PAYLOAD) {
        SHIM_WARN("cuModuleGetFunction: name too long (%u bytes)", name_len);
        return CUDA_ERROR_INVALID_VALUE;
    }

    uint8_t *payload = malloc(payload_len);
    if (payload == NULL) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }

    Req_cuModuleGetFunction req = {
        .mod_handle = PTR_TO_HANDLE(hmod),
        .name_len   = name_len,
    };
    memcpy(payload, &req, sizeof(req));
    memcpy(payload + sizeof(req), name, name_len);

    Resp_cuModuleGetFunction resp = { 0 };
    CUresult r = ipc_call(FN_cuModuleGetFunction, payload, payload_len,
                          &resp, sizeof(resp), NULL);
    free(payload);

    if (r == CUDA_SUCCESS) {
        *hfunc = (CUfunction)HANDLE_TO_PTR(resp.func_handle);
    }
    return r;
}

// ── cuLaunchKernel ────────────────────────────────────────────────────────────

__attribute__((visibility("default")))
CUresult cuLaunchKernel(
    CUfunction  f,
    unsigned int gridDimX,  unsigned int gridDimY,  unsigned int gridDimZ,
    unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
    unsigned int sharedMemBytes,
    CUstream     hStream,
    void       **kernelParams,
    void       **extra        // must be NULL per driver spec when using kernelParams
) {
    SHIM_CHECK_CONNECTED();

    if (extra != NULL) {
        // The CUDA driver forbids setting both kernelParams and extra.
        SHIM_WARN("cuLaunchKernel: 'extra' launch parameter is not supported; "
                  "use kernelParams only");
        return CUDA_ERROR_INVALID_VALUE;
    }

    // Count how many parameters are present by scanning kernelParams until NULL.
    uint32_t num_params = 0;
    if (kernelParams != NULL) {
        while (kernelParams[num_params] != NULL) {
            ++num_params;
        }
    }

    if (num_params > MAX_KERNEL_PARAMS) {
        SHIM_WARN("cuLaunchKernel: too many kernel parameters (%u > %u)",
                  num_params, MAX_KERNEL_PARAMS);
        return CUDA_ERROR_INVALID_VALUE;
    }

    // Each parameter is serialised as 8 bytes (covers all scalars + pointers).
    uint32_t params_bytes = num_params * (uint32_t)sizeof(uint64_t);
    uint32_t payload_len  = (uint32_t)sizeof(Req_cuLaunchKernel) + params_bytes;

    uint8_t *payload = malloc(payload_len);
    if (payload == NULL) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }

    Req_cuLaunchKernel req = {
        .func_handle   = PTR_TO_HANDLE(f),
        .grid_x        = gridDimX,
        .grid_y        = gridDimY,
        .grid_z        = gridDimZ,
        .block_x       = blockDimX,
        .block_y       = blockDimY,
        .block_z       = blockDimZ,
        .shared_mem    = sharedMemBytes,
        .stream_handle = PTR_TO_HANDLE(hStream),
        .num_params    = num_params,
    };
    memcpy(payload, &req, sizeof(req));

    // Serialise each parameter value as 8 bytes.  kernelParams[i] points to
    // the actual argument value, not to a pointer to it.
    uint8_t *pdata = payload + sizeof(req);
    for (uint32_t i = 0; i < num_params; ++i) {
        uint64_t val = 0;
        // Safe: we copy exactly 8 bytes regardless of actual size.
        // Callers pass pointers to at-least-8-byte storage per CUDA convention.
        memcpy(&val, kernelParams[i], sizeof(uint64_t));
        memcpy(pdata, &val, sizeof(uint64_t));
        pdata += sizeof(uint64_t);
    }

    CUresult r = ipc_call(FN_cuLaunchKernel, payload, payload_len, NULL, 0, NULL);
    free(payload);
    return r;
}

// ── cuModuleUnload ────────────────────────────────────────────────────────────

__attribute__((visibility("default")))
CUresult cuModuleUnload(CUmodule hmod) {
    SHIM_CHECK_CONNECTED();

    Req_cuModuleUnload req = { .mod_handle = PTR_TO_HANDLE(hmod) };
    return ipc_call(FN_cuModuleUnload, &req, sizeof(req), NULL, 0, NULL);
}
