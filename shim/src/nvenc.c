// Copyright (C) 2026 AniiGIT
// SPDX-License-Identifier: AGPL-3.0-or-later

/*
 * nvenc.c - Tier 1: NVENC API stubs.
 *
 * The NVENC API uses a function-pointer table rather than direct symbol
 * exports.  Applications call NvEncodeAPICreateInstance() to obtain a
 * NV_ENCODE_API_FUNCTION_LIST struct filled with function pointers.
 *
 * Phase 2D: All 10 NVENC functions are stubbed to return
 * NV_ENC_ERR_UNIMPLEMENTED (error code 15).  The full forwarding
 * implementation is deferred to Phase 2E once the sidecar NVENC
 * executor is complete.
 *
 * Phase 2E will replace these stubs with IPC-forwarded implementations
 * following the same request/response pattern as the CUDA functions.
 *
 * Functions (10):
 *   NvEncOpenEncodeSession, NvEncInitializeEncoder,
 *   NvEncCreateInputBuffer, NvEncCreateBitstreamBuffer,
 *   NvEncEncodePicture, NvEncLockBitstream, NvEncUnlockBitstream,
 *   NvEncDestroyInputBuffer, NvEncDestroyBitstreamBuffer,
 *   NvEncDestroyEncoder
 *
 * The stub for NvEncodeAPICreateInstance intercepts the entry point that
 * applications call to obtain function pointers, returning NOT_SUPPORTED.
 */

#include "idlegpu_shim.h"

// NVENC status codes (defined by the NVENC SDK; reproduced here to avoid
// a hard dependency on the NVENC headers in Phase 2D).
#define NV_ENC_SUCCESS           0
#define NV_ENC_ERR_UNIMPLEMENTED 15

// Opaque NVENC encoder handle type (void*-compatible).
typedef void* NV_ENC_SESSION_HANDLE;

// ── NvEncodeAPICreateInstance ─────────────────────────────────────────────────
//
// This is the sole public symbol that applications dlopen to bootstrap NVENC.
// Return NOT_SUPPORTED so applications fail immediately with a clear error
// rather than obtaining a function-pointer table that would crash later.

__attribute__((visibility("default")))
int NvEncodeAPICreateInstance(void *functionList) {
    (void)functionList;
    SHIM_WARN("NvEncodeAPICreateInstance: NVENC forwarding not yet implemented "
              "(Phase 2E); returning NV_ENC_ERR_UNIMPLEMENTED");
    return NV_ENC_ERR_UNIMPLEMENTED;
}

// ── Individual NVENC stubs ────────────────────────────────────────────────────
//
// These are not normally called directly by applications (they go through
// the function-pointer table), but some statically linked encoders or
// wrappers may look up these symbols by name.  Stub them defensively.

__attribute__((visibility("default")))
int NvEncOpenEncodeSession(void *device, unsigned int deviceType,
                           NV_ENC_SESSION_HANDLE *encoder) {
    (void)device; (void)deviceType; (void)encoder;
    SHIM_WARN("NvEncOpenEncodeSession: unimplemented (Phase 2E)");
    return NV_ENC_ERR_UNIMPLEMENTED;
}

__attribute__((visibility("default")))
int NvEncInitializeEncoder(NV_ENC_SESSION_HANDLE encoder, void *createEncodeParams) {
    (void)encoder; (void)createEncodeParams;
    SHIM_WARN("NvEncInitializeEncoder: unimplemented (Phase 2E)");
    return NV_ENC_ERR_UNIMPLEMENTED;
}

__attribute__((visibility("default")))
int NvEncCreateInputBuffer(NV_ENC_SESSION_HANDLE encoder, void *createInputBufferParams) {
    (void)encoder; (void)createInputBufferParams;
    SHIM_WARN("NvEncCreateInputBuffer: unimplemented (Phase 2E)");
    return NV_ENC_ERR_UNIMPLEMENTED;
}

__attribute__((visibility("default")))
int NvEncCreateBitstreamBuffer(NV_ENC_SESSION_HANDLE encoder,
                               void *createBitstreamBufferParams) {
    (void)encoder; (void)createBitstreamBufferParams;
    SHIM_WARN("NvEncCreateBitstreamBuffer: unimplemented (Phase 2E)");
    return NV_ENC_ERR_UNIMPLEMENTED;
}

__attribute__((visibility("default")))
int NvEncEncodePicture(NV_ENC_SESSION_HANDLE encoder, void *encodePicParams) {
    (void)encoder; (void)encodePicParams;
    SHIM_WARN("NvEncEncodePicture: unimplemented (Phase 2E)");
    return NV_ENC_ERR_UNIMPLEMENTED;
}

__attribute__((visibility("default")))
int NvEncLockBitstream(NV_ENC_SESSION_HANDLE encoder, void *lockBitstreamBufferParams) {
    (void)encoder; (void)lockBitstreamBufferParams;
    SHIM_WARN("NvEncLockBitstream: unimplemented (Phase 2E)");
    return NV_ENC_ERR_UNIMPLEMENTED;
}

__attribute__((visibility("default")))
int NvEncUnlockBitstream(NV_ENC_SESSION_HANDLE encoder, void *bitstreamBuffer) {
    (void)encoder; (void)bitstreamBuffer;
    SHIM_WARN("NvEncUnlockBitstream: unimplemented (Phase 2E)");
    return NV_ENC_ERR_UNIMPLEMENTED;
}

__attribute__((visibility("default")))
int NvEncDestroyInputBuffer(NV_ENC_SESSION_HANDLE encoder, void *inputBuffer) {
    (void)encoder; (void)inputBuffer;
    SHIM_WARN("NvEncDestroyInputBuffer: unimplemented (Phase 2E)");
    return NV_ENC_ERR_UNIMPLEMENTED;
}

__attribute__((visibility("default")))
int NvEncDestroyBitstreamBuffer(NV_ENC_SESSION_HANDLE encoder, void *bitstreamBuffer) {
    (void)encoder; (void)bitstreamBuffer;
    SHIM_WARN("NvEncDestroyBitstreamBuffer: unimplemented (Phase 2E)");
    return NV_ENC_ERR_UNIMPLEMENTED;
}

__attribute__((visibility("default")))
int NvEncDestroyEncoder(NV_ENC_SESSION_HANDLE encoder) {
    (void)encoder;
    SHIM_WARN("NvEncDestroyEncoder: unimplemented (Phase 2E)");
    return NV_ENC_ERR_UNIMPLEMENTED;
}
