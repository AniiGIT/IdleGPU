// Copyright (C) 2026 AniiGIT
// SPDX-License-Identifier: AGPL-3.0-or-later

/*
 * cuda_stream.c - Tier 1: Streams and synchronisation.
 *
 * Intercepted functions (8):
 *   cuStreamCreate, cuStreamDestroy, cuStreamSynchronize, cuStreamWaitEvent,
 *   cuEventCreate, cuEventDestroy, cuEventRecord, cuEventSynchronize
 *
 * All stream and event objects are represented as opaque 64-bit handles on
 * the sidecar side.  The shim never allocates or interprets them locally.
 *
 * _v2 aliases are provided for applications linked against versioned symbols.
 */

#include "idlegpu_shim.h"

#define HANDLE_TO_PTR(h)  ((void *)(uintptr_t)(h))
#define PTR_TO_HANDLE(p)  ((uint64_t)(uintptr_t)(p))

// ── cuStreamCreate ────────────────────────────────────────────────────────────

__attribute__((visibility("default")))
CUresult cuStreamCreate(CUstream *phStream, unsigned int Flags) {
    SHIM_REQUIRE_IPC(g_real.cuStreamCreate, phStream, Flags);

    Req_cuStreamCreate  req  = { .flags = Flags };
    Resp_cuStreamCreate resp = { 0 };

    CUresult r = ipc_call(FN_cuStreamCreate, &req, sizeof(req),
                          &resp, sizeof(resp), NULL);
    if (r == CUDA_SUCCESS && phStream != NULL) {
        *phStream = (CUstream)HANDLE_TO_PTR(resp.stream_handle);
    }
    return r;
}

// ── cuStreamDestroy ───────────────────────────────────────────────────────────

__attribute__((visibility("default")))
CUresult cuStreamDestroy(CUstream hStream) {
    SHIM_REQUIRE_IPC(g_real.cuStreamDestroy, hStream);

    Req_cuStreamDestroy req = { .stream_handle = PTR_TO_HANDLE(hStream) };
    return ipc_call(FN_cuStreamDestroy, &req, sizeof(req), NULL, 0, NULL);
}

__attribute__((visibility("default")))
CUresult cuStreamDestroy_v2(CUstream hStream) {
    return cuStreamDestroy(hStream);
}

// ── cuStreamSynchronize ───────────────────────────────────────────────────────

__attribute__((visibility("default")))
CUresult cuStreamSynchronize(CUstream hStream) {
    SHIM_REQUIRE_IPC(g_real.cuStreamSynchronize, hStream);

    Req_cuStreamSynchronize req = { .stream_handle = PTR_TO_HANDLE(hStream) };
    return ipc_call(FN_cuStreamSynchronize, &req, sizeof(req), NULL, 0, NULL);
}

// ── cuStreamWaitEvent ─────────────────────────────────────────────────────────

__attribute__((visibility("default")))
CUresult cuStreamWaitEvent(CUstream hStream, CUevent hEvent, unsigned int Flags) {
    SHIM_REQUIRE_IPC(g_real.cuStreamWaitEvent, hStream, hEvent, Flags);

    Req_cuStreamWaitEvent req = {
        .stream_handle = PTR_TO_HANDLE(hStream),
        .event_handle  = PTR_TO_HANDLE(hEvent),
        .flags         = Flags,
    };
    return ipc_call(FN_cuStreamWaitEvent, &req, sizeof(req), NULL, 0, NULL);
}

// ── cuEventCreate ─────────────────────────────────────────────────────────────

__attribute__((visibility("default")))
CUresult cuEventCreate(CUevent *phEvent, unsigned int Flags) {
    SHIM_REQUIRE_IPC(g_real.cuEventCreate, phEvent, Flags);

    Req_cuEventCreate  req  = { .flags = Flags };
    Resp_cuEventCreate resp = { 0 };

    CUresult r = ipc_call(FN_cuEventCreate, &req, sizeof(req),
                          &resp, sizeof(resp), NULL);
    if (r == CUDA_SUCCESS && phEvent != NULL) {
        *phEvent = (CUevent)HANDLE_TO_PTR(resp.event_handle);
    }
    return r;
}

// ── cuEventDestroy ────────────────────────────────────────────────────────────

__attribute__((visibility("default")))
CUresult cuEventDestroy(CUevent hEvent) {
    SHIM_REQUIRE_IPC(g_real.cuEventDestroy, hEvent);

    Req_cuEventDestroy req = { .event_handle = PTR_TO_HANDLE(hEvent) };
    return ipc_call(FN_cuEventDestroy, &req, sizeof(req), NULL, 0, NULL);
}

__attribute__((visibility("default")))
CUresult cuEventDestroy_v2(CUevent hEvent) {
    return cuEventDestroy(hEvent);
}

// ── cuEventRecord ─────────────────────────────────────────────────────────────

__attribute__((visibility("default")))
CUresult cuEventRecord(CUevent hEvent, CUstream hStream) {
    SHIM_REQUIRE_IPC(g_real.cuEventRecord, hEvent, hStream);

    Req_cuEventRecord req = {
        .event_handle  = PTR_TO_HANDLE(hEvent),
        .stream_handle = PTR_TO_HANDLE(hStream),
    };
    return ipc_call(FN_cuEventRecord, &req, sizeof(req), NULL, 0, NULL);
}

// ── cuEventSynchronize ────────────────────────────────────────────────────────

__attribute__((visibility("default")))
CUresult cuEventSynchronize(CUevent hEvent) {
    SHIM_REQUIRE_IPC(g_real.cuEventSynchronize, hEvent);

    Req_cuEventSynchronize req = { .event_handle = PTR_TO_HANDLE(hEvent) };
    return ipc_call(FN_cuEventSynchronize, &req, sizeof(req), NULL, 0, NULL);
}
