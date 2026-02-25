// Copyright (C) 2026 AniiGIT
// SPDX-License-Identifier: AGPL-3.0-or-later

/*
 * shim_main.c - Global state and library constructor/destructor.
 *
 * The constructor (called by the dynamic linker on LD_PRELOAD injection)
 * connects to the sidecar's Unix socket.  The destructor disconnects.
 * If the sidecar is not running at inject time, g_ipc_connected remains 0
 * and all CUDA functions return CUDA_ERROR_NOT_INITIALIZED (except cuInit,
 * which returns CUDA_ERROR_NO_DEVICE so the app fails gracefully).
 */

#include <stdarg.h>
#include <stdio.h>
#include <pthread.h>

#include "idlegpu_shim.h"

// ── Global state ─────────────────────────────────────────────────────────────

volatile int      g_ipc_connected = 0;
uint32_t          g_call_id       = 0;
pthread_mutex_t   g_ipc_mutex     = PTHREAD_MUTEX_INITIALIZER;

// ── Logging ───────────────────────────────────────────────────────────────────

void idlegpu_log(const char *level, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    // Write to stderr so it doesn't interfere with the intercepted app's stdout.
    fprintf(stderr, "[idlegpu-cuda] %s  ", level);
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    va_end(ap);
}

// ── Constructor / destructor ──────────────────────────────────────────────────

__attribute__((constructor))
static void shim_init(void) {
    SHIM_INFO("loading (build " __DATE__ " " __TIME__ ")");

    if (ipc_connect() == 0) {
        g_ipc_connected = 1;
        SHIM_INFO("connected to sidecar IPC socket");
    } else {
        SHIM_WARN("sidecar not available -- all CUDA calls will fail gracefully");
    }
}

__attribute__((destructor))
static void shim_fini(void) {
    if (g_ipc_connected) {
        ipc_disconnect();
        g_ipc_connected = 0;
    }
    SHIM_INFO("unloaded");
}
