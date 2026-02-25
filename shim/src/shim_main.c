// Copyright (C) 2026 AniiGIT
// SPDX-License-Identifier: AGPL-3.0-or-later

/*
 * shim_main.c - Global state and library constructor/destructor.
 *
 * The constructor (called by the dynamic linker on LD_PRELOAD injection):
 *   1. Bootstraps real_cuda_init() with the real dlsym pointer so that the
 *      g_real fallback table and dlopen/dlsym overrides are ready before any
 *      application code runs.
 *   2. Connects to the sidecar's Unix socket (ipc_connect).
 *
 * If the sidecar is not running at inject time, g_ipc_connected remains 0
 * and CUDA functions fall back to g_real (the local GPU) when available.
 *
 * dlvsym bootstrap note:
 *   We use dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5") rather than calling
 *   dlsym() directly.  By the time the constructor runs, our dlsym override
 *   is already exported, so calling dlsym(RTLD_NEXT, "dlsym") would recurse
 *   into our own override.  dlvsym bypasses the override by requesting the
 *   versioned glibc symbol directly.
 */

#define _GNU_SOURCE
#include <dlfcn.h>
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

    // Bootstrap the real CUDA function pointer table and dlopen/dlsym overrides.
    // dlvsym with an explicit version string bypasses our own dlsym override.
    typedef void *(*dlsym_fn_t)(void *, const char *);
    dlsym_fn_t bootstrap = (dlsym_fn_t)dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
    real_cuda_init(bootstrap);

    if (ipc_connect() == 0) {
        g_ipc_connected = 1;
        SHIM_INFO("connected to sidecar IPC socket");
    } else {
        SHIM_WARN("sidecar not available -- CUDA calls will use local GPU fallback");
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
