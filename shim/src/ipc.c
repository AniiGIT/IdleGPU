// Copyright (C) 2026 AniiGIT
// SPDX-License-Identifier: AGPL-3.0-or-later

/*
 * ipc.c - Unix socket IPC transport between the shim and the sidecar.
 *
 * The sidecar listens on a Unix domain socket (SOCK_STREAM).  The shim
 * connects once at constructor time and reuses the connection for all
 * subsequent CUDA calls.  All calls are serialised under g_ipc_mutex so
 * that multi-threaded CUDA applications are handled correctly.
 *
 * Wire format:
 *   Send: IpcReqHeader (16 bytes) + payload_len bytes
 *   Recv: IpcRespHeader (12 bytes) + payload_len bytes
 *
 * Framing guarantees:
 *   ipc_send_all() / ipc_recv_all() loop until all bytes are transferred,
 *   retrying on EINTR.  Any other error closes the socket and sets
 *   g_ipc_connected = 0.
 */

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include "idlegpu_shim.h"

// File descriptor for the Unix socket connection to the sidecar.
static int s_fd = -1;

// ── Low-level I/O helpers ─────────────────────────────────────────────────────

// Write exactly len bytes to fd, retrying on EINTR.  Returns 0 or -1.
static int ipc_send_all(int fd, const void *buf, size_t len) {
    const uint8_t *p = (const uint8_t *)buf;
    while (len > 0) {
        ssize_t n = send(fd, p, len, MSG_NOSIGNAL);
        if (n < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        p   += (size_t)n;
        len -= (size_t)n;
    }
    return 0;
}

// Read exactly len bytes from fd, retrying on EINTR.  Returns 0 or -1.
static int ipc_recv_all(int fd, void *buf, size_t len) {
    uint8_t *p = (uint8_t *)buf;
    while (len > 0) {
        ssize_t n = recv(fd, p, len, MSG_WAITALL);
        if (n < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (n == 0) {
            // Peer closed connection.
            return -1;
        }
        p   += (size_t)n;
        len -= (size_t)n;
    }
    return 0;
}

// ── Public API ────────────────────────────────────────────────────────────────

int ipc_connect(void) {
    const char *path = getenv("IDLEGPU_SOCKET");
    if (!path || path[0] == '\0') {
        path = IDLEGPU_SOCKET_DEFAULT;
    }

    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) {
        SHIM_WARN("ipc_connect: socket() failed: %s", strerror(errno));
        return -1;
    }

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    // Path must fit — truncate with a clear log if it doesn't.
    if (strlen(path) >= sizeof(addr.sun_path)) {
        SHIM_WARN("ipc_connect: socket path too long: %s", path);
        close(fd);
        return -1;
    }
    strncpy(addr.sun_path, path, sizeof(addr.sun_path) - 1);

    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        SHIM_WARN("ipc_connect: connect(%s) failed: %s", path, strerror(errno));
        close(fd);
        return -1;
    }

    s_fd = fd;
    SHIM_DEBUG("ipc_connect: connected to %s (fd=%d)", path, fd);
    return 0;
}

void ipc_disconnect(void) {
    if (s_fd >= 0) {
        close(s_fd);
        s_fd = -1;
        SHIM_DEBUG("ipc_disconnect: socket closed");
    }
}

// Mark the connection dead on any I/O error.
static void ipc_mark_dead(void) {
    SHIM_WARN("IPC connection lost -- closing socket");
    if (s_fd >= 0) {
        close(s_fd);
        s_fd = -1;
    }
    g_ipc_connected = 0;
}

CUresult ipc_call(
    uint32_t    func_id,
    const void *req_payload,
    uint32_t    req_payload_len,
    void       *resp_payload,
    uint32_t    resp_payload_max,
    uint32_t   *resp_payload_len_out
) {
    pthread_mutex_lock(&g_ipc_mutex);

    if (s_fd < 0) {
        pthread_mutex_unlock(&g_ipc_mutex);
        return CUDA_ERROR_NOT_INITIALIZED;
    }

    // ── Send request ─────────────────────────────────────────────────────────

    IpcReqHeader req_hdr = {
        .magic       = IPC_MAGIC,
        .func_id     = func_id,
        .call_id     = shim_next_call_id(),
        .payload_len = req_payload_len,
    };

    if (ipc_send_all(s_fd, &req_hdr, sizeof(req_hdr)) < 0) {
        ipc_mark_dead();
        pthread_mutex_unlock(&g_ipc_mutex);
        return CUDA_ERROR_UNKNOWN;
    }

    if (req_payload_len > 0 && req_payload != NULL) {
        if (ipc_send_all(s_fd, req_payload, req_payload_len) < 0) {
            ipc_mark_dead();
            pthread_mutex_unlock(&g_ipc_mutex);
            return CUDA_ERROR_UNKNOWN;
        }
    }

    // ── Receive response ──────────────────────────────────────────────────────

    IpcRespHeader resp_hdr;
    if (ipc_recv_all(s_fd, &resp_hdr, sizeof(resp_hdr)) < 0) {
        ipc_mark_dead();
        pthread_mutex_unlock(&g_ipc_mutex);
        return CUDA_ERROR_UNKNOWN;
    }

    if (resp_hdr.call_id != req_hdr.call_id) {
        SHIM_WARN("ipc_call: response call_id mismatch (expected %u, got %u)",
                  req_hdr.call_id, resp_hdr.call_id);
        ipc_mark_dead();
        pthread_mutex_unlock(&g_ipc_mutex);
        return CUDA_ERROR_UNKNOWN;
    }

    if (resp_hdr.payload_len > IPC_MAX_PAYLOAD) {
        SHIM_WARN("ipc_call: response payload too large (%u bytes)", resp_hdr.payload_len);
        ipc_mark_dead();
        pthread_mutex_unlock(&g_ipc_mutex);
        return CUDA_ERROR_UNKNOWN;
    }

    // Read response payload (even if the caller doesn't want it, drain it).
    if (resp_hdr.payload_len > 0) {
        if (resp_payload != NULL && resp_hdr.payload_len <= resp_payload_max) {
            if (ipc_recv_all(s_fd, resp_payload, resp_hdr.payload_len) < 0) {
                ipc_mark_dead();
                pthread_mutex_unlock(&g_ipc_mutex);
                return CUDA_ERROR_UNKNOWN;
            }
        } else {
            // Drain into a temporary buffer so the stream stays in sync.
            uint8_t drain[256];
            uint32_t remaining = resp_hdr.payload_len;
            while (remaining > 0) {
                uint32_t chunk = remaining < sizeof(drain) ? remaining : sizeof(drain);
                if (ipc_recv_all(s_fd, drain, chunk) < 0) {
                    ipc_mark_dead();
                    pthread_mutex_unlock(&g_ipc_mutex);
                    return CUDA_ERROR_UNKNOWN;
                }
                remaining -= chunk;
            }
            if (resp_payload != NULL) {
                SHIM_WARN("ipc_call func_id=%u: response payload %u > max %u; discarded",
                          func_id, resp_hdr.payload_len, resp_payload_max);
            }
        }
    }

    if (resp_payload_len_out != NULL) {
        *resp_payload_len_out = resp_hdr.payload_len;
    }

    pthread_mutex_unlock(&g_ipc_mutex);
    return (CUresult)resp_hdr.cuda_result;
}
