// Copyright (C) 2026 AniiGIT
// SPDX-License-Identifier: AGPL-3.0-or-later

/*
 * idlegpu_shim.h - Internal API for the IdleGPU CUDA intercept shim.
 *
 * Covers:
 *   - IPC function IDs (one per CUDA/NVENC function intercepted)
 *   - IPC request/response payload structs (POD; serialised by value)
 *   - IPC transport API (ipc.c)
 *   - Logging macros
 *   - Global state declarations
 *
 * IPC wire format (Unix stream socket, host byte order)
 * ─────────────────────────────────────────────────────
 * Request:
 *   IpcReqHeader  (16 bytes)
 *   [payload_len bytes: function-specific payload]
 *
 * Response:
 *   IpcRespHeader (12 bytes)
 *   [payload_len bytes: function-specific output data]
 *
 * All structs are packed to avoid padding surprises.
 * Both shim and sidecar run on the same machine (Unix socket), so byte
 * order conversion is not required.
 */

#pragma once

#include <stdint.h>
#include <stddef.h>
#include <pthread.h>

#include "cuda_compat.h"

// ── IPC configuration ─────────────────────────────────────────────────────────

// Default socket path. Override with $IDLEGPU_SOCKET environment variable.
#define IDLEGPU_SOCKET_DEFAULT "/var/run/idlegpu/cuda.sock"

// Frame magic — "IGPU" — present in every request header.
#define IPC_MAGIC 0x49475055u

#define IPC_VERSION 1u

// Maximum payload size (16 MiB; large enough for PTX modules).
#define IPC_MAX_PAYLOAD (16u * 1024u * 1024u)

// Maximum device name length (matches CUDA_MAX_NAME_LEN).
#define IPC_MAX_NAME 256u

// ── Function IDs ──────────────────────────────────────────────────────────────

// Initialization and device management (1–10)
#define FN_cuInit                   1u
#define FN_cuDeviceGet              2u
#define FN_cuDeviceGetCount         3u
#define FN_cuDeviceGetName          4u
#define FN_cuDeviceGetAttribute     5u
#define FN_cuDeviceTotalMem         6u
#define FN_cuCtxCreate              7u
#define FN_cuCtxDestroy             8u
#define FN_cuCtxSetCurrent          9u
#define FN_cuCtxGetCurrent         10u

// Memory management (11–19)
#define FN_cuMemAlloc              11u
#define FN_cuMemFree               12u
#define FN_cuMemcpyHtoD            13u
#define FN_cuMemcpyDtoH            14u
#define FN_cuMemcpyDtoD            15u
#define FN_cuMemsetD8              16u
#define FN_cuMemsetD16             17u
#define FN_cuMemsetD32             18u
#define FN_cuMemGetInfo            19u

// Driver version (20)
#define FN_cuDriverGetVersion      20u

// Module and kernel execution (21–25)
#define FN_cuModuleLoad            21u
#define FN_cuModuleLoadData        22u
#define FN_cuModuleGetFunction     23u
#define FN_cuLaunchKernel          24u
#define FN_cuModuleUnload          25u

// Streams and synchronisation (31–38)
#define FN_cuStreamCreate          31u
#define FN_cuStreamDestroy         32u
#define FN_cuStreamSynchronize     33u
#define FN_cuStreamWaitEvent       34u
#define FN_cuEventCreate           35u
#define FN_cuEventDestroy          36u
#define FN_cuEventRecord           37u
#define FN_cuEventSynchronize      38u

// NVENC (41–50)
#define FN_NvEncOpenEncodeSession      41u
#define FN_NvEncInitializeEncoder      42u
#define FN_NvEncCreateInputBuffer      43u
#define FN_NvEncCreateBitstreamBuffer  44u
#define FN_NvEncEncodePicture          45u
#define FN_NvEncLockBitstream          46u
#define FN_NvEncUnlockBitstream        47u
#define FN_NvEncDestroyInputBuffer     48u
#define FN_NvEncDestroyBitstreamBuffer 49u
#define FN_NvEncDestroyEncoder         50u

// Runtime internals (51–)
// cuGetExportTable is an internal CUDA driver function used by the CUDA
// runtime and ffmpeg to discover driver capability tables at runtime.
// Reserved here even though the current implementation is a stub so that
// future IPC forwarding can use a stable function ID.
#define FN_cuGetExportTable            51u

// Extended device / context (52–58) — promoted from stubs in Phase 2E.
// Required by ffmpeg's NVENC encoder initialisation path.
#define FN_cuDeviceComputeCapability   52u
#define FN_cuDeviceGetUuid             53u
#define FN_cuCtxPushCurrent            54u
#define FN_cuCtxPopCurrent             55u
#define FN_cuDeviceGetLuid             56u
#define FN_cuCtxSetLimit               57u
#define FN_cuCtxGetLimit               58u

// Extended memory (59–60) — pitched 2D and managed allocations.
#define FN_cuMemAllocPitch             59u
#define FN_cuMemAllocManaged           60u

// Async memset (61–63) — stream-ordered variants of cuMemsetD8/D16/D32.
#define FN_cuMemsetD8Async             61u
#define FN_cuMemsetD16Async            62u
#define FN_cuMemsetD32Async            63u

// Async device-to-device copy (64).
#define FN_cuMemcpyDtoDAsync           64u

// Async host↔device copies (79–80).
// Request for HtoD: _Req_HtoDAsync_Hdr (24 B) + byte_count bytes of data.
// Request for DtoH: _Req_DtoHAsync_Hdr (24 B); response is byte_count bytes.
#define FN_cuMemcpyHtoDAsync           79u
#define FN_cuMemcpyDtoHAsync           80u

// Stream creation with priority (65).
#define FN_cuStreamCreateWithPriority  65u

// Context introspection (66–73) — queried/set on the calling thread's context.
#define FN_cuCtxSynchronize            66u
#define FN_cuCtxGetDevice              67u
#define FN_cuCtxGetFlags               68u
#define FN_cuCtxGetApiVersion          69u
#define FN_cuCtxGetCacheConfig         70u
#define FN_cuCtxSetCacheConfig         71u
#define FN_cuCtxGetSharedMemConfig     72u
#define FN_cuCtxSetSharedMemConfig     73u

// Kernel / function attribute queries and settings (74–77).
#define FN_cuFuncGetAttribute          74u
#define FN_cuFuncSetAttribute          75u
#define FN_cuFuncSetCacheConfig        76u
#define FN_cuFuncSetSharedMemConfig    77u

// Occupancy (78) — requires function handle (from cuModuleGetFunction).
#define FN_cuOccupancyMaxActiveBlocksPerMultiprocessor 78u

// ── IPC frame headers ─────────────────────────────────────────────────────────

typedef struct __attribute__((packed)) {
    uint32_t magic;        // IPC_MAGIC
    uint32_t func_id;      // FN_cu* constant
    uint32_t call_id;      // monotonically increasing call identifier
    uint32_t payload_len;  // byte count of payload that follows
} IpcReqHeader;            // 16 bytes

typedef struct __attribute__((packed)) {
    uint32_t call_id;      // echoes request call_id
    uint32_t cuda_result;  // CUresult (0 = success)
    uint32_t payload_len;  // byte count of output payload that follows
} IpcRespHeader;           // 12 bytes

// ── Per-function payload structs ──────────────────────────────────────────────
// Convention: Req_* = request payload; Resp_* = response payload.
// Variable-length data (strings, host buffers) follows the fixed struct.

// cuInit(flags)
typedef struct __attribute__((packed)) { uint32_t flags; } Req_cuInit;

// cuDeviceGet(ordinal)
typedef struct __attribute__((packed)) { int32_t  ordinal; } Req_cuDeviceGet;
typedef struct __attribute__((packed)) { int32_t  device;  } Resp_cuDeviceGet;

// cuDeviceGetCount — no payload fields

// cuDeviceGetName(len, dev)  → followed by `len` bytes of name (response)
typedef struct __attribute__((packed)) {
    uint32_t len;
    int32_t  device;
} Req_cuDeviceGetName;
// Resp: [len bytes of null-terminated name]

// cuDeviceGetAttribute(attrib, dev)
typedef struct __attribute__((packed)) {
    int32_t  attrib;
    int32_t  device;
} Req_cuDeviceGetAttribute;
typedef struct __attribute__((packed)) { int32_t value; } Resp_cuDeviceGetAttribute;

// cuDeviceTotalMem(dev)
typedef struct __attribute__((packed)) { int32_t device; } Req_cuDeviceTotalMem;
typedef struct __attribute__((packed)) { uint64_t bytes;  } Resp_cuDeviceTotalMem;

// cuCtxCreate(flags, dev)
typedef struct __attribute__((packed)) {
    uint32_t flags;
    int32_t  device;
} Req_cuCtxCreate;
typedef struct __attribute__((packed)) { uint64_t ctx_handle; } Resp_cuCtxCreate;

// cuCtxDestroy(ctx)
typedef struct __attribute__((packed)) { uint64_t ctx_handle; } Req_cuCtxDestroy;

// cuCtxSetCurrent(ctx)
typedef struct __attribute__((packed)) { uint64_t ctx_handle; } Req_cuCtxSetCurrent;

// cuCtxGetCurrent — no request payload
typedef struct __attribute__((packed)) { uint64_t ctx_handle; } Resp_cuCtxGetCurrent;

// cuMemAlloc(bytesize)
typedef struct __attribute__((packed)) { uint64_t bytesize;   } Req_cuMemAlloc;
typedef struct __attribute__((packed)) { uint64_t dptr;        } Resp_cuMemAlloc;

// cuMemFree(dptr)
typedef struct __attribute__((packed)) { uint64_t dptr; } Req_cuMemFree;

// cuMemcpyHtoD(dst, ByteCount)  → followed by ByteCount bytes of host data
typedef struct __attribute__((packed)) {
    uint64_t dst;
    uint64_t byte_count;
} Req_cuMemcpyHtoD;

// cuMemcpyDtoH(src, ByteCount)  → response followed by ByteCount bytes
typedef struct __attribute__((packed)) {
    uint64_t src;
    uint64_t byte_count;
} Req_cuMemcpyDtoH;

// cuMemcpyDtoD(dst, src, ByteCount)
typedef struct __attribute__((packed)) {
    uint64_t dst;
    uint64_t src;
    uint64_t byte_count;
} Req_cuMemcpyDtoD;

// cuMemsetD8(dstDevice, uc, N)
typedef struct __attribute__((packed)) {
    uint64_t dst;
    uint8_t  value;
    uint8_t  _pad[7];
    uint64_t count;
} Req_cuMemsetD8;

// cuMemsetD16(dstDevice, us, N)
typedef struct __attribute__((packed)) {
    uint64_t dst;
    uint16_t value;
    uint8_t  _pad[6];
    uint64_t count;
} Req_cuMemsetD16;

// cuMemsetD32(dstDevice, ui, N)
typedef struct __attribute__((packed)) {
    uint64_t dst;
    uint32_t value;
    uint8_t  _pad[4];
    uint64_t count;
} Req_cuMemsetD32;

// cuMemGetInfo — no request payload
typedef struct __attribute__((packed)) {
    uint64_t free;
    uint64_t total;
} Resp_cuMemGetInfo;

// cuDriverGetVersion — no request payload
typedef struct __attribute__((packed)) { int32_t version; } Resp_cuDriverGetVersion;

// cuModuleLoad(fname)  → request payload is Req_cuModuleLoad + fname bytes
typedef struct __attribute__((packed)) { uint32_t fname_len; } Req_cuModuleLoad;
typedef struct __attribute__((packed)) { uint64_t mod_handle; } Resp_cuModuleLoad;

// cuModuleLoadData(image)  → request payload is Req_cuModuleLoadData + image bytes
typedef struct __attribute__((packed)) { uint64_t image_len; } Req_cuModuleLoadData;
typedef struct __attribute__((packed)) { uint64_t mod_handle; } Resp_cuModuleLoadData;

// cuModuleGetFunction(hmod, name)  → request payload is Req_cuModuleGetFunction + name bytes
typedef struct __attribute__((packed)) {
    uint64_t mod_handle;
    uint32_t name_len;
} Req_cuModuleGetFunction;
typedef struct __attribute__((packed)) { uint64_t func_handle; } Resp_cuModuleGetFunction;

// cuLaunchKernel(f, grid, block, sharedMem, stream, kernelParams)
// Variable-length tail: Req_cuLaunchKernel + [num_params * 8 bytes of param data]
typedef struct __attribute__((packed)) {
    uint64_t func_handle;
    uint32_t grid_x,  grid_y,  grid_z;
    uint32_t block_x, block_y, block_z;
    uint32_t shared_mem;
    uint64_t stream_handle;
    uint32_t num_params;    // count of 8-byte values that follow
} Req_cuLaunchKernel;

// cuModuleUnload(mod)
typedef struct __attribute__((packed)) { uint64_t mod_handle; } Req_cuModuleUnload;

// cuStreamCreate(flags)
typedef struct __attribute__((packed)) { uint32_t flags; } Req_cuStreamCreate;
typedef struct __attribute__((packed)) { uint64_t stream_handle; } Resp_cuStreamCreate;

// cuStreamDestroy(stream)
typedef struct __attribute__((packed)) { uint64_t stream_handle; } Req_cuStreamDestroy;

// cuStreamSynchronize(stream)
typedef struct __attribute__((packed)) { uint64_t stream_handle; } Req_cuStreamSynchronize;

// cuStreamWaitEvent(stream, event, flags)
typedef struct __attribute__((packed)) {
    uint64_t stream_handle;
    uint64_t event_handle;
    uint32_t flags;
} Req_cuStreamWaitEvent;

// cuEventCreate(flags)
typedef struct __attribute__((packed)) { uint32_t flags; } Req_cuEventCreate;
typedef struct __attribute__((packed)) { uint64_t event_handle; } Resp_cuEventCreate;

// cuEventDestroy(event)
typedef struct __attribute__((packed)) { uint64_t event_handle; } Req_cuEventDestroy;

// cuEventRecord(event, stream)
typedef struct __attribute__((packed)) {
    uint64_t event_handle;
    uint64_t stream_handle;
} Req_cuEventRecord;

// cuEventSynchronize(event)
typedef struct __attribute__((packed)) { uint64_t event_handle; } Req_cuEventSynchronize;

// cuGetExportTable(ppExportTable, pExportTableId)
//
// Request payload: 16-byte UUID identifying the capability table.
// Response payload: Resp_cuGetExportTable header (4 bytes) followed by
//   entry_count * 8 bytes — the raw 64-bit function pointer values from
//   the agent's libcuda.so.1, serialised as little-endian uint64.
//   The shim malloc's a persistent buffer and writes entry_count pointers
//   into it, then assigns *ppExportTable to that buffer.
//
// Note: the pointer values originate from the agent's address space.
//   They are valid as remote GPU-side addresses but NOT callable locally.
//   Use only for capability table presence checks, not for calling entries.
typedef struct __attribute__((packed)) { uint8_t uuid[16]; } Req_cuGetExportTable;
typedef struct __attribute__((packed)) { uint32_t entry_count; } Resp_cuGetExportTable;

// Maximum number of entries we read from an export table.
#define EXPORT_TABLE_MAX_ENTRIES 256u

// cuDeviceComputeCapability(major, minor, dev)
typedef struct __attribute__((packed)) { int32_t  device; } Req_cuDeviceComputeCapability;
typedef struct __attribute__((packed)) { int32_t  major; int32_t minor; } Resp_cuDeviceComputeCapability;

// cuDeviceGetUuid(uuid, dev)  → response: 16-byte UUID
typedef struct __attribute__((packed)) { int32_t  device; } Req_cuDeviceGetUuid;
typedef struct __attribute__((packed)) { uint8_t  bytes[16]; } Resp_cuDeviceGetUuid;

// cuDeviceGetLuid(luid, deviceNodeMask, dev)  → response: 8-byte LUID + uint32
typedef struct __attribute__((packed)) { int32_t  device; } Req_cuDeviceGetLuid;
typedef struct __attribute__((packed)) {
    uint8_t  luid[8];
    uint32_t device_node_mask;
} Resp_cuDeviceGetLuid;

// cuCtxPushCurrent(ctx)
typedef struct __attribute__((packed)) { uint64_t ctx_handle; } Req_cuCtxPushCurrent;

// cuCtxPopCurrent — no request payload
typedef struct __attribute__((packed)) { uint64_t ctx_handle; } Resp_cuCtxPopCurrent;

// cuCtxSetLimit(limit, value)
typedef struct __attribute__((packed)) {
    int32_t  limit;   // CUlimit enum (forwarded numerically)
    uint64_t value;   // new limit value (size_t on 64-bit)
} Req_cuCtxSetLimit;

// cuCtxGetLimit(pvalue, limit)
typedef struct __attribute__((packed)) { int32_t  limit; } Req_cuCtxGetLimit;
typedef struct __attribute__((packed)) { uint64_t value; } Resp_cuCtxGetLimit;

// cuMemAllocPitch_v2(dptr, pPitch, WidthInBytes, Height, ElementSizeBytes)
// Allocates pitched 2D device memory.  Pitch alignment is chosen by the driver
// to satisfy ElementSizeBytes alignment and coalescing requirements.
typedef struct __attribute__((packed)) {
    uint64_t width_bytes;   // WidthInBytes
    uint64_t height;        // Height (number of rows)
    uint32_t element_size;  // ElementSizeBytes (1, 2, 4, 8, or 16)
} Req_cuMemAllocPitch;      // 20 bytes
typedef struct __attribute__((packed)) {
    uint64_t dptr;   // allocated device pointer
    uint64_t pitch;  // actual row pitch in bytes
} Resp_cuMemAllocPitch;     // 16 bytes

// cuMemAllocManaged(dptr, bytesize, flags)
// Allocates unified managed memory accessible from both CPU and GPU.
// flags: CU_MEM_ATTACH_GLOBAL (1) or CU_MEM_ATTACH_HOST (2).
typedef struct __attribute__((packed)) {
    uint64_t bytesize;  // allocation size in bytes
    uint32_t flags;     // CU_MEM_ATTACH_* flags
} Req_cuMemAllocManaged;    // 12 bytes
typedef struct __attribute__((packed)) {
    uint64_t dptr;      // managed device/host pointer
} Resp_cuMemAllocManaged;   // 8 bytes

// cuMemsetD8Async / cuMemsetD16Async / cuMemsetD32Async
// Stream-ordered memset: fills N elements on dstDevice using hStream.
// Struct layout mirrors Req_cuMemsetD8/D16/D32 with stream_handle appended.
typedef struct __attribute__((packed)) {
    uint64_t dst;
    uint8_t  value;
    uint8_t  _pad[7];
    uint64_t count;
    uint64_t stream_handle;
} Req_cuMemsetD8Async;   // 32 bytes

typedef struct __attribute__((packed)) {
    uint64_t dst;
    uint16_t value;
    uint8_t  _pad[6];
    uint64_t count;
    uint64_t stream_handle;
} Req_cuMemsetD16Async;  // 32 bytes

typedef struct __attribute__((packed)) {
    uint64_t dst;
    uint32_t value;
    uint8_t  _pad[4];
    uint64_t count;
    uint64_t stream_handle;
} Req_cuMemsetD32Async;  // 32 bytes

// cuMemcpyDtoDAsync_v2(dstDevice, srcDevice, ByteCount, hStream)
typedef struct __attribute__((packed)) {
    uint64_t dst;
    uint64_t src;
    uint64_t byte_count;
    uint64_t stream_handle;
} Req_cuMemcpyDtoDAsync;  // 32 bytes

// cuStreamCreateWithPriority(phStream, flags, priority)
typedef struct __attribute__((packed)) {
    uint32_t flags;
    int32_t  priority;
} Req_cuStreamCreateWithPriority;   // 8 bytes
typedef struct __attribute__((packed)) {
    uint64_t stream_handle;
} Resp_cuStreamCreateWithPriority;  // 8 bytes

// cuCtxSynchronize — no request payload, no response payload (result only)

// cuCtxGetDevice — no request payload
typedef struct __attribute__((packed)) { int32_t  device; } Resp_cuCtxGetDevice;

// cuCtxGetFlags — no request payload
typedef struct __attribute__((packed)) { uint32_t flags; } Resp_cuCtxGetFlags;

// cuCtxGetApiVersion(ctx, version)
typedef struct __attribute__((packed)) { uint64_t ctx_handle; } Req_cuCtxGetApiVersion;
typedef struct __attribute__((packed)) { uint32_t version; } Resp_cuCtxGetApiVersion;

// cuCtxGetCacheConfig — no request payload  (CUfunc_cache enum)
typedef struct __attribute__((packed)) { int32_t  config; } Resp_cuCtxGetCacheConfig;

// cuCtxSetCacheConfig(config)
typedef struct __attribute__((packed)) { int32_t  config; } Req_cuCtxSetCacheConfig;

// cuCtxGetSharedMemConfig — no request payload  (CUsharedconfig enum)
typedef struct __attribute__((packed)) { int32_t  config; } Resp_cuCtxGetSharedMemConfig;

// cuCtxSetSharedMemConfig(config)
typedef struct __attribute__((packed)) { int32_t  config; } Req_cuCtxSetSharedMemConfig;

// cuFuncGetAttribute(pi, attrib, hfunc)
typedef struct __attribute__((packed)) {
    int32_t  attrib;
    uint64_t func_handle;
} Req_cuFuncGetAttribute;   // 12 bytes
typedef struct __attribute__((packed)) { int32_t value; } Resp_cuFuncGetAttribute;

// cuFuncSetAttribute(hfunc, attrib, value)
typedef struct __attribute__((packed)) {
    uint64_t func_handle;
    int32_t  attrib;
    int32_t  value;
} Req_cuFuncSetAttribute;   // 16 bytes

// cuFuncSetCacheConfig(hfunc, config)
typedef struct __attribute__((packed)) {
    uint64_t func_handle;
    int32_t  config;
} Req_cuFuncSetCacheConfig;  // 12 bytes

// cuFuncSetSharedMemConfig(hfunc, config)
typedef struct __attribute__((packed)) {
    uint64_t func_handle;
    int32_t  config;
} Req_cuFuncSetSharedMemConfig;  // 12 bytes

// cuOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, func, blockSize, dynSmem)
typedef struct __attribute__((packed)) {
    uint64_t func_handle;
    int32_t  block_size;
    uint64_t dynamic_smem_size;
} Req_cuOccupancyMaxActiveBlocksPerMultiprocessor;  // 20 bytes
typedef struct __attribute__((packed)) { int32_t num_blocks; } Resp_cuOccupancyMaxActiveBlocksPerMultiprocessor;

// ── IPC transport API (implemented in ipc.c) ──────────────────────────────────

// Connect to the sidecar Unix socket. Returns 0 on success, -1 on error.
int  ipc_connect(void);

// Disconnect.
void ipc_disconnect(void);

// Send request + receive response. Thread-safe (internal mutex).
// Returns the CUresult from the response header.
// resp_payload may be NULL if resp_payload_max is 0 (no output expected).
// On timeout: marks the IPC connection dead, returns CUDA_ERROR_NOT_INITIALIZED.
CUresult ipc_call(
    uint32_t    func_id,
    const void *req_payload,
    uint32_t    req_payload_len,
    void       *resp_payload,
    uint32_t    resp_payload_max,
    uint32_t   *resp_payload_len_out   // may be NULL
);

// Variant of ipc_call for optional IPC calls where a timeout should not be
// treated as a fatal connection error.  On timeout: does NOT call ipc_mark_dead();
// returns CUDA_ERROR_NOT_SUPPORTED so the caller can fall back gracefully.
// Non-timeout transport errors (broken socket) still mark the connection dead.
// Use only for functions that have a viable local-driver fallback path.
CUresult ipc_call_optional(
    uint32_t    func_id,
    const void *req_payload,
    uint32_t    req_payload_len,
    void       *resp_payload,
    uint32_t    resp_payload_max,
    uint32_t   *resp_payload_len_out   // may be NULL
);

// ── Global state (defined in shim_main.c) ────────────────────────────────────

// Set to 1 after successful ipc_connect() in the constructor.
extern volatile int g_ipc_connected;

// Monotonically increasing call ID counter.
extern uint32_t g_call_id;

// Mutex serialising IPC calls (single connection shared by all threads).
extern pthread_mutex_t g_ipc_mutex;

// Number of local CUDA devices detected at shim startup via g_real.cuDeviceGetCount.
// 0 if no local NVIDIA driver is present.  Used by device-routing functions in
// cuda_init.c to map virtual device indices:
//   0 .. g_local_device_count-1  → local real driver
//   g_local_device_count .. N    → remote agent via IPC
extern int g_local_device_count;

static inline uint32_t shim_next_call_id(void) {
    // Protected by g_ipc_mutex in ipc_call(); safe for single-threaded use.
    return ++g_call_id;
}

// ── Logging ───────────────────────────────────────────────────────────────────

void idlegpu_log(const char *level, const char *fmt, ...);

#define SHIM_INFO(fmt, ...)  idlegpu_log("INFO",  fmt, ##__VA_ARGS__)
#define SHIM_WARN(fmt, ...)  idlegpu_log("WARN",  fmt, ##__VA_ARGS__)
#define SHIM_DEBUG(fmt, ...) idlegpu_log("DEBUG", fmt, ##__VA_ARGS__)

// Log a call to an unimplemented function and return NOT_SUPPORTED.
// Used in stubs.c and nvenc.c.
#define SHIM_UNIMPLEMENTED(fn_name) \
    do { \
        idlegpu_log("WARN", "unimplemented: " fn_name \
                    " -- returning CUDA_ERROR_NOT_SUPPORTED"); \
        return CUDA_ERROR_NOT_SUPPORTED; \
    } while (0)

// Return CUDA_ERROR_NOT_INITIALIZED if the IPC connection is down.
// Used in nvenc.c and stubs.c where no real-CUDA fallback exists.
#define SHIM_CHECK_CONNECTED() \
    do { \
        if (!g_ipc_connected) { \
            return CUDA_ERROR_NOT_INITIALIZED; \
        } \
    } while (0)

// ── Real CUDA function pointer table (defined in real_cuda.c) ─────────────────
//
// Loaded at shim startup via RTLD_NEXT.  Used as local-GPU fallback when the
// IPC connection to the sidecar is not available.  All pointers are NULL if
// libcuda.so.1 is not in the process's link chain at construction time.

typedef struct {
    // Tier 1 init / device
    CUresult (*cuInit)(unsigned int flags);
    CUresult (*cuDeviceGet)(CUdevice *device, int ordinal);
    CUresult (*cuDeviceGetCount)(int *count);
    CUresult (*cuDeviceGetName)(char *name, int len, CUdevice dev);
    CUresult (*cuDeviceGetAttribute)(int *pi, CUdevice_attribute attrib, CUdevice dev);
    CUresult (*cuDeviceTotalMem)(size_t *bytes, CUdevice dev);
    CUresult (*cuCtxCreate)(CUcontext *pctx, unsigned int flags, CUdevice dev);
    CUresult (*cuCtxDestroy)(CUcontext ctx);
    CUresult (*cuCtxSetCurrent)(CUcontext ctx);
    CUresult (*cuCtxGetCurrent)(CUcontext *pctx);
    // Tier 1 memory
    CUresult (*cuMemAlloc)(CUdeviceptr *dptr, size_t bytesize);
    CUresult (*cuMemFree)(CUdeviceptr dptr);
    CUresult (*cuMemcpyHtoD)(CUdeviceptr dst, const void *src, size_t n);
    CUresult (*cuMemcpyDtoH)(void *dst, CUdeviceptr src, size_t n);
    CUresult (*cuMemcpyDtoD)(CUdeviceptr dst, CUdeviceptr src, size_t n);
    CUresult (*cuMemsetD8)(CUdeviceptr dst, unsigned char val, size_t n);
    CUresult (*cuMemsetD16)(CUdeviceptr dst, unsigned short val, size_t n);
    CUresult (*cuMemsetD32)(CUdeviceptr dst, unsigned int val, size_t n);
    CUresult (*cuMemGetInfo)(size_t *free, size_t *total);
    // Tier 1 module / kernel
    CUresult (*cuModuleLoad)(CUmodule *mod, const char *fname);
    CUresult (*cuModuleLoadData)(CUmodule *mod, const void *image);
    CUresult (*cuModuleGetFunction)(CUfunction *fn, CUmodule mod, const char *name);
    CUresult (*cuLaunchKernel)(CUfunction f,
                               unsigned int gx, unsigned int gy, unsigned int gz,
                               unsigned int bx, unsigned int by, unsigned int bz,
                               unsigned int sharedMem, CUstream stream,
                               void **params, void **extra);
    CUresult (*cuModuleUnload)(CUmodule mod);
    // Tier 1 stream / sync
    CUresult (*cuStreamCreate)(CUstream *phStream, unsigned int flags);
    CUresult (*cuStreamDestroy)(CUstream stream);
    CUresult (*cuStreamSynchronize)(CUstream stream);
    CUresult (*cuStreamWaitEvent)(CUstream stream, CUevent event, unsigned int flags);
    CUresult (*cuEventCreate)(CUevent *phEvent, unsigned int flags);
    CUresult (*cuEventDestroy)(CUevent event);
    CUresult (*cuEventRecord)(CUevent event, CUstream stream);
    CUresult (*cuEventSynchronize)(CUevent event);
    // Runtime internals with local-driver fallback
    CUresult (*cuGetExportTable)(const void **ppExportTable, const CUuuid *pExportTableId);
    CUresult (*cuDriverGetVersion)(int *driverVersion);
    // Extended device / context — promoted from stubs in Phase 2E
    CUresult (*cuDeviceComputeCapability)(int *major, int *minor, CUdevice dev);
    CUresult (*cuDeviceGetUuid)(CUuuid *uuid, CUdevice dev);
    CUresult (*cuDeviceGetLuid)(char *luid, unsigned int *deviceNodeMask, CUdevice dev);
    CUresult (*cuCtxPushCurrent_v2)(CUcontext ctx);
    CUresult (*cuCtxPopCurrent_v2)(CUcontext *pctx);
    CUresult (*cuCtxSetLimit)(CUlimit limit, size_t value);
    CUresult (*cuCtxGetLimit)(size_t *pvalue, CUlimit limit);
    // Extended memory — pitched 2D and managed allocations
    CUresult (*cuMemAllocPitch_v2)(CUdeviceptr *dptr, size_t *pPitch,
                                   size_t WidthInBytes, size_t Height,
                                   unsigned int ElementSizeBytes);
    CUresult (*cuMemAllocManaged)(CUdeviceptr *dptr, size_t bytesize,
                                  unsigned int flags);
    // Async memset
    CUresult (*cuMemsetD8Async)(CUdeviceptr dst, unsigned char uc,
                                size_t N, CUstream hStream);
    CUresult (*cuMemsetD16Async)(CUdeviceptr dst, unsigned short us,
                                 size_t N, CUstream hStream);
    CUresult (*cuMemsetD32Async)(CUdeviceptr dst, unsigned int ui,
                                 size_t N, CUstream hStream);
    // Async copy
    CUresult (*cuMemcpyDtoDAsync_v2)(CUdeviceptr dstDevice, CUdeviceptr srcDevice,
                                      size_t ByteCount, CUstream hStream);
    // Stream creation with priority
    CUresult (*cuStreamCreateWithPriority)(CUstream *phStream,
                                           unsigned int flags, int priority);
    // Stream query (local-only; not forwarded via IPC)
    CUresult (*cuStreamQuery)(CUstream hStream);
    CUresult (*cuStreamGetCtx)(CUstream hStream, CUcontext *pctx);
    CUresult (*cuStreamGetFlags)(CUstream hStream, unsigned int *flags);
    CUresult (*cuStreamGetPriority)(CUstream hStream, int *priority);
    CUresult (*cuStreamAddCallback)(CUstream hStream, void *callback,
                                    void *userData, unsigned int flags);
    // Event query / timing (local-only; not forwarded via IPC)
    CUresult (*cuEventQuery)(CUevent hEvent);
    CUresult (*cuEventElapsedTime)(float *pMilliseconds,
                                   CUevent hStart, CUevent hEnd);
    // Context introspection
    CUresult (*cuCtxSynchronize)(void);
    CUresult (*cuCtxGetDevice)(CUdevice *device);
    CUresult (*cuCtxGetFlags)(unsigned int *flags);
    CUresult (*cuCtxGetApiVersion)(CUcontext ctx, unsigned int *version);
    CUresult (*cuCtxGetCacheConfig)(int *pconfig);
    CUresult (*cuCtxSetCacheConfig)(int config);
    CUresult (*cuCtxGetSharedMemConfig)(int *pConfig);
    CUresult (*cuCtxSetSharedMemConfig)(int config);
    // Kernel / function attributes
    CUresult (*cuFuncGetAttribute)(int *pi, int attrib, CUfunction hfunc);
    CUresult (*cuFuncSetAttribute)(CUfunction hfunc, int attrib, int value);
    CUresult (*cuFuncSetCacheConfig)(CUfunction hfunc, int config);
    CUresult (*cuFuncSetSharedMemConfig)(CUfunction hfunc, int config);
    // Occupancy
    CUresult (*cuOccupancyMaxActiveBlocksPerMultiprocessor)(
                  int *numBlocks, CUfunction func,
                  int blockSize, size_t dynamicSMemSize);
    CUresult (*cuOccupancyMaxPotentialBlockSize)(
                  int *minGridSize, int *blockSize,
                  CUfunction func, void *blockSizeToDynamicSMemSize,
                  size_t dynamicSMemSize, int blockSizeLimit);
} RealCuda;

// Global real CUDA function pointer table (defined in real_cuda.c).
extern RealCuda g_real;

// Populate g_real from the real libcuda.so.1 via RTLD_NEXT.
// bootstrap_dlsym: the real dlsym() obtained via dlvsym() before our dlsym
//   override is active — avoids infinite recursion.
void real_cuda_init(void *(*bootstrap_dlsym)(void *, const char *));

// If IPC is connected, fall through to the IPC path.
// If IPC is NOT connected, call the real CUDA function (local GPU fallback) if
// available, otherwise return CUDA_ERROR_NOT_INITIALIZED.
#define SHIM_REQUIRE_IPC(fn_ptr, ...) \
    do { \
        if (!g_ipc_connected) { \
            if ((fn_ptr) != NULL) return (fn_ptr)(__VA_ARGS__); \
            return CUDA_ERROR_NOT_INITIALIZED; \
        } \
    } while (0)

// After an ipc_call() that may have timed out and marked the connection dead,
// attempt a local GPU fallback if the function pointer is available.
// Only triggers when g_ipc_connected has just been set to 0 by ipc_mark_dead().
#define SHIM_FALLBACK_IF_DEAD(fn_ptr, ...) \
    do { \
        if (!g_ipc_connected && (fn_ptr) != NULL) { \
            return (fn_ptr)(__VA_ARGS__); \
        } \
    } while (0)
