// Copyright (C) 2026 AniiGIT
// SPDX-License-Identifier: AGPL-3.0-or-later

/*
 * cuda_compat.h - Minimal CUDA Driver API type definitions.
 *
 * Defines only the types needed by the Tier 1 intercept functions.
 * Compatible with CUDA 12.x headers but does not require the CUDA SDK.
 * Source of truth: CUDA Driver API documentation.
 */

#pragma once

#include <stddef.h>
#include <stdint.h>

// ── Opaque handle types ───────────────────────────────────────────────────────

// Forward-declare structs so handle types are distinct and type-safe.
struct CUctx_st;
struct CUmod_st;
struct CUfunc_st;
struct CUstream_st;
struct CUevent_st;

typedef struct CUctx_st    *CUcontext;
typedef struct CUmod_st    *CUmodule;
typedef struct CUfunc_st   *CUfunction;
typedef struct CUstream_st *CUstream;
typedef struct CUevent_st  *CUevent;

// Device ordinal (integer index into the list of CUDA devices).
typedef int CUdevice;

// 64-bit GPU-side virtual address.
typedef unsigned long long CUdeviceptr;

// ── CUresult ──────────────────────────────────────────────────────────────────

typedef enum CUresult_enum {
    CUDA_SUCCESS                       = 0,
    CUDA_ERROR_INVALID_VALUE           = 1,
    CUDA_ERROR_OUT_OF_MEMORY           = 2,
    CUDA_ERROR_NOT_INITIALIZED         = 3,
    CUDA_ERROR_DEINITIALIZED           = 4,
    CUDA_ERROR_PROFILER_DISABLED       = 5,
    CUDA_ERROR_NO_DEVICE               = 100,
    CUDA_ERROR_INVALID_DEVICE          = 101,
    CUDA_ERROR_INVALID_IMAGE           = 200,
    CUDA_ERROR_INVALID_CONTEXT         = 201,
    CUDA_ERROR_NOT_MAPPED              = 211,
    CUDA_ERROR_INVALID_HANDLE          = 400,
    CUDA_ERROR_NOT_FOUND               = 500,
    CUDA_ERROR_NOT_READY               = 600,
    CUDA_ERROR_LAUNCH_FAILED           = 719,
    CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES = 701,
    CUDA_ERROR_NOT_SUPPORTED           = 801,
    CUDA_ERROR_UNKNOWN                 = 999,
} CUresult;

// ── Context / stream / event flags ────────────────────────────────────────────

typedef enum CUctx_flags_enum {
    CU_CTX_SCHED_AUTO          = 0x00,
    CU_CTX_SCHED_SPIN          = 0x01,
    CU_CTX_SCHED_YIELD         = 0x02,
    CU_CTX_SCHED_BLOCKING_SYNC = 0x04,
    CU_CTX_BLOCKING_SYNC       = 0x04,
    CU_CTX_MAP_HOST            = 0x08,
    CU_CTX_LMEM_RESIZE_TO_MAX  = 0x10,
} CUctx_flags;

#define CU_STREAM_DEFAULT      0x0u
#define CU_STREAM_NON_BLOCKING 0x1u

#define CU_EVENT_DEFAULT        0x0u
#define CU_EVENT_BLOCKING_SYNC  0x1u
#define CU_EVENT_DISABLE_TIMING 0x2u
#define CU_EVENT_INTERPROCESS   0x4u

// ── CUdevice_attribute ────────────────────────────────────────────────────────
//
// Forwarded numerically to the agent; the shim doesn't inspect these values.

typedef enum CUdevice_attribute_enum {
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK    = 1,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X          = 2,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y          = 3,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z          = 4,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X           = 5,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y           = 6,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z           = 7,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8,
    CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY    = 9,
    CU_DEVICE_ATTRIBUTE_WARP_SIZE                = 10,
    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT     = 16,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76,
} CUdevice_attribute;

// ── CUlimit ───────────────────────────────────────────────────────────────────
//
// Forwarded numerically to the agent; the shim doesn't inspect these values.

typedef enum CUlimit_enum {
    CU_LIMIT_STACK_SIZE                       = 0x00,
    CU_LIMIT_PRINTF_FIFO_SIZE                 = 0x01,
    CU_LIMIT_MALLOC_HEAP_SIZE                 = 0x02,
    CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH           = 0x03,
    CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT = 0x04,
    CU_LIMIT_MAX_L2_FETCH_GRANULARITY         = 0x05,
    CU_LIMIT_PERSISTING_L2_CACHE_SIZE         = 0x06,
} CUlimit;

// ── CUuuid ────────────────────────────────────────────────────────────────────
//
// 16-byte UUID used by cuGetExportTable to identify capability tables.

typedef struct { unsigned char bytes[16]; } CUuuid;

// ── cuLaunchKernel extra parameter keys ───────────────────────────────────────

#define CU_LAUNCH_PARAM_END             ((void *)0x00)
#define CU_LAUNCH_PARAM_BUFFER_POINTER  ((void *)0x01)
#define CU_LAUNCH_PARAM_BUFFER_SIZE     ((void *)0x02)
