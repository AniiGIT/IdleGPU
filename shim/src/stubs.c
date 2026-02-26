// Copyright (C) 2026 AniiGIT
// SPDX-License-Identifier: AGPL-3.0-or-later

/*
 * stubs.c - Stubs for all CUDA Driver API functions NOT in the Tier 1 set.
 *
 * Every stub:
 *   1. Logs the function name at WARN level (once per call, always — so
 *      production workloads that hit unimplemented paths are visible).
 *   2. Returns CUDA_ERROR_NOT_SUPPORTED.
 *
 * This ensures that applications which call an unimplemented function get
 * a deterministic, visible error instead of falling through to the real
 * CUDA driver (which would execute on the local GPU, defeating the purpose
 * of the shim).
 *
 * Tier 1 functions (implemented in cuda_init.c, cuda_mem.c,
 * cuda_module.c, cuda_stream.c) are NOT listed here.
 *
 * Organisation: grouped loosely by CUDA Driver API section.  New stubs
 * can be added as Phase 3 Tier 2 functions are promoted to full
 * implementations.
 */

#include <stdlib.h>
#include <string.h>

#include "idlegpu_shim.h"

// ── Context management (non-Tier-1) ──────────────────────────────────────────

__attribute__((visibility("default")))
CUresult cuCtxPopCurrent(CUcontext *pctx) {
    (void)pctx;
    SHIM_UNIMPLEMENTED("cuCtxPopCurrent");
}

__attribute__((visibility("default")))
CUresult cuCtxPopCurrent_v2(CUcontext *pctx) {
    (void)pctx;
    SHIM_UNIMPLEMENTED("cuCtxPopCurrent_v2");
}

__attribute__((visibility("default")))
CUresult cuCtxPushCurrent(CUcontext ctx) {
    (void)ctx;
    SHIM_UNIMPLEMENTED("cuCtxPushCurrent");
}

__attribute__((visibility("default")))
CUresult cuCtxPushCurrent_v2(CUcontext ctx) {
    (void)ctx;
    SHIM_UNIMPLEMENTED("cuCtxPushCurrent_v2");
}

__attribute__((visibility("default")))
CUresult cuCtxSynchronize(void) {
    SHIM_UNIMPLEMENTED("cuCtxSynchronize");
}

__attribute__((visibility("default")))
CUresult cuCtxGetDevice(int *device) {
    (void)device;
    SHIM_UNIMPLEMENTED("cuCtxGetDevice");
}

__attribute__((visibility("default")))
CUresult cuCtxGetApiVersion(CUcontext ctx, unsigned int *version) {
    (void)ctx; (void)version;
    SHIM_UNIMPLEMENTED("cuCtxGetApiVersion");
}

__attribute__((visibility("default")))
CUresult cuCtxGetFlags(unsigned int *flags) {
    (void)flags;
    SHIM_UNIMPLEMENTED("cuCtxGetFlags");
}

__attribute__((visibility("default")))
CUresult cuCtxGetLimit(size_t *pvalue, int limit) {
    (void)pvalue; (void)limit;
    SHIM_UNIMPLEMENTED("cuCtxGetLimit");
}

__attribute__((visibility("default")))
CUresult cuCtxSetLimit(int limit, size_t value) {
    (void)limit; (void)value;
    SHIM_UNIMPLEMENTED("cuCtxSetLimit");
}

__attribute__((visibility("default")))
CUresult cuCtxGetCacheConfig(int *pconfig) {
    (void)pconfig;
    SHIM_UNIMPLEMENTED("cuCtxGetCacheConfig");
}

__attribute__((visibility("default")))
CUresult cuCtxSetCacheConfig(int config) {
    (void)config;
    SHIM_UNIMPLEMENTED("cuCtxSetCacheConfig");
}

__attribute__((visibility("default")))
CUresult cuCtxGetSharedMemConfig(int *pConfig) {
    (void)pConfig;
    SHIM_UNIMPLEMENTED("cuCtxGetSharedMemConfig");
}

__attribute__((visibility("default")))
CUresult cuCtxSetSharedMemConfig(int config) {
    (void)config;
    SHIM_UNIMPLEMENTED("cuCtxSetSharedMemConfig");
}

__attribute__((visibility("default")))
CUresult cuCtxAttach(CUcontext *pctx, unsigned int flags) {
    (void)pctx; (void)flags;
    SHIM_UNIMPLEMENTED("cuCtxAttach");
}

__attribute__((visibility("default")))
CUresult cuCtxDetach(CUcontext ctx) {
    (void)ctx;
    SHIM_UNIMPLEMENTED("cuCtxDetach");
}

// ── Device management (non-Tier-1) ────────────────────────────────────────────

__attribute__((visibility("default")))
CUresult cuDriverGetVersion(int *driverVersion) {
    (void)driverVersion;
    SHIM_UNIMPLEMENTED("cuDriverGetVersion");
}

__attribute__((visibility("default")))
CUresult cuDeviceGetUuid(void *uuid, int dev) {
    (void)uuid; (void)dev;
    SHIM_UNIMPLEMENTED("cuDeviceGetUuid");
}

__attribute__((visibility("default")))
CUresult cuDeviceGetByPCIBusId(int *dev, const char *pciBusId) {
    (void)dev; (void)pciBusId;
    SHIM_UNIMPLEMENTED("cuDeviceGetByPCIBusId");
}

__attribute__((visibility("default")))
CUresult cuDeviceGetPCIBusId(char *pciBusId, int len, int dev) {
    (void)pciBusId; (void)len; (void)dev;
    SHIM_UNIMPLEMENTED("cuDeviceGetPCIBusId");
}

__attribute__((visibility("default")))
CUresult cuDeviceCanAccessPeer(int *canAccessPeer, int dev, int peerDev) {
    (void)canAccessPeer; (void)dev; (void)peerDev;
    SHIM_UNIMPLEMENTED("cuDeviceCanAccessPeer");
}

__attribute__((visibility("default")))
CUresult cuDeviceGetP2PAttribute(int *value, int attrib, int srcDevice, int dstDevice) {
    (void)value; (void)attrib; (void)srcDevice; (void)dstDevice;
    SHIM_UNIMPLEMENTED("cuDeviceGetP2PAttribute");
}

__attribute__((visibility("default")))
CUresult cuDeviceComputeCapability(int *major, int *minor, int dev) {
    (void)major; (void)minor; (void)dev;
    SHIM_UNIMPLEMENTED("cuDeviceComputeCapability");
}

__attribute__((visibility("default")))
CUresult cuDeviceGetProperties(void *prop, int dev) {
    (void)prop; (void)dev;
    SHIM_UNIMPLEMENTED("cuDeviceGetProperties");
}

// ── Memory management (non-Tier-1) ────────────────────────────────────────────

__attribute__((visibility("default")))
CUresult cuMemAllocPitch(CUdeviceptr *dptr, size_t *pPitch,
                         size_t WidthInBytes, size_t Height, unsigned int ElementSizeBytes) {
    (void)dptr; (void)pPitch; (void)WidthInBytes; (void)Height; (void)ElementSizeBytes;
    SHIM_UNIMPLEMENTED("cuMemAllocPitch");
}

__attribute__((visibility("default")))
CUresult cuMemAllocPitch_v2(CUdeviceptr *dptr, size_t *pPitch,
                             size_t WidthInBytes, size_t Height, unsigned int ElementSizeBytes) {
    (void)dptr; (void)pPitch; (void)WidthInBytes; (void)Height; (void)ElementSizeBytes;
    SHIM_UNIMPLEMENTED("cuMemAllocPitch_v2");
}

__attribute__((visibility("default")))
CUresult cuMemAllocHost(void **pp, size_t bytesize) {
    (void)pp; (void)bytesize;
    SHIM_UNIMPLEMENTED("cuMemAllocHost");
}

__attribute__((visibility("default")))
CUresult cuMemAllocHost_v2(void **pp, size_t bytesize) {
    (void)pp; (void)bytesize;
    SHIM_UNIMPLEMENTED("cuMemAllocHost_v2");
}

__attribute__((visibility("default")))
CUresult cuMemFreeHost(void *p) {
    (void)p;
    SHIM_UNIMPLEMENTED("cuMemFreeHost");
}

__attribute__((visibility("default")))
CUresult cuMemAllocManaged(CUdeviceptr *dptr, size_t bytesize, unsigned int flags) {
    (void)dptr; (void)bytesize; (void)flags;
    SHIM_UNIMPLEMENTED("cuMemAllocManaged");
}

__attribute__((visibility("default")))
CUresult cuMemPrefetchAsync(CUdeviceptr devPtr, size_t count,
                            int dstDevice, CUstream hStream) {
    (void)devPtr; (void)count; (void)dstDevice; (void)hStream;
    SHIM_UNIMPLEMENTED("cuMemPrefetchAsync");
}

__attribute__((visibility("default")))
CUresult cuMemAdvise(CUdeviceptr devPtr, size_t count, int advice, int device) {
    (void)devPtr; (void)count; (void)advice; (void)device;
    SHIM_UNIMPLEMENTED("cuMemAdvise");
}

__attribute__((visibility("default")))
CUresult cuMemHostAlloc(void **pp, size_t bytesize, unsigned int Flags) {
    (void)pp; (void)bytesize; (void)Flags;
    SHIM_UNIMPLEMENTED("cuMemHostAlloc");
}

__attribute__((visibility("default")))
CUresult cuMemHostRegister(void *p, size_t bytesize, unsigned int Flags) {
    (void)p; (void)bytesize; (void)Flags;
    SHIM_UNIMPLEMENTED("cuMemHostRegister");
}

__attribute__((visibility("default")))
CUresult cuMemHostRegister_v2(void *p, size_t bytesize, unsigned int Flags) {
    (void)p; (void)bytesize; (void)Flags;
    SHIM_UNIMPLEMENTED("cuMemHostRegister_v2");
}

__attribute__((visibility("default")))
CUresult cuMemHostUnregister(void *p) {
    (void)p;
    SHIM_UNIMPLEMENTED("cuMemHostUnregister");
}

__attribute__((visibility("default")))
CUresult cuMemHostGetDevicePointer(CUdeviceptr *pdptr, void *p, unsigned int Flags) {
    (void)pdptr; (void)p; (void)Flags;
    SHIM_UNIMPLEMENTED("cuMemHostGetDevicePointer");
}

__attribute__((visibility("default")))
CUresult cuMemHostGetDevicePointer_v2(CUdeviceptr *pdptr, void *p, unsigned int Flags) {
    (void)pdptr; (void)p; (void)Flags;
    SHIM_UNIMPLEMENTED("cuMemHostGetDevicePointer_v2");
}

__attribute__((visibility("default")))
CUresult cuMemHostGetFlags(unsigned int *pFlags, void *p) {
    (void)pFlags; (void)p;
    SHIM_UNIMPLEMENTED("cuMemHostGetFlags");
}

__attribute__((visibility("default")))
CUresult cuIpcGetMemHandle(void *pHandle, CUdeviceptr dptr) {
    (void)pHandle; (void)dptr;
    SHIM_UNIMPLEMENTED("cuIpcGetMemHandle");
}

__attribute__((visibility("default")))
CUresult cuIpcOpenMemHandle(CUdeviceptr *pdptr, void *handle, unsigned int Flags) {
    (void)pdptr; (void)handle; (void)Flags;
    SHIM_UNIMPLEMENTED("cuIpcOpenMemHandle");
}

__attribute__((visibility("default")))
CUresult cuIpcCloseMemHandle(CUdeviceptr dptr) {
    (void)dptr;
    SHIM_UNIMPLEMENTED("cuIpcCloseMemHandle");
}

__attribute__((visibility("default")))
CUresult cuMemGetAddressRange(CUdeviceptr *pbase, size_t *psize, CUdeviceptr dptr) {
    (void)pbase; (void)psize; (void)dptr;
    SHIM_UNIMPLEMENTED("cuMemGetAddressRange");
}

__attribute__((visibility("default")))
CUresult cuMemGetAddressRange_v2(CUdeviceptr *pbase, size_t *psize, CUdeviceptr dptr) {
    (void)pbase; (void)psize; (void)dptr;
    SHIM_UNIMPLEMENTED("cuMemGetAddressRange_v2");
}

__attribute__((visibility("default")))
CUresult cuMemcpyHtoDAsync(CUdeviceptr dstDevice, const void *srcHost,
                           size_t ByteCount, CUstream hStream) {
    (void)dstDevice; (void)srcHost; (void)ByteCount; (void)hStream;
    SHIM_UNIMPLEMENTED("cuMemcpyHtoDAsync");
}

__attribute__((visibility("default")))
CUresult cuMemcpyHtoDAsync_v2(CUdeviceptr dstDevice, const void *srcHost,
                               size_t ByteCount, CUstream hStream) {
    (void)dstDevice; (void)srcHost; (void)ByteCount; (void)hStream;
    SHIM_UNIMPLEMENTED("cuMemcpyHtoDAsync_v2");
}

__attribute__((visibility("default")))
CUresult cuMemcpyDtoHAsync(void *dstHost, CUdeviceptr srcDevice,
                           size_t ByteCount, CUstream hStream) {
    (void)dstHost; (void)srcDevice; (void)ByteCount; (void)hStream;
    SHIM_UNIMPLEMENTED("cuMemcpyDtoHAsync");
}

__attribute__((visibility("default")))
CUresult cuMemcpyDtoHAsync_v2(void *dstHost, CUdeviceptr srcDevice,
                               size_t ByteCount, CUstream hStream) {
    (void)dstHost; (void)srcDevice; (void)ByteCount; (void)hStream;
    SHIM_UNIMPLEMENTED("cuMemcpyDtoHAsync_v2");
}

__attribute__((visibility("default")))
CUresult cuMemcpyDtoDAsync(CUdeviceptr dstDevice, CUdeviceptr srcDevice,
                           size_t ByteCount, CUstream hStream) {
    (void)dstDevice; (void)srcDevice; (void)ByteCount; (void)hStream;
    SHIM_UNIMPLEMENTED("cuMemcpyDtoDAsync");
}

__attribute__((visibility("default")))
CUresult cuMemcpyDtoDAsync_v2(CUdeviceptr dstDevice, CUdeviceptr srcDevice,
                               size_t ByteCount, CUstream hStream) {
    (void)dstDevice; (void)srcDevice; (void)ByteCount; (void)hStream;
    SHIM_UNIMPLEMENTED("cuMemcpyDtoDAsync_v2");
}

__attribute__((visibility("default")))
CUresult cuMemcpy(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount) {
    (void)dst; (void)src; (void)ByteCount;
    SHIM_UNIMPLEMENTED("cuMemcpy");
}

__attribute__((visibility("default")))
CUresult cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src,
                       size_t ByteCount, CUstream hStream) {
    (void)dst; (void)src; (void)ByteCount; (void)hStream;
    SHIM_UNIMPLEMENTED("cuMemcpyAsync");
}

__attribute__((visibility("default")))
CUresult cuMemcpyPeer(CUdeviceptr dstDevice, CUcontext dstContext,
                      CUdeviceptr srcDevice, CUcontext srcContext, size_t ByteCount) {
    (void)dstDevice; (void)dstContext; (void)srcDevice; (void)srcContext; (void)ByteCount;
    SHIM_UNIMPLEMENTED("cuMemcpyPeer");
}

__attribute__((visibility("default")))
CUresult cuMemcpyPeerAsync(CUdeviceptr dstDevice, CUcontext dstContext,
                           CUdeviceptr srcDevice, CUcontext srcContext,
                           size_t ByteCount, CUstream hStream) {
    (void)dstDevice; (void)dstContext; (void)srcDevice; (void)srcContext;
    (void)ByteCount; (void)hStream;
    SHIM_UNIMPLEMENTED("cuMemcpyPeerAsync");
}

__attribute__((visibility("default")))
CUresult cuMemsetD8Async(CUdeviceptr dstDevice, unsigned char uc,
                         size_t N, CUstream hStream) {
    (void)dstDevice; (void)uc; (void)N; (void)hStream;
    SHIM_UNIMPLEMENTED("cuMemsetD8Async");
}

__attribute__((visibility("default")))
CUresult cuMemsetD16Async(CUdeviceptr dstDevice, unsigned short us,
                          size_t N, CUstream hStream) {
    (void)dstDevice; (void)us; (void)N; (void)hStream;
    SHIM_UNIMPLEMENTED("cuMemsetD16Async");
}

__attribute__((visibility("default")))
CUresult cuMemsetD32Async(CUdeviceptr dstDevice, unsigned int ui,
                          size_t N, CUstream hStream) {
    (void)dstDevice; (void)ui; (void)N; (void)hStream;
    SHIM_UNIMPLEMENTED("cuMemsetD32Async");
}

__attribute__((visibility("default")))
CUresult cuMemset2D(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui,
                    size_t Width, size_t Height) {
    (void)dstDevice; (void)dstPitch; (void)ui; (void)Width; (void)Height;
    SHIM_UNIMPLEMENTED("cuMemset2D");
}

__attribute__((visibility("default")))
CUresult cuMemset2DAsync(CUdeviceptr dstDevice, size_t dstPitch, unsigned int ui,
                         size_t Width, size_t Height, CUstream hStream) {
    (void)dstDevice; (void)dstPitch; (void)ui; (void)Width; (void)Height; (void)hStream;
    SHIM_UNIMPLEMENTED("cuMemset2DAsync");
}

// ── Module management (non-Tier-1) ────────────────────────────────────────────

__attribute__((visibility("default")))
CUresult cuModuleGetGlobal(CUdeviceptr *dptr, size_t *bytes,
                           CUmodule hmod, const char *name) {
    (void)dptr; (void)bytes; (void)hmod; (void)name;
    SHIM_UNIMPLEMENTED("cuModuleGetGlobal");
}

__attribute__((visibility("default")))
CUresult cuModuleGetGlobal_v2(CUdeviceptr *dptr, size_t *bytes,
                               CUmodule hmod, const char *name) {
    (void)dptr; (void)bytes; (void)hmod; (void)name;
    SHIM_UNIMPLEMENTED("cuModuleGetGlobal_v2");
}

__attribute__((visibility("default")))
CUresult cuModuleGetTexRef(void **pTexRef, CUmodule hmod, const char *name) {
    (void)pTexRef; (void)hmod; (void)name;
    SHIM_UNIMPLEMENTED("cuModuleGetTexRef");
}

__attribute__((visibility("default")))
CUresult cuModuleGetSurfRef(void **pSurfRef, CUmodule hmod, const char *name) {
    (void)pSurfRef; (void)hmod; (void)name;
    SHIM_UNIMPLEMENTED("cuModuleGetSurfRef");
}

__attribute__((visibility("default")))
CUresult cuModuleLoadDataEx(CUmodule *module, const void *image,
                             unsigned int numOptions, int *options,
                             void **optionValues) {
    (void)module; (void)image; (void)numOptions; (void)options; (void)optionValues;
    SHIM_UNIMPLEMENTED("cuModuleLoadDataEx");
}

__attribute__((visibility("default")))
CUresult cuModuleLoadFatBinary(CUmodule *module, const void *fatCubin) {
    (void)module; (void)fatCubin;
    SHIM_UNIMPLEMENTED("cuModuleLoadFatBinary");
}

__attribute__((visibility("default")))
CUresult cuLinkCreate(unsigned int numOptions, int *options,
                      void **optionValues, void **stateOut) {
    (void)numOptions; (void)options; (void)optionValues; (void)stateOut;
    SHIM_UNIMPLEMENTED("cuLinkCreate");
}

__attribute__((visibility("default")))
CUresult cuLinkAddData(void *state, int type, void *data, size_t size,
                       const char *name, unsigned int numOptions,
                       int *options, void **optionValues) {
    (void)state; (void)type; (void)data; (void)size; (void)name;
    (void)numOptions; (void)options; (void)optionValues;
    SHIM_UNIMPLEMENTED("cuLinkAddData");
}

__attribute__((visibility("default")))
CUresult cuLinkAddFile(void *state, int type, const char *path,
                       unsigned int numOptions, int *options, void **optionValues) {
    (void)state; (void)type; (void)path; (void)numOptions; (void)options; (void)optionValues;
    SHIM_UNIMPLEMENTED("cuLinkAddFile");
}

__attribute__((visibility("default")))
CUresult cuLinkComplete(void *state, void **cubinOut, size_t *sizeOut) {
    (void)state; (void)cubinOut; (void)sizeOut;
    SHIM_UNIMPLEMENTED("cuLinkComplete");
}

__attribute__((visibility("default")))
CUresult cuLinkDestroy(void *state) {
    (void)state;
    SHIM_UNIMPLEMENTED("cuLinkDestroy");
}

// ── Kernel / function attributes (non-Tier-1) ─────────────────────────────────

__attribute__((visibility("default")))
CUresult cuFuncGetAttribute(int *pi, int attrib, CUfunction hfunc) {
    (void)pi; (void)attrib; (void)hfunc;
    SHIM_UNIMPLEMENTED("cuFuncGetAttribute");
}

__attribute__((visibility("default")))
CUresult cuFuncSetAttribute(CUfunction hfunc, int attrib, int value) {
    (void)hfunc; (void)attrib; (void)value;
    SHIM_UNIMPLEMENTED("cuFuncSetAttribute");
}

__attribute__((visibility("default")))
CUresult cuFuncSetCacheConfig(CUfunction hfunc, int config) {
    (void)hfunc; (void)config;
    SHIM_UNIMPLEMENTED("cuFuncSetCacheConfig");
}

__attribute__((visibility("default")))
CUresult cuFuncSetSharedMemConfig(CUfunction hfunc, int config) {
    (void)hfunc; (void)config;
    SHIM_UNIMPLEMENTED("cuFuncSetSharedMemConfig");
}

__attribute__((visibility("default")))
CUresult cuParamSetSize(CUfunction hfunc, unsigned int numbytes) {
    (void)hfunc; (void)numbytes;
    SHIM_UNIMPLEMENTED("cuParamSetSize");
}

__attribute__((visibility("default")))
CUresult cuParamSetv(CUfunction hfunc, int offset, void *ptr, unsigned int numbytes) {
    (void)hfunc; (void)offset; (void)ptr; (void)numbytes;
    SHIM_UNIMPLEMENTED("cuParamSetv");
}

__attribute__((visibility("default")))
CUresult cuLaunch(CUfunction f) {
    (void)f;
    SHIM_UNIMPLEMENTED("cuLaunch");
}

__attribute__((visibility("default")))
CUresult cuLaunchGrid(CUfunction f, int grid_width, int grid_height) {
    (void)f; (void)grid_width; (void)grid_height;
    SHIM_UNIMPLEMENTED("cuLaunchGrid");
}

__attribute__((visibility("default")))
CUresult cuLaunchCooperativeKernel(CUfunction f,
                                   unsigned int gridDimX, unsigned int gridDimY,
                                   unsigned int gridDimZ,
                                   unsigned int blockDimX, unsigned int blockDimY,
                                   unsigned int blockDimZ,
                                   unsigned int sharedMemBytes, CUstream hStream,
                                   void **kernelParams) {
    (void)f; (void)gridDimX; (void)gridDimY; (void)gridDimZ;
    (void)blockDimX; (void)blockDimY; (void)blockDimZ;
    (void)sharedMemBytes; (void)hStream; (void)kernelParams;
    SHIM_UNIMPLEMENTED("cuLaunchCooperativeKernel");
}

__attribute__((visibility("default")))
CUresult cuLaunchHostFunc(CUstream hStream, void (*fn)(void *), void *userData) {
    (void)hStream; (void)fn; (void)userData;
    SHIM_UNIMPLEMENTED("cuLaunchHostFunc");
}

__attribute__((visibility("default")))
CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, CUfunction func,
                                                      int blockSize, size_t dynamicSMemSize) {
    (void)numBlocks; (void)func; (void)blockSize; (void)dynamicSMemSize;
    SHIM_UNIMPLEMENTED("cuOccupancyMaxActiveBlocksPerMultiprocessor");
}

__attribute__((visibility("default")))
CUresult cuOccupancyMaxPotentialBlockSize(int *minGridSize, int *blockSize,
                                          CUfunction func, void *blockSizeToDynamicSMemSize,
                                          size_t dynamicSMemSize, int blockSizeLimit) {
    (void)minGridSize; (void)blockSize; (void)func;
    (void)blockSizeToDynamicSMemSize; (void)dynamicSMemSize; (void)blockSizeLimit;
    SHIM_UNIMPLEMENTED("cuOccupancyMaxPotentialBlockSize");
}

// ── Streams (non-Tier-1) ──────────────────────────────────────────────────────

__attribute__((visibility("default")))
CUresult cuStreamAddCallback(CUstream hStream, void *callback,
                             void *userData, unsigned int flags) {
    (void)hStream; (void)callback; (void)userData; (void)flags;
    SHIM_UNIMPLEMENTED("cuStreamAddCallback");
}

__attribute__((visibility("default")))
CUresult cuStreamAttachMemAsync(CUstream hStream, CUdeviceptr dptr,
                                size_t length, unsigned int flags) {
    (void)hStream; (void)dptr; (void)length; (void)flags;
    SHIM_UNIMPLEMENTED("cuStreamAttachMemAsync");
}

__attribute__((visibility("default")))
CUresult cuStreamQuery(CUstream hStream) {
    (void)hStream;
    SHIM_UNIMPLEMENTED("cuStreamQuery");
}

__attribute__((visibility("default")))
CUresult cuStreamGetCtx(CUstream hStream, CUcontext *pctx) {
    (void)hStream; (void)pctx;
    SHIM_UNIMPLEMENTED("cuStreamGetCtx");
}

__attribute__((visibility("default")))
CUresult cuStreamGetFlags(CUstream hStream, unsigned int *flags) {
    (void)hStream; (void)flags;
    SHIM_UNIMPLEMENTED("cuStreamGetFlags");
}

__attribute__((visibility("default")))
CUresult cuStreamGetPriority(CUstream hStream, int *priority) {
    (void)hStream; (void)priority;
    SHIM_UNIMPLEMENTED("cuStreamGetPriority");
}

__attribute__((visibility("default")))
CUresult cuStreamCreateWithPriority(CUstream *phStream, unsigned int flags, int priority) {
    (void)phStream; (void)flags; (void)priority;
    SHIM_UNIMPLEMENTED("cuStreamCreateWithPriority");
}

// ── Events (non-Tier-1) ───────────────────────────────────────────────────────

__attribute__((visibility("default")))
CUresult cuEventQuery(CUevent hEvent) {
    (void)hEvent;
    SHIM_UNIMPLEMENTED("cuEventQuery");
}

__attribute__((visibility("default")))
CUresult cuEventElapsedTime(float *pMilliseconds, CUevent hStart, CUevent hEnd) {
    (void)pMilliseconds; (void)hStart; (void)hEnd;
    SHIM_UNIMPLEMENTED("cuEventElapsedTime");
}

__attribute__((visibility("default")))
CUresult cuIpcGetEventHandle(void *pHandle, CUevent event) {
    (void)pHandle; (void)event;
    SHIM_UNIMPLEMENTED("cuIpcGetEventHandle");
}

__attribute__((visibility("default")))
CUresult cuIpcOpenEventHandle(CUevent *phEvent, void *handle) {
    (void)phEvent; (void)handle;
    SHIM_UNIMPLEMENTED("cuIpcOpenEventHandle");
}

// ── Peer access ───────────────────────────────────────────────────────────────

__attribute__((visibility("default")))
CUresult cuCtxEnablePeerAccess(CUcontext peerContext, unsigned int Flags) {
    (void)peerContext; (void)Flags;
    SHIM_UNIMPLEMENTED("cuCtxEnablePeerAccess");
}

__attribute__((visibility("default")))
CUresult cuCtxDisablePeerAccess(CUcontext peerContext) {
    (void)peerContext;
    SHIM_UNIMPLEMENTED("cuCtxDisablePeerAccess");
}

// ── CUDA Graph API ────────────────────────────────────────────────────────────

__attribute__((visibility("default")))
CUresult cuGraphCreate(void **phGraph, unsigned int flags) {
    (void)phGraph; (void)flags;
    SHIM_UNIMPLEMENTED("cuGraphCreate");
}

__attribute__((visibility("default")))
CUresult cuGraphLaunch(void *hGraphExec, CUstream hStream) {
    (void)hGraphExec; (void)hStream;
    SHIM_UNIMPLEMENTED("cuGraphLaunch");
}

__attribute__((visibility("default")))
CUresult cuGraphExecDestroy(void *hGraphExec) {
    (void)hGraphExec;
    SHIM_UNIMPLEMENTED("cuGraphExecDestroy");
}

__attribute__((visibility("default")))
CUresult cuGraphDestroy(void *hGraph) {
    (void)hGraph;
    SHIM_UNIMPLEMENTED("cuGraphDestroy");
}

__attribute__((visibility("default")))
CUresult cuGraphInstantiate(void **phGraphExec, void *hGraph,
                             void **phErrorNode, char *logBuffer, size_t bufferSize) {
    (void)phGraphExec; (void)hGraph; (void)phErrorNode; (void)logBuffer; (void)bufferSize;
    SHIM_UNIMPLEMENTED("cuGraphInstantiate");
}

__attribute__((visibility("default")))
CUresult cuGraphAddKernelNode(void **phGraphNode, void *hGraph,
                               const void *const *dependencies,
                               size_t numDependencies, const void *nodeParams) {
    (void)phGraphNode; (void)hGraph; (void)dependencies; (void)numDependencies; (void)nodeParams;
    SHIM_UNIMPLEMENTED("cuGraphAddKernelNode");
}

__attribute__((visibility("default")))
CUresult cuGraphAddMemcpyNode(void **phGraphNode, void *hGraph,
                               const void *const *dependencies,
                               size_t numDependencies, const void *copyParams,
                               CUcontext ctx) {
    (void)phGraphNode; (void)hGraph; (void)dependencies; (void)numDependencies;
    (void)copyParams; (void)ctx;
    SHIM_UNIMPLEMENTED("cuGraphAddMemcpyNode");
}

__attribute__((visibility("default")))
CUresult cuGraphAddMemsetNode(void **phGraphNode, void *hGraph,
                               const void *const *dependencies,
                               size_t numDependencies, const void *memsetParams,
                               CUcontext ctx) {
    (void)phGraphNode; (void)hGraph; (void)dependencies; (void)numDependencies;
    (void)memsetParams; (void)ctx;
    SHIM_UNIMPLEMENTED("cuGraphAddMemsetNode");
}

// ── Error reporting ───────────────────────────────────────────────────────────

__attribute__((visibility("default")))
CUresult cuGetErrorName(CUresult error, const char **pStr) {
    (void)error; (void)pStr;
    SHIM_UNIMPLEMENTED("cuGetErrorName");
}

__attribute__((visibility("default")))
CUresult cuGetErrorString(CUresult error, const char **pStr) {
    (void)error; (void)pStr;
    SHIM_UNIMPLEMENTED("cuGetErrorString");
}

// ── Runtime internals ─────────────────────────────────────────────────────────

// cuGetExportTable — full IPC forwarding implementation.
//
// The CUDA runtime and applications such as ffmpeg call this function at
// startup to discover internal driver capability tables.  Returning
// CUDA_ERROR_NOT_SUPPORTED causes the runtime to abort GPU detection, so
// we must forward the call to the agent and return the real table bytes.
//
// The agent serialises the export table as raw 64-bit function pointer
// values from its own address space.  The shim malloc's a persistent
// buffer, writes the received pointer values into it, and sets
// *ppExportTable to that buffer.  The pointers are agent-side virtual
// addresses and are NOT callable in the sidecar process — they exist
// solely to satisfy non-NULL presence checks in the CUDA runtime.
__attribute__((visibility("default")))
CUresult cuGetExportTable(const void **ppExportTable, const CUuuid *pExportTableId) {
    if (ppExportTable == NULL || pExportTableId == NULL) {
        return CUDA_ERROR_INVALID_VALUE;
    }

    if (!g_ipc_connected) {
        // No local CUDA driver fallback for export tables — the sidecar
        // container has no direct GPU access.
        return CUDA_ERROR_NOT_SUPPORTED;
    }

    // Build request: pack the 16-byte UUID.
    Req_cuGetExportTable req;
    memcpy(req.uuid, pExportTableId->bytes, 16);

    // Response buffer: fixed header + up to EXPORT_TABLE_MAX_ENTRIES pointers.
    uint32_t resp_max = (uint32_t)(sizeof(Resp_cuGetExportTable)
                                   + EXPORT_TABLE_MAX_ENTRIES * sizeof(uint64_t));
    uint8_t resp_buf[sizeof(Resp_cuGetExportTable)
                     + EXPORT_TABLE_MAX_ENTRIES * sizeof(uint64_t)];

    uint32_t resp_len = 0;
    CUresult r = ipc_call(FN_cuGetExportTable, &req, sizeof(req),
                          resp_buf, resp_max, &resp_len);
    if (r != CUDA_SUCCESS) {
        return r;
    }

    if (resp_len < sizeof(Resp_cuGetExportTable)) {
        SHIM_WARN("cuGetExportTable: truncated response (%u bytes)", resp_len);
        return CUDA_ERROR_UNKNOWN;
    }

    Resp_cuGetExportTable hdr;
    memcpy(&hdr, resp_buf, sizeof(hdr));
    uint32_t entry_count = hdr.entry_count;

    if (entry_count == 0) {
        // Agent returned an empty table: driver supports the UUID but has
        // no entries.  Return a non-NULL sentinel so the runtime sees success.
        *ppExportTable = (const void *)1;
        return CUDA_SUCCESS;
    }

    uint32_t table_bytes = entry_count * (uint32_t)sizeof(uint64_t);
    if (resp_len < sizeof(Resp_cuGetExportTable) + table_bytes) {
        SHIM_WARN("cuGetExportTable: response too short for %u entries", entry_count);
        return CUDA_ERROR_UNKNOWN;
    }

    // Allocate a persistent buffer for the table.  Intentionally never freed
    // — export tables are expected to live for the duration of the process.
    void *table = malloc(table_bytes);
    if (table == NULL) {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }

    memcpy(table, resp_buf + sizeof(Resp_cuGetExportTable), table_bytes);
    *ppExportTable = table;

    SHIM_DEBUG("cuGetExportTable: forwarded %u entries for UUID %02x%02x…",
               entry_count,
               pExportTableId->bytes[0], pExportTableId->bytes[1]);
    return CUDA_SUCCESS;
}

// ── Profiler ──────────────────────────────────────────────────────────────────

__attribute__((visibility("default")))
CUresult cuProfilerInitialize(const char *configFile, const char *outputFile, int outputMode) {
    (void)configFile; (void)outputFile; (void)outputMode;
    SHIM_UNIMPLEMENTED("cuProfilerInitialize");
}

__attribute__((visibility("default")))
CUresult cuProfilerStart(void) {
    SHIM_UNIMPLEMENTED("cuProfilerStart");
}

__attribute__((visibility("default")))
CUresult cuProfilerStop(void) {
    SHIM_UNIMPLEMENTED("cuProfilerStop");
}
