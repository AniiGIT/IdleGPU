# Copyright (C) 2026 AniiGIT
# SPDX-License-Identifier: AGPL-3.0-or-later

cmake_minimum_required(VERSION 3.20)

project(idlegpu_cuda_shim
    VERSION 0.2.0
    DESCRIPTION "IdleGPU CUDA intercept shim (LD_PRELOAD library)"
    LANGUAGES C
)

# ── Standards ────────────────────────────────────────────────────────────────

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# ── Sources ──────────────────────────────────────────────────────────────────

add_library(idlegpu-cuda SHARED
    src/shim_main.c
    src/ipc.c
    src/real_cuda.c
    src/cuda_init.c
    src/cuda_mem.c
    src/cuda_module.c
    src/cuda_stream.c
    src/nvenc.c
    src/stubs.c
)

target_include_directories(idlegpu-cuda PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# ── Compile options ───────────────────────────────────────────────────────────

target_compile_options(idlegpu-cuda PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Werror=implicit-function-declaration
    -Wno-unused-parameter   # CUDA stubs have intentionally unused params
    -Wno-gnu-zero-variadic-macro-arguments  # ##__VA_ARGS__ GCC extension in logging macros
)

# ── Link libraries ────────────────────────────────────────────────────────────

# -ldl for dlopen/dlsym (optional passthrough to real libcuda.so)
# -lpthread for the IPC serialisation mutex
target_link_libraries(idlegpu-cuda PRIVATE dl pthread)

# ── Shared library properties ─────────────────────────────────────────────────

# Hide all internal symbols; only exported cu*/NVENC symbols are visible.
set_target_properties(idlegpu-cuda PROPERTIES
    C_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    OUTPUT_NAME "idlegpu-cuda"
    PREFIX "lib"
)

# ── Installation ──────────────────────────────────────────────────────────────

include(GNUInstallDirs)
install(TARGETS idlegpu-cuda
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

# ── Build summary ─────────────────────────────────────────────────────────────

message(STATUS "idlegpu-cuda shim v${PROJECT_VERSION}")
message(STATUS "  Build type  : ${CMAKE_BUILD_TYPE}")
message(STATUS "  Install dir : ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
message(STATUS "  Usage       : LD_PRELOAD=libidlegpu-cuda.so <cuda-app>")
