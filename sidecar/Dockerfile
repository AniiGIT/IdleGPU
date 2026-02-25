# Copyright (C) 2026 AniiGIT
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# sidecar/Dockerfile - IdleGPU sidecar container image.
#
# This container runs alongside a CUDA application container.  It:
#   1. Provides libidlegpu-cuda.so at /opt/idlegpu/lib/libidlegpu-cuda.so
#   2. Runs the sidecar Python process (idlegpu-sidecar)
#
# LD_PRELOAD injection is done at the application container level via
# docker-compose volumes and environment variables — see docker-compose.example.yml.
#
# Build arguments
# ───────────────
#   PYTHON_VERSION   Python version tag (default: 3.11-slim)
#   UID              UID for the in-container idlegpu user (default: 1000).
#                    Set to $(id -u) to match the host user that owns the
#                    volume-mounted cert directory so file permissions align.
#   GID              GID for the in-container idlegpu group (default: 1000).
#                    Set to $(id -g) to match the host group.
#
# Runtime environment variables (all required except IDLEGPU_SOCKET)
# ───────────────────────────────────────────────────────────────────
#   IDLEGPU_BROKER_HOST   broker hostname or IP
#   IDLEGPU_AGENT_ID      agent UUID to pair with
#   IDLEGPU_BROKER_PORT   broker port (default: 8765)
#   IDLEGPU_SOCKET        Unix socket path (default: /var/run/idlegpu/cuda.sock)
#   IDLEGPU_CA_CERT       path to CA cert PEM inside container
#   IDLEGPU_SIDECAR_CERT  path to sidecar client cert PEM inside container
#   IDLEGPU_SIDECAR_KEY   path to sidecar client key PEM inside container
#   IDLEGPU_DEV           "1" for plaintext ws:// dev mode

ARG PYTHON_VERSION=3.11-slim

# ── Stage 1: build the C shim ─────────────────────────────────────────────────
FROM debian:bookworm-slim AS shim-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY shim/ ./shim/

RUN cmake -S shim -B shim/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/idlegpu \
    && cmake --build shim/build --parallel \
    && cmake --install shim/build

# ── Stage 2: Python sidecar runtime ───────────────────────────────────────────
FROM python:${PYTHON_VERSION}

# Copy the compiled shim from stage 1.
COPY --from=shim-builder /opt/idlegpu/lib/libidlegpu-cuda.so /opt/idlegpu/lib/libidlegpu-cuda.so

# Install Python dependencies.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy the sidecar and agent packages (agent needed for frame.py / cuda_driver.py
# is not used in the sidecar, but the package must be present for imports).
WORKDIR /app
COPY agent/      ./agent/
COPY broker/     ./broker/
COPY sidecar/    ./sidecar/
COPY pyproject.toml ./

RUN pip install --no-cache-dir --no-deps -e .

# Create the socket directory with appropriate permissions.
RUN mkdir -p /var/run/idlegpu && chmod 0755 /var/run/idlegpu

# Non-root user — UID/GID default to 1000; pass --build-arg UID=$(id -u)
# GID=$(id -g) to match the host user that owns volume-mounted cert files.
ARG UID=1000
ARG GID=1000
RUN groupadd --system --gid "${GID}" idlegpu \
    && useradd --system --no-create-home --shell /bin/false \
               --uid "${UID}" --gid "${GID}" idlegpu \
    && chown idlegpu:idlegpu /var/run/idlegpu

USER idlegpu

# The sidecar reads its configuration from environment variables.
CMD ["idlegpu-sidecar"]
