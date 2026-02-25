# IdleGPU

> Your gaming PC's GPU, put to work while you sleep — securely shared with your self-hosted stack over your local network.

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Status: Early Development](https://img.shields.io/badge/Status-Early%20Development-orange)]()
[![Security: First Class](https://img.shields.io/badge/Security-First%20Class-green)]()

---

## What is IdleGPU?

IdleGPU is an open-source daemon that detects when your gaming PC is idle and dynamically makes its GPU available to your self-hosted server — transparently, securely, and with minimal overhead.

When you sit down to game, the GPU is instantly reclaimed. No reboots, no configuration changes, no interruptions.

It works at the CUDA level, meaning **any application that uses CUDA or NVENC** — AI inference, video encoding, image generation, transcription — gets access to your gaming GPU as if it were locally installed. Containers like Unmanic, Ollama, Jellyfin, and Stable Diffusion work without modification.

---

## How It Works

```
Gaming PC (idle)                        Self-Hosted Server
─────────────────────                   ──────────────────────────────
  IdleGPU Agent                           IdleGPU Broker
  ┌─────────────────┐                     ┌──────────────────────────┐
  │ • GPU util < 10%│                     │ • Agent registry         │
  │ • CPU util < 20%│  mTLS WebSocket     │ • Job routing            │
  │ • No input 5min │ ──────────────────▶ │ • Status dashboard       │
  │                 │                     └────────────┬─────────────┘
  │ CUDA Server     │                                  │
  │ (cricket-based) │ ◀── CUDA calls forwarded ────────┤
  └─────────────────┘                                  │
        │                                   ┌──────────▼─────────────┐
        │ Real GPU executes                 │  IdleGPU Sidecar       │
        │ Results return                    │  (Docker container)    │
        ▼                                   │                        │
   RTX 4090 / etc.                          │  LD_PRELOAD CUDA shim  │
                                            │  intercepts calls from │
                                            │  sibling containers    │
                                            └──────────┬─────────────┘
                                                       │
                                            ┌──────────▼─────────────┐
                                            │  Unmanic / Ollama /    │
                                            │  Jellyfin / SD / etc.  │
                                            │  (zero modification)   │
                                            └────────────────────────┘
```

The gaming PC **never has an open inbound port**. The agent always dials out to the broker. The broker never dials in.

---

## Quick Start

### On your gaming PC (agent)

```bash
pip install idlegpu-agent
idlegpu-agent start --broker broker.yourdomain.local
```

Or as a Windows Service / systemd unit (Phase 3):

```bash
idlegpu-agent install --broker broker.yourdomain.local
```

### On your server (broker + sidecar)

Add the sidecar to any existing compose file:

```yaml
services:
  unmanic:
    image: josh5/unmanic:latest
    depends_on:
      idlegpu:
        condition: service_healthy
    environment:
      - NVIDIA_VISIBLE_DEVICES=all # no changes needed
      - NVIDIA_DRIVER_CAPABILITIES=all

  idlegpu:
    image: idlegpu/sidecar:latest
    environment:
      - IDLEGPU_BROKER=broker.yourdomain.local
      - IDLEGPU_SECRET=${IDLEGPU_SECRET} # from .env
    volumes:
      - idlegpu-lib:/usr/local/lib/idlegpu
    healthcheck:
      test: ["CMD", "idlegpu", "health"]
      interval: 10s
      timeout: 3s
      retries: 3

volumes:
  idlegpu-lib:
```

That is the entire integration. No CUDA drivers needed on the server. No GPU needed on the server. No changes to Unmanic, Ollama, or any other container.

---

## Getting Started: Certificates

IdleGPU uses mutual TLS (mTLS) so both the agent and the broker authenticate each other. Run these steps once before starting the broker in production.

### 1. Generate the CA and broker certificate (on the broker host)

```bash
idlegpu-broker setup
```

This generates a local CA, a broker TLS certificate, and a one-time enrollment token. It automatically writes the `[tls]` paths into the system config file (`/etc/idlegpu/broker.toml` on Linux, `%PROGRAMDATA%\idlegpu\broker.toml` on Windows) — no manual editing required.

It prints:
- The CA certificate **fingerprint** (SHA-256) — share this out-of-band with anyone enrolling an agent
- The exact `idlegpu-agent enroll` command pre-filled with the token and broker hostname
- The enrollment port (plain HTTP, default: main port + 1, e.g. 8766)

### 2. Start the broker (or pass `--start` to setup)

```bash
idlegpu-broker start
# or in one step:
idlegpu-broker setup --start
```

The broker listens on two ports:
- **Main port** (default 8765): mTLS WebSocket and REST API — agents connect here after enrollment
- **Enrollment port** (default 8766): plain HTTP, serves only `/ca.crt` and `/enroll` — used once to bootstrap agent certificates before they hold client credentials

### 3. Enroll each agent (on the gaming PC)

```bash
idlegpu-agent enroll --broker broker.yourdomain.local --token <token>
```

The agent:
1. Fetches the CA certificate from the broker's enrollment port (plain HTTP)
2. Prints the CA fingerprint and prompts you to confirm it matches the one printed by `idlegpu-broker setup`
3. Generates an ECDSA P-256 key pair and CSR
4. Submits the CSR to the broker with the one-time token
5. Saves `ca.crt`, `agent.crt`, and `agent.key` to the agent data directory
6. Automatically writes `[broker]` (host and port) and `[tls]` paths into the agent config file — no manual editing required

The fingerprint confirmation is your trust anchor — it prevents a machine-in-the-middle from substituting its own CA during bootstrap. If the fingerprints don't match, press Enter (default `N`) to abort.

Start the agent (or pass `--start` to enroll):

```bash
idlegpu-agent start
# or in one step:
idlegpu-agent enroll --broker broker.yourdomain.local --token <token> --start
```

### Dev mode (no certificates required)

```bash
idlegpu-broker start --dev
idlegpu-agent start --dev
```

Dev mode uses plaintext `ws://` with no authentication. Never use it on an untrusted network.

---

## Supported Workloads

| Application            | Type                   | Status       |
| ---------------------- | ---------------------- | ------------ |
| Unmanic                | Video encoding (NVENC) | Planned v0.2 |
| Ollama                 | LLM inference          | Planned v0.2 |
| Jellyfin               | Hardware transcoding   | Planned v0.3 |
| Stable Diffusion WebUI | Image generation       | Planned v0.3 |
| Whisper                | Audio transcription    | Planned v0.3 |
| Any CUDA application   | General compute        | Planned v0.3 |

---

## Security

Security is not an afterthought in IdleGPU — it is the primary design constraint. Every architectural decision is made with security first.

### What the agent can access

The agent is a restricted process that only ever reads:

- GPU utilization percentage (read-only, via NVML)
- CPU and RAM utilization (read-only, via psutil)
- Time since last user input (a single integer — not _what_ was typed or clicked)
- Its own designated scratch directory for temporary job files

The agent **cannot** and **does not**:

- Read files on your system
- Observe keyboard or mouse input content
- Capture your screen
- Inspect running processes
- Access your network beyond the broker connection

### Network security

- All agent ↔ broker communication uses **mutual TLS (mTLS)** — both sides present certificates. A rogue device on your network cannot impersonate your broker or inject jobs.
- The gaming PC **never opens an inbound port**. The agent dials out only. Your gaming PC's firewall requires zero changes.
- Job payloads are **signed** with a shared secret so they cannot be tampered with in transit.
- The broker API only binds to your local network interface by default — not exposed to the internet unless you explicitly configure it.

### Container isolation

- All compute workloads run inside Docker containers with `--gpus all` and **no host filesystem access** beyond an explicit scratch volume.
- The sidecar runs as a non-root user.
- No `privileged: true` required anywhere.
- The CUDA shim library is open source and auditable — you can read every line of what is injected into sibling containers.

### Agent process isolation

The agent runs as a dedicated low-privilege service account (`idlegpu`) created at install time. This account has no home directory, no shell access, and can only write to `/var/lib/idlegpu` (Linux) or the designated data directory (Windows).

### Transparency log

Every job the agent accepts is written to a local transparency log:

```
[2025-03-01 03:14:22] Job accepted from broker @ 192.168.1.10
[2025-03-01 03:14:22] Backend: cuda-server
[2025-03-01 03:14:22] Duration: 00:14:33
[2025-03-01 03:28:55] Job completed — GPU released
```

You always know exactly what your GPU was used for and for how long.

### Trust model

| What                          | How                               |
| ----------------------------- | --------------------------------- |
| Agent → Broker authentication | mTLS client certificate           |
| Broker → Agent authentication | mTLS server certificate           |
| Job payload integrity         | HMAC-SHA256 signature             |
| Container filesystem access   | Explicit volume mounts only       |
| Agent system access           | Restricted service account        |
| Inbound attack surface        | Zero — no open ports on gaming PC |

### Verifying it yourself

IdleGPU is fully open source under AGPL-3.0. Every security claim above corresponds to auditable code. If you find a security issue, please see [SECURITY.md](SECURITY.md).

---

## Roadmap

### v0.1 — Foundation (complete)

- [x] Idle detection (GPU %, CPU %, input idle time)
- [x] Agent ↔ Broker WebSocket connection with mTLS
- [x] Broker agent registry and status API
- [x] Transparency log
- [x] Basic CLI (`idlegpu-agent start/stop/status/enroll`)
- [x] Windows and Linux support

### v0.2 — CUDA Forwarding (current)

- [x] mTLS certificate provisioning (`idlegpu-broker setup`, `idlegpu-agent enroll`)
- [ ] CUDA intercept shim (`libidlegpu-cuda.so`)
- [ ] Docker sidecar image
- [ ] FFmpeg/NVENC forwarding for encoding workloads
- [ ] Unmanic integration tested and documented
- [ ] Ollama integration tested and documented
- [ ] Graceful reconnect and job pause on GPU reclaim

### v0.3 — Polish

- [ ] Web dashboard (node status, job history, live utilization)
- [ ] Multi-node support (multiple gaming PCs)
- [ ] Per-app graceful shutdown profiles
- [ ] `docker-compose` plugin for automatic sidecar injection
- [ ] AMD GPU support (ROCm)
- [ ] Automatic mDNS broker discovery (no manual IP config)

### v0.4 — Ecosystem

- [ ] Plugin system for custom compute backends
- [ ] Jellyfin, Plex, Whisper integrations
- [ ] Stable Diffusion WebUI integration
- [ ] Packaged installers (.exe, .deb, .rpm)
- [ ] Helm chart for Kubernetes deployments

---

## Architecture Deep Dive

### Idle Detection

The agent polls three independent signals every 10 seconds. All three must be below their thresholds simultaneously before the GPU is offered:

```
GPU utilization  < 10%   (via NVML — NVIDIA Management Library)
CPU utilization  < 20%   (via psutil)
Input idle time  > 5min  (via GetLastInputInfo on Windows, xprintidle on Linux)
```

When the user returns, activity is detected within 3 seconds and the GPU is reclaimed immediately. Any running CUDA workload receives a graceful shutdown signal before the connection closes.

### CUDA Forwarding

The sidecar injects `libidlegpu-cuda.so` into sibling containers via a shared volume and `LD_PRELOAD`. This library overrides CUDA API entry points and forwards calls over a Unix socket to the sidecar process, which forwards them over the mTLS connection to the agent on the gaming PC. Results travel the same path in reverse.

The application sees a fully functional GPU. CUDA calls execute on real hardware. The only overhead is network round-trip time for API calls — the actual compute runs natively on the gaming PC's GPU.

### Graceful Reclaim

When the gaming PC goes active, the sequence is:

```
1. Agent sends GOING_ACTIVE signal to broker (3 second warning)
2. Broker notifies sidecar
3. Sidecar sends SIGTERM to in-progress CUDA kernels (they finish naturally)
4. New CUDA calls return CUDA_ERROR_NO_DEVICE
5. Application pauses or falls back to CPU gracefully
6. mTLS connection closes cleanly
7. Gaming PC GPU fully released — zero residual load
```

---

## Contributing

IdleGPU is currently not open for general contributions or pull requests. See [CONTRIBUTING.md](CONTRIBUTING.md) for more details.

---

## Security Disclosure

If you find a security vulnerability in IdleGPU, please **do not open a public issue**. See [SECURITY.md](SECURITY.md) for the responsible disclosure process.

---

## License

IdleGPU is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.

This means:

- You can use, modify, and distribute IdleGPU freely
- If you modify IdleGPU and offer it as a hosted service, you must open source your modifications
- The self-hosted path is and will always remain completely free

See [LICENSE](LICENSE) for the full text.

---

## Acknowledgements

IdleGPU builds on ideas from [Cricket](https://github.com/RWTH-AVC/cricket) (CUDA API interception), [rCUDA](http://rcuda.net/) (remote CUDA execution research), and the broader self-hosted community whose frustration with underutilised hardware inspired this project.

---

_IdleGPU is not affiliated with NVIDIA Corporation._
