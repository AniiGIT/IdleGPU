# Security Policy

## Our Commitment

Security is the primary design constraint of IdleGPU — not an afterthought. This means we take vulnerability reports seriously, respond promptly, and are committed to full transparency about any issues discovered.

---

## Supported Versions

As IdleGPU is in early development, only the latest release receives security fixes. Once the project reaches a stable v1.0, a formal support window will be defined here.

| Version | Supported |
|---|---|
| Latest | ✅ |
| Older releases | ❌ |

---

## Reporting a Vulnerability

**Please do not open a public GitHub issue for security vulnerabilities.** Public disclosure before a fix is available puts users at risk.

### Preferred method — GitHub Private Vulnerability Reporting

Use GitHub's built-in private reporting tool:

1. Go to the **Security** tab of this repository
2. Click **"Report a vulnerability"**
3. Fill in the details

This keeps the report private between you and the maintainers until a fix is ready.

### What to include

A good report helps us respond faster. Please include as much of the following as possible:

- A clear description of the vulnerability
- The component affected (agent, broker, sidecar, CUDA shim, etc.)
- Steps to reproduce the issue
- The potential impact in your assessment
- Any suggested mitigations if you have them

---

## What to Expect

| Timeline | Action |
|---|---|
| Within 48 hours | Acknowledgement of your report |
| Within 7 days | Initial assessment and severity classification |
| Within 30 days | Fix developed and tested (critical issues may be faster) |
| After fix ships | Public disclosure with credit to the reporter |

We will keep you updated throughout the process. If you do not hear back within 48 hours, please follow up.

---

## Disclosure Policy

IdleGPU follows **coordinated disclosure**. This means:

- We ask that you give us reasonable time to fix the issue before public disclosure
- We will work with you to agree on a disclosure timeline
- We will credit you in the release notes and security advisory unless you prefer to remain anonymous
- We will never take legal action against good-faith security researchers

---

## Scope

The following are in scope for security reports:

- **Agent** — idle detection, CUDA server, broker connection
- **Broker** — agent registry, job routing, API endpoints
- **Sidecar** — CUDA shim, LD_PRELOAD injection, container isolation
- **mTLS implementation** — certificate handling, authentication logic
- **Job signing** — HMAC verification, payload integrity
- **Docker security** — privilege escalation, container escape vectors
- **Network security** — traffic interception, man-in-the-middle vectors

The following are out of scope:

- Vulnerabilities in third-party dependencies (report these upstream)
- Issues requiring physical access to the machine
- Social engineering attacks
- Denial of service against your own infrastructure

---

## Security Design Principles

For transparency, the core security decisions in IdleGPU are:

- The agent runs as a restricted low-privilege service account with no shell access
- The gaming PC never opens inbound ports — the agent always dials out
- All agent ↔ broker traffic uses mutual TLS (mTLS) — both sides authenticate
- Certificates use ECDSA P-256 keys; the local CA and all agent certs are generated
  by the broker via `idlegpu-broker setup` and `idlegpu-agent enroll`
- Enrollment tokens are 32-byte random secrets, single-use, and compared with a
  constant-time digest to prevent timing attacks
- Private key files are written with mode 600 on Linux; the CA private key never
  leaves the broker host
- Job payloads are signed with HMAC-SHA256 — tampering is detectable
- Docker containers run with no host filesystem access beyond explicit scratch volumes
- The CUDA shim is fully open source — every line of what is injected is auditable
- A local transparency log records every job accepted by the agent

---

## Recognition

We maintain a hall of fame for security researchers who have responsibly disclosed vulnerabilities to IdleGPU. Your contribution to the security of this project is genuinely appreciated.

---

*This policy is adapted from the [GitHub Security Advisory](https://docs.github.com/en/code-security/security-advisories) best practices.*
