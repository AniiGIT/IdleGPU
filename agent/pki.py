# Copyright (C) 2026 AniiGIT
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
agent/pki.py - PKI utilities for the IdleGPU agent.

Handles agent-side certificate operations:
  - Generating an ECDSA P-256 keypair and Certificate Signing Request (CSR)
  - Enrolling with the broker over HTTP to obtain a signed certificate
  - Loading the mTLS SSLContext used for wss:// connections
  - Checking certificate expiry so the agent can warn before a cert expires

Enrollment is a two-step HTTP bootstrap (plaintext) that runs once:
  1. Fetch the broker's CA certificate via fetch_ca_cert() from the enrollment
     port (plain HTTP, enroll_port = main_port + 1 by default).
  2. Present the fingerprint to the operator for out-of-band verification.
  3. POST the agent's CSR with the one-time token via enroll(), which receives
     the pre-fetched ca_cert_pem so it can persist the CA cert alongside the
     newly signed agent certificate.

The fetch and interactive fingerprint confirmation are the caller's
responsibility (see agent/__main__.py cmd_enroll). This keeps PKI functions
pure and testable without interactive I/O.

After enrollment all communication uses mTLS over wss://.
"""

from __future__ import annotations

import json
import logging
import os
import ssl
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.ec import SECP256R1, generate_private_key
from cryptography.x509.oid import NameOID

from .config import TlsSection

# Plain str prevents static analysers from narrowing platform branches.
_OS_NAME: str = os.name

logger = logging.getLogger(__name__)

# Certificate expiry warning threshold (days).
EXPIRY_WARN_DAYS = 30


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _write_restricted(path: Path, data: bytes) -> None:
    """Write *data* to *path* and restrict permissions to owner-only on Linux."""
    path.write_bytes(data)
    if _OS_NAME == "posix":
        path.chmod(0o600)


def _check_permissions(path: Path) -> None:
    """
    Refuse to continue if a cert file is world-readable on Linux.

    Raises PermissionError with a clear remediation message.
    On Windows the permission model is different -- log a reminder instead.
    """
    if _OS_NAME == "posix":
        mode = path.stat().st_mode
        if mode & 0o177:  # any group or other bits set
            raise PermissionError(
                f"cert file {path} is too permissive "
                f"(mode {oct(mode & 0o777)}). "
                f"Run: chmod 600 {path}"
            )
    else:
        logger.debug(
            "permission check skipped on non-POSIX platform for %s", path
        )


# ---------------------------------------------------------------------------
# Keypair and CSR generation
# ---------------------------------------------------------------------------


def generate_agent_keypair_and_csr(agent_id: str) -> tuple[bytes, bytes]:
    """
    Generate an ECDSA P-256 private key and a Certificate Signing Request.

    The CSR subject is:
      O=IdleGPU, CN=idlegpu-agent-<agent_id>

    Returns (key_pem, csr_pem) as bytes. Neither is written to disk here --
    the caller handles persistence.
    """
    key = generate_private_key(SECP256R1())

    csr = (
        x509.CertificateSigningRequestBuilder()
        .subject_name(x509.Name([
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "IdleGPU"),
            x509.NameAttribute(NameOID.COMMON_NAME, f"idlegpu-agent-{agent_id}"),
        ]))
        .sign(key, hashes.SHA256())
    )

    key_pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    csr_pem = csr.public_bytes(serialization.Encoding.PEM)

    return key_pem, csr_pem


# ---------------------------------------------------------------------------
# Enrollment helpers
# ---------------------------------------------------------------------------


def fetch_ca_cert(broker_host: str, enroll_port: int) -> bytes:
    """
    Fetch the broker's CA certificate from the plain HTTP enrollment port.

    Returns the PEM-encoded CA certificate bytes.
    Raises urllib.error.URLError on network failure.
    """
    url = f"http://{broker_host}:{enroll_port}/ca.crt"
    logger.info("fetching CA cert from %s", url)
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            return resp.read()
    except urllib.error.URLError as exc:
        raise urllib.error.URLError(
            f"could not reach broker enrollment port at {url}: {exc.reason}"
        ) from exc


def cert_fingerprint(cert_pem: bytes) -> str:
    """
    Return the SHA-256 fingerprint of a PEM certificate as a colon-separated
    hex string, e.g. ``AA:BB:CC:...``.
    """
    cert = x509.load_pem_x509_certificate(cert_pem)
    return cert.fingerprint(hashes.SHA256()).hex(":")


# ---------------------------------------------------------------------------
# Enrollment
# ---------------------------------------------------------------------------


def enroll(
    broker_host: str,
    enroll_port: int,
    token: str,
    agent_id: str,
    data_dir: Path,
    ca_cert_pem: bytes,
) -> None:
    """
    Enroll this agent with the broker and install certificate files.

    The caller is responsible for fetching the CA cert (via fetch_ca_cert())
    and confirming the fingerprint with the operator before calling this.

    Steps:
      1. Generate an ECDSA P-256 keypair and CSR for this agent_id
      2. POST the CSR and token to http://broker:enroll_port/enroll
      3. Save ca.crt, agent.crt, and agent.key to data_dir with mode 600

    Raises urllib.error.URLError on network failure.
    Raises RuntimeError on broker-reported errors (bad token, invalid CSR, etc.).
    """
    data_dir.mkdir(parents=True, exist_ok=True)

    base_url = f"http://{broker_host}:{enroll_port}"

    # Step 1: generate agent keypair and CSR.
    logger.info("generating ECDSA P-256 keypair for agent_id=%s", agent_id)
    key_pem, csr_pem = generate_agent_keypair_and_csr(agent_id)

    # Step 2: POST CSR to broker enrollment endpoint.
    logger.info("submitting CSR to %s/enroll", base_url)
    payload = json.dumps({
        "token": token,
        "agent_id": agent_id,
        "csr_pem": csr_pem.decode("ascii"),
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{base_url}/enroll",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"enrollment rejected by broker (HTTP {exc.code}): {body}"
        ) from exc

    agent_cert_pem = result.get("agent_cert_pem", "").encode("ascii")
    if not agent_cert_pem:
        raise RuntimeError("broker returned empty agent_cert_pem")

    # Step 3: persist certificate files.
    ca_path = data_dir / "ca.crt"
    cert_path = data_dir / "agent.crt"
    key_path = data_dir / "agent.key"

    ca_path.write_bytes(ca_cert_pem)
    _write_restricted(cert_path, agent_cert_pem)
    _write_restricted(key_path, key_pem)

    logger.info(
        "enrollment complete: ca=%s cert=%s key=%s", ca_path, cert_path, key_path
    )


# ---------------------------------------------------------------------------
# mTLS SSL context
# ---------------------------------------------------------------------------


def load_tls_context(cfg_tls: TlsSection) -> ssl.SSLContext:
    """
    Build and return an SSLContext for mTLS wss:// connections.

    Checks that all three cert files exist and are not world-readable (Linux).
    Raises PermissionError if a file's permissions are too open.
    Raises FileNotFoundError if any cert file is missing.
    """
    assert cfg_tls.ca_cert is not None
    assert cfg_tls.agent_cert is not None
    assert cfg_tls.agent_key is not None

    ca_path = Path(cfg_tls.ca_cert)
    cert_path = Path(cfg_tls.agent_cert)
    key_path = Path(cfg_tls.agent_key)

    for path in (ca_path, cert_path, key_path):
        if not path.exists():
            raise FileNotFoundError(
                f"TLS file not found: {path}. "
                "Run `idlegpu-agent enroll` to obtain certificates."
            )
        _check_permissions(path)

    # PROTOCOL_TLS_CLIENT sets verify_mode=CERT_REQUIRED and check_hostname=True.
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.load_verify_locations(cafile=str(ca_path))
    ctx.load_cert_chain(certfile=str(cert_path), keyfile=str(key_path))

    return ctx


# ---------------------------------------------------------------------------
# Certificate expiry check
# ---------------------------------------------------------------------------


def check_cert_expiry(cert_pem: bytes) -> int:
    """Return the number of days until the certificate expires (may be negative)."""
    cert = x509.load_pem_x509_certificate(cert_pem)
    delta = cert.not_valid_after_utc - datetime.now(timezone.utc)
    return delta.days
