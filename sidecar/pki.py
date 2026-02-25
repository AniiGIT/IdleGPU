# Copyright (C) 2026 AniiGIT
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
sidecar/pki.py - PKI utilities for the IdleGPU sidecar.

Handles sidecar-side certificate operations:
  - Generating an ECDSA P-256 keypair and Certificate Signing Request (CSR)
  - Enrolling with the broker over HTTP to obtain a signed certificate
  - Writing sidecar.env for docker-compose env_file loading

Enrollment runs once on the host (before starting the sidecar container),
storing certificate files in a directory that is later volume-mounted into
the container.  The generated sidecar.env maps the three TLS environment
variables to the in-container cert paths so docker-compose can load them
with env_file:.

Enrollment flow (see cmd_enroll in __main__.py):
  1. Fetch CA cert from broker's plain HTTP enrollment port
  2. Present fingerprint for TOFU verification
  3. Generate ECDSA P-256 keypair and CSR (CN=idlegpu-sidecar-<id>)
  4. POST CSR + token to POST /enroll/sidecar
  5. Write ca.crt, sidecar.crt, sidecar.key and sidecar.env to data_dir
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.ec import SECP256R1, generate_private_key
from cryptography.x509.oid import NameOID

# Plain str prevents static analysers from narrowing platform branches.
_OS_NAME: str = os.name

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _write_restricted(path: Path, data: bytes) -> None:
    """Write *data* to *path* and restrict permissions to owner-only on Linux."""
    path.write_bytes(data)
    if _OS_NAME == "posix":
        path.chmod(0o600)


def default_data_dir() -> Path:
    """Return the platform-appropriate default directory for sidecar cert files."""
    if _OS_NAME == "nt":
        data = os.environ.get("PROGRAMDATA", r"C:\ProgramData")
        return Path(data) / "idlegpu" / "tls"
    return Path("/etc/idlegpu/tls")


# ---------------------------------------------------------------------------
# Keypair and CSR generation
# ---------------------------------------------------------------------------


def generate_sidecar_keypair_and_csr(sidecar_id: str) -> tuple[bytes, bytes]:
    """
    Generate an ECDSA P-256 private key and a Certificate Signing Request.

    The CSR subject is:
      O=IdleGPU, CN=idlegpu-sidecar-<sidecar_id>

    Returns (key_pem, csr_pem) as bytes. Neither is written to disk here --
    the caller handles persistence.
    """
    key = generate_private_key(SECP256R1())

    csr = (
        x509.CertificateSigningRequestBuilder()
        .subject_name(x509.Name([
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "IdleGPU"),
            x509.NameAttribute(NameOID.COMMON_NAME, f"idlegpu-sidecar-{sidecar_id}"),
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


def enroll_sidecar(
    broker_host: str,
    enroll_port: int,
    token: str,
    sidecar_id: str,
    data_dir: Path,
    ca_cert_pem: bytes,
    env_cert_dir: Path | None = None,
) -> Path:
    """
    Enroll this sidecar with the broker and install certificate files.

    The caller is responsible for fetching the CA cert (via fetch_ca_cert())
    and confirming the fingerprint with the operator before calling this.

    *data_dir* is where cert files are written on the host.
    *env_cert_dir* is the path where those files will appear inside the
    container (used in sidecar.env). Defaults to *data_dir* when not set
    (appropriate when the host path and container path are the same).

    Steps:
      1. Generate an ECDSA P-256 keypair and CSR for this sidecar_id
      2. POST the CSR and token to http://broker:enroll_port/enroll/sidecar
      3. Save ca.crt, sidecar.crt, sidecar.key to data_dir (mode 600 on Linux)
      4. Write sidecar.env to data_dir with IDLEGPU_CA_CERT, IDLEGPU_SIDECAR_CERT,
         and IDLEGPU_SIDECAR_KEY using env_cert_dir paths (so docker-compose
         can load it with env_file:)

    Returns the path to the written sidecar.env file.
    Raises urllib.error.URLError on network failure.
    Raises RuntimeError on broker-reported errors (bad token, invalid CSR, etc.).
    """
    data_dir.mkdir(parents=True, exist_ok=True)

    # cert_dir is the path prefix seen inside the container; may differ from
    # data_dir (host) when the volume mount uses different source/target paths.
    cert_dir = env_cert_dir if env_cert_dir is not None else data_dir

    base_url = f"http://{broker_host}:{enroll_port}"

    # Step 1: generate sidecar keypair and CSR.
    logger.info("generating ECDSA P-256 keypair for sidecar_id=%s", sidecar_id)
    key_pem, csr_pem = generate_sidecar_keypair_and_csr(sidecar_id)

    # Step 2: POST CSR to broker sidecar enrollment endpoint.
    logger.info("submitting CSR to %s/enroll/sidecar", base_url)
    payload = json.dumps({
        "token": token,
        "sidecar_id": sidecar_id,
        "csr_pem": csr_pem.decode("ascii"),
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{base_url}/enroll/sidecar",
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

    sidecar_cert_pem = result.get("sidecar_cert_pem", "").encode("ascii")
    if not sidecar_cert_pem:
        raise RuntimeError("broker returned empty sidecar_cert_pem")

    # Step 3: persist certificate files.
    ca_path = data_dir / "ca.crt"
    cert_path = data_dir / "sidecar.crt"
    key_path = data_dir / "sidecar.key"

    ca_path.write_bytes(ca_cert_pem)
    _write_restricted(cert_path, sidecar_cert_pem)
    _write_restricted(key_path, key_pem)

    logger.info(
        "enrollment complete: ca=%s cert=%s key=%s", ca_path, cert_path, key_path
    )

    # Step 4: write sidecar.env for docker-compose env_file loading.
    # Paths in the env file use cert_dir (container-side), not data_dir (host).
    env_path = data_dir / "sidecar.env"
    env_content = (
        f"IDLEGPU_CA_CERT={cert_dir / 'ca.crt'}\n"
        f"IDLEGPU_SIDECAR_CERT={cert_dir / 'sidecar.crt'}\n"
        f"IDLEGPU_SIDECAR_KEY={cert_dir / 'sidecar.key'}\n"
    )
    env_path.write_text(env_content, encoding="ascii")
    logger.info("sidecar.env written to %s", env_path)

    return env_path
