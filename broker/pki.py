# Copyright (C) 2026 AniiGIT
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
broker/pki.py - PKI utilities for the IdleGPU broker.

Generates and manages the local CA, broker certificate, and agent enrollment
tokens. All keys use ECDSA P-256 (secp256r1) -- equivalent security to
RSA-3072, faster, and produces smaller certificates.

Key files written to data_dir:
  ca.crt              CA certificate (public -- shared with agents via /ca.crt)
  ca.key              CA private key  (mode 600 -- never leaves the broker host)
  broker.crt          Broker TLS certificate (public)
  broker.key          Broker TLS private key (mode 600)
  enrollment_token    One-time enrollment token (deleted after first use)

None of these files are stored in broker.toml. The config file only holds paths.
"""

from __future__ import annotations

import logging
import os
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.ec import SECP256R1, generate_private_key
from cryptography.x509.oid import NameOID

# Plain str prevents static analysers from narrowing platform branches.
_OS_NAME: str = os.name

logger = logging.getLogger(__name__)

# Certificate lifetimes.
_CA_LIFETIME_DAYS = 3650      # 10 years
_CERT_LIFETIME_DAYS = 365     # 1 year

# Enrollment token: 32 random bytes = 64 hex chars.
_TOKEN_FILENAME = "enrollment_token"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _generate_ec_key():  # type: ignore[return]
    """Generate an ECDSA P-256 private key."""
    return generate_private_key(SECP256R1())


def _key_to_pem(key) -> bytes:  # type: ignore[return, type-arg]
    """Serialize a private key to unencrypted PKCS8 PEM bytes."""
    return key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )


def _cert_to_pem(cert: x509.Certificate) -> bytes:
    """Serialize a certificate to PEM bytes."""
    return cert.public_bytes(serialization.Encoding.PEM)


def _write_restricted(path: Path, data: bytes) -> None:
    """Write *data* to *path* and restrict permissions to owner-only on Linux."""
    path.write_bytes(data)
    if _OS_NAME == "posix":
        path.chmod(0o600)


def _now() -> datetime:
    """Return the current UTC time as a timezone-aware datetime."""
    return datetime.now(timezone.utc)


def _idlegpu_name(cn: str) -> x509.Name:
    return x509.Name([
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "IdleGPU"),
        x509.NameAttribute(NameOID.COMMON_NAME, cn),
    ])


# ---------------------------------------------------------------------------
# CA generation
# ---------------------------------------------------------------------------


def generate_ca() -> tuple[bytes, bytes]:
    """
    Generate a new ECDSA P-256 CA key and self-signed certificate.

    Returns (cert_pem, key_pem) as bytes. The caller is responsible for
    writing the files to disk with appropriate permissions.

    The CA has BasicConstraints(ca=True) and a 10-year lifetime.
    """
    key = _generate_ec_key()
    name = _idlegpu_name("IdleGPU Local CA")
    now = _now()

    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + timedelta(days=_CA_LIFETIME_DAYS))
        .add_extension(
            x509.BasicConstraints(ca=True, path_length=0), critical=True
        )
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(key.public_key()),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )

    return _cert_to_pem(cert), _key_to_pem(key)


# ---------------------------------------------------------------------------
# Broker certificate
# ---------------------------------------------------------------------------


def generate_broker_cert(
    ca_cert_pem: bytes,
    ca_key_pem: bytes,
    hostname: str,
) -> tuple[bytes, bytes]:
    """
    Generate an ECDSA P-256 broker certificate signed by the local CA.

    The certificate includes the hostname as a Subject Alternative Name so
    that agents can verify the broker's identity when connecting via mTLS.

    Returns (cert_pem, key_pem) as bytes.
    """
    ca_cert = x509.load_pem_x509_certificate(ca_cert_pem)
    ca_key = serialization.load_pem_private_key(ca_key_pem, password=None)

    key = _generate_ec_key()
    now = _now()

    cert = (
        x509.CertificateBuilder()
        .subject_name(_idlegpu_name(hostname))
        .issuer_name(ca_cert.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + timedelta(days=_CERT_LIFETIME_DAYS))
        .add_extension(
            x509.BasicConstraints(ca=False, path_length=None), critical=True
        )
        .add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName(hostname),
                x509.IPAddress(__import__("ipaddress").ip_address("127.0.0.1")),
            ]),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256())
    )

    return _cert_to_pem(cert), _key_to_pem(key)


# ---------------------------------------------------------------------------
# CSR signing (agent enrollment)
# ---------------------------------------------------------------------------


def sign_csr(ca_cert_pem: bytes, ca_key_pem: bytes, csr_pem: bytes) -> bytes:
    """
    Sign an agent's Certificate Signing Request with the local CA.

    Validates the CSR signature before signing. Raises ValueError if the
    CSR is malformed or the signature is invalid.

    Returns the signed agent certificate as PEM bytes.
    """
    try:
        csr = x509.load_pem_x509_csr(csr_pem)
    except Exception as exc:
        raise ValueError(f"malformed CSR: {exc}") from exc

    if not csr.is_signature_valid:
        raise ValueError("CSR signature is invalid")

    ca_cert = x509.load_pem_x509_certificate(ca_cert_pem)
    ca_key = serialization.load_pem_private_key(ca_key_pem, password=None)
    now = _now()

    agent_cert = (
        x509.CertificateBuilder()
        .subject_name(csr.subject)
        .issuer_name(ca_cert.subject)
        .public_key(csr.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + timedelta(days=_CERT_LIFETIME_DAYS))
        .add_extension(
            x509.BasicConstraints(ca=False, path_length=None), critical=True
        )
        .sign(ca_key, hashes.SHA256())
    )

    return _cert_to_pem(agent_cert)


# ---------------------------------------------------------------------------
# Enrollment token
# ---------------------------------------------------------------------------


def create_enrollment_token(data_dir: Path) -> str:
    """
    Generate a new one-time enrollment token and persist it to data_dir.

    The token is a 64-character hex string (32 random bytes). It is written
    to data_dir/enrollment_token with mode 600 on Linux.

    Overwrites any existing token -- call this only from `broker setup`.
    """
    token = secrets.token_hex(32)
    token_file = data_dir / _TOKEN_FILENAME
    data_dir.mkdir(parents=True, exist_ok=True)
    _write_restricted(token_file, (token + "\n").encode("ascii"))
    logger.debug("enrollment token written to %s", token_file)
    return token


def consume_enrollment_token(data_dir: Path, provided_token: str) -> bool:
    """
    Validate *provided_token* against the stored token and consume it.

    Returns True and deletes the token file if the token matches.
    Returns False if the file is absent, unreadable, or the token does not match.

    Uses secrets.compare_digest to prevent timing attacks.
    """
    token_file = data_dir / _TOKEN_FILENAME

    try:
        stored = token_file.read_text(encoding="ascii").strip()
    except OSError:
        logger.warning("enrollment token file not found or unreadable: %s", token_file)
        return False

    if not secrets.compare_digest(stored, provided_token.strip()):
        logger.warning("enrollment token mismatch -- rejected")
        return False

    # Token is valid -- delete immediately to enforce one-time use.
    try:
        token_file.unlink()
    except OSError as exc:
        logger.warning("could not remove enrollment token file: %s", exc)

    return True


# ---------------------------------------------------------------------------
# Certificate expiry check
# ---------------------------------------------------------------------------


def check_cert_expiry(cert_pem: bytes) -> int:
    """Return the number of days until the certificate expires (may be negative)."""
    cert = x509.load_pem_x509_certificate(cert_pem)
    delta = cert.not_valid_after_utc - _now()
    return delta.days


# ---------------------------------------------------------------------------
# Setup helper -- writes all files to data_dir
# ---------------------------------------------------------------------------


def setup(data_dir: Path, hostname: str) -> tuple[str, str, str, str, str]:
    """
    Generate CA, broker cert, and enrollment token; write everything to data_dir.

    Returns (ca_cert_path, broker_cert_path, broker_key_path, token, fingerprint)
    where fingerprint is the SHA-256 hex fingerprint of the CA cert for
    out-of-band verification.
    """
    data_dir.mkdir(parents=True, exist_ok=True)

    ca_cert_pem, ca_key_pem = generate_ca()
    broker_cert_pem, broker_key_pem = generate_broker_cert(ca_cert_pem, ca_key_pem, hostname)

    ca_cert_path = data_dir / "ca.crt"
    ca_key_path = data_dir / "ca.key"
    broker_cert_path = data_dir / "broker.crt"
    broker_key_path = data_dir / "broker.key"

    # Public certs: readable by any process; keys: owner-only.
    ca_cert_path.write_bytes(ca_cert_pem)
    _write_restricted(ca_key_path, ca_key_pem)
    broker_cert_path.write_bytes(broker_cert_pem)
    _write_restricted(broker_key_path, broker_key_pem)

    token = create_enrollment_token(data_dir)

    # Compute CA cert fingerprint for out-of-band verification.
    ca_cert_obj = x509.load_pem_x509_certificate(ca_cert_pem)
    fingerprint = ca_cert_obj.fingerprint(hashes.SHA256()).hex(":")

    logger.info(
        "PKI setup complete: CA and broker cert written to %s", data_dir
    )

    return (
        str(ca_cert_path),
        str(broker_cert_path),
        str(broker_key_path),
        token,
        fingerprint,
    )
