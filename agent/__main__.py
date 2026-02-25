# Copyright (C) 2026 AniiGIT
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
agent/__main__.py - CLI entry point for the IdleGPU agent.

Usage:
  python -m agent start   [--broker HOST] [--port PORT] [--dev]
  python -m agent stop
  python -m agent enroll  --broker HOST [--port PORT] --token TOKEN [--data-dir DIR]
  python -m agent status

Commands:
  start    Load config, open the transparency log, and connect to the broker.
           Connects via mTLS (wss://) when [tls] is configured in agent.toml.
           Falls back to plaintext ws:// in dev mode or when TLS is unconfigured.
           Runs until interrupted (Ctrl+C or SIGTERM). Reconnects automatically.
  stop     Signal a running agent to exit using the PID file written by start.
  enroll   Enroll this agent with the broker to obtain mTLS certificates.
           Fetches the CA cert from the plain HTTP enrollment port, presents
           its fingerprint for out-of-band verification, submits a CSR with
           the one-time token, saves ca.crt, agent.crt, and agent.key, and
           writes [broker] host/port and [tls] paths to agent.toml.
           Pass --start to connect immediately after enrollment.
  status   One-shot idle check: prints current GPU, CPU, and input idle metrics.
           Does not require the broker to be reachable.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import ssl
import sys
import tomllib
from pathlib import Path

from .config import AgentConfig, load_config, system_config_path
from .transparency_log import TransparencyLog


# ---------------------------------------------------------------------------
# PID file helpers
# ---------------------------------------------------------------------------


def _data_dir(cfg: AgentConfig) -> Path:
    """Return the data directory derived from the transparency log path."""
    return Path(cfg.logging.transparency_log).parent


def _pid_file(cfg: AgentConfig) -> Path:
    """Return the path to the agent PID file."""
    return _data_dir(cfg) / "agent.pid"


def _write_pid(pid_file: Path) -> None:
    """Write the current process PID to pid_file."""
    try:
        pid_file.parent.mkdir(parents=True, exist_ok=True)
        pid_file.write_text(str(os.getpid()) + "\n", encoding="ascii")
    except OSError as exc:
        logging.getLogger(__name__).warning(
            "could not write PID file %s: %s; stop command will not work",
            pid_file,
            exc,
        )


def _remove_pid(pid_file: Path) -> None:
    """Remove the PID file. Best effort - silently ignores errors."""
    try:
        pid_file.unlink(missing_ok=True)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Config file helpers
# ---------------------------------------------------------------------------


def _patch_toml_sections(sections: dict[str, dict], data_dir: Path) -> Path | None:
    """
    Write or update one or more sections in agent.toml.

    Tries the system config path first; falls back to data_dir/agent.toml
    on permission errors. Returns the path that was successfully written, or
    None when all candidates fail (prints manual fallback instructions then).

    Reads the existing file (if present), merges each section dict into the
    parsed data, and writes back using tomli_w. All other sections are
    preserved. Note: tomli_w does not round-trip comments -- they are not
    preserved on re-write, but all key/value pairs remain intact.
    """
    try:
        import tomli_w  # noqa: PLC0415
    except ImportError:
        print(
            "error: tomli-w is not installed. Run: pip install 'tomli-w>=1.0'",
            file=sys.stderr,
        )
        sys.exit(1)

    candidates = [system_config_path(), data_dir / "agent.toml"]
    for config_path in candidates:
        data: dict = {}
        if config_path.exists():
            try:
                with open(config_path, "rb") as fh:
                    data = tomllib.load(fh)
            except (OSError, tomllib.TOMLDecodeError):
                pass  # start fresh; existing file could not be parsed

        data.update(sections)

        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, "wb") as fh:
                tomli_w.dump(data, fh)
            return config_path
        except OSError:
            continue

    # All candidates failed — print manual fallback.
    print(
        "Warning: could not write config to any location.",
        file=sys.stderr,
    )
    print(
        "Add the following to your agent.toml manually:",
        file=sys.stderr,
    )
    for section_name, section_values in sections.items():
        print(f"[{section_name}]", file=sys.stderr)
        for key, val in section_values.items():
            val_repr = f'"{val}"' if isinstance(val, str) else str(val)
            print(f"{key} = {val_repr}", file=sys.stderr)
    return None


# ---------------------------------------------------------------------------
# Subcommand: start
# ---------------------------------------------------------------------------


async def _run_agent(cfg: AgentConfig, ssl_ctx: ssl.SSLContext | None) -> None:
    """Async entry point for the start command."""
    # Import here so startup errors are clear even if deps are missing.
    from .connection import BrokerConnection  # noqa: PLC0415

    tlog = TransparencyLog(cfg.logging)
    conn = BrokerConnection(cfg, tlog, ssl_ctx=ssl_ctx)

    logger = logging.getLogger(__name__)
    logger.info(
        "agent starting: broker=%s:%d tls=%s",
        cfg.broker.host, cfg.broker.port,
        "enabled" if ssl_ctx else "disabled (plaintext)",
    )

    await conn.run()


def cmd_start(args: argparse.Namespace) -> None:
    """Load config and run the agent until interrupted."""
    cfg = load_config()

    # Allow CLI flags to override config values without modifying the file.
    if args.broker:
        cfg.broker.host = args.broker
    if args.port:
        cfg.broker.port = args.port

    log_level = logging.DEBUG if args.dev else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger(__name__)

    if args.dev:
        logger.warning(
            "dev mode: plaintext WebSocket, no mTLS. "
            "Do not use on untrusted networks."
        )

    # Build mTLS context unless dev mode is explicitly requested.
    ssl_ctx: ssl.SSLContext | None = None
    if not args.dev and cfg.tls.is_configured():
        from .pki import check_cert_expiry, load_tls_context  # noqa: PLC0415
        try:
            ssl_ctx = load_tls_context(cfg.tls)
        except (FileNotFoundError, PermissionError) as exc:
            logger.error("TLS setup failed: %s", exc)
            sys.exit(1)
        # Warn if cert is approaching expiry.
        assert cfg.tls.agent_cert is not None
        days = check_cert_expiry(Path(cfg.tls.agent_cert).read_bytes())
        if days < 30:
            logger.warning(
                "agent cert expires in %d day(s) -- re-enroll soon: "
                "idlegpu-agent enroll",
                days,
            )
    elif not args.dev and not cfg.tls.is_configured():
        logger.warning(
            "TLS not configured -- connecting in plaintext mode. "
            "Run `idlegpu-agent enroll` to set up mTLS."
        )

    pid_file = _pid_file(cfg)
    _write_pid(pid_file)

    try:
        asyncio.run(_run_agent(cfg, ssl_ctx))
    except KeyboardInterrupt:
        print("\nidlegpu-agent stopped.")
    finally:
        # Remove the PID file on clean exit (KeyboardInterrupt or exception).
        # Note: if the process is killed by an OS signal (e.g. SIGTERM on
        # Linux), this finally block does not run; cmd_stop removes the file
        # after signaling instead.
        _remove_pid(pid_file)


# ---------------------------------------------------------------------------
# Subcommand: stop
# ---------------------------------------------------------------------------


def cmd_stop(_args: argparse.Namespace) -> None:
    """
    Signal a running agent to exit using the PID file.

    Reads <data_dir>/agent.pid, sends SIGTERM to that process, then removes
    the PID file. On Windows, SIGTERM calls TerminateProcess (abrupt but
    reliable). On Linux, it delivers SIGTERM for graceful shutdown.
    """
    cfg = load_config()
    pid_file = _pid_file(cfg)

    if not pid_file.exists():
        print(f"No PID file found at {pid_file}.")
        print(
            "The agent may not be running, or was started without write "
            "access to that directory."
        )
        sys.exit(1)

    try:
        raw = pid_file.read_text(encoding="ascii").strip()
        pid = int(raw)
    except (OSError, ValueError) as exc:
        print(f"Could not read PID from {pid_file}: {exc}")
        sys.exit(1)

    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        print(f"No process with PID {pid}; removing stale PID file.")
        _remove_pid(pid_file)
        sys.exit(1)
    except PermissionError:
        print(f"Permission denied sending signal to PID {pid}.")
        sys.exit(1)

    # Remove PID file after a successful signal. The process may still be
    # running briefly, but the file is no longer valid.
    _remove_pid(pid_file)
    print(f"Sent SIGTERM to agent process {pid}.")


# ---------------------------------------------------------------------------
# Subcommand: enroll
# ---------------------------------------------------------------------------


def cmd_enroll(args: argparse.Namespace) -> None:
    """
    Enroll this agent with the broker to obtain mTLS certificates.

    Fetches the CA cert from the plain HTTP enrollment port, presents its
    SHA-256 fingerprint to the operator for out-of-band verification, then
    generates an ECDSA P-256 keypair and CSR, submits it with the one-time
    token, saves the certificate files, and writes [tls] paths to agent.toml.

    Pass --start to connect to the broker immediately after enrollment.
    """
    from .pki import cert_fingerprint, enroll, fetch_ca_cert  # noqa: PLC0415

    cfg = load_config()

    # Derive data directory from transparency log path (same as connection.py).
    data_dir = _data_dir(cfg)
    if args.data_dir is not None:
        data_dir = Path(args.data_dir)

    # Allow the agent_id to be pre-loaded from the existing identity file if
    # it already exists, so enrolled certs match the running agent.
    agent_id_file = data_dir / "agent_id"
    if agent_id_file.exists():
        agent_id = agent_id_file.read_text(encoding="ascii").strip()
    else:
        import uuid  # noqa: PLC0415
        agent_id = str(uuid.uuid4())

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Effective enrollment port: explicit flag, or main port + 1.
    enroll_port = args.enroll_port if args.enroll_port is not None else args.port + 1

    print(f"Enrolling agent {agent_id}")
    print(f"  Broker      : {args.broker}:{args.port}")
    print(f"  Enroll port : {enroll_port}")
    print()

    # Step 1: fetch CA cert from the plain HTTP enrollment port.
    try:
        ca_cert_pem = fetch_ca_cert(args.broker, enroll_port)
    except Exception as exc:
        print(f"error: could not fetch CA cert: {exc}", file=sys.stderr)
        sys.exit(1)

    # Step 2: interactive fingerprint confirmation (TOFU).
    fingerprint = cert_fingerprint(ca_cert_pem)
    print("CA certificate fingerprint (SHA-256):")
    print(f"  {fingerprint}")
    print()
    print("Compare this fingerprint with the one printed by `idlegpu-broker setup`.")
    print("If they do not match, abort and investigate before proceeding.")
    print()

    try:
        answer = input("Does this fingerprint match? [y/N] ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nAborted.")
        sys.exit(1)

    if answer.lower() != "y":
        print("Enrollment aborted.")
        sys.exit(1)

    print()

    # Step 3: generate keypair/CSR, POST to broker, save certs.
    try:
        enroll(
            broker_host=args.broker,
            enroll_port=enroll_port,
            token=args.token,
            agent_id=agent_id,
            data_dir=data_dir,
            ca_cert_pem=ca_cert_pem,
        )
    except Exception as exc:
        print(f"Enrollment failed: {exc}", file=sys.stderr)
        sys.exit(1)

    ca_path = data_dir / "ca.crt"
    cert_path = data_dir / "agent.crt"
    key_path = data_dir / "agent.key"

    config_path = _patch_toml_sections(
        {
            "broker": {
                "host": args.broker,
                "port": args.port,
            },
            "tls": {
                "ca_cert":    str(ca_path),
                "agent_cert": str(cert_path),
                "agent_key":  str(key_path),
            },
        },
        data_dir,
    )

    print("Enrollment successful.")
    if config_path is not None:
        print(f"[broker] and [tls] written to {config_path}.")
    print()

    if args.start:
        print("Starting agent (Ctrl+C to stop)...")
        print()
        cmd_start(argparse.Namespace(broker=None, port=None, dev=False))
    else:
        print("The agent is ready. Start it with:")
        print("  idlegpu-agent start")


# ---------------------------------------------------------------------------
# Subcommand: status
# ---------------------------------------------------------------------------


def cmd_status(_args: argparse.Namespace) -> None:
    """Print current idle metrics without connecting to the broker."""
    from .idle_monitor import cpu_utilization, gpu_utilization, idle_seconds, warmup_cpu  # noqa: PLC0415

    cfg = load_config()

    warmup_cpu()
    secs = idle_seconds()
    gpu = gpu_utilization()
    cpu = cpu_utilization()

    idle_thresh = cfg.idle.input_idle_seconds
    gpu_thresh = cfg.idle.gpu_threshold
    cpu_thresh = cfg.idle.cpu_threshold

    gpu_str = f"{gpu}%" if gpu is not None else "n/a (NVML unavailable)"
    gpu_idle = gpu is None or gpu < gpu_thresh
    cpu_idle = cpu < cpu_thresh
    input_idle = secs >= idle_thresh

    print(f"GPU utilization : {gpu_str:<20} threshold < {gpu_thresh}%"
          f"  {'[idle]' if gpu_idle else '[active]'}")
    print(f"CPU utilization : {cpu:.1f}%{'':<18} threshold < {cpu_thresh}%"
          f"  {'[idle]' if cpu_idle else '[active]'}")
    print(f"Input idle time : {secs}s{'':<19} threshold > {idle_thresh}s"
          f"  {'[idle]' if input_idle else '[active]'}")
    print()

    if gpu_idle and cpu_idle and input_idle:
        print("System state: IDLE - GPU would be offered to broker")
    else:
        reasons = []
        if not gpu_idle:
            reasons.append(f"GPU at {gpu}% (>= {gpu_thresh}%)")
        if not cpu_idle:
            reasons.append(f"CPU at {cpu:.1f}% (>= {cpu_thresh}%)")
        if not input_idle:
            remaining = idle_thresh - secs
            reasons.append(f"input active ({remaining}s until idle threshold)")
        print(f"System state: ACTIVE - {'; '.join(reasons)}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="idlegpu-agent",
        description="IdleGPU agent - idle detection and GPU forwarding.",
    )
    sub = parser.add_subparsers(dest="command", metavar="command")
    sub.required = True

    # start
    p_start = sub.add_parser("start", help="Connect to broker and begin monitoring.")
    p_start.add_argument(
        "--broker",
        metavar="HOST",
        default=None,
        help="Broker hostname or IP (overrides agent.toml).",
    )
    p_start.add_argument(
        "--port",
        type=int,
        metavar="PORT",
        default=None,
        help="Broker port (overrides agent.toml).",
    )
    p_start.add_argument(
        "--dev",
        action="store_true",
        help="Enable debug logging and force plaintext ws:// (no mTLS).",
    )

    # stop
    sub.add_parser("stop", help="Signal a running agent to exit.")

    # enroll
    p_enroll = sub.add_parser(
        "enroll",
        help="Obtain mTLS certificates from the broker.",
    )
    p_enroll.add_argument(
        "--broker",
        required=True,
        metavar="HOST",
        help="Broker hostname or IP.",
    )
    p_enroll.add_argument(
        "--port",
        type=int,
        default=8765,
        metavar="PORT",
        help="Broker main port (default: 8765).",
    )
    p_enroll.add_argument(
        "--enroll-port",
        type=int,
        default=None,
        metavar="PORT",
        dest="enroll_port",
        help=(
            "Broker enrollment port for CA cert fetch and CSR submission "
            "(plain HTTP). Default: main port + 1."
        ),
    )
    p_enroll.add_argument(
        "--token",
        required=True,
        metavar="TOKEN",
        help="One-time enrollment token from `idlegpu-broker setup`.",
    )
    p_enroll.add_argument(
        "--data-dir",
        default=None,
        metavar="DIR",
        dest="data_dir",
        help="Directory to save certificates (default: agent data dir).",
    )
    p_enroll.add_argument(
        "--start",
        action="store_true",
        help="Start the agent immediately after enrollment completes.",
    )

    # status
    sub.add_parser("status", help="Print current idle metrics (no broker needed).")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    dispatch = {
        "start": cmd_start,
        "stop": cmd_stop,
        "enroll": cmd_enroll,
        "status": cmd_status,
    }

    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    handler(args)


if __name__ == "__main__":
    main()
