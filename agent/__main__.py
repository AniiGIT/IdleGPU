# Copyright (C) 2026 AniiGIT
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
agent/__main__.py - CLI entry point for the IdleGPU agent.

Usage:
  python -m agent start   [--broker HOST] [--port PORT] [--dev]
  python -m agent stop
  python -m agent status

Commands:
  start    Load config, open the transparency log, and connect to the broker.
           Runs until interrupted (Ctrl+C or SIGTERM). Reconnects automatically.
  stop     Signal a running agent to exit using the PID file written by start.
  status   One-shot idle check: prints current GPU, CPU, and input idle metrics.
           Does not require the broker to be reachable.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
from pathlib import Path

from .config import AgentConfig, load_config
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
# Subcommand: start
# ---------------------------------------------------------------------------


async def _run_agent(cfg: AgentConfig) -> None:
    """Async entry point for the start command."""
    # Import here so startup errors are clear even if deps are missing.
    from .connection import BrokerConnection  # noqa: PLC0415

    tlog = TransparencyLog(cfg.logging)
    conn = BrokerConnection(cfg, tlog)

    logger = logging.getLogger(__name__)
    logger.info(
        "agent starting: broker=%s:%d", cfg.broker.host, cfg.broker.port
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

    if args.dev:
        logging.getLogger(__name__).warning(
            "dev mode: plaintext WebSocket, no mTLS. "
            "Do not use on untrusted networks."
        )

    pid_file = _pid_file(cfg)
    _write_pid(pid_file)

    try:
        asyncio.run(_run_agent(cfg))
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
        help="Enable debug logging (plaintext mode, no mTLS).",
    )

    # stop
    sub.add_parser("stop", help="Signal a running agent to exit.")

    # status
    sub.add_parser("status", help="Print current idle metrics (no broker needed).")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    dispatch = {
        "start": cmd_start,
        "stop": cmd_stop,
        "status": cmd_status,
    }

    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    handler(args)


if __name__ == "__main__":
    main()
