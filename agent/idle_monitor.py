# Copyright (C) 2026 AniiGIT
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
agent/idle_monitor.py - Three-signal system idle detection.

All three signals must satisfy their thresholds simultaneously before the
GPU is offered to the broker:

  GPU utilization  < idle.gpu_threshold      (pynvml - NVIDIA Management Library)
  CPU utilization  < idle.cpu_threshold      (psutil)
  Input idle time  > idle.input_idle_seconds (GetLastInputInfo on Windows,
                                              xprintidle on Linux)

All thresholds and poll intervals come from an IdleSection loaded by
agent.config.load_config(). No hardcoded values here.
"""

from __future__ import annotations

import asyncio
import ctypes
import logging
import os
import subprocess
from typing import Callable

from .config import IdleSection

# Plain str prevents static analysers from narrowing platform branches.
_OS_NAME: str = os.name

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Input idle time
# ---------------------------------------------------------------------------


# LASTINPUTINFO uses only standard ctypes types; no ctypes.wintypes import needed.
class _LASTINPUTINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize", ctypes.c_uint),
        ("dwTime", ctypes.c_uint32),
    ]


def _idle_seconds_windows() -> int:
    info = _LASTINPUTINFO()
    info.cbSize = ctypes.sizeof(_LASTINPUTINFO)
    if not ctypes.windll.user32.GetLastInputInfo(ctypes.byref(info)):  # type: ignore[attr-defined]
        raise OSError("GetLastInputInfo failed")
    elapsed_ms = ctypes.windll.kernel32.GetTickCount() - info.dwTime  # type: ignore[attr-defined]
    # GetTickCount wraps at ~49.7 days; treat wrap-around as no elapsed time.
    if elapsed_ms < 0:
        return 0
    return int(elapsed_ms) // 1000


def _idle_seconds_linux() -> int:
    try:
        result = subprocess.run(
            ["xprintidle"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        return int(result.stdout.strip()) // 1000
    except FileNotFoundError:
        raise OSError(
            "xprintidle not found; install it (apt install xprintidle) "
            "and ensure DISPLAY is set"
        ) from None
    except (ValueError, subprocess.TimeoutExpired) as exc:
        raise OSError(f"xprintidle error: {exc}") from exc


def idle_seconds() -> int:
    """Return whole seconds since the last keyboard or mouse input event.

    Only the elapsed-time integer is used. Input content is never read,
    logged, or transmitted.
    """
    if _OS_NAME == "nt":
        return _idle_seconds_windows()
    else:
        return _idle_seconds_linux()


# ---------------------------------------------------------------------------
# GPU and CPU utilization
# ---------------------------------------------------------------------------


def gpu_utilization() -> int | None:
    """Return GPU utilization % for the first NVIDIA device, or None if unavailable.

    Returns None when pynvml is missing or NVML cannot initialise (no NVIDIA
    driver, no GPU, etc.). The caller treats None as idle for that signal.
    """
    try:
        import pynvml  # noqa: PLC0415  # type: ignore[import-not-found]

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return int(util.gpu)
    except Exception:
        return None


def cpu_utilization() -> float:
    """Return CPU utilization % averaged across all cores since the last call.

    Uses psutil.cpu_percent(interval=None), non-blocking, to avoid stalling
    the async event loop. psutil computes utilization as the delta between
    two snapshots; the very first call in a process has no previous snapshot
    and always returns 0.0. Call warmup_cpu() once at startup to seed the
    baseline so that all readings inside the monitor loop are accurate.
    """
    import psutil  # noqa: PLC0415 - optional dep, imported lazily

    return psutil.cpu_percent(interval=None)


def warmup_cpu() -> None:
    """Seed the psutil CPU baseline so the first real reading is not 0.0.

    Call this once before starting the monitor loop. The throwaway snapshot
    is taken here; after a short interval the first cpu_utilization() call
    in the loop will have a valid delta to measure against.
    """
    import psutil  # noqa: PLC0415 - optional dep, imported lazily

    psutil.cpu_percent(interval=None)  # throwaway; seeds the internal counter


# ---------------------------------------------------------------------------
# Composite idle check
# ---------------------------------------------------------------------------


def is_system_idle(cfg: IdleSection) -> tuple[bool, int, int | None, float]:
    """
    Evaluate all three idle signals against the thresholds in *cfg*.

    Returns:
        (idle, input_secs, gpu_pct, cpu_pct)

    *idle* is True only when every signal satisfies its threshold.
    *gpu_pct* is None when NVML is unavailable; that signal is treated as
    idle so a machine without a recognised GPU is not permanently blocked.
    """
    input_secs = idle_seconds()
    gpu_pct = gpu_utilization()
    cpu_pct = cpu_utilization()

    gpu_ok = gpu_pct is None or gpu_pct < cfg.gpu_threshold
    cpu_ok = cpu_pct < cfg.cpu_threshold
    input_ok = input_secs >= cfg.input_idle_seconds

    return (gpu_ok and cpu_ok and input_ok), input_secs, gpu_pct, cpu_pct


# ---------------------------------------------------------------------------
# Async monitor loop
# ---------------------------------------------------------------------------


IdleCallback = Callable[[int], None]


async def monitor(
    cfg: IdleSection,
    on_idle: IdleCallback,
    on_active: IdleCallback,
) -> None:
    """
    Poll system idle state and fire callbacks on state transitions.

    Poll timing (from *cfg*):
    - ``poll_idle_seconds``: while the machine is active (waiting for idle).
    - ``poll_active_seconds``: while the GPU is offered (fast reclaim on return).

    Callbacks receive the current ``idle_seconds()`` value as their sole argument.
    ``on_idle`` is called every tick while idle; ``on_active`` fires once on the
    idle → active transition.
    """
    warmup_cpu()
    was_idle = False

    while True:
        idle, input_secs, gpu_pct, cpu_pct = is_system_idle(cfg)

        if idle:
            if not was_idle:
                logger.info(
                    "system idle: input=%ds gpu=%s%% cpu=%.1f%%",
                    input_secs,
                    gpu_pct if gpu_pct is not None else "n/a",
                    cpu_pct,
                )
            on_idle(input_secs)
            was_idle = True
            interval = cfg.poll_active_seconds  # poll fast to detect user return
        else:
            if was_idle:
                logger.info(
                    "system active: input=%ds gpu=%s%% cpu=%.1f%%",
                    input_secs,
                    gpu_pct if gpu_pct is not None else "n/a",
                    cpu_pct,
                )
                on_active(input_secs)
                was_idle = False
            interval = cfg.poll_idle_seconds  # poll slower while waiting for idle

        await asyncio.sleep(interval)
