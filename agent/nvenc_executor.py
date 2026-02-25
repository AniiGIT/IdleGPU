# Copyright (C) 2026 AniiGIT
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
agent/nvenc_executor.py - NVENC API forwarding on the agent side.

Phase 2F status
───────────────
This module provides the infrastructure for NVENC forwarding but full
execution is deferred until the shim-side NVENC interception is updated
(Phase 2G) to actually fill function pointer tables rather than returning
NV_ENC_ERR_UNIMPLEMENTED.

When that change lands, this module will:
  1. Load the real NVENC library (libnvidia-encode.so.1 on Linux,
     nvEncodeAPI64.dll on Windows).
  2. Call NvEncodeAPICreateInstance() to obtain the real function table.
  3. Dispatch each NVENC call frame to the corresponding function table entry.
  4. Return results as response dicts.

Currently: all NVENC calls return NV_ENC_ERR_UNIMPLEMENTED (15) so that
the sidecar returns a clear error to the application rather than silently
hanging.

Public API
──────────
  dispatch(req: dict) -> dict
      Called by CudaExecutor.dispatch_nvenc() for every NVENC_CALL frame.
"""

from __future__ import annotations

import ctypes
import logging
import sys
from typing import Any

logger = logging.getLogger(__name__)

# NVENC status codes.
NV_ENC_SUCCESS           = 0
NV_ENC_ERR_UNIMPLEMENTED = 15
NV_ENC_ERR_INVALID_PTR   = 6

# Lazy-loaded NVENC function table handle.
_nvenc_lib: Any = None
_nvenc_fn_table: Any = None  # NV_ENCODE_API_FUNCTION_LIST (Phase 2G)
_nvenc_checked = False


def _try_load_nvenc() -> bool:
    """Attempt to load and initialise the NVENC library.

    Returns True on success.  Logs a warning and returns False if the
    library is unavailable (no GPU, or GPU without NVENC support).
    """
    global _nvenc_lib, _nvenc_checked
    if _nvenc_checked:
        return _nvenc_lib is not None
    _nvenc_checked = True

    if sys.platform == "win32":
        candidates = ["nvEncodeAPI64.dll", "nvEncodeAPI.dll"]
    else:
        candidates = ["libnvidia-encode.so.1", "libnvidia-encode.so"]

    for name in candidates:
        try:
            _nvenc_lib = ctypes.CDLL(name)
            logger.info("nvenc_executor: loaded %s", name)
            return True
        except OSError:
            continue

    logger.warning(
        "nvenc_executor: could not load NVENC library (tried %s); "
        "NVENC forwarding unavailable", candidates
    )
    return False


def dispatch(req: dict) -> dict:  # type: ignore[type-arg]
    """Dispatch a single NVENC call frame.

    Phase 2F: returns NV_ENC_ERR_UNIMPLEMENTED for all calls, confirming
    that the library is reachable but full forwarding is not yet wired up.

    Phase 2G will replace this with real function table dispatch.
    """
    func: str = req.get("func", "<unknown>")
    available = _try_load_nvenc()

    if not available:
        logger.warning("nvenc_executor: NVENC library unavailable, cannot execute %s", func)
        return {"result": NV_ENC_ERR_UNIMPLEMENTED}

    # Phase 2G: dispatch to real function table here.
    # For now, log the call and return NOT_IMPLEMENTED so callers get a
    # deterministic error rather than an unexpected hang.
    logger.info(
        "nvenc_executor: %s called — full forwarding not yet implemented (Phase 2G)",
        func,
    )
    return {"result": NV_ENC_ERR_UNIMPLEMENTED}
