# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
JIT wrapper for the fused reduce-scatter + residual + RMSNorm kernel.
"""

from __future__ import annotations

import os
from pathlib import Path

_lib = None


def _load_lib():
    global _lib
    if _lib is not None:
        return _lib

    from torch.utils.cpp_extension import load

    src = str(Path(__file__).with_name("moe_rs_fused.cu"))
    _lib = load(
        name="moe_rs_fused_kernel",
        sources=[src],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=os.environ.get("MOE_RS_FUSED_VERBOSE", "") == "1",
    )
    return _lib
