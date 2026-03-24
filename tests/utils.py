"""Shared test utilities."""

from __future__ import annotations

import pytest

try:
    import torch  # noqa: F401
    _torch_available = True
except ImportError:
    _torch_available = False

try:
    import torch
    _cuda_available = torch.cuda.is_available()
except ImportError:
    _cuda_available = False


requires_torch = pytest.mark.skipif(not _torch_available, reason="torch not installed")
requires_cuda = pytest.mark.skipif(not _cuda_available, reason="CUDA not available")
