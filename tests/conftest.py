"""Shared fixtures: deterministic seeds and a CPU-only device."""

from __future__ import annotations

import pytest
import torch


@pytest.fixture(autouse=True)
def _seed() -> None:
    torch.manual_seed(0)


@pytest.fixture(params=[torch.float32, torch.float64], ids=["f32", "f64"])
def dtype(request: pytest.FixtureRequest) -> torch.dtype:
    return request.param
