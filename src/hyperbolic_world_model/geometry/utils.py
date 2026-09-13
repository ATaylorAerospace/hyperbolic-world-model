"""Numerically stable primitives shared by every manifold.

All hyperbolic geometries hit the same three hazards: ``artanh`` diverges at ``|x| -> 1``,
``arcosh`` has an infinite derivative at ``x = 1``, and points drift outside their domain under
float32 rounding. The helpers here clamp consistently so that each manifold's ``expmap``,
``logmap`` and ``dist`` are finite and differentiable in both float32 and float64.
"""

from __future__ import annotations

import torch
from torch import Tensor

# Distance from the domain boundary at which we clamp. Float32 has ~7 significant digits, so a
# margin of 4e-3 keeps ``artanh`` away from the region where 1 - x underflows to 0. Float64 can
# get much closer. These values follow the conventions of geoopt / hyperbolic-nn.
BOUNDARY_EPS: dict[torch.dtype, float] = {
    torch.float16: 4e-3,
    torch.bfloat16: 4e-3,
    torch.float32: 4e-3,
    torch.float64: 1e-5,
}

# Smallest positive value we divide by. Used for ``x / max(norm, MIN_NORM)``.
MIN_NORM: dict[torch.dtype, float] = {
    torch.float16: 1e-4,
    torch.bfloat16: 1e-4,
    torch.float32: 1e-7,
    torch.float64: 1e-15,
}


def eps(dtype: torch.dtype, table: dict[torch.dtype, float] = BOUNDARY_EPS) -> float:
    """Return the tolerance for ``dtype`` from ``table`` (falls back to the float32 entry)."""
    return table.get(dtype, table[torch.float32])


def min_norm(dtype: torch.dtype) -> float:
    """Smallest norm we are willing to divide by for ``dtype``."""
    return eps(dtype, MIN_NORM)


class _Artanh(torch.autograd.Function):
    """``artanh`` with input clamped to ``(-1 + eps, 1 - eps)`` and an exact gradient."""

    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, x: Tensor) -> Tensor:  # type: ignore[override]
        e = eps(x.dtype)
        x = x.clamp(-1 + e, 1 - e)
        ctx.save_for_backward(x)
        return 0.5 * (torch.log1p(x) - torch.log1p(-x))

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_output: Tensor) -> Tensor:  # type: ignore[override]
        (x,) = ctx.saved_tensors
        return grad_output / (1 - x**2)


class _Arcosh(torch.autograd.Function):
    """``arcosh`` with input clamped to ``[1 + tiny, inf)`` and a finite gradient at the clamp.

    The clamp margin is :data:`MIN_NORM` (not :data:`BOUNDARY_EPS`): ``arcosh(1 + e) ~ sqrt(2e)``,
    so a margin of ``1e-5`` would turn ``dist(x, x)`` into ``4.5e-3``. With ``1e-7`` (float32) the
    self-distance is ``4.5e-4`` and the gradient at the clamp is ``~2e3``, both acceptable.
    """

    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, x: Tensor) -> Tensor:  # type: ignore[override]
        e = min_norm(x.dtype)
        x = x.clamp_min(1 + e)
        z = torch.sqrt(x * x - 1)
        ctx.save_for_backward(z)
        return torch.log(x + z)

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_output: Tensor) -> Tensor:  # type: ignore[override]
        (z,) = ctx.saved_tensors
        return grad_output / z


def artanh(x: Tensor) -> Tensor:
    """Inverse hyperbolic tangent, safe at the boundary of the unit interval."""
    return _Artanh.apply(x)


def arcosh(x: Tensor) -> Tensor:
    """Inverse hyperbolic cosine, safe at ``x = 1``."""
    return _Arcosh.apply(x)


def tanh(x: Tensor, clamp: float = 15.0) -> Tensor:
    """``tanh`` with the argument clamped so the output never saturates to exactly +-1."""
    return torch.tanh(x.clamp(-clamp, clamp))


def sinh(x: Tensor, clamp: float = 15.0) -> Tensor:
    """``sinh`` with the argument clamped to avoid overflow in float32."""
    return torch.sinh(x.clamp(-clamp, clamp))


def cosh(x: Tensor, clamp: float = 15.0) -> Tensor:
    """``cosh`` with the argument clamped to avoid overflow in float32."""
    return torch.cosh(x.clamp(-clamp, clamp))


def safe_norm(x: Tensor, dim: int = -1, keepdim: bool = True) -> Tensor:
    """Euclidean norm clamped from below so it can be used as a divisor."""
    return x.norm(dim=dim, keepdim=keepdim).clamp_min(min_norm(x.dtype))


def clip_norm(x: Tensor, max_norm: float | Tensor, dim: int = -1) -> Tensor:
    """Rescale ``x`` along ``dim`` so its norm is at most ``max_norm`` (no-op if already smaller).

    Used for boundary clipping in the Poincaré ball and for bounding tangent vectors before
    ``expmap`` so ``tanh``/``sinh`` do not saturate.
    """
    norm = x.norm(dim=dim, keepdim=True).clamp_min(min_norm(x.dtype))
    max_norm_t = torch.as_tensor(max_norm, dtype=x.dtype, device=x.device)
    scale = torch.where(norm > max_norm_t, max_norm_t / norm, torch.ones_like(norm))
    return x * scale


def as_curvature_tensor(curvature: float | Tensor, like: Tensor) -> Tensor:
    """Return ``curvature`` as a tensor with the dtype/device of ``like``."""
    return torch.as_tensor(curvature, dtype=like.dtype, device=like.device)


__all__ = [
    "BOUNDARY_EPS",
    "MIN_NORM",
    "arcosh",
    "artanh",
    "as_curvature_tensor",
    "clip_norm",
    "cosh",
    "eps",
    "min_norm",
    "safe_norm",
    "sinh",
    "tanh",
]
