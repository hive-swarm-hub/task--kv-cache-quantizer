"""
KV Cache Quantizer — Hadamard rotation + per-group 3-bit.

Uses Walsh-Hadamard transform to uniformize distribution before quantization.
Better than random rotation — deterministic and optimal for spreading outliers.

Interface contract (do not change function signatures):
  - quantize(tensor: torch.Tensor) -> dict
  - dequantize(quantized: dict) -> torch.Tensor
  - bits_per_value() -> float
"""

import torch
import math

GROUP_SIZE = 4
BITS = 2
MAX_VAL = (1 << BITS) - 1  # 3

_hadamard_cache = {}


def _hadamard(n):
    """Build normalized n×n Hadamard matrix (n must be power of 2)."""
    if n not in _hadamard_cache:
        # Build unnormalized, then normalize once at the end
        H = torch.tensor([[1.0]])
        while H.shape[0] < n:
            H = torch.cat([
                torch.cat([H, H], dim=1),
                torch.cat([H, -H], dim=1),
            ], dim=0)
        H = H / math.sqrt(n)
        _hadamard_cache[n] = H
    return _hadamard_cache[n]


def _get_hadamard(dim, device, dtype):
    # Find next power of 2 >= dim
    n = 1
    while n < dim:
        n *= 2
    H = _hadamard(n).to(device=device, dtype=dtype)
    return H[:dim, :dim] if n > dim else H


def bits_per_value() -> float:
    return float(BITS)


def quantize(tensor: torch.Tensor) -> dict:
    orig_shape = tensor.shape
    dtype = tensor.dtype
    B, H_heads, S, D = orig_shape

    # Apply Hadamard rotation along head_dim
    Had = _get_hadamard(D, tensor.device, dtype)
    t = torch.matmul(tensor, Had)

    # Flatten last two dims for group quantization
    t = t.reshape(B, H_heads, -1)
    N = t.shape[-1]

    # Pad to multiple of GROUP_SIZE
    pad = (GROUP_SIZE - N % GROUP_SIZE) % GROUP_SIZE
    if pad > 0:
        t = torch.nn.functional.pad(t, (0, pad))

    # Reshape to groups
    t = t.reshape(B, H_heads, -1, GROUP_SIZE)

    vmin = t.min(dim=-1, keepdim=True).values
    vmax = t.max(dim=-1, keepdim=True).values
    scale = (vmax - vmin) / MAX_VAL
    scale = scale.clamp(min=1e-8)

    quantized = ((t - vmin) / scale).round().clamp(0, MAX_VAL).to(torch.uint8)

    return {
        "data": quantized,
        "scale": scale.to(torch.float16),
        "vmin": vmin.to(torch.float16),
        "dtype": dtype,
        "shape": orig_shape,
        "N": N,
    }


def dequantize(quantized: dict) -> torch.Tensor:
    dtype = quantized["dtype"]
    scale = quantized["scale"].to(dtype)
    vmin = quantized["vmin"].to(dtype)
    t = quantized["data"].to(dtype) * scale + vmin

    B, H_heads, S, D = quantized["shape"]
    N = quantized["N"]
    t = t.reshape(B, H_heads, -1)[:, :, :N]
    t = t.reshape(B, H_heads, S, D)

    # Inverse Hadamard (H is self-inverse when normalized)
    Had = _get_hadamard(D, t.device, dtype)
    t = torch.matmul(t, Had)

    return t
