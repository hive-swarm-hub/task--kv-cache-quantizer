"""
KV Cache Quantizer — rotation + per-group 3-bit.

Apply random Hadamard-like rotation to spread outliers before quantization.
Inspired by TurboQuant's PolarQuant approach.

Interface contract (do not change function signatures):
  - quantize(tensor: torch.Tensor) -> dict
  - dequantize(quantized: dict) -> torch.Tensor
  - bits_per_value() -> float
"""

import torch

GROUP_SIZE = 16
BITS = 3
MAX_VAL = (1 << BITS) - 1  # 7

# Fixed random rotation matrix (seeded for reproducibility)
_rotation_cache = {}


def _get_rotation(dim, device, dtype):
    key = (dim, device, dtype)
    if key not in _rotation_cache:
        gen = torch.Generator(device='cpu')
        gen.manual_seed(42)
        # Random orthogonal matrix via QR decomposition
        rand_mat = torch.randn(dim, dim, generator=gen)
        Q, _ = torch.linalg.qr(rand_mat)
        _rotation_cache[key] = Q.to(device=device, dtype=dtype)
    return _rotation_cache[key]


def bits_per_value() -> float:
    return float(BITS)


def quantize(tensor: torch.Tensor) -> dict:
    orig_shape = tensor.shape
    dtype = tensor.dtype
    B, H, S, D = orig_shape

    # Apply rotation along head_dim to spread outliers
    R = _get_rotation(D, tensor.device, dtype)
    t = torch.matmul(tensor, R)  # (B, H, S, D) @ (D, D) -> (B, H, S, D)

    # Flatten last two dims for group quantization
    t = t.reshape(B, H, -1)
    N = t.shape[-1]

    # Pad to multiple of GROUP_SIZE
    pad = (GROUP_SIZE - N % GROUP_SIZE) % GROUP_SIZE
    if pad > 0:
        t = torch.nn.functional.pad(t, (0, pad))

    # Reshape to groups
    t = t.reshape(B, H, -1, GROUP_SIZE)

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

    B, H, S, D = quantized["shape"]
    N = quantized["N"]
    t = t.reshape(B, H, -1)[:, :, :N]
    t = t.reshape(B, H, S, D)

    # Inverse rotation
    R = _get_rotation(D, t.device, dtype)
    t = torch.matmul(t, R.T)

    return t
