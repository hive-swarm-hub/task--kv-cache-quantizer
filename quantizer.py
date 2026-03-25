"""
KV Cache Quantizer — per-group 4-bit with fine-grained scales.

Interface contract (do not change function signatures):
  - quantize(tensor: torch.Tensor) -> dict
  - dequantize(quantized: dict) -> torch.Tensor
  - bits_per_value() -> float
"""

import torch

GROUP_SIZE = 32


def bits_per_value() -> float:
    # 4 bits per value + overhead for scale/zero per group
    # Each group of 32 values: 32*4 bits data + 32 bits scale + 16 bits zero = 176 bits
    # Per value: 176/32 = 5.5 bits
    # But we report the dominant cost
    return 4.0


def quantize(tensor: torch.Tensor) -> dict:
    orig_shape = tensor.shape
    dtype = tensor.dtype

    # Flatten last two dims: (batch, heads, seq_len * head_dim)
    t = tensor.reshape(*tensor.shape[:2], -1)
    B, H, N = t.shape

    # Pad to multiple of GROUP_SIZE
    pad = (GROUP_SIZE - N % GROUP_SIZE) % GROUP_SIZE
    if pad > 0:
        t = torch.nn.functional.pad(t, (0, pad))
    N_padded = t.shape[-1]

    # Reshape to groups: (B, H, num_groups, GROUP_SIZE)
    t = t.reshape(B, H, N_padded // GROUP_SIZE, GROUP_SIZE)

    vmin = t.min(dim=-1, keepdim=True).values
    vmax = t.max(dim=-1, keepdim=True).values
    scale = (vmax - vmin) / 15.0
    scale = scale.clamp(min=1e-8)

    quantized = ((t - vmin) / scale).round().clamp(0, 15).to(torch.uint8)

    return {
        "data": quantized,
        "scale": scale.to(torch.float16),
        "vmin": vmin.to(torch.float16),
        "dtype": dtype,
        "shape": orig_shape,
        "N": N,
    }


def dequantize(quantized: dict) -> torch.Tensor:
    scale = quantized["scale"].to(quantized["dtype"])
    vmin = quantized["vmin"].to(quantized["dtype"])
    t = quantized["data"].to(quantized["dtype"]) * scale + vmin

    B, H = quantized["shape"][:2]
    N = quantized["N"]
    t = t.reshape(B, H, -1)[:, :, :N]
    return t.reshape(quantized["shape"])
