"""Comprehensive xformers compatibility shim for MegaSaM/UniDepth on sm_120+ GPUs.

Modern xformers CUDA kernels don't support compute capability 12.0 (RTX 5060 Ti).
UniDepth's bundled DINOv2 fork has try/except fallbacks for ImportError, but the
real xformers imports fine — it just crashes at runtime when dispatching to a
missing CUDA kernel.

This shim (run BEFORE UniDepth imports):
1. Provides xformers.components.attention.NystromAttention (removed in xformers >=0.0.23)
2. Monkey-patches xformers.ops.memory_efficient_attention → PyTorch SDPA
3. Monkey-patches xformers.ops.unbind → torch.unbind

The nested-tensor ops (fmha.BlockDiagonalMask, index_select_cat, scaled_index_add)
are NOT patched — they are only used during training, never during MegaSaM inference.
"""
from __future__ import annotations

import math
import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── 1. NystromAttention (xformers.components.attention) ──────────────────

class NystromAttention(nn.Module):
    """Pure-PyTorch Nystrom attention matching the old xformers API."""

    def __init__(self, num_landmarks: int = 128, num_heads: int = 1,
                 dropout: float = 0.0, **kwargs):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, q, k, v, key_padding_mask=None):
        B, N, H, D = q.shape
        m = min(self.num_landmarks, N)
        if m >= N or N <= 256:
            return self._exact(q, k, v, key_padding_mask)

        if key_padding_mask is not None:
            mask = key_padding_mask.unsqueeze(-1).unsqueeze(-1)
            k = k.masked_fill(mask, 0.0)
            v = v.masked_fill(mask, 0.0)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        scale = math.sqrt(D)
        indices = torch.linspace(0, N - 1, m, device=q.device).long()
        q_l, k_l = q[:, :, indices, :], k[:, :, indices, :]

        ker1 = F.softmax(q @ k_l.transpose(-1, -2) / scale, dim=-1)
        ker2 = F.softmax(q_l @ k_l.transpose(-1, -2) / scale, dim=-1)
        ker3 = F.softmax(q_l @ k.transpose(-1, -2) / scale, dim=-1)
        ker2_inv = self._iterative_pinv(ker2)
        out = self.dropout(ker1 @ ker2_inv @ (ker3 @ v))
        return out.transpose(1, 2)

    def _exact(self, q, k, v, mask):
        B, N, H, D = q.shape
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        attn = q @ k.transpose(-1, -2) / math.sqrt(D)
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2), float("-inf"))
        out = self.dropout(F.softmax(attn, dim=-1)) @ v
        return out.transpose(1, 2)

    @staticmethod
    def _iterative_pinv(mat, n_iter=6):
        norm = mat.norm(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        Z = mat.transpose(-1, -2) / (norm * norm)
        for _ in range(n_iter):
            Z = 2 * Z - Z @ mat @ Z
        return Z


# ── 2. memory_efficient_attention → PyTorch SDPA ────────────────────────

def _memory_efficient_attention(q, k, v, attn_bias=None, p=0.0, scale=None):
    """Drop-in for xformers.ops.memory_efficient_attention.

    xformers layout: (B, M, H, D).  PyTorch SDPA layout: (B, H, M, D).
    """
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        attn_mask=None, dropout_p=p, scale=scale,
    )
    return out.transpose(1, 2)


def _unbind(x, dim=0):
    """Drop-in for xformers.ops.unbind."""
    return torch.unbind(x, dim=dim)


# ── Patch sys.modules ───────────────────────────────────────────────────

# 1. Provide xformers.components.attention.NystromAttention
_comp = types.ModuleType("xformers.components")
_comp.__path__ = []
_attn_mod = types.ModuleType("xformers.components.attention")
_attn_mod.NystromAttention = NystromAttention
_comp.attention = _attn_mod
sys.modules.setdefault("xformers.components", _comp)
sys.modules.setdefault("xformers.components.attention", _attn_mod)

# 2. Monkey-patch xformers.ops — replace CUDA-dependent attention with SDPA
try:
    import xformers.ops as _ops
    _ops.memory_efficient_attention = _memory_efficient_attention
    _ops.unbind = _unbind
except ImportError:
    pass
