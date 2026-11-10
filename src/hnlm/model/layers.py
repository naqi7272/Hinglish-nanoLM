"""The building blocks of the GPT, each explained in plain words.

Shapes used in comments:  B = batch, T = new tokens in this call, S = all tokens attended to
(past + new, with a KV cache), H = heads, D = d_model, Dh = head_dim = D / H.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from hnlm.model.config import GPTConfig

# ---- RMSNorm ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """Rescale each vector to have root-mean-square 1, then multiply by a learned gain.

    LayerNorm also subtracts the mean and adds a bias; RMSNorm skips both. It is cheaper and
    works just as well (used by Llama, Qwen, Mistral). Keeping activations at a steady scale
    is what lets deep stacks of layers train without blowing up.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute in float32 even under mixed precision: squaring small fp16 values underflows.
        x32 = x.float()
        normed = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return normed.type_as(x) * self.weight


# ---- RoPE ------------------------------------------------------------------------------


class RotaryEmbedding(nn.Module):
    """Rotary position embeddings.

    Attention compares tokens with dot products q·k, which by themselves ignore order. RoPE
    rotates each (q, k) vector by an angle proportional to its position. Rotating q by angle
    m·θ and k by n·θ makes q·k depend only on (m - n): the RELATIVE distance. Different pairs
    of dimensions rotate at different speeds θ_i (fast ones for nearby detail, slow ones for
    long range), like the hands of a clock.
    """

    def __init__(self, head_dim: int, max_len: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))  # (Dh/2,)
        angles = torch.outer(torch.arange(max_len).float(), inv_freq)  # (max_len, Dh/2)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        """x: (B, H, T, Dh) for positions start_pos .. start_pos + T - 1."""
        t = x.shape[-2]
        cos = self.cos[start_pos : start_pos + t].to(x.dtype)  # (T, Dh/2)
        sin = self.sin[start_pos : start_pos + t].to(x.dtype)
        x1, x2 = x[..., ::2], x[..., 1::2]  # pair up dimensions (0,1), (2,3), ...
        # 2-D rotation of every pair: [x1, x2] -> [x1 cos - x2 sin, x1 sin + x2 cos]
        out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
        return out.flatten(-2)


# ---- KV cache --------------------------------------------------------------------------


class KVCache:
    """Remembers each layer's keys and values for tokens already processed.

    Without it, generating token number t recomputes attention inputs for all t previous
    tokens, again and again: O(t^2) work per sequence. With it, each new token only computes
    its own q, k, v and attends to the stored k, v. Memory per layer: 2 * B * H * max_len * Dh.
    """

    def __init__(self, batch: int, n_head: int, max_len: int, head_dim: int, device, dtype):
        shape = (batch, n_head, max_len, head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.length = 0  # how many positions are filled

    def append(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        t = k.shape[-2]
        if self.length + t > self.k.shape[-2]:
            raise ValueError("KV cache is full (sequence longer than ctx_len)")
        self.k[:, :, self.length : self.length + t] = k
        self.v[:, :, self.length : self.length + t] = v
        self.length += t
        return self.k[:, :, : self.length], self.v[:, :, : self.length]


# ---- attention -------------------------------------------------------------------------


def causal_mask(t: int, s: int, device) -> torch.Tensor:
    """(T, S) boolean mask, True = allowed. The T new queries sit at the END of the S keys,
    so query i (absolute position s - t + i) may look at keys 0 .. s - t + i."""
    return torch.ones(t, s, dtype=torch.bool, device=device).tril(diagonal=s - t)


class CausalSelfAttention(nn.Module):
    """Every token looks back at earlier tokens and pulls in information from them.

    1. Project each token into a query (what am I looking for?), a key (what do I contain?)
       and a value (what do I pass on?), split across H heads that each attend separately.
    2. score[i, j] = q_i · k_j / sqrt(Dh): how relevant token j is to token i. Dividing by
       sqrt(Dh) keeps scores from growing with dimension, so softmax doesn't saturate.
    3. Causal mask: token i may only see j <= i. Future scores are set to -inf, so they get
       zero weight after softmax. This is what makes it a LANGUAGE MODEL: no peeking ahead.
    4. softmax over j turns scores into weights that sum to 1; output_i = Σ_j weight_ij · v_j.
    5. Concatenate the heads and mix them with an output projection.
    """

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.n_head, self.head_dim = cfg.n_head, cfg.head_dim
        self.use_sdpa = cfg.use_sdpa
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(
        self, x: torch.Tensor, rope: RotaryEmbedding, cache: KVCache | None = None
    ) -> torch.Tensor:
        b, t, d = x.shape
        start_pos = cache.length if cache is not None else 0

        q, k, v = self.qkv(x).split(d, dim=-1)  # each (B, T, D)
        # (B, T, D) -> (B, H, T, Dh): each head gets its own slice of the dimensions.
        q, k, v = (z.view(b, t, self.n_head, self.head_dim).transpose(1, 2) for z in (q, k, v))
        q, k = rope(q, start_pos), rope(k, start_pos)
        if cache is not None:
            k, v = cache.append(k, v)  # now (B, H, S, Dh) with S = start_pos + T
        s = k.shape[-2]

        dropout = self.dropout if self.training else 0.0
        if self.use_sdpa:
            # PyTorch's is_causal assumes T == S, so with a cache we pass the mask explicitly
            # (a single new token may attend to everything: no mask needed at all).
            if s == t:
                out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=dropout)
            else:
                mask = None if t == 1 else causal_mask(t, s, x.device)
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout)
        else:
            # The same thing written out by hand (used in tests to check the fused kernel).
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, H, T, S)
            scores = scores.masked_fill(~causal_mask(t, s, x.device), float("-inf"))
            weights = F.dropout(scores.softmax(dim=-1), p=dropout, training=self.training)
            out = weights @ v  # (B, H, T, Dh)

        out = out.transpose(1, 2).reshape(b, t, d)  # heads back side by side
        return self.proj(out)


# ---- feed-forward ----------------------------------------------------------------------


class SwiGLU(nn.Module):
    """Per-token MLP: attention mixes information BETWEEN tokens, this processes each token.

    out = W_down( SiLU(W_gate x) * (W_up x) ). The gate decides, per hidden unit, how much of
    the "up" signal passes. Gated MLPs consistently beat plain ReLU/GELU MLPs (Shazeer 2020)
    and are used in Llama, Qwen, Mistral.
    """

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.down = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


# ---- transformer block -----------------------------------------------------------------


class Block(nn.Module):
    """Pre-norm residual block:  x = x + Attn(Norm(x));  x = x + MLP(Norm(x)).

    The residual "highway" (x + ...) lets gradients flow straight through many layers; each
    layer only has to learn a small update. Normalizing BEFORE each sub-layer (pre-norm) is
    what makes deep transformers stable without careful learning-rate tricks.
    """

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, rope: RotaryEmbedding, cache: KVCache | None = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), rope, cache)
        return x + self.mlp(self.norm2(x))
