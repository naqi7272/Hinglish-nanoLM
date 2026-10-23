"""Model hyper-parameters, and the three sizes we train."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class GPTConfig:
    vocab_size: int = 16384
    ctx_len: int = 512  # maximum sequence length the model sees
    n_layer: int = 6
    n_head: int = 4
    d_model: int = 256
    dropout: float = 0.0  # small models on lots of data don't need it
    rope_theta: float = 10000.0
    use_sdpa: bool = True  # PyTorch's fused attention kernel; same math, much faster on GPU

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head

    @property
    def ffn_hidden(self) -> int:
        # SwiGLU has 3 weight matrices instead of 2, so the hidden size is 8/3 * d instead of
        # 4 * d to keep the parameter count the same as a classic MLP. Rounded up to a
        # multiple of 64 for GPU-friendly shapes.
        h = int(8 * self.d_model / 3)
        return 64 * ((h + 63) // 64)

    def __post_init__(self) -> None:
        if self.d_model % self.n_head:
            raise ValueError("d_model must be divisible by n_head")
        if self.head_dim % 2:
            raise ValueError("head_dim must be even for RoPE")

    def to_dict(self) -> dict:
        return asdict(self)


# Non-embedding parameters per layer ≈ 12 * d^2 (attention 4d^2 + SwiGLU 8d^2).
SIZES = {
    "5m": dict(n_layer=6, n_head=4, d_model=256),  # ~4.8M non-embedding
    "15m": dict(n_layer=8, n_head=6, d_model=384),  # ~14.2M
    "30m": dict(n_layer=10, n_head=8, d_model=512),  # ~31.5M
}
