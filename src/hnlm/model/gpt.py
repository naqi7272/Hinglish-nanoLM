"""Decoder-only GPT in plain PyTorch.

token ids -> embedding -> N x Block -> RMSNorm -> linear head (tied to the embedding)
          -> logits: a score for every vocabulary entry as the NEXT token, at every position.

Training objective: cross-entropy between the logits at position t and the real token at
t + 1. That's it: everything a language model "knows" comes from predicting the next token.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from hnlm.model.config import GPTConfig
from hnlm.model.layers import Block, KVCache, RMSNorm, RotaryEmbedding


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        # Weight tying: the input embedding ("id -> vector") and the output layer
        # ("vector -> score per id") share ONE matrix. It saves vocab * d parameters (4-8M
        # here, a big share of a small model) and usually improves quality slightly.
        self.lm_head.weight = self.tok_emb.weight
        self.rope = RotaryEmbedding(cfg.head_dim, cfg.ctx_len, cfg.rope_theta)
        self.apply(self._init_weights)
        # GPT-2 trick: scale down the layers that write into the residual stream, so the
        # stream's variance doesn't grow with depth at initialization.
        for name, p in self.named_parameters():
            if name.endswith(("attn.proj.weight", "mlp.down.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())  # tied weights are counted once
        return n - self.tok_emb.weight.numel() if non_embedding else n

    def new_kv_cache(self, batch: int, device=None, dtype=None) -> list[KVCache]:
        p = next(self.parameters())
        return [
            KVCache(
                batch,
                self.cfg.n_head,
                self.cfg.ctx_len,
                self.cfg.head_dim,
                device or p.device,
                dtype or p.dtype,
            )
            for _ in self.blocks
        ]

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        cache: list[KVCache] | None = None,
        loss_mask: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """idx: (B, T) token ids. targets: (B, T) the next token for each position.

        Returns logits (B, T, vocab) and the loss (None without targets). With
        reduction="none" the loss is per position (B, T), which bits-per-byte needs.
        """
        start = cache[0].length if cache is not None else 0
        if start + idx.shape[1] > self.cfg.ctx_len:
            raise ValueError(f"sequence longer than ctx_len={self.cfg.ctx_len}")

        x = self.drop(self.tok_emb(idx))
        for i, block in enumerate(self.blocks):
            x = block(x, self.rope, cache[i] if cache is not None else None)
        logits = self.lm_head(self.norm(x))

        loss = None
        if targets is not None:
            per_pos = F.cross_entropy(
                logits.float().view(-1, logits.size(-1)), targets.reshape(-1), reduction="none"
            ).view(targets.shape)
            if reduction == "none":
                loss = per_pos
            elif loss_mask is not None:
                loss = (per_pos * loss_mask).sum() / loss_mask.sum().clamp(min=1)
            else:
                loss = per_pos.mean()
        return logits, loss
