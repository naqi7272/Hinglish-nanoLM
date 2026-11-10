"""Turning next-token scores into text.

The model gives logits (one score per vocabulary entry). To pick the next token:
  * temperature: divide logits by T before softmax. T < 1 sharpens the distribution (safer,
    more repetitive), T > 1 flattens it (more creative, more mistakes). T = 0 means greedy.
  * top-k: keep only the k highest-scoring tokens, so a long tail of junk can't be sampled.
  * top-p (nucleus): keep the smallest set of tokens whose probabilities add up to p.
    Adapts to the situation: few candidates when the model is sure, many when it isn't.
"""

from __future__ import annotations

import torch

from hnlm.model.gpt import GPT


def sample_next(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """logits: (B, vocab) -> next token ids (B, 1)."""
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits.float() / temperature
    if top_k is not None:
        kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p is not None:
        sorted_logits, order = logits.sort(dim=-1, descending=True)
        probs = sorted_logits.softmax(dim=-1)
        # Drop a token if the tokens ranked ABOVE it already reach p (the top token always stays).
        drop = probs.cumsum(dim=-1) - probs >= top_p
        sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, order, sorted_logits)
    probs = logits.softmax(dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)


@torch.no_grad()
def generate(
    model: GPT,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    use_cache: bool = True,
    stop_token: int | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Extend idx (B, T) by up to max_new_tokens tokens.

    use_cache=True : process the prompt once ("prefill"), then feed ONE new token per step,
                     reusing stored keys/values.
    use_cache=False: re-run the whole sequence every step. Same output, much more compute;
                     kept for the speed comparison and as a correctness reference.
    """
    model.eval()
    ctx = model.cfg.ctx_len
    max_new_tokens = min(max_new_tokens, ctx - idx.shape[1])  # the cache can't grow past ctx
    cache = model.new_kv_cache(idx.shape[0]) if use_cache else None

    logits, _ = model(idx, cache=cache)  # prefill
    for _ in range(max_new_tokens):
        nxt = sample_next(logits[:, -1, :], temperature, top_k, top_p, generator)
        idx = torch.cat([idx, nxt], dim=1)
        if stop_token is not None and bool((nxt == stop_token).all()):
            break
        if use_cache:
            logits, _ = model(nxt, cache=cache)  # only the new token
        else:
            logits, _ = model(idx)  # everything again
    return idx
