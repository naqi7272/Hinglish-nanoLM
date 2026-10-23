"""Validation loss and bits-per-byte.

Why bits-per-byte (BPB)? The training loss is "nats per TOKEN". A tokenizer that cuts text
into fewer, longer tokens makes each prediction harder but needs fewer of them, so per-token
losses of two different tokenizers can't be compared. BPB divides the total information by
the size of the TEXT instead:

    BPB = (sum of losses over all tokens, in nats) / ln(2) / (number of UTF-8 bytes)

Same text, same bytes, whatever the tokenizer: so it's a fair comparison. (Lower is better;
it's the size a perfect compressor built on this model would shrink the text to.)

To get the bytes of each predicted token we decode every vocabulary entry once
(`token_byte_lengths`). Special tokens like <|endoftext|> have 0 bytes and are left out of
both sums.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from tokenizers import Tokenizer


def _byte_decoder() -> dict[str, int]:
    """Inverse of GPT-2's bytes_to_unicode: byte-level BPE shows each byte as a printable
    character (e.g. space -> 'Ġ'); this maps those characters back to byte values."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
    bs += list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def token_byte_lengths(tok: Tokenizer) -> np.ndarray:
    dec = _byte_decoder()
    special = {t.content for t in tok.get_added_tokens_decoder().values()}
    out = np.zeros(tok.get_vocab_size(), dtype=np.int64)
    for piece, idx in tok.get_vocab().items():
        out[idx] = 0 if piece in special else len(piece)
        if piece not in special:
            assert all(ch in dec for ch in piece), piece
    return out


@torch.no_grad()
def evaluate(
    model,
    tokens: np.ndarray,
    token_bytes: torch.Tensor,
    ctx_len: int,
    batch_size: int,
    device,
    autocast,
    max_tokens: int | None = None,
) -> dict:
    """Average loss per token and BPB over (the first max_tokens of) a token file, read as
    consecutive non-overlapping windows."""
    model.eval()
    n = len(tokens) - 1 if max_tokens is None else min(len(tokens) - 1, max_tokens)
    win = min(ctx_len, n)
    starts = list(range(0, n - win + 1, win))
    nats, nbytes, ntok = 0.0, 0, 0
    for i in range(0, len(starts), batch_size):
        chunk = starts[i : i + batch_size]
        x = np.stack([tokens[s : s + win] for s in chunk]).astype(np.int64)
        y = np.stack([tokens[s + 1 : s + 1 + win] for s in chunk]).astype(np.int64)
        x, y = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
        with autocast():
            _, loss = model(x, targets=y, reduction="none")
        b = token_bytes[y]
        mask = b > 0
        nats += (loss * mask).sum().item()
        nbytes += b.sum().item()
        ntok += mask.sum().item()
    model.train()
    return {
        "loss": nats / max(ntok, 1),
        "bpb": nats / math.log(2) / max(nbytes, 1),
        "tokens": ntok,
    }
