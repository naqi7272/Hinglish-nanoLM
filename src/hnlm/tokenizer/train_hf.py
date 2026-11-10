"""Production byte-level BPE with Hugging Face `tokenizers` (Rust, so ~1000x faster than
bpe_scratch.py), set up to behave like our scratch version:

  * same pre-tokenization regex (GPT-4's pattern + \\p{M}, so Hindi words stay whole)
  * byte-level: every string is representable, nothing is ever "unknown"
  * no normalizer: the corpus pipeline already applied NFC, and leaving text untouched
    keeps the round-trip guarantee decode(encode(x)) == x for ANY input

The only conceptual difference from bpe_scratch.py is the byte-to-character trick that
HF/GPT-2 use internally (each byte is shown as a printable character such as "Ġ" for space);
it doesn't change which merges are learned.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Iterator
from pathlib import Path

from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

from hnlm.tokenizer.pretokenize import SPLIT_PATTERN


def build_tokenizer() -> Tokenizer:
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(SPLIT_PATTERN), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tok.decoder = decoders.ByteLevel()
    return tok


def train_tokenizer(
    texts: Iterable[str],
    vocab_size: int,
    special_tokens: list[str],
    min_frequency: int = 2,
) -> Tokenizer:
    tok = build_tokenizer()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # all 256 bytes, always
        show_progress=True,
    )
    tok.train_from_iterator(texts, trainer=trainer)
    return tok


def sample_texts(path: Path, sample_mb: float, seed: int) -> Iterator[str]:
    """Yield a random ~sample_mb MB subset of a .jsonl file, spread over the whole file
    (not just its beginning, which could be all from one day of tweets)."""
    keep = min(1.0, sample_mb * 1e6 / path.stat().st_size)
    rng = random.Random(f"{seed}:{path.name}")
    with open(path, encoding="utf-8") as f:
        for line in f:
            if rng.random() < keep:
                yield json.loads(line)["text"]
