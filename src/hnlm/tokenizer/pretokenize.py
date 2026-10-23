"""Pre-tokenization: chop text into "words" before BPE runs.

BPE never merges across these chunk boundaries. That stops it from learning silly tokens
like "dog." or "the dog" (word + punctuation, or two words glued together), and makes
training fast: we only need to count each distinct chunk once.

The pattern is GPT-4's (cl100k) split pattern with one change: everywhere it says
\\p{L} (a letter) we also accept \\p{M} (a combining mark). Devanagari vowel signs such as
ि ी ु ं are marks, not letters. With the original GPT-4 pattern, "किताबें" is split into
["क", "ि", "त", "ा", "ब", "े", "ं"]-style fragments at every vowel sign, so BPE can never learn
whole Hindi syllables. (This is one real reason GPT-4's tokenizer was bad at Hindi.)
"""

from __future__ import annotations

import regex

# Read it piece by piece:
#   '(?i:[sdmt]|ll|ve|re)        English contractions: 's 'll 've ...
#   [^\r\n\p{L}\p{M}\p{N}]?+[\p{L}\p{M}]+   a word, optionally with ONE leading space/punct
#   \p{N}{1,3}                   numbers in groups of up to 3 digits
#    ?[^\s\p{L}\p{M}\p{N}]++[\r\n]*   punctuation runs (incl. Devanagari danda "।")
#   \s*[\r\n] | \s+(?!\S) | \s+  newlines and other whitespace
SPLIT_PATTERN = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{M}\p{N}]?+[\p{L}\p{M}]+|\p{N}{1,3}|"""
    r""" ?[^\s\p{L}\p{M}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)
_SPLIT_RE = regex.compile(SPLIT_PATTERN)


def pretokenize(text: str) -> list[str]:
    """'Yaar, मैं Delhi जा रहा हूँ!' -> ['Yaar', ',', ' मैं', ' Delhi', ' जा', ' रहा', ' हूँ', '!']"""
    return _SPLIT_RE.findall(text)
