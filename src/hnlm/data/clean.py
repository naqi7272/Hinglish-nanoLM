"""Text cleaning and script detection.

Goal: keep natural Hindi / Hinglish / English text and throw away noise, without
destroying anything a tokenizer needs to learn. Each rule below says why it exists.

A note on Devanagari: a syllable like "कि" is TWO code points, the consonant क (a Letter)
plus the vowel sign ि (a combining Mark, Unicode category "Mc"). So anywhere we ask
"is this a letter?" we must also accept marks, or we'd chop Hindi words in half.
"""

from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass

# ---- character classes ---------------------------------------------------------------

DEVANAGARI = re.compile(r"[ऀ-ॿ꣠-ꣿ]")
LATIN = re.compile(r"[A-Za-zÀ-ɏ]")

URL = re.compile(r"(https?://\S+|www\.\S+)")
MENTION = re.compile(r"@\w+")
RETWEET = re.compile(r"^RT\b\s*(@\w+)?:?\s*")
HASHTAG = re.compile(r"#(\w+)")
# Zero-width space, BOM, soft hyphen, word joiner. We KEEP ZWJ (U+200D) and ZWNJ (U+200C):
# in Devanagari they change how conjunct consonants render, so they carry meaning.
INVISIBLE = re.compile(r"[​﻿­⁠]")
SPACES = re.compile(r"[^\S\n]+")  # any whitespace except newline
MANY_NEWLINES = re.compile(r"\n{3,}")


@dataclass(frozen=True)
class CleanConfig:
    min_chars: int = 20
    min_words: int = 3
    max_chars: int = 100_000
    max_repeat_char: int = 3
    min_allowed_letter_frac: float = 0.9
    max_digit_symbol_frac: float = 0.3
    drop_emoji: bool = True


def _is_emoji_or_symbol(ch: str) -> bool:
    # "So" = Symbol, other (emoji, dingbats). "Cs"/"Co" = surrogates / private use (garbage).
    return unicodedata.category(ch) in ("So", "Cs", "Co")


def script_stats(text: str) -> dict[str, int]:
    """Count letters by script. Combining marks count with the script they belong to."""
    deva = latn = other = 0
    for ch in text:
        cat = unicodedata.category(ch)
        if not (cat.startswith("L") or cat.startswith("M")):
            continue
        if DEVANAGARI.match(ch):
            deva += 1
        elif LATIN.match(ch):
            latn += 1
        else:
            other += 1
    return {"deva": deva, "latn": latn, "other": other}


def script_label(stats: dict[str, int]) -> str:
    """'deva', 'latn', 'mixed' (both scripts present in bulk), or 'other'."""
    total = stats["deva"] + stats["latn"] + stats["other"]
    if total == 0:
        return "other"
    deva, latn = stats["deva"] / total, stats["latn"] / total
    if deva >= 0.8:
        return "deva"
    if latn >= 0.8:
        return "latn"
    if deva + latn >= 0.8:
        return "mixed"
    return "other"


def normalize(text: str, cfg: CleanConfig) -> str:
    text = html.unescape(text)  # "&amp;" -> "&" (common in tweets)
    # NFC: one canonical byte sequence per visible character. Without it, the same Hindi
    # word can arrive in two different byte forms and the tokenizer learns both.
    text = unicodedata.normalize("NFC", text)
    text = INVISIBLE.sub("", text)
    text = RETWEET.sub("", text)
    text = URL.sub(" ", text)
    text = MENTION.sub(" ", text)
    text = HASHTAG.sub(r"\1", text)  # keep the word, drop the '#'
    text = "".join(
        ch
        for ch in text
        if ch in "\n\t"
        or (unicodedata.category(ch) != "Cc" and not (cfg.drop_emoji and _is_emoji_or_symbol(ch)))
    )
    if cfg.max_repeat_char:
        n = cfg.max_repeat_char
        text = re.sub(r"(.)\1{" + str(n) + ",}", lambda m: m.group(1) * n, text)
    text = SPACES.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = MANY_NEWLINES.sub("\n\n", text)
    return text.strip()


def clean_document(text: str, cfg: CleanConfig) -> tuple[str, str] | None:
    """Return (clean_text, script_label), or None if the document should be dropped."""
    text = normalize(text, cfg)
    if not (cfg.min_chars <= len(text) <= cfg.max_chars):
        return None
    if len(text.split()) < cfg.min_words:
        return None

    stats = script_stats(text)
    letters = stats["deva"] + stats["latn"] + stats["other"]
    if letters == 0:
        return None
    if (stats["deva"] + stats["latn"]) / letters < cfg.min_allowed_letter_frac:
        return None  # e.g. Urdu, Bengali, Chinese: out of scope for this project

    non_space = sum(1 for ch in text if not ch.isspace())
    if (non_space - letters) / max(non_space, 1) > cfg.max_digit_symbol_frac:
        return None  # tables, phone numbers, scores, ASCII art

    return text, script_label(stats)
