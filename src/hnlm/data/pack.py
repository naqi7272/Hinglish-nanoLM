"""Tokenize the text splits once and store them as flat binary files of token ids.

data/bins/{tokenizer}/{source}.{split}.bin   uint16 token ids, documents separated by
                                             <|endoftext|> (16k vocab < 65,536 fits uint16)
data/bins/{tokenizer}/meta.json              token / byte / doc counts per file

Training then just reads random windows from these files via numpy memmap: no text
processing in the training loop at all, and nothing has to fit in RAM.
"""

from __future__ import annotations

import json
from itertools import islice
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm

EOT = "<|endoftext|>"


def _batches(path: Path, size: int):
    with open(path, encoding="utf-8") as f:
        while batch := [json.loads(line)["text"] for line in islice(f, size)]:
            yield batch


def pack_file(tok: Tokenizer, src: Path, dst: Path, batch_docs: int = 5000) -> dict:
    eot = tok.token_to_id(EOT)
    if eot is None or tok.get_vocab_size() > 2**16:
        raise ValueError("tokenizer needs <|endoftext|> and a vocab that fits in uint16")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".bin.tmp")
    stats = {"tokens": 0, "bytes": 0, "docs": 0}
    with open(tmp, "wb") as out:
        for batch in tqdm(_batches(src, batch_docs), desc=f"pack {src.name}", unit="batch"):
            encs = tok.encode_batch(batch, add_special_tokens=False)  # multi-threaded (Rust)
            for text, enc in zip(batch, encs):
                ids = np.array(enc.ids + [eot], dtype=np.uint16)
                out.write(ids.tobytes())
                stats["tokens"] += len(ids)
                stats["bytes"] += len(text.encode("utf-8"))
                stats["docs"] += 1
    tmp.replace(dst)
    return stats


def pack_all(tokenizer_path: Path, splits_dir: Path, out_dir: Path, force: bool = False) -> dict:
    tok = Tokenizer.from_file(str(tokenizer_path))
    meta_path = out_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.update(
        tokenizer=tokenizer_path.stem, vocab_size=tok.get_vocab_size(), eot_id=tok.token_to_id(EOT)
    )
    files = meta.setdefault("files", {})
    for src in sorted(splits_dir.glob("*.jsonl")):
        name = src.stem  # e.g. "hingcorpus.train"
        dst = out_dir / f"{name}.bin"
        if dst.exists() and name in files and not force:
            print(f"  skip: {dst} already exists")
            continue
        files[name] = pack_file(tok, src, dst)
        print(f"  {name}: {files[name]}")
        meta_path.write_text(json.dumps(meta, indent=2))
    return meta
