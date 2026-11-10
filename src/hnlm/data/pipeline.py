"""The corpus pipeline, one function per stage. Each stage reads the previous stage's
files and writes its own, so any stage can be re-run on its own:

    raw/{src}.jsonl  --clean-->  clean/{src}.jsonl  --dedup-->  dedup/{src}.jsonl
                     --split-->  splits/{src}.{train,val}.jsonl   (or {src}.eval.jsonl)
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
from collections import Counter
from itertools import islice
from pathlib import Path

import numpy as np
from tqdm import tqdm

from hnlm.data.clean import CleanConfig, clean_document
from hnlm.data.dedup import MinHasher, exact_hash, find_duplicates
from hnlm.data.sources import Source, iter_texts, read_jsonl, write_jsonl
from hnlm.data.split import assign_split

STAGES = ("peek", "download", "clean", "dedup", "split", "stats")


def _path(data_dir: Path, stage: str, name: str) -> Path:
    return data_dir / stage / f"{name}.jsonl"


def _count_lines(path: Path) -> int:
    with open(path, "rb") as f:
        return sum(1 for _ in f)


def _done(path: Path, force: bool) -> bool:
    """Stages skip work whose output already exists, so after a disconnect you can simply
    re-run the same command. Outputs are written atomically, so 'exists' means 'complete'."""
    if path.exists() and not force:
        print(f"  skip: {path} already exists (use --force to redo)")
        return True
    return False


def _pool(workers: int | None, initializer, initargs):
    # "spawn" (not fork): forking a process that has already started torch threads can deadlock.
    return mp.get_context("spawn").Pool(workers or os.cpu_count() or 1, initializer, initargs)


# ---- stages ----------------------------------------------------------------------------


def peek(src: Source, n: int = 3) -> None:
    print(f"\n== {src.name} ({src.license}) ==")
    for i, text in enumerate(islice(iter_texts(src), n)):
        print(f"[{i}] {text[:300]!r}")


def download(src: Source, data_dir: Path, force: bool = False) -> dict:
    if _done(_path(data_dir, "raw", src.name), force):
        return {}
    budget = int(src.budget_mb * 1e6)
    used = 0

    def rows():
        nonlocal used
        bar = tqdm(total=budget, unit="B", unit_scale=True, desc=f"download {src.name}")
        for text in iter_texts(src):
            size = len(text.encode("utf-8"))
            yield {"text": text}
            used += size
            bar.update(size)
            if used >= budget:
                break
        bar.close()

    docs = write_jsonl(_path(data_dir, "raw", src.name), rows())
    return {"docs": docs, "bytes": used}


_CLEAN_CFG: CleanConfig | None = None


def _init_clean_worker(cfg: CleanConfig) -> None:
    global _CLEAN_CFG
    _CLEAN_CFG = cfg


def _clean_line(line: str) -> tuple[str, str] | None:
    return clean_document(json.loads(line)["text"], _CLEAN_CFG)


def clean(
    src: Source, data_dir: Path, cfg: CleanConfig, workers: int | None = None, force: bool = False
) -> dict:
    out_path = _path(data_dir, "clean", src.name)
    if _done(out_path, force):
        return {}
    in_path = _path(data_dir, "raw", src.name)
    n = _count_lines(in_path)
    reasons = Counter()

    def rows():
        with (
            open(in_path, encoding="utf-8") as f,
            _pool(workers, _init_clean_worker, (cfg,)) as pool,
        ):
            results = pool.imap(_clean_line, f, chunksize=2000)
            for out in tqdm(results, total=n, desc=f"clean {src.name}"):
                if out is None:
                    reasons["dropped"] += 1
                    continue
                text, script = out
                reasons[f"script_{script}"] += 1
                yield {"text": text, "script": script}

    kept = write_jsonl(out_path, rows())
    return {"kept": kept, **reasons}


_HASHER: MinHasher | None = None


def _init_worker(hasher: MinHasher) -> None:
    global _HASHER
    _HASHER = hasher


def _hash_line(line: str) -> tuple[np.ndarray, int]:
    text = json.loads(line)["text"]
    return _HASHER.signature(text), exact_hash(text)


def dedup(
    src: Source,
    data_dir: Path,
    num_perm: int = 64,
    bands: int = 16,
    ngram: int = 5,
    max_chars: int = 4000,
    jaccard_threshold: float = 0.7,
    workers: int | None = None,
    force: bool = False,
) -> dict:
    out_path = _path(data_dir, "dedup", src.name)
    if _done(out_path, force):
        return {}
    in_path = _path(data_dir, "clean", src.name)
    n = _count_lines(in_path)
    hasher = MinHasher(num_perm=num_perm, ngram=ngram, max_chars=max_chars)

    # Pass 1: signatures for every doc (preallocated: N x num_perm uint32).
    sigs = np.empty((n, num_perm), dtype=np.uint32)
    exact = np.empty(n, dtype=np.uint64)
    with open(in_path, encoding="utf-8") as f, _pool(workers, _init_worker, (hasher,)) as pool:
        results = pool.imap(_hash_line, f, chunksize=512)
        for i, (sig, h) in enumerate(tqdm(results, total=n, desc=f"minhash {src.name}")):
            sigs[i], exact[i] = sig, h

    keep = find_duplicates(sigs, exact, bands=bands, jaccard_threshold=jaccard_threshold)

    # Pass 2: copy the kept docs.
    def rows():
        for i, row in enumerate(read_jsonl(in_path)):
            if keep[i]:
                yield row

    kept = write_jsonl(out_path, rows())
    return {"in": n, "kept": kept, "removed": n - kept}


def split(src: Source, data_dir: Path, val_fraction: float, seed: int, force: bool = False) -> dict:
    in_path = _path(data_dir, "dedup", src.name)
    out_dir = data_dir / "splits"
    marker = out_dir / f"{src.name}.{'eval' if src.role == 'eval' else 'val'}.jsonl"
    if _done(marker, force):
        return {}
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    if src.role == "eval":
        counts["eval"] = write_jsonl(out_dir / f"{src.name}.eval.jsonl", read_jsonl(in_path))
        return dict(counts)

    final = {s: out_dir / f"{src.name}.{s}.jsonl" for s in ("train", "val")}
    tmp = {s: p.with_suffix(".jsonl.tmp") for s, p in final.items()}
    files = {s: open(p, "w", encoding="utf-8", buffering=1 << 20) for s, p in tmp.items()}
    try:
        for row in read_jsonl(in_path):
            s = assign_split(row["text"], val_fraction, seed)
            files[s].write(json.dumps(row, ensure_ascii=False) + "\n")
            counts[s] += 1
    finally:
        for f in files.values():
            f.close()
    tmp["train"].replace(final["train"])
    tmp["val"].replace(final["val"])  # renamed last: its existence means "split finished"
    return dict(counts)


def stats(sources: list[Source], data_dir: Path) -> dict:
    """Docs / MB / script mix at every stage, so the README can show the funnel."""
    out: dict = {}
    for src in sources:
        out[src.name] = {}
        for stage in ("raw", "clean", "dedup"):
            path = _path(data_dir, stage, src.name)
            if not path.exists():
                continue
            docs, nbytes, scripts = 0, 0, Counter()
            for row in read_jsonl(path):
                docs += 1
                nbytes += len(row["text"].encode("utf-8"))
                if "script" in row:
                    scripts[row["script"]] += 1
            out[src.name][stage] = {"docs": docs, "mb": round(nbytes / 1e6, 2), **scripts}
    (data_dir / "stats.json").write_text(json.dumps(out, indent=2))
    return out
