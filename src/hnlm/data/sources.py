"""Read raw documents from a source described in configs/data.yaml.

Two loaders:
  * hf    : stream from the Hugging Face Hub with `datasets` (no full download needed).
  * local : read files from disk (.txt one doc per line, .jsonl, .csv/.tsv; optionally
            .gz/.bz2/.xz compressed; a directory is read recursively in sorted order).
"""

from __future__ import annotations

import bz2
import csv
import gzip
import io
import json
import lzma
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class Source:
    name: str
    role: str  # "train" or "eval"
    loader: str  # "hf" or "local"
    license: str
    script: str = ""
    url: str = ""
    hf_id: str | None = None
    hf_kwargs: dict = field(default_factory=dict)
    path: str | None = None
    text_field: str | None = "text"
    budget_mb: float = 100.0


def load_data_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["sources"] = [Source(**s) for s in cfg["sources"]]
    return cfg


def iter_texts(src: Source) -> Iterator[str]:
    if src.loader == "hf":
        yield from _iter_hf(src)
    elif src.loader == "local":
        yield from _iter_local(src)
    else:
        raise ValueError(f"unknown loader {src.loader!r} for source {src.name}")


def _iter_hf(src: Source) -> Iterator[str]:
    from datasets import load_dataset  # optional dependency: `uv sync --extra data`

    ds = load_dataset(src.hf_id, streaming=True, **src.hf_kwargs)
    for row in ds:
        if src.text_field not in row:
            raise KeyError(
                f"{src.name}: field {src.text_field!r} not found; columns are {list(row)}. "
                "Fix text_field in configs/data.yaml."
            )
        text = row[src.text_field]
        if isinstance(text, str) and text.strip():
            yield text


_OPENERS = {".gz": gzip.open, ".bz2": bz2.open, ".xz": lzma.open}


def _open_text(path: Path) -> io.TextIOBase:
    opener = _OPENERS.get(path.suffix)
    if opener:
        return opener(path, "rt", encoding="utf-8", errors="replace")
    return open(path, encoding="utf-8", errors="replace")


def _logical_suffix(path: Path) -> str:
    """'corpus.csv.gz' -> '.csv'."""
    suffixes = [s for s in path.suffixes if s not in _OPENERS]
    return suffixes[-1] if suffixes else ".txt"


def _iter_local(src: Source) -> Iterator[str]:
    root = Path(src.path)
    if not root.exists():
        raise FileNotFoundError(f"{src.name}: {root} does not exist (download it first)")
    files = sorted(p for p in root.rglob("*") if p.is_file()) if root.is_dir() else [root]
    # Allow very long CSV cells. 2**31 - 1, not sys.maxsize: a C long is 32-bit on Windows.
    csv.field_size_limit(2**31 - 1)
    for path in files:
        kind = _logical_suffix(path)
        with _open_text(path) as f:
            if kind == ".jsonl" or kind == ".json":
                for line in f:
                    if line.strip():
                        text = json.loads(line).get(src.text_field or "text")
                        if text:
                            yield text
            elif kind in (".csv", ".tsv"):
                reader = csv.reader(f, delimiter="\t" if kind == ".tsv" else ",")
                header = next(reader, None)
                col = header.index(src.text_field) if src.text_field and header else 0
                if not src.text_field and header:
                    yield header[col]  # no header declared: first row is data
                for row in reader:
                    if len(row) > col and row[col].strip():
                        yield row[col]
            elif kind == ".txt":
                for line in f:
                    if line.strip():
                        yield line.rstrip("\n")
            # anything else (e.g. a README in the folder) is skipped


def write_jsonl(path: Path, rows: Iterator[dict]) -> int:
    """Write rows atomically (tmp file + rename) so a crash never leaves a half file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    n = 0
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    tmp.replace(path)
    return n


def read_jsonl(path: Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)
