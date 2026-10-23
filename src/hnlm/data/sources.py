"""Read raw documents from a source described in configs/data.yaml.

Two loaders:
  * hf    : stream from the Hugging Face Hub with `datasets` (no full download needed).
  * local : read files from disk (.txt one doc per line, .jsonl, .csv/.tsv; optionally
            .gz/.bz2/.xz compressed; .tar/.tar.gz archives are streamed without unpacking;
            a directory is read recursively in sorted order). A relative `path` is
            resolved against the data directory.
"""

from __future__ import annotations

import bz2
import csv
import gzip
import io
import json
import lzma
import tarfile
from collections.abc import Iterable, Iterator
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


def load_data_config(path: str | Path, data_dir: str | Path | None = None) -> dict:
    """Load configs/data.yaml. `data_dir` overrides the file's data_dir, and relative local
    source paths are resolved against it (so they follow you onto Google Drive)."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["data_dir"] = str(data_dir or cfg["data_dir"])
    cfg["sources"] = [Source(**s) for s in cfg["sources"]]
    for src in cfg["sources"]:
        if src.path and not Path(src.path).is_absolute():
            src.path = str(Path(cfg["data_dir"]) / src.path)
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


_TAR_SUFFIXES = (".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz")


def _iter_records(f: Iterable[str], kind: str, text_field: str | None) -> Iterator[str]:
    """Yield documents from an open text file (or any iterable of lines) by file type."""
    if kind in (".jsonl", ".json"):
        for line in f:
            if line.strip():
                text = json.loads(line).get(text_field or "text")
                if text:
                    yield text
    elif kind in (".csv", ".tsv"):
        reader = csv.reader(f, delimiter="\t" if kind == ".tsv" else ",")
        header = next(reader, None)
        col = header.index(text_field) if text_field and header else 0
        if not text_field and header:
            yield header[col]  # no header declared: first row is data
        for row in reader:
            if len(row) > col and row[col].strip():
                yield row[col]
    elif kind == ".txt":
        for line in f:
            if line.strip():
                yield line.rstrip("\n")
    # anything else (e.g. a README) is skipped


def _iter_tar(path: Path, text_field: str | None) -> Iterator[str]:
    """Stream documents straight out of a .tar(.gz) archive, without unpacking it to disk.

    Mode "r|*" reads the archive front to back once (no seeking), which is all we need and
    works on slow network drives like a mounted Google Drive.
    """
    with tarfile.open(path, mode="r|*") as tar:
        for member in tar:
            name = Path(member.name)
            if not member.isfile() or name.name.startswith(".") or name.suffix in _OPENERS:
                continue
            raw = tar.extractfile(member)
            # A streamed member can't be wrapped in TextIOWrapper (it needs seekable()), so
            # decode it line by line ourselves. Nested compression inside the tar isn't
            # supported; such members are skipped like any unknown file type.
            lines = (line.decode("utf-8", errors="replace") for line in raw)
            yield from _iter_records(lines, _logical_suffix(name), text_field)


def _iter_local(src: Source) -> Iterator[str]:
    root = Path(src.path)
    if not root.exists():
        raise FileNotFoundError(f"{src.name}: {root} does not exist (download it first)")
    files = sorted(p for p in root.rglob("*") if p.is_file()) if root.is_dir() else [root]
    # Allow very long CSV cells. 2**31 - 1, not sys.maxsize: a C long is 32-bit on Windows.
    csv.field_size_limit(2**31 - 1)
    for path in files:
        if path.name.endswith(_TAR_SUFFIXES):
            yield from _iter_tar(path, src.text_field)
        else:
            with _open_text(path) as f:
                yield from _iter_records(f, _logical_suffix(path), src.text_field)


def write_jsonl(path: Path, rows: Iterator[dict]) -> int:
    """Write rows atomically (tmp file + rename) so a crash never leaves a half file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    n = 0
    # Big write buffer: far fewer, larger writes (matters a lot on network drives).
    with open(tmp, "w", encoding="utf-8", buffering=1 << 20) as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    tmp.replace(path)
    return n


def read_jsonl(path: Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)
