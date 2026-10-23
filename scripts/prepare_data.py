"""Build the corpus: download -> clean -> dedup -> split -> stats.

Examples:
    uv run python scripts/prepare_data.py --stages peek
    uv run python scripts/prepare_data.py --stages download,clean,dedup,split,stats
    uv run python scripts/prepare_data.py --stages clean,dedup --only hingcorpus
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hnlm.data import pipeline
from hnlm.data.clean import CleanConfig
from hnlm.data.sources import load_data_config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--stages", default="download,clean,dedup,split,stats")
    ap.add_argument("--only", default=None, help="comma-separated source names")
    ap.add_argument("--data-dir", default=None, help="override data_dir (e.g. a Drive folder)")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    cfg = load_data_config(args.config)
    data_dir = Path(args.data_dir or cfg["data_dir"])
    sources = cfg["sources"]
    if args.only:
        wanted = set(args.only.split(","))
        sources = [s for s in sources if s.name in wanted]

    stages = args.stages.split(",")
    unknown = set(stages) - set(pipeline.STAGES)
    if unknown:
        ap.error(f"unknown stages {unknown}; choose from {pipeline.STAGES}")

    clean_cfg = CleanConfig(**cfg.get("clean", {}))
    dedup_cfg = cfg.get("dedup", {})

    for src in sources:
        for stage in stages:
            if stage == "peek":
                pipeline.peek(src)
            elif stage == "download":
                print(src.name, "download", pipeline.download(src, data_dir))
            elif stage == "clean":
                print(src.name, "clean", pipeline.clean(src, data_dir, clean_cfg))
            elif stage == "dedup":
                print(
                    src.name,
                    "dedup",
                    pipeline.dedup(src, data_dir, workers=args.workers, **dedup_cfg),
                )
            elif stage == "split":
                print(
                    src.name,
                    "split",
                    pipeline.split(src, data_dir, cfg["val_fraction"], cfg["seed"]),
                )

    if "stats" in stages:
        print(json.dumps(pipeline.stats(sources, data_dir), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
