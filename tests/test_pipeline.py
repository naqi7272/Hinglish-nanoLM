import json

from hnlm.data import pipeline
from hnlm.data.clean import CleanConfig
from hnlm.data.sources import Source


def test_pipeline_end_to_end(tmp_path, tiny_corpus):
    raw = tmp_path / "corpus.txt"
    lines = tiny_corpus + tiny_corpus[:5] + ["ok", "https://spam.example 🎉"]
    raw.write_text("\n".join(lines), encoding="utf-8")
    src = Source(
        name="toy", role="train", loader="local", license="test", path=str(raw), text_field=None
    )

    assert pipeline.download(src, tmp_path)["docs"] == len(lines)
    assert pipeline.clean(src, tmp_path, CleanConfig())["kept"] == len(tiny_corpus) + 5
    assert pipeline.dedup(src, tmp_path, workers=1)["kept"] == len(tiny_corpus)
    counts = pipeline.split(src, tmp_path, val_fraction=0.2, seed=0)
    assert counts["train"] + counts["val"] == len(tiny_corpus)

    # The split is deterministic: running it again gives exactly the same files.
    first = (tmp_path / "splits" / "toy.val.jsonl").read_text()
    pipeline.split(src, tmp_path, val_fraction=0.2, seed=0)
    assert (tmp_path / "splits" / "toy.val.jsonl").read_text() == first
    assert all("script" in json.loads(line) for line in first.splitlines())
