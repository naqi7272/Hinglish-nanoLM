import io
import tarfile

from hnlm.data.sources import Source, iter_texts, load_data_config


def _add(tar: tarfile.TarFile, name: str, text: str) -> None:
    data = text.encode("utf-8")
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def test_tar_gz_is_streamed_without_unpacking(tmp_path):
    archive = tmp_path / "corpus_roman.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        _add(tar, "corpus/part1.txt", "pehli line\n\ndoosri line\n")
        _add(tar, "corpus/part2.txt", "teesri line\n")
        _add(tar, "corpus/._part1.txt", "macOS junk")  # hidden files are skipped
    src = Source(name="t", role="train", loader="local", license="x", path=str(tmp_path))
    src.text_field = None
    assert list(iter_texts(src)) == ["pehli line", "doosri line", "teesri line"]
    assert not (tmp_path / "corpus").exists()  # nothing was extracted


def test_relative_local_paths_follow_data_dir(tmp_path):
    cfg = load_data_config("configs/data.yaml", data_dir=tmp_path)
    hc = next(s for s in cfg["sources"] if s.name == "hingcorpus")
    assert hc.path == str(tmp_path / "downloads" / "hingcorpus")
