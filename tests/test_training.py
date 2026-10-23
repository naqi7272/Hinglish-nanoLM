"""End-to-end on CPU: tokenize -> pack -> train -> checkpoint -> resume -> evaluate."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer

from hnlm.data.pack import pack_all
from hnlm.eval.bpb import token_byte_lengths
from hnlm.tokenizer.train_hf import train_tokenizer
from hnlm.train.trainer import TrainConfig, load_model, lr_at, train

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_corpus.txt"


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("data")
    lines = FIXTURE.read_text("utf-8").splitlines()
    (root / "splits").mkdir()
    files = {
        "hingcorpus": lines[:10] + lines[22:],
        "sangraha_hin": lines[10:18],
        "fineweb_edu_en": lines[18:22],
    }
    for name, docs in files.items():
        for split, reps in (("train", 6), ("val", 2)):
            with open(root / "splits" / f"{name}.{split}.jsonl", "w", encoding="utf-8") as f:
                for d in docs * reps:
                    f.write(json.dumps({"text": d}, ensure_ascii=False) + "\n")
    (root / "tokenizers").mkdir()
    train_tokenizer(lines, 320, ["<|endoftext|>"]).save(str(root / "tokenizers" / "toy.json"))
    pack_all(root / "tokenizers" / "toy.json", root / "splits", root / "bins" / "toy")
    return root


def cfg(**kw) -> TrainConfig:
    base = dict(
        run_name="t",
        tokenizer="toy",
        model_size="5m",
        model={"n_layer": 2, "n_head": 2, "d_model": 32},
        ctx_len=16,
        batch_size=2,
        grad_accum=2,
        max_tokens=64 * 8,
        lr=3e-3,
        eval_interval=4,
        eval_tokens=256,
        ckpt_interval=2,
        log_interval=2,
        wandb_project=None,
        device="cpu",
    )
    return TrainConfig(**{**base, **kw})


def test_token_bytes_add_up(data_dir):
    tok = Tokenizer.from_file(str(data_dir / "tokenizers" / "toy.json"))
    nbytes = token_byte_lengths(tok)
    text = "Yaar, मैं कल Delhi जा रहा हूँ! 😀"
    assert nbytes[tok.encode(text).ids].sum() == len(text.encode("utf-8"))
    assert nbytes[tok.token_to_id("<|endoftext|>")] == 0


def test_pack_layout(data_dir):
    meta = json.loads((data_dir / "bins" / "toy" / "meta.json").read_text())
    arr = np.memmap(data_dir / "bins" / "toy" / "sangraha_hin.train.bin", dtype=np.uint16)
    assert len(arr) == meta["files"]["sangraha_hin.train"]["tokens"]
    assert (arr == meta["eot_id"]).sum() == meta["files"]["sangraha_hin.train"]["docs"]


def test_lr_schedule_warms_up_then_decays():
    c = TrainConfig(max_tokens=100 * 65536, warmup_frac=0.1, lr=1e-3)
    lrs = [lr_at(s, c) for s in range(100)]
    assert lrs[0] < lrs[9] == pytest.approx(1e-3)
    assert lrs[-1] == pytest.approx(1e-4, rel=0.05) and lrs[50] < lrs[20]


def test_train_logs_and_loss_goes_down(data_dir, tmp_path):
    run = train(cfg(max_tokens=64 * 40, eval_interval=20), data_dir, tmp_path)
    rows = list(csv.DictReader(open(run / "train.csv")))
    assert float(rows[-1]["loss"]) < float(rows[0]["loss"])
    evals = list(csv.DictReader(open(run / "eval.csv")))
    assert {"bpb_hingcorpus", "bpb_sangraha_hin", "bpb_fineweb_edu_en"} <= set(evals[-1])
    assert (run / "final.pt").exists()
    model = load_model(run / "final.pt")
    assert model.cfg.d_model == 32


def test_resume_is_exact(data_dir, tmp_path):
    """Stopping at step 4 and resuming must give the SAME weights as never stopping."""
    a = train(cfg(), data_dir, tmp_path / "a")
    b_dir = tmp_path / "b"
    train(cfg(), data_dir, b_dir, max_steps=4)  # "disconnect" after step 4
    b = train(cfg(), data_dir, b_dir)  # same command again
    wa = torch.load(a / "final.pt", weights_only=False)["model"]
    wb = torch.load(b / "final.pt", weights_only=False)["model"]
    assert all(torch.equal(wa[k], wb[k]) for k in wa)
    steps = [int(r["step"]) for r in csv.DictReader(open(b / "train.csv"))]
    assert steps == sorted(set(steps))  # no duplicated rows after the resume
