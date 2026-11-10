"""The training loop.

One optimizer step:
  1. Gradient accumulation: run `grad_accum` micro-batches forward + backward, adding up
     their gradients, so the effective batch is bigger than what fits in GPU memory at once.
  2. Mixed precision: the forward pass runs in 16-bit under autocast; with float16 the loss
     is scaled up before backward so tiny gradients don't round to zero (GradScaler).
  3. Gradient clipping: if the total gradient norm exceeds 1.0, shrink it. One bad batch then
     can't throw the weights far off (loss spikes).
  4. AdamW step with the learning rate from the schedule: linear warmup (Adam's statistics
     are unreliable for the first steps, so start gently) then cosine decay down to 10%
     (big steps early to make progress, small steps late to settle into a good minimum).

Every `eval_interval` steps: validation loss + bits-per-byte per source.
Every `ckpt_interval` steps: an atomic checkpoint, so a disconnect loses minutes, not hours.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import yaml
from tokenizers import Tokenizer

from hnlm.device import get_device, seed_everything
from hnlm.eval.bpb import evaluate, token_byte_lengths
from hnlm.model.config import SIZES, GPTConfig
from hnlm.model.gpt import GPT
from hnlm.train.checkpoint import load_checkpoint, save_checkpoint
from hnlm.train.data import MixtureLoader
from hnlm.train.logger import RunLogger


@dataclass
class TrainConfig:
    run_name: str = "hinglish16k-5m"
    tokenizer: str = "hinglish16k"
    model_size: str = "5m"
    model: dict = field(default_factory=dict)  # overrides on top of SIZES[model_size]
    ctx_len: int = 512
    train_sources: list = field(
        default_factory=lambda: ["hingcorpus", "sangraha_hin", "fineweb_edu_en"]
    )
    mix: dict | None = None  # None = proportional to token counts
    # Budget. tokens per step = batch_size * grad_accum * ctx_len (default 32*4*512 = 65,536)
    max_tokens: int = 100_000_000
    batch_size: int = 32
    grad_accum: int = 4
    # Optimizer
    lr: float = 2e-3
    min_lr_ratio: float = 0.1
    warmup_frac: float = 0.02
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    grad_clip: float = 1.0
    # Bookkeeping
    eval_interval: int = 250
    eval_tokens: int = 250_000  # per val file during training; the final eval uses all
    ckpt_interval: int = 250
    log_interval: int = 10
    seed: int = 1337
    device: str = "auto"
    compile: bool = False
    wandb_project: str | None = "hinglish-nanolm"

    @classmethod
    def from_yaml(cls, path: str | Path, **overrides) -> TrainConfig:
        raw = yaml.safe_load(open(path, encoding="utf-8")) or {}
        raw.update({k: v for k, v in overrides.items() if v is not None})
        if "betas" in raw:
            raw["betas"] = tuple(raw["betas"])
        return cls(**raw)

    @property
    def tokens_per_step(self) -> int:
        return self.batch_size * self.grad_accum * self.ctx_len

    @property
    def total_steps(self) -> int:
        return max(1, self.max_tokens // self.tokens_per_step)


def lr_at(step: int, cfg: TrainConfig) -> float:
    warmup = max(1, int(cfg.warmup_frac * cfg.total_steps))
    if step < warmup:
        return cfg.lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, cfg.total_steps - warmup)
    min_lr = cfg.lr * cfg.min_lr_ratio
    return min_lr + 0.5 * (cfg.lr - min_lr) * (1 + math.cos(math.pi * min(progress, 1.0)))


def make_optimizer(model: GPT, cfg: TrainConfig, fused: bool) -> torch.optim.AdamW:
    # Weight decay pulls weights toward zero (a mild regularizer). Apply it to the big
    # matrices only; decaying norm gains would fight their purpose.
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=cfg.betas, fused=fused)


def build_model(cfg: TrainConfig, vocab_size: int) -> GPT:
    shape = {**SIZES[cfg.model_size], **cfg.model}
    return GPT(GPTConfig(vocab_size=vocab_size, ctx_len=cfg.ctx_len, **shape))


def run_validation(model, val_files, token_bytes, cfg, dev, max_tokens) -> dict:
    row = {}
    for name, tokens in val_files.items():
        r = evaluate(
            model,
            tokens,
            token_bytes,
            cfg.ctx_len,
            cfg.batch_size,
            dev.device,
            dev.autocast,
            max_tokens,
        )
        row[f"loss_{name}"] = round(r["loss"], 5)
        row[f"bpb_{name}"] = round(r["bpb"], 5)
    return row


def train(cfg: TrainConfig, data_dir: Path, runs_dir: Path, max_steps: int | None = None) -> Path:
    """Train (or resume) cfg.run_name. Returns the run directory."""
    dev = get_device(cfg.device)
    seed_everything(cfg.seed)
    run_dir = runs_dir / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    bins = data_dir / "bins" / cfg.tokenizer
    meta = json.loads((bins / "meta.json").read_text())
    tok = Tokenizer.from_file(str(data_dir / "tokenizers" / f"{cfg.tokenizer}.json"))

    model = build_model(cfg, meta["vocab_size"]).to(dev.device)
    opt = make_optimizer(model, cfg, fused=dev.type == "cuda")
    scaler = torch.amp.GradScaler(dev.type, enabled=dev.needs_grad_scaler)

    start_step, wandb_id = 0, None
    ckpt = load_checkpoint(run_dir / "latest.pt", map_location=dev.device)
    if ckpt is not None:
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        if ckpt["scaler"]:  # empty if saved without fp16 (precision may change on resume)
            scaler.load_state_dict(ckpt["scaler"])
        start_step, wandb_id = ckpt["step"], ckpt.get("wandb_id")
        print(f"resumed {cfg.run_name} from step {start_step}")

    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "train": asdict(cfg),
                "model": model.cfg.to_dict(),
                "non_embedding_params": model.num_params(),
                "total_params": model.num_params(False),
            },
            indent=2,
        )
    )
    logger = RunLogger(
        run_dir,
        cfg.wandb_project,
        cfg.run_name,
        {**asdict(cfg), "params": model.num_params()},
        wandb_id,
    )
    logger.truncate_after(start_step)

    train_data = MixtureLoader(bins, cfg.train_sources, "train", cfg.mix)
    val_files = {
        s: np.memmap(bins / f"{s}.val.bin", dtype=np.uint16, mode="r") for s in cfg.train_sources
    }
    eval_file = bins / "phinc.eval.bin"
    if eval_file.exists():
        val_files["phinc"] = np.memmap(eval_file, dtype=np.uint16, mode="r")
    token_bytes = torch.from_numpy(token_byte_lengths(tok)).to(dev.device)

    fwd = torch.compile(model) if cfg.compile else model
    end_step = cfg.total_steps if max_steps is None else min(cfg.total_steps, max_steps)
    print(
        f"{cfg.run_name}: {model.num_params() / 1e6:.2f}M non-embedding params, "
        f"{cfg.total_steps} steps x {cfg.tokens_per_step:,} tokens, on {dev.describe()}; "
        f"train data {train_data.total_tokens() / 1e6:.0f}M tokens"
    )

    def checkpoint(step: int) -> None:
        save_checkpoint(
            run_dir / "latest.pt",
            {
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scaler": scaler.state_dict(),
                "step": step,
                "wandb_id": logger.wandb_id,
                "model_config": model.cfg.to_dict(),
                "tokenizer": cfg.tokenizer,
            },
        )

    model.train()
    t0 = time.time()
    for step in range(start_step, end_step):
        lr = lr_at(step, cfg)
        for g in opt.param_groups:
            g["lr"] = lr
        rng = np.random.default_rng([cfg.seed, step])
        loss_sum = 0.0
        for _ in range(cfg.grad_accum):
            x, y = train_data.get_batch(rng, cfg.batch_size, cfg.ctx_len, dev.device)
            with dev.autocast():
                _, loss = fwd(x, targets=y)
            scaler.scale(loss / cfg.grad_accum).backward()
            loss_sum += loss.item() / cfg.grad_accum
        scaler.unscale_(opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip).item()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)

        done = step + 1
        tokens = done * cfg.tokens_per_step
        if done % cfg.log_interval == 0 or done == end_step:
            dt = time.time() - t0
            tps = cfg.tokens_per_step * cfg.log_interval / dt if dt > 0 else 0.0
            t0 = time.time()
            logger.log(
                "train",
                {
                    "step": done,
                    "tokens": tokens,
                    "lr": round(lr, 7),
                    "loss": round(loss_sum, 5),
                    "grad_norm": round(grad_norm, 4),
                    "tokens_per_sec": round(tps),
                },
            )
            print(
                f"step {done}/{cfg.total_steps} loss {loss_sum:.4f} lr {lr:.2e} "
                f"gnorm {grad_norm:.2f} {tps / 1e3:.0f}k tok/s"
            )
        is_last = done == cfg.total_steps
        if done % cfg.eval_interval == 0 or is_last:
            row = run_validation(
                model, val_files, token_bytes, cfg, dev, None if is_last else cfg.eval_tokens
            )
            logger.log("eval", {"step": done, "tokens": tokens, **row})
            print("  eval", row)
        if done % cfg.ckpt_interval == 0 or done == end_step:
            checkpoint(done)

    if end_step == cfg.total_steps:
        save_checkpoint(
            run_dir / "final.pt",
            {
                "model": model.state_dict(),
                "model_config": model.cfg.to_dict(),
                "tokenizer": cfg.tokenizer,
            },
        )
    logger.finish()
    return run_dir


def load_model(ckpt_path: Path, device="cpu") -> GPT:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = GPT(GPTConfig(**ckpt["model_config"]))
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()
