# Hinglish-nanoLM

A from-scratch learning project: train and analyze a **Hinglish tokenizer**, then pretrain a
**small GPT** (5M–30M parameters) on Hinglish + Hindi text, in plain PyTorch.

> Work in progress. This README becomes the full write-up (every component explained,
> plots, sample generations, lessons learned) at the end of the project.

## Status

| Milestone | State |
|---|---|
| M0 Scaffold (uv, device auto-detect, tests) | ✅ |
| M1 Corpus pipeline (clean, MinHash dedup, split) | ✅ code + tests · ⏳ run on Colab/Kaggle |
| M2a From-scratch BPE | 📝 skeleton ready, merge loop is a TODO for me to write |
| M2b Production tokenizer + comparison vs GPT-4o / Llama 3 / Qwen3 | — |
| M3 GPT model (RoPE, RMSNorm, SwiGLU, KV cache, sampling) | — |
| M4 Trainer (AdamW, cosine LR, AMP, resumable checkpoints, W&B) | — |
| M5 Experiments (3 sizes, bits-per-byte, KV-cache speed) | — |
| M6 Write-up | — |

## Quick start (laptop, CPU)

```bash
uv sync                      # Python 3.12 + deps
uv run pytest                # all tests run on CPU in seconds
uv run python scripts/train_bpe_scratch.py   # after implementing the BPE TODOs
```

Heavy work (corpus download, training) runs in `notebooks/` on Colab or Kaggle.

## Data

| Source | Script | Role | License |
|---|---|---|---|
| [L3Cube-HingCorpus](https://github.com/l3cube-pune/code-mixed-nlp) | Roman Hinglish (tweets) | train | CC BY-NC-SA 4.0 |
| [AI4Bharat Sangraha](https://huggingface.co/datasets/ai4bharat/sangraha) (verified, Hindi) | Devanagari | train | CC BY 4.0 |
| [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (small slice) | English | train | ODC-By 1.0 |
| [PHINC](https://huggingface.co/datasets/veezbo/phinc) | Roman Hinglish | held-out eval | CC BY 4.0 |

Because HingCorpus is **non-commercial, share-alike**, this project and anything trained on it
(tokenizer, model weights) is for non-commercial use and shared under CC BY-NC-SA 4.0.

## Layout

```
configs/      data + model/training recipes (YAML)
src/hnlm/
  data/       clean.py, dedup.py (MinHash+LSH from scratch), split.py, pipeline.py
  tokenizer/  pretokenize.py, bpe_scratch.py, (train_hf.py, compare.py)
  model/      (GPT in plain PyTorch)
  train/      (trainer, checkpointing, logging)
  eval/       (bits-per-byte, generation speed)
  device.py   cuda / mps / cpu + mixed-precision auto-detect
scripts/      command-line entry points
notebooks/    Colab/Kaggle notebooks for the heavy runs
tests/        pytest suite (CPU-only, fast)
```
