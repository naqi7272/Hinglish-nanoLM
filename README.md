# Hinglish-nanoLM

A from-scratch learning project: build a **Hinglish tokenizer** and pretrain a **small GPT**
(5M–30M parameters) on Roman Hinglish, Devanagari Hindi and a little English, in plain
PyTorch. Then measure two things:

1. **How unfair big-model tokenizers are to Hindi/Hinglish** (GPT-4o, Llama 3, Qwen3 vs mine).
2. **How loss scales with model size**, and whether a Hinglish tokenizer beats an
   English-only one at the same model size (measured fairly, in bits per byte).

Everything except the reference tokenizers is written by hand: data cleaning, MinHash
deduplication, BPE, the transformer, the training loop, evaluation.

## Status

| Milestone | State |
|---|---|
| M0 Scaffold (uv, device auto-detect, tests) | ✅ |
| M1 Corpus pipeline (clean, MinHash dedup, split) | ✅ built: 4.6M docs, 1.45 GB |
| M2a From-scratch BPE (merge loop written by me) | ✅ |
| M2b Production tokenizer + comparison vs GPT-4o / Llama 3 / Qwen3 | ✅ |
| M3 GPT model (RoPE, RMSNorm, SwiGLU, KV cache, sampling) | ✅ code + tests |
| M4 Trainer (AdamW, cosine LR, AMP, resumable checkpoints, W&B) | ✅ code + tests |
| M5 Experiments (model sizes, bits-per-byte, KV-cache speed) | ✅ 5M + 15M trained (30M skipped: GPU time) |
| M6 Write-up | ✅ |

---

## Results

**In one line:** a 16k-token tokenizer trained on Hinglish needs **fewer tokens per word than
GPT-4o's 200k tokenizer** on Hindi and Roman Hinglish, and at the same model size and training
budget it gives **4–14% lower bits per byte** than an English-only tokenizer, on every
language including English.

### 1. Tokenizers: tokens per word (lower is better)

![fertility](plots/fertility.png)

| Tokenizer | Vocab | English | Hindi (Devanagari) | Roman Hinglish |
|---|---:|---:|---:|---:|
| **hinglish16k (mine)** | 16,384 | 1.60 | **1.50** | **1.55** |
| english16k (baseline) | 16,384 | 1.38 | 8.00 | 2.18 |
| GPT-2 | 50,257 | 1.30 | 7.65 | 1.86 |
| GPT-4o | 200,019 | **1.25** | 1.63 | 1.59 |
| Llama 3 | 128,256 | 1.26 | 2.64 | 1.73 |
| Qwen3 | 151,669 | 1.28 | 4.65 | 1.75 |

*(English: 155 FineWeb-Edu docs · Hindi: 808 Sangraha docs · Roman Hinglish: 2,000 PHINC
sentences, all held out from training.)*

What it shows:
- **The "Hindi tax" is real and varies a lot.** For the same Hindi text, Qwen3 needs 3.1× and
  Llama 3 1.8× as many tokens as mine; GPT-2 and an English-only tokenizer need over 5×
  (they fall back to raw bytes, see `english16k`'s 67 tokens for one short sentence below).
- **GPT-4o is the exception**: its 200k vocabulary covers Hindi well (1.63). Mine reaches 1.50
  with a vocabulary 12× smaller, because all of its 16k entries are spent on these languages.
- **The trade-off**: on English mine needs 28% more tokens than GPT-4o (1.60 vs 1.25). A small
  vocabulary can't be best at everything.

How six tokenizers cut the same sentences ([full table](results/segmentations.md)):

| Sentence | mine | GPT-4o | Llama 3 | Qwen3 | GPT-2 | english16k |
|---|---:|---:|---:|---:|---:|---:|
| The weather is really nice today, let's go for a walk. | 14 | 13 | 14 | 14 | 14 | 14 |
| आज मौसम बहुत अच्छा है, चलो टहलने चलते हैं। | 14 | 14 | 22 | 42 | 66 | 67 |
| Aaj mausam bahut accha hai, chalo walk pe chalte hain. | 14 | 18 | 20 | 20 | 23 | 23 |
| मैं कल Delhi जा रहा हूँ, meeting है। | 10 | 10 | 14 | 25 | 33 | 36 |

### 2. Same model, two tokenizers (bits per byte, lower is better)

![bpb](plots/bpb_tokenizers.png)

Two 4.8M-parameter models, identical except the tokenizer, each trained on 100M tokens of
the same corpus:

| Validation set | hinglish16k | english16k | Improvement |
|---|---:|---:|---:|
| Roman Hinglish (tweets) | **2.010** | 2.131 | 5.7% |
| Hindi (Devanagari) | **0.714** | 0.826 | 13.5% |
| English | **1.738** | 1.837 | 5.4% |
| Roman Hinglish (PHINC, unseen source) | **2.554** | 2.671 | 4.4% |

Bits per byte is measured per byte of *text*, so it's fair across tokenizers. The Hinglish
tokenizer wins everywhere, even on English. The likely reason: with the same budget of 100M
tokens, a tokenizer that packs more text into each token lets the model **see more text**,
and the English-only tokenizer spends a large share of its tokens spelling Hindi out byte by
byte. (BPB values are not comparable *across* languages: one Devanagari letter is 3 bytes.)

### 3. Bigger model, lower loss

![loss vs tokens](plots/loss_vs_tokens.png) ![loss vs size](plots/loss_vs_size.png)

| Model | Non-embedding params | Training tokens | Mean validation BPB |
|---|---:|---:|---:|
| hinglish16k-5m | 4.8M | 100M | 1.487 |
| hinglish16k-15m | 14.2M | 300M | **1.269** (−15%) |

Even at the same point in training (~100M tokens), the 15M model was already lower (1.38 vs
1.49). Two sizes show the direction but not a scaling law; the 30M run (~2× the compute
again) didn't fit in free GPU time.

### 4. Generation speed with and without KV cache

![kv cache](plots/kv_cache_speed.png)

| 15M model, batch 1 | 64 tokens | 128 | 256 | 480 |
|---|---:|---:|---:|---:|
| CPU: speedup from KV cache | 3.3× | 5.5× | 5.9× | **10.9×** |
| T4 GPU: speedup from KV cache | 0.94× | 1.04× | 0.92× | 1.03× |

A surprise worth understanding: **on the GPU the cache made no difference.** For a model this
small with one sequence, each generation step is dominated by fixed overhead (Python and
launching ~100 small GPU kernels), not by math: re-processing 500 tokens costs the GPU about
the same as processing 1. On the CPU, where the math really is the bottleneck, skipping the
recomputation gives up to 10.9×, and the gap grows with length (without a cache, step *t*
redoes *t* tokens of work). The cache pays off on GPUs once models, batches or contexts are
big enough that compute dominates, which is the normal case for real LLMs.

### 5. Samples (15M model, temperature 0.8, top-k 50, top-p 0.95)

> **Aaj office mein** bhi aise hi kaam karte hain...but at least he can also help.
>
> **Mummy ne bola ki** nahi toh main kabka kha jau?
>
> **मुझे लगता है कि** वह इस क्रीम को देखने के लिए एकदम तैयार नहीं हैं। उनकी हर जगह एक बहुत अलग जगह है।
>
> **The best way to learn** about a different topic in our brain. The same way they can work at a different age.

Fluent-looking Hinglish, Hindi and English, switching languages naturally, with little
meaning: about what 14M parameters and a few hours of training can do. All samples:
`results/samples_*.md`.

---

## How it works, in simple words

### Data (`src/hnlm/data/`)

| Source | Script | Role | License |
|---|---|---|---|
| [L3Cube-HingCorpus](https://github.com/l3cube-pune/code-mixed-nlp) | Roman Hinglish (tweets) | train | CC BY-NC-SA 4.0 |
| [AI4Bharat Sangraha](https://huggingface.co/datasets/ai4bharat/sangraha) (verified, Hindi) | Devanagari | train | CC BY 4.0 |
| [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (small slice) | English | train | ODC-By 1.0 |
| [PHINC](https://huggingface.co/datasets/veezbo/phinc) | Roman Hinglish | held-out eval | CC BY 4.0 |

After cleaning and MinHash dedup:

| Source | Raw docs | Kept docs | Kept MB | Removed as duplicates |
|---|---:|---:|---:|---:|
| HingCorpus | 6,917,003 | 4,474,333 | 447 | 1,980,280 (31%) |
| Sangraha Hindi | 160,326 | 157,070 | 859 | 3,060 (2%) |
| FineWeb-Edu | 30,491 | 30,436 | 144 | 5 |
| PHINC (eval) | 13,738 | 12,806 | 0.9 | 179 |

- **Cleaning** (`clean.py`) removes URLs, @mentions, emoji and "RT", shortens "sooooo" to
  "sooo", applies Unicode NFC (one byte form per visible character), keeps the invisible
  joiners that change how Hindi conjuncts render, and labels each doc as Devanagari / Latin /
  mixed. Gotcha: Hindi vowel signs (ि ी ं) are Unicode *marks*, not letters, so every
  "is this a letter?" check must accept marks too.
- **Near-duplicate removal** (`dedup.py`, MinHash + LSH written in numpy): turn each doc into
  a set of 5-character pieces; squash the set into 64 numbers (MinHash) so that *the fraction
  of matching numbers ≈ how similar two docs are*; group docs whose signatures match in at
  least one of 16 bands (fast, by sorting); verify; keep one per group. Tweets lost 31%
  (retweets, forwards), curated web text only 2%.
- **Train/val split** (`split.py`) is decided by a hash of the text, so re-running never moves
  a document from validation into training.

### Tokenizer (`src/hnlm/tokenizer/`)

- **Byte-level BPE** (`bpe_scratch.py`, merge loop written by me): start from the 256 byte
  values, so nothing is ever "unknown". Repeatedly find the most frequent neighbouring pair
  and merge it into a new token. The very first merge learned on our data is `E0 A4`: the
  first two bytes of most Devanagari letters (each Hindi letter is 3 bytes in UTF-8).
- **Pre-tokenization** (`pretokenize.py`): split text into words before BPE. I use GPT-4's
  pattern with one fix: GPT-4's `\p{L}` (letters) misses Hindi vowel signs, which cuts every
  Hindi word apart at each matra. Adding `\p{M}` keeps Hindi words whole.
- **Production version** (`train_hf.py`): the same recipe in Hugging Face `tokenizers` (Rust,
  ~1000× faster), 16,384 tokens. `hinglish16k` trains on equal parts Roman Hinglish and Hindi
  plus some English; `english16k` (baseline) on English only.
- **Why 16k?** In a small model the embedding table is huge relative to the rest: a 32k
  vocab at width 256 is 8.4M parameters, more than the whole 5M model.
- **Metrics** (`compare.py`): *fertility* = tokens per word; *chars per token*; plus example
  segmentations.

### Model (`src/hnlm/model/`), a decoder-only GPT

```
token ids → embedding → [ RMSNorm → attention → + ] → [ RMSNorm → SwiGLU → + ] × N
          → RMSNorm → output layer (shares weights with the embedding) → next-token scores
```

- **Embedding**: each token id becomes a vector of `d_model` numbers.
- **Causal self-attention**: every token builds a *query* ("what am I looking for?"), a *key*
  ("what do I contain?") and a *value* ("what do I pass on?"). Query·key scores say how much
  each earlier token matters; softmax turns them into weights; the output is the weighted
  sum of values. A **causal mask** hides future tokens: no peeking at the answer. Several
  **heads** do this in parallel, each free to track something different.
- **RoPE**: instead of adding a position vector, rotate queries and keys by an angle that
  grows with position. Then q·k depends only on the *distance* between two tokens.
- **RMSNorm**: rescale each vector to a steady size before each sub-layer, which keeps deep
  networks stable. Cheaper than LayerNorm and just as good.
- **SwiGLU**: the per-token MLP, with a learned gate deciding what passes. Used by Llama/Qwen.
- **Residual connections**: each layer *adds* its update to a running "stream", so gradients
  flow easily through many layers.
- **Weight tying**: the input embedding and the output layer share one matrix (saves
  4–8M parameters here).
- **KV cache**: during generation, store each layer's keys and values so every new token only
  computes its own, instead of recomputing the whole sequence.
- **Sampling**: *temperature* (sharper or flatter choices), *top-k* (only the k best),
  *top-p* (smallest set covering p probability).

| Size | Layers | Heads | Width | Non-embedding params | Total (with embedding) |
|---|---:|---:|---:|---:|---:|
| 5m | 6 | 4 | 256 | 4.82M | 9.01M |
| 15m | 8 | 6 | 384 | 14.16M | 20.45M |
| 30m | 10 | 8 | 512 | 32.12M | 40.51M |

### Training (`src/hnlm/train/`)

- **Objective**: predict the next token (cross-entropy). That's all pretraining is.
- **Data**: the corpus is tokenized once into flat `uint16` files; each batch picks random
  512-token windows, mixing sources in proportion to their size.
- **AdamW** (β = 0.9, 0.95; weight decay 0.1 on matrices only), **warmup + cosine** learning
  rate, **gradient clipping** at 1.0, **gradient accumulation** (4 × 32 sequences = 65,536
  tokens per step), **mixed precision** (bfloat16 when the GPU supports it, else float16 with
  loss scaling).
- **Budget**: ~20 tokens per non-embedding parameter (the Chinchilla rule of thumb): 100M /
  300M / 600M tokens. The corpus is a few hundred million tokens, so the 30M run sees it
  about twice, which is still fine.
- **Checkpoints that survive Colab disconnects**: written atomically to Google Drive every
  250 steps. Batches are seeded by step number, so a resumed run is **bit-for-bit identical**
  to an uninterrupted one (there's a test for this).
- **Logging**: CSV always, Weights & Biases when a key is set.

### Evaluation (`src/hnlm/eval/`)

- **Bits per byte (BPB)** = total loss in bits ÷ number of UTF-8 bytes of the text. Per-token
  loss can't compare two tokenizers (one makes fewer, harder predictions); per *byte* of the
  same text it can. Reported separately for Roman Hinglish, Hindi, English and PHINC.

---

## Reproduce

Step-by-step (with secrets setup and what to do on disconnects): [`docs/RUN_CHECKLIST.md`](docs/RUN_CHECKLIST.md).

| Step | Where | Notebook | Time |
|---|---|---|---|
| 1. Build corpus | Colab CPU | `notebooks/01_prepare_data.ipynb` | ~1 h |
| 2. Train + compare tokenizers | Colab CPU | `notebooks/02_tokenizer.ipynb` | ~20 min |
| 3. Train models + experiments | Colab **T4 GPU** | `notebooks/03_train.ipynb` | many hours (the 15M run alone takes ~6 h on a free T4), resumable |

Laptop (CPU) development:

```bash
uv sync                      # Python 3.12 + deps
uv run pytest                # 54 tests, all on CPU, ~10 s
uv run python scripts/train_bpe_scratch.py --vocab-size 400
```

### Tests (`tests/`)

Tokenizer round-trip (scratch + production, incl. Hindi, emoji, odd whitespace) · attention
causality (changing a future token never changes earlier outputs) · KV-cache output equals
full forward · overfit one batch · RoPE depends only on relative position · sampling
controls · cleaning / dedup / split · byte accounting for BPB · LR schedule · end-to-end
training · bit-exact resume.

---

## What I learned

- **Tokenizers carry hidden assumptions.** One regex class (`\p{L}` without `\p{M}`) is enough
  to cut every Hindi word apart. Choosing the training data for a tokenizer decides which
  languages are cheap, and that cost follows the language through training and inference.
- **A small vocabulary spent well beats a big one spent elsewhere**, for the languages you
  care about (16k vs GPT-4o's 200k on Hindi), at a visible price on the others (English).
- **Measure the right thing.** Per-token loss would have hidden the tokenizer effect; bits
  per byte made it comparable.
- **Data work is most of the work.** 31% of tweets were near-duplicates. Cleaning ran 18×
  slower on Colab than on my laptop because writing to Google Drive was the bottleneck, not
  the code.
- **Performance needs measuring, not assuming.** Training ran at only 14k tokens/s; the
  likely cause was that PyTorch reports bfloat16 as "supported" on a T4, where it's only
  emulated, so the code now picks float16 there. And the KV cache, a big win on CPU, did
  nothing for a tiny model on a GPU.
- **Real LLM training is enormous.** My 14M model took hours on a free GPU. Llama 3 8B is
  ~500× bigger and saw ~50,000× more tokens, on thousands of GPUs.
- **Reliability is a feature.** Atomic checkpoints, batches seeded by step and a test proving
  that a resumed run is bit-identical made Colab disconnects a non-event.

## Limitations

- Small models (5M, 15M) and a few hundred million tokens: good for learning and for
  *relative* comparisons, not a useful assistant. Only two model sizes, one seed each.
- The tokenizer A/B test is at 5M parameters only; the 15M baseline didn't fit in free GPU
  time.
- Tweets dominate the Roman-script data, so the model's Roman Hinglish is "Twitter style".
- Vocab sizes differ between my tokenizer (16k) and the big ones (128k–200k), so tokenizer
  comparisons mix "what it was trained on" with "how big it is". The 16k-vs-16k BPB
  experiment controls for that.

## License

HingCorpus is **CC BY-NC-SA 4.0** (non-commercial, share-alike), so the trained tokenizers and
models from this project are shared under the same terms. Code: same, for simplicity.
Other datasets: see the table above.
