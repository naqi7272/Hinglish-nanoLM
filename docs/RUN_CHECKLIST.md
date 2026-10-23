# Run checklist: everything you need to do, in order

All code is written and tested. What's left is running it on free Colab hardware.
Step 1 (corpus) is already **done** ✅.

---

## Step 0: one-time setup (10 min)

### Hugging Face token (for the official Llama 3 tokenizer)
1. Go to https://huggingface.co/settings/tokens → **Create new token** → type **Read** → copy it.
2. Optional, for the *official* Llama 3 repo: open https://huggingface.co/meta-llama/Meta-Llama-3-8B
   and click **Agree / request access**. Approval can take a while. Without it, the code
   automatically uses a public mirror with the identical tokenizer, so you can skip this.
3. In Colab, click the 🔑 **Secrets** icon in the left sidebar → **Add new secret**:
   name `HF_TOKEN`, value = your token, and switch on **Notebook access**.
   Never paste the token into a notebook cell or a chat.

### Weights & Biases (optional, for live training dashboards)
1. Sign up at https://wandb.ai, then copy your key from https://wandb.ai/authorize.
2. Colab 🔑 Secrets → name `WANDB_API_KEY` → paste → Notebook access on.

Without W&B everything still works: all plots are made from local CSV logs.

### Free up Drive space (if not done yet)
In any Colab cell:
```python
!rm -rf /content/drive/MyDrive/hinglish-nanolm/data/raw /content/drive/MyDrive/hinglish-nanolm/data/clean /content/drive/MyDrive/hinglish-nanolm/data/dedup
```
Then empty the trash at https://drive.google.com/drive/trash.

---

## Step 2: tokenizers (CPU runtime, ~20 min)

1. Open https://colab.research.google.com/github/naqi7272/Hinglish-nanoLM/blob/claude/hinglish-nanolm-setup-finaye/notebooks/02_tokenizer.ipynb
2. **Runtime → Run all.**
3. Expect: "HF_TOKEN loaded", two plots, a table, and `m2b_results.zip` downloaded.

---

## Step 3: train the models (T4 GPU runtime, ~5 h total, can be split over days)

1. Open https://colab.research.google.com/github/naqi7272/Hinglish-nanoLM/blob/claude/hinglish-nanolm-setup-finaye/notebooks/03_train.ipynb
2. **Runtime → Change runtime type → T4 GPU → Save.**
3. Run the cells **one by one** from the top:
   - **Setup cells:** you should see the GPU name, e.g. `Tesla T4`.
   - **Tokenize cell:** about 10–20 min, only the first time.
   - **Smoke test:** about 1 min. The loss should go down. **If this cell fails, stop and paste me the error.**
   - **Training cells:** one per run.

   | Run | Time on a T4 |
   |---|---|
   | 5m | ~10 min |
   | 15m | ~45 min |
   | english16k-15m | ~45 min |
   | 30m | ~3 h |

   These times are rough estimates; the tokens/s in the log shows the real speed.
4. **If Colab disconnects or hits its usage limit:** come back later and reconnect. Run the 4 setup cells, then the training cell that was interrupted. It prints `resumed ... from step N` and continues. Finished runs skip instantly.
5. Run the last cells: KV-cache benchmark, samples, plots, and the `m5_results.zip` download.

Tip: Kaggle gives 30 free GPU hours per week if Colab's limit runs out. Tell me and I'll adapt the notebook.

---

## Step 4: put results in the repo (laptop, 5 min)

1. Right-click each zip → **Extract All** → into `C:\Users\PS\Hinglish-nanoLM`. Allow it to merge folders.
2. In the terminal:
```bash
cd ~/Hinglish-nanoLM
git pull --no-edit
git add results plots tokenizers
git commit -m "Add tokenizer and training results"
git push
```

## Step 5: tell Claude "results pushed"

Claude then fills in all the numbers in `README.md` and `docs/PROJECT_NOTES_HINGLISH.md`, writes
the "what I learned" section from your real results, and updates your resume bullets.
