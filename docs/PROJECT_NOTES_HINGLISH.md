# Hinglish-nanoLM: Meri Project Notes (Hinglish mein)

> Ye notes mere liye hain: kya kiya, kyun kiya, aur interview mein kaise samjhaun.
> Project abhi chal raha hai. Jo part complete hai wo ✅, jo baaki hai wo ⏳.

---

## 0. Ek line mein project

**"Maine Hinglish (Roman + Devanagari) ke liye tokenizer aur ek chhota GPT model bilkul
scratch se banaya, taaki samajh sakun ki LLM andar se kaise kaam karta hai, aur measure kiya
ki bade models (GPT-4o, Llama 3, Qwen3) ke tokenizers Hindi/Hinglish ke saath kitna
unfair hain."**

### 30-second interview pitch
"Hinglish India mein crores log daily likhte hain, lekin bade LLM tokenizers English ke liye
optimize hain. Hindi ka ek letter UTF-8 mein 3 bytes ka hota hai, toh same baat kehne mein
Hindi zyada tokens khaati hai, matlab zyada cost, slow generation aur chhota effective context.
Maine data pipeline (cleaning + MinHash dedup), BPE tokenizer, aur GPT (RoPE, RMSNorm, SwiGLU,
KV cache) khud likha, 3 sizes ke models train kiye, aur bits-per-byte se fair comparison kiya."

---

## 1. Status: abhi tak kya hua

| Step | Kya | Status |
|---|---|---|
| M0 | Project setup: uv, tests, CPU/GPU auto-detect | ✅ |
| M1 | Corpus pipeline: download → clean → dedup → split | ✅ code, ⏳ Colab run chal raha hai |
| M2a | BPE tokenizer scratch se (merge loop **maine khud likha**) | ✅ 16/16 tests pass |
| M2b | Production tokenizer (16k) + GPT-4o / Llama 3 / Qwen3 se comparison | ⏳ |
| M3 | GPT model plain PyTorch mein (attention **main khud likhunga**) | ⏳ |
| M4 | Training loop: AdamW, cosine LR, mixed precision, checkpoints | ⏳ |
| M5 | Experiments: 5M / 15M / 30M models, scaling plots, KV-cache speed | ⏳ |
| M6 | README write-up | ⏳ |

---

## 2. Project setup (M0)

### Kya kiya
- `uv` se Python 3.12 project banaya, `pytest` se tests, `ruff` se lint.
- `device.py`: GPU hai toh GPU, warna CPU. GPU type ke hisaab se precision bhi khud choose
  karta hai (T4 pe float16 + GradScaler, naye GPUs pe bfloat16, CPU pe float32).

### Kyun kiya
Development mere laptop (CPU) pe hota hai aur training Colab/Kaggle (free GPU) pe. Same code
dono jagah bina change ke chalna chahiye.

### Interview Q&A
**Q: float16 aur bfloat16 mein kya fark hai? GradScaler kyun chahiye?**
A: Dono 16-bit hain. bfloat16 ki range float32 jitni badi hai, toh values overflow/underflow
nahi hoti. float16 ki range chhoti hai, toh bahut chhote gradients zero ho jaate hain
(underflow). GradScaler loss ko ek bade number se multiply karta hai backward se pehle, phir
optimizer step se pehle gradients ko wapas divide karta hai. T4 GPU bfloat16 fast nahi chalata,
isliye wahan float16 + GradScaler.

**Q: Tests kyun likhe jab ye learning project hai?**
A: ML code mein bugs silent hote hain: loss thoda kam girta hai aur pata nahi chalta kyun.
Tests (round-trip, causality, KV-cache equality, overfit-one-batch) bugs ko turant pakadte hain.

---

## 3. Data choose karna (M1)

### Kya kiya
| Source | Kya hai | Kyun | License |
|---|---|---|---|
| L3Cube-HingCorpus | Roman Hinglish tweets (6.9M docs liye, 700 MB) | Asli Roman Hinglish ka sabse bada source | CC BY-NC-SA 4.0 |
| Sangraha (verified Hindi) | Devanagari Hindi web text | High quality, permissive license | CC BY 4.0 |
| FineWeb-Edu (chhota slice) | English | Model + plots English bhi dekhein | ODC-By |
| PHINC | Clean Hinglish sentences | Sirf evaluation ke liye (held-out) | CC BY 4.0 |

### Kyun kiya
Real Hinglish do scripts mein likhi jaati hai: "main kal Delhi ja raha hoon" aur "मैं कल
Delhi जा रहा हूँ". Isliye dono scripts aur thodi English chahiye.

### Interview Q&A
**Q: License kyun check kiya?**
A: HingCorpus NonCommercial + ShareAlike hai. Iska matlab project aur trained weights bhi
same license ke saath hi share kar sakte hain, commercial use nahi. Data ka license
check karna real ML kaam ka important part hai.

**Q: Eval set alag source se kyun liya?**
A: Agar eval bhi tweets se ho, toh model "Twitter style" mein accha dikh sakta hai bina
generally accha hue. Alag clean source se honest measurement milta hai.

---

## 4. Cleaning (M1)

### Kya kiya
- URLs, @mentions, "RT", emoji hataye. "sooooo" ko "sooo" kiya.
- Unicode **NFC normalization** kiya.
- **ZWJ/ZWNJ** characters rakhe (Devanagari mein ye conjunct letters ka look badalte hain).
- Har document ka script label: `deva`, `latn`, `mixed`, `other`. Other scripts (Urdu,
  Chinese…) hata diye.

### Kyun kiya
Tweets mein bahut noise hai. Noise pe trained model noise hi generate karega.

### Interview Q&A
**Q: NFC normalization kyun?**
A: Unicode mein same dikhne wala character do alag byte sequences mein likha ja sakta hai.
Normalize na karein toh tokenizer ek hi word ke do versions seekhta hai aur vocab waste hota hai.

**Q: Hindi ke saath "letter" check karne mein kya trap hai?**
A: "कि" do code points hai: क (Letter) + ि (Mark, category Mc). Agar sirf letters count karo,
toh vowel signs gayab ya alag ho jaate hain. Isliye hamesha `\p{L}` ke saath `\p{M}` bhi
lena padta hai.

---

## 5. MinHash dedup (M1), numpy mein scratch se

### Kya kiya
4 steps, sab khud likhe:
1. **Shingles:** har document ko character 5-grams ke set mein todo.
2. **MinHash:** har set ko 64 numbers ke signature mein compress karo.
3. **LSH banding:** signature ko 16 bands × 4 rows mein todo; jo docs kisi ek band mein
   bilkul same hain wo "candidate pair" bante hain (sorting se, fast).
4. **Verify + union-find:** candidates ka similarity ≥ 0.7 ho toh same cluster; har cluster
   ka pehla doc rakho, baaki hatao.

### Kyun kiya
Retweets aur copy-paste forwards bahut hote hain. Duplicates se model text ratta maarta hai,
aur agar same text train aur val dono mein aa gaya toh val loss jhootha accha dikhta hai
(**data leakage**).

### Interview Q&A
**Q: MinHash kaam kyun karta hai?**
A: Ek random hash function ke liye, P(min-hash of A == min-hash of B) = Jaccard(A, B).
Isliye 64 hash functions mein se kitni jagah signature match hua, wo Jaccard similarity ka
estimate hai.

**Q: LSH banding kyun? Seedha sab pairs compare kyun nahi?**
A: Millions docs ke liye O(N²) pairs compare karna impossible hai. Banding se sirf likely
similar pairs check hote hain. Probability = 1 − (1 − s^r)^b. b=16, r=4 pe, s=0.8 wala pair
99.98% baar pakda jaata hai. Phir verification step se false positives hat jaate hain.

**Q: Dedup split se pehle kyun?**
A: Warna ek copy train mein aur doosri val mein ja sakti hai, aur model ko "already dekha
hua" text pe test karoge.

**Q: Library (datasketch) kyun nahi use ki?**
A: Seekhne ke liye, aur memory ke liye. Mera version sirf signatures numpy array mein rakhta
hai aur sorting se buckets banata hai, jo millions tweets ke liye halka hai.

---

## 6. Train/val split (M1)

### Kya kiya
Har document ka split uske text ke **hash** se decide kiya, random number se nahi.

### Kyun kiya
Pipeline dobara chalao ya naya data add karo, toh bhi koi document val se train mein nahi
jaayega. Split stable rehta hai.

### Interview Q&A
**Q: Random split mein kya problem hai?**
A: Seed ya data order badla toh split badal jaata hai. Purane val docs train mein aa sakte
hain aur results compare karna galat ho jaata hai.

---

## 7. Pre-tokenization: Hindi ka ek real bug (M2a)

### Kya kiya
GPT-4 ka text-splitting regex use kiya, lekin `\p{L}` (letter) ke saath `\p{M}` (mark) bhi
add kiya.

### Kyun kiya
GPT-4 ke original pattern se "किताबें" har vowel sign pe toot jaata hai, kyunki matras
Unicode mein "letter" nahi, "mark" hain. Phir BPE kabhi poore Hindi syllables nahi seekh pata.
**Ye ek real reason hai ki purane tokenizers Hindi mein kharab the.**

### Interview Q&A
**Q: Pre-tokenization kya hai, kyun zaroori hai?**
A: BPE se pehle text ko "words" mein todna. Isse BPE "dog." ya "the dog" jaise ajeeb tokens
nahi banata, aur training fast hoti hai kyunki har unique word ek hi baar count hota hai.

---

## 8. BPE tokenizer scratch se (M2a), merge loop maine likha

### Kya kiya
Maine teen functions khud likhe:
- `count_pairs`: har adjacent pair kitni baar aaya (chunk ki frequency se multiply).
- `merge`: list mein pair ko naye id se replace (left to right, overlap nahi).
- `train` loop: count → sabse frequent pair → naya token → har jagah merge → repeat.

**Haath se example:** `"aaabdaaabac"` mein sabse pehle "aa" (count 4) merge hota hai.

### Mere output se kya seekha
- **Pehla merge `(224, 164)` tha**, jo Devanagari letters ke pehle 2 bytes hain. Hindi letter
  3 bytes ka hota hai, toh BPE sabse pehle unhe jodta hai.
- Phir letters → matras → syllables → poore words: `' है'`, `' में'`, `' hai'`, `' mein'`.
- Sample sentence: **73 bytes → 35 tokens**. Chhote data pe Hindi abhi bhi bytes mein
  tooti thi; bade data aur 16k vocab pe poore words banenge.

### Interview Q&A
**Q: Byte-level BPE kya hai?**
A: Start 256 bytes se karo, isliye koi bhi text "unknown" nahi hota (emoji, Hindi, kuch bhi).
Phir sabse frequent adjacent pair ko baar baar naye token mein merge karo. Merges ki ordered
list hi tokenizer hai.

**Q: Encode kaise hota hai?**
A: Seekhe hue merges usi order mein apply hote hain: har step pe wo pair jo sabse pehle seekha
gaya tha (lowest rank).

**Q: Naive BPE training slow kyun hai, kaise fast karein?**
A: Har merge pe poora corpus dobara count hota hai. Fast versions sirf affected pairs ke counts
update karte hain aur priority queue use karte hain (Hugging Face `tokenizers` Rust mein ye
karta hai).

**Q: BPE vs WordPiece vs Unigram?**
A: BPE frequency se merge karta hai. WordPiece likelihood badhane wala pair chunta hai. Unigram
bade vocab se shuru karke tokens hatata hai aur probabilistic segmentation deta hai.
GPT/Llama family byte-level BPE use karti hai.

**Q: Vocab size kaise choose kiya?**
A: Chhote model mein embedding table hi bahut parameters kha leta hai: 32k × 256 = 8.4M, jo
poore 5M model se bada hai! Isliye 16k rakha.

---

## 9. Engineering lessons (real bugs jo aaye)

| Problem | Kya hua | Fix |
|---|---|---|
| Windows pe test fail | `csv.field_size_limit(sys.maxsize)`: Windows pe C long 32-bit hota hai | `2**31 - 1` |
| Colab mein file nahi mili | Config ka relative path code folder se resolve ho raha tha, Drive se nahi | Path ko data dir se resolve kiya |
| 2.4 GB tar.gz unpack | ~5 GB extra Drive space lagta | Archive ko bina unpack kiye stream kiya |
| Cleaning bahut slow (941 docs/s, ~2 ghante) | Google Drive pe lakhon chhoti lines likhna slow hai | Local disk pe kaam, end mein Drive pe copy, 2 CPU workers: ~5x fast |
| Colab disconnect | Kaam dobara karna padta | Har stage complete output skip karta hai: same command dobara chalao |

### Interview Q&A
**Q: Is project mein sabse mushkil engineering problem kya thi?**
A: I/O. Code fast tha (local 17k docs/s), lekin Colab pe Drive pe likhne se ~20x slow ho gaya.
Profile karke pata chala bottleneck CPU nahi, storage tha. Lesson: pehle measure karo, phir
optimize.

---

## 10. Aage kya aayega (short preview)

- **Tokenizer comparison:** fertility (tokens per word) aur bytes per token: English vs
  Hindi vs Roman Hinglish, mera tokenizer vs GPT-4o / Llama 3 / Qwen3, plots ke saath.
- **GPT model:** causal self-attention, RoPE, RMSNorm, SwiGLU, weight tying, KV cache,
  sampling (temperature, top-k, top-p).
- **Training:** AdamW, warmup + cosine LR, gradient clipping, gradient accumulation, mixed
  precision, Colab-disconnect-safe checkpoints, W&B logging.
- **Experiments:** 3 model sizes, loss vs size aur tokens seen; **bits-per-byte** se
  tokenizers ka fair comparison; KV cache se generation kitni fast hoti hai.

**Q (aane wala): Loss ki jagah bits-per-byte kyun?**
A: Alag tokenizers alag number of tokens banate hain, toh per-token loss compare karna unfair
hai. Bits-per-byte = total loss (bits mein) ÷ text ke UTF-8 bytes, jo tokenizer pe depend nahi
karta.

---

## 11. Kya ye project karne layak hai?

**Haan, agar goal seekhna aur portfolio hai.** Honestly:

✅ **Fayde**
- LLM ka har part khud likha: data, tokenizer, model, training. Interview mein "kaise kaam
  karta hai" ka jawab ratta nahi, experience se aata hai.
- Real problem: Indian languages ke saath tokenizer unfairness measurable hai, aur ye topic
  research mein bhi discuss hota hai.
- Real engineering: messy data, licenses, slow storage, Windows bugs, disconnecting GPUs.
- Results aur plots dikhane layak hain: GitHub + README.

⚠️ **Limits (inhe chhupana nahi, interview mein khud bolna)**
- 30M parameter model ChatGPT jaisa smart nahi hoga. Ye seekhne ka model hai, product nahi.
- Data license non-commercial hai, toh commercial use nahi.
- Free GPU pe experiments chhote hain, toh conclusions "small scale pe" hain.

**Interview tip:** limits khud bolna maturity dikhata hai. "Maine ye measure kiya, ye
limitation hai, scale badhaun toh ye try karunga."

---

## 12. Engineering project hai ya research?

**Dono ka mix, lekin zyada engineering (~70%) + thoda research (~30%).**

| Engineering part | Research part |
|---|---|
| Data pipeline, MinHash dedup | Tokenizer fertility analysis (Hindi vs English vs Hinglish) |
| BPE + GPT scratch se | Model size vs loss (mini scaling study) |
| Training loop, checkpoints, mixed precision | Bits-per-byte se fair tokenizer comparison |
| Tests, Colab/Kaggle compatibility | Results ko interpret karna aur conclusions likhna |

Sahi naam: **"Research Engineering / Applied ML project."** Ye ML Engineer, Research
Engineer, ya LLM/NLP roles ke liye best fit hai.

---

## 13. Resume ke liye bullet points (project complete hone ke baad numbers update karna)

- Built a Hinglish corpus pipeline (6.9M+ docs) with Unicode-aware cleaning and from-scratch
  **MinHash-LSH deduplication** in numpy; deterministic hash-based train/val split.
- Implemented **byte-level BPE from scratch**; found and fixed a Devanagari pre-tokenization
  issue (combining marks split by GPT-4's regex).
- Compared tokenizer fertility vs **GPT-4o, Llama 3, Qwen3** on English / Hindi / Roman Hinglish. *(numbers TBD)*
- Implemented a decoder-only GPT in plain PyTorch (**RoPE, RMSNorm, SwiGLU, KV cache**);
  trained 5M/15M/30M models on free GPUs with resumable checkpoints. *(results TBD)*

---

## 14. Extra tricky interview questions (practice ke liye)

1. **Hindi ko English se zyada tokens kyun lagte hain?** UTF-8 mein Devanagari 3 bytes/char, aur
   English-heavy training data ki wajah se tokenizer ne Hindi ke kam merges seekhe.
2. **Zyada tokens ka nuksaan?** Zyada API cost, slow generation, aur same context window mein
   kam actual text.
3. **Dedup na karo toh?** Memorization, val leakage, aur repeated text pe compute waste.
4. **Tokenizer ka vocab bahut bada ho toh?** Embedding/output layer bada, rare tokens kam
   train hote hain. Bahut chhota ho toh sequences lambi, attention cost badhta hai.
5. **Agar tumhe ek hafte aur milta toh kya karte?** Bada data, transliteration-aware
   evaluation (Roman ↔ Devanagari), aur tokenizer ka downstream task pe effect measure karna.
