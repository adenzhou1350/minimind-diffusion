# minimind-diffusion

> **Less is more — a diffusion-model take on minimind & minimind-v**

`minimind-diffusion` ports [minimind](https://github.com/jingyaogong/minimind)'s "train a small LLM from
scratch in PyTorch" minimalism onto a **Diffusion Language Model (DLM)**: instead of generating tokens
left-to-right, it starts from an all-`<mask>` sequence and **iteratively unmasks** it, like drafting and
then refining.

It implements the **full-sequence masked diffusion** recipe of **LLaDA v1** (Nie et al., 2025), and adds a
vision path on top ([minimind-v](https://github.com/jingyaogong/minimind-v) style) — going all the way from
a **text DLM** to an **image-text multimodal VLM**.

One repo, two tracks:

- **Part 1 · Text diffusion LM** — pretrain → SFT → inference → web demo (§3)
- **Part 2 · Multimodal extension (VLM)** — frozen SigLIP2 + projector, two-stage align → SFT (§4)

---

## Table of contents

- [1. What it is](#1-what-it-is)
- [2. Core principles](#2-core-principles)
- [3. Quick start: text DLM](#3-quick-start-text-dlm)
- [4. Extension: multimodal VLM](#4-extension-multimodal-vlm)
- [5. Config](#5-config)
- [6. Differences from minimind](#6-differences-from-minimind)
- [7. Known issues / honest notes](#7-known-issues--honest-notes)
- [8. Structure](#8-structure)
- [9. Acknowledgments](#9-acknowledgments)

---

## 1. What it is

A **minimal, runnable, principle-first** diffusion LM project, aligned with minimind's style:

| Dimension | minimind (AR) | minimind-diffusion (DLM) |
|---|---|---|
| Generation | autoregressive (L→R next token) | masked diffusion (all-mask → iterative unmask) |
| Attention | causal | **bidirectional** (no causal mask) |
| KV cache | yes | no (whole sequence each step) |
| Timestep t | none | yes (mask ratio), but **not fed to the model** |
| Loss | next-token CE | **1/t-weighted masked CE** (likelihood bound) |
| generate | AR loop + KV cache | diffusion sampling + remasking |
| Vocab | 6400 | 6401 (+`<mask>`) |

Style parity: single-file `model_dlm.py`, one `train_*.py` per stage, one-line cosine `get_lr`, JSONL data,
6400-BPE vocab, bilingual comments, `🌏🌎🌍` section markers. The multimodal side ([Part 2](#4-extension-multimodal-vlm))
likewise adds only a <50-line vision path on top of the text DLM, mirroring minimind-v's "inherit the LLM +
minimal vision" pattern.

**Who this is for**: anyone who wants to **verify from scratch that a diffusion model can actually be trained**,
and see every engineering difference between masked diffusion and autoregression. It does not chase SOTA — it
lays out both the minimal working pipeline and the **real scale wall** (§7).

---

## 2. Core principles

### 2.1 Diffusion vs autoregression
- **Autoregressive (AR)**: left to right, each step predicts the next token from what's already generated. `P(x) = ∏ P(x_i | x_{<i})`.
- **Masked diffusion (DLM)**: randomly **replace tokens with `<mask>`** (forward noising), train a
  **bidirectional** transformer to **predict the original tokens at masked positions**; at inference, start
  from all-`<mask>` and unmask over N steps.

### 2.2 Forward noising q(x_t | x_0)
Each sequence samples a mask ratio `t ~ U(0,1)`; each real token position is **independently** masked to
`<mask>` with probability `t`:
- `t=0`: all clean; `t=1`: all masked.
- The mask ratio **grows linearly with t** — the key difference from BERT (fixed 15%).

### 2.3 Training loss (1/t-weighted, a likelihood bound)
```
L(θ) = - E[ (1/t) · Σ_{masked i} CE(p_θ(x_0^i | x_t), x_0^i) ]
```
- Cross-entropy only at **masked positions**.
- **The 1/t weight is the soul**: small t (few masked, easy) → large weight; large t (many masked, hard) →
  small weight. This makes L(θ) an **upper bound** on the negative log-likelihood — unlike BERT/MaskGIT's
  uniform-weight heuristic.

> In practice this "rigorous" recipe **does not work** on a small model, so it was changed to uniform
> weighting — see the honest note in §7.1.

### 2.4 Architecture: LLaMA without the causal mask
The bidirectional transformer is a standard LLaMA decoder with the **causal mask turned off** (`is_causal=False`).
It keeps RoPE + RMSNorm + SwiGLU + GQA + per-head QK-norm. **No KV cache** (bidirectional + non-AR, none
needed). **The timestep t is not fed to the model** (time-free): the transformer only sees "what x_t looks
like", not "which step it is" — so one set of weights covers both pretrain and SFT.

### 2.5 Sampling: semi-autoregressive iterative unmasking
Start from all-`<mask>`, run N uniform steps `t_k = 1 - k/N` (`s` decays 1→0):
1. Run the bidirectional transformer → **argmax** the original token at each `<mask>` position.
2. Compute **confidence** (low-confidence strategy = softmax probability).
3. Temporarily write back all predictions; then, in ascending confidence, **re-mask the lowest `floor(L·s)`**
   back to `<mask>` (low-confidence remasking) — re-guessed next step with more context.
4. `s` decays 1→0, masked count shrinks linearly to 0; the last step reveals everything.

> **Emergent behavior**: low-confidence remasking can make "reasoning steps that popped up mid-draft get
> masked again" — visible live in the `web_demo` streaming output. LLaDA's paper notes this too.

---

## 3. Quick start: text DLM

### 3.1 Install
```bash
pip install -r requirements.txt
```

### 3.2 Place tokenizer + data (reuse minimind)
Copy from the [minimind](https://github.com/jingyaogong/minimind) repo:
- `model/tokenizer.json` + `model/tokenizer_config.json` (6400-BPE vocab)
- `dataset/pretrain_t2t_mini.jsonl` (~1.2GB)
- `dataset/sft_t2t_mini.jsonl` (~1.6GB)

(See `dataset/dataset.md`. These two files are gitignored.)

### 3.3 Pretrain
```bash
python trainer/train_pretrain.py
# small profile (8GB GPU / CPU smoke):
python trainer/train_pretrain.py --hidden_size 512 --num_hidden_layers 6 --batch_size 4 --max_seq_len 128
```
Produces `out/pretrain_768.pth` (or `_512.pth`).

### 3.4 SFT (conditional generation)
```bash
python trainer/train_sft.py
# small profile:
python trainer/train_sft.py --hidden_size 512 --num_hidden_layers 6 --batch_size 4
```
Produces `out/sft_768.pth`. SFT masks only the assistant answer positions (prompt never masked), using
minimind's chat template.

### 3.5 Inference
```bash
python eval_dlm.py --from_weight sft
# small profile:
python eval_dlm.py --hidden_size 512 --num_hidden_layers 6 --from_weight sft
```
Prints generations for Chinese prompts + tokens/s. For interactive multi-turn chat: `python chat_dlm.py`.

### 3.6 Web demo (streaming diffusion sampling)
```bash
streamlit run scripts/web_demo.py
```
Refreshes the currently-unmasked tokens each step — watch "draft → refine" directly.

---

## 4. Extension: multimodal VLM

On top of the text DLM, add a vision path mirroring [minimind-v](https://github.com/jingyaogong/minimind-v):
a frozen SigLIP2 (95M, 256px, 8×8=64 tokens, 768-dim) + MLP projector, with vision tokens filling the
`<|image_pad|>` placeholders (an observation condition, **never masked**); the diffusion loss stays on text
only. LLaVA-style prefix injection (not cross-attn). `vocab_size=6401` is unchanged (image_pad reuses
minimind's reserved id 12, no new token).

> **Only a vision path is added; the text diffusion core is untouched**: `DLMForVLM` inherits `DLMForMD`,
> and the vision side is a thin encoder + projector layer. It shares the same sampling / loss / tokenizer as Part 1.

### 4.1 Place data + vision encoder (download yourself, gitignored)
- `dataset/pretrain_i2t.parquet` (~4.1GB, image-text pairs, 1.27M rows) + `dataset/sft_i2t.parquet` (~4.6GB, multi-turn instructions with images, 2.9M rows)
- `model/siglip2-base-p32-256-ve/` (download from [jingyaogong/siglip2-base-p32-256-ve](https://huggingface.co/jingyaogong/siglip2-base-p32-256-ve))

### 4.2 Two-stage training
```bash
# Stage 1 alignment: freeze the whole LLM, train projector only (lr 4e-4, freeze_mode=2)
python trainer/train_pretrain_vlm.py
# Stage 2 SFT: LLM first/last layers + final norm + lm_head + projector (lr 5e-6, freeze_mode=1)
python trainer/train_sft_vlm.py
# small profile (8GB GPU): --max_seq_len 256 --batch_size 16 --accumulation_steps 8
```
Stage 1 → `out/vlm_align_768.pth`, Stage 2 → `out/vlm_sft_768.pth`.
Stage 2 checkpoints every 1000 steps (one epoch ~47h, so a crash won't lose much);
`--skip N --from_weight vlm_sft` resumes from the N-th accum step.

### 4.3 VLM inference
```bash
python eval_dlm_vlm.py --from_weight vlm_sft --sample_idx 0
```
Takes the image at row `--sample_idx` from `dataset/sft_i2t.parquet`, runs 3 Chinese prompts
(describe / what's in it / main color), prints generation + tokens/s, and prints the reference
prompt/answer from the parquet for comparison. The sample image is saved to `out/eval_sample.jpg`.

### 4.4 VLM web demo
```bash
streamlit run scripts/web_demo_vlm.py
```
Upload an image + text prompt, watch diffusion unmask live (unrevealed `<mask>` shown as ▍). Sliders for
gen_length / steps / temperature / repetition penalty.

---

## 5. Config

Default profile (aligned with minimind):
```
hidden_size=768, num_hidden_layers=8, num_attention_heads=8,
num_key_value_heads=4 (GQA 2:1), vocab_size=6401,
intermediate_size = ceil(π·768/64)*64, rope_theta=1e6, tied embeddings
```
Small smoke profile (8GB / CPU): `hidden_size=512, num_hidden_layers=6`.

Sampling defaults: `steps=128, gen_length=128, temperature=0.7, repetition_penalty=1.3, low_confidence=True`.
- The paper's optimum is `steps ≈ gen_length` (slow); 128 is enough for a teaching demo.
- `temperature=0` collapses into repeated high-frequency tokens (a diffusion trait, opposite to AR); use **0.6–0.9**, >0.9 starts rambling.
- `repetition_penalty` 1.2–1.5 breaks up repetition (bidirectional diffusion has no generation history, loops easily); 1.0 deadlocks, 2.0 forces gibberish.
- `block_length` >0 enables semi-AR block generation (LLaDA 2 idea), must divide gen_length; more anti-repetition but choppier.

---

## 6. Differences from minimind

| Dimension | minimind | minimind-diffusion |
|---|---|---|
| Generation | autoregressive | masked diffusion |
| Attention | causal | bidirectional |
| KV cache | yes | no |
| Timestep t | none | yes, not fed to model |
| Loss | next-token CE | uniform-weighted masked CE (original 1/t fails on small models, see §7.1) |
| dataset returns | (input_ids, labels) | (input_ids, attention_mask[, response_mask]) |
| Where masking happens | none | sampled live in forward (random each step) |
| generate | AR + KV cache | diffusion sampling + low-conf remasking |
| Vocab | 6400 | 6401 |
| Sampling knobs | temp/topk/topp/rep | temperature + repetition_penalty + block |
| Training stages | pretrain/sft/lora/dpo/ppo/grpo/agent/distill | **pretrain/sft** (text) + **align/sft** (multimodal) |

> **Masking sampled in forward**: minimind's dataset returns `labels` (target tokens); the diffusion mask is
> random each step, so **putting it in the dataset is meaningless** — the dataset returns only the clean
> sequence + mask range, and the masking + loss happen in `DLMForMD.forward`.

---

## 7. Known issues / honest notes

> This section is the most valuable part of the repo: an honest record of what a 64M diffusion model
> **can and cannot learn**.

### 7.1 Loss recipe: from LLaDA's 1/t to uniform weighting (key finding)
The original LLaDA uses `1/t`-weighted masked CE (mathematically a likelihood bound, rigorous), `t~U(0,1)`.
But **it does not work on a small model + little data**: held-out mask-reconstruction accuracy is only **0–5%**.

Diagnosis: the 1/t weight lets gradients be dominated by "lightly-masked easy cases" (1/t explodes at small t),
so the model only learns the token marginal, not "reconstruct from context". After screening 4 controls,
**uniform weight + `t~U(0.1,0.5)`** pushes accuracy to **41.5%** (64M), with a healthy curve (loss 7.4→1.7).

Cost: no longer a strict diffusion likelihood bound (NELBO) — more like BERT-style MLM + diffusion sampling.
**The denoising-matching objective is still there, the sampling math is unchanged**; only the trade-off weight
is no longer optimal. LLaDA 2 itself clamps the mask ratio to `[α_min,α_max]`, on the same path. Honest label:
deviates from the original for small-model learnability.

### 7.2 Sampling: repetition penalty + block (anti-loop)
A small diffusion LM loops on high-frequency patterns ("the sky is blue, the sky is blue...") — bidirectional
sampling has no generation history (each position predicted independently), unlike AR's causal constraint.
- `repetition_penalty` (default 1.3): down-weights already-revealed response tokens. 1.0 deadlocks, 1.2–1.5 sweet spot, 2.0 forces gibberish.
- `block_length` (default 0): >0 enables semi-AR block generation (LLaDA 2 idea) — AR across blocks, diffusion within, later blocks see earlier clean blocks.

Effect: from "deadlocked repetition" → "coherent multi-sentence Chinese".

### 7.3 The scale wall: can talk, can't converse (text side)
64M + mini corpus (900k samples) trained to 2 epochs, base model:
- ✅ Produces **coherent Chinese** (language modeling learned)
- ✅ **Answers some questions for real** ("what's your name" → self-intro; "how to learn programming" → Python/Codecademy tips)
- ❌ **Unstable**: still slips into continuation mode ("what's 1+1" → starts listing "-2 squared is..." math problems)
- ❌ **Knowledge hallucination** ("1+1=0.2133", "Mercury reflects the sun")
- ❌ **Ignores context across turns**

Diagnosis: a **train/inference distribution gap** (training masks part, inference masks the whole start; a small
model can't bridge it) + **insufficient scale** (LLaDA's 8B+2.3T pins it, 64M can't). Multi-epoch has diminishing
returns (1→2 epoch, acc 41→44, +3 pts) — it hits the scale wall. A qualitative jump to "stable conversation"
needs a bigger model / more data, beyond minimind's tier.

### 7.4 VLM: learns the format, not how to see (multimodal side)
Ported minimind-v's "inherit the LLM + <50-line vision path" to the diffusion side: vision tokens injected as
an **observation condition** (fill placeholders, never masked), diffusion loss still on text. Two stages
(alignment trains projector only → SFT unfreezes first/last layers).

Stage 2 SFT is complete (end of epoch, 22691 steps, ~30h, final loss 2.14, a noisy 1.8–2.4 plateau with no
downward trend). Honest final-eval conclusions:

- ✅ **Learned the VQA format/behavior**: clean ChatML, no token leakage, coherent sentences, on-topic (with emotional associations) for color questions
- ❌ **Did not learn grounding**: swapping images produces the same hallucination (cookies on a table → "summer night / nature scene"; a classical oil painting → "oranges in a forest"; heart + text → "white background / font") — the image is essentially ignored
- ⚠️ **Collapses without an image**: with `<|image_pad|>` placeholders but no image → collapses into an `<|im_start|>assistant` loop (placeholder positions are always overwritten by vision features during training, so bare placeholders are out-of-distribution); pure text with no placeholders → falls back to the text-side scale wall (§7.3)

Diagnosis: the same scale wall as §7.3 — 64M can't pin multimodal alignment, and the vision tokens act more
like a "**format trigger**" (keeping the model in the descriptive-ChatML region) than a real grounding signal.
The achieved goal is a **working minimal "diffusion + visual condition" pipeline**, not a SOTA VLM. Real
grounding likewise needs a bigger base + more image-text data.

### 7.5 Other
- **`<mask>` init**: standard learnable init (matching LLaDA official; mean/zero init showed no improvement after screening, see `screen_init.py`)
- **Not in this repo**: DPO/LoRA/RL/distillation (LLaDA 2 block diffusion partially implemented) — independent follow-ups, outside the minimal teaching scope
- **Data loading (text)**: stdlib json + byte-offset index, no datasets/pandas/pyarrow dependency (pyarrow crashes on import under Python 3.14 in this environment)
- **Data loading (VLM)**: parquet read **row-group by row-group** via pyarrow (`read_row_group` into Python lists), not a one-shot full-column `read` — the latter asks Arrow for a single 20GB+ contiguous buffer and fails realloc (`ArrowMemoryError`), which was the earlier SFT hang

---

## 8. Structure
```
model/
  model_dlm.py         # DLMConfig + bidirectional transformer + DLMForMD (mask + loss + generate)
  model_dlm_v.py       # DLMVLMConfig + MMVisionProjector + DLMForVLM (inherits DLMForMD, adds vision path)
  tokenizer_loader.py  # reuse minimind BPE + <mask> (<|image_pad|> uses reserved id 12)
dataset/
  lm_dataset.py         # PretrainDataset + SFTDataset (text)
  vlm_dataset.py        # PretrainVLMDataset + SFTVLMDataset (parquet+PIL, row-group streaming)
  dataset.md            # data placement notes
trainer/
  trainer_utils.py      # get_lr / SkipBatchSampler / init_model / init_vlm_model / freeze_vlm / vlm_checkpoint
  train_pretrain.py     # full-sequence random masking + uniform-weight loss
  train_sft.py          # [prompt;response], mask response only
  train_pretrain_vlm.py # Stage 1 alignment: freeze whole LLM, train projector only
  train_sft_vlm.py      # Stage 2 SFT: LLM first/last layers + projector, checkpoint every 1000 steps
scripts/
  web_demo.py           # Streamlit streaming diffusion sampling (text)
  web_demo_vlm.py       # Streamlit streaming diffusion sampling (image+text)
eval_dlm.py             # Chinese prompts + tokens/s (cf. minimind eval_llm.py)
eval_dlm_vlm.py         # image + Chinese prompt diffusion sampling (cf. minimind-v eval)
chat_dlm.py             # interactive multi-turn chat
screen_*.py             # training-recipe screening scripts (loss weight / mask init / scale, see §7.1)
tests/                  # pytest: model / loss / sampling / dataset / vlm / trainer_utils
```

---

## 9. Acknowledgments
- [minimind](https://github.com/jingyaogong/minimind) — the source of the minimalist style and "train from scratch" philosophy.
- [minimind-v](https://github.com/jingyaogong/minimind-v) — the "inherit the LLM + minimal vision path" pattern for the multimodal extension.
- [LLaDA](https://github.com/ML-GSAI/LLaDA) (Nie et al., 2025, arXiv:2502.09992) — the method and official implementation of masked diffusion language models.

## License
Apache-2.0 (same as minimind).
