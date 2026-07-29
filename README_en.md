# mind-diffusion

> **The Great Way is Simple — a diffusion-flavored minimind**

`mind-diffusion` ports the "train a small LLM from scratch with raw PyTorch" philosophy of
[minimind](https://github.com/jingyaogong/minimind) onto a **Diffusion Language Model (DLM)**:
instead of generating tokens left-to-right, it starts from a **fully-`<mask>`ed** sequence and
**iteratively unmasks** it — like drafting, then refining.

It implements the **LLaDA v1** (Nie et al., 2025) **full-sequence masked diffusion** recipe.

---

## 1. What it is

A **minimal, runnable, principle-clear** diffusion language model, styled after minimind:

| Axis | minimind (AR) | mind-diffusion (DLM) |
|---|---|---|
| Generation | autoregressive (left→right next token) | masked diffusion (full mask → iterative unmask) |
| Attention | causal | **bidirectional** (no causal mask) |
| KV cache | yes | no (full sequence each step) |
| Timestep t | n/a | yes (mask ratio), but **not fed to model** |
| Loss | next-token CE | **1/t-weighted masked CE** (likelihood bound) |
| generate | AR loop + KV cache | diffusion sampling + remasking |
| Vocab | 6400 | 6401 (+`<mask>`) |

Style parity: single-file `model_dlm.py`, one `train_*.py` per stage, one-line cosine `get_lr`,
JSONL data, vocab-6400 BPE, bilingual comments.

---

## 2. Core principles

### 2.1 Diffusion vs autoregression
- **AR**: left-to-right; each step predicts the next token given the past. `P(x) = ∏ P(x_i | x_{<i})`.
- **Masked diffusion (DLM)**: randomly **replace tokens with `<mask>`** (forward noising); train a
  **bidirectional** transformer to **predict the original tokens at masked positions**; at inference,
  start from **all-`<mask>`** and iteratively unmask over N steps.

### 2.2 Forward noising q(x_t | x_0)
Sample a mask ratio `t ~ U(0,1)` per sequence; each real token is independently masked w.p. `t`:
- `t=0`: fully clean; `t=1`: fully masked.
- The masked fraction **grows linearly in t** — the key contrast with BERT's fixed 15%.

### 2.3 Training loss (1/t-weighted, a likelihood bound)
```
L(θ) = - E[ (1/t) · Σ_{masked i} CE(p_θ(x_0^i | x_t), x_0^i) ]
```
- Cross-entropy only at **masked positions**.
- **The 1/t weight is the soul**: small t (few masks, easy restore) gets high weight; large t
  (many masks, hard) gets low weight. This makes L(θ) an **upper bound on negative log-likelihood** —
  unlike BERT/MaskGIT's uniform-weighting heuristic.

### 2.4 Architecture: LLaMA without the causal mask
A bidirectional transformer = standard LLaMA decoder with **causal mask removed** (`is_causal=False`).
Keeps RoPE + RMSNorm + SwiGLU + GQA + per-head QK-norm. **No KV cache** (bidirectional +
non-autoregressive don't use one). **No timestep fed to the model** (time-free): the transformer
sees only "what x_t looks like", not "which step we're on".

### 2.5 Sampling: semi-autoregressive iterative unmasking
From all-`<mask>`, N uniform timesteps `t_k = 1 - k/N` (`s` goes 1 → 0):
1. Run the bidirectional transformer → **argmax**-predict every masked position.
2. Score **confidence** (low-confidence = softmax probability).
3. Write all predictions back; then by ascending confidence, **re-mask the lowest `floor(L·s)`**
   back to `<mask>` (low-confidence remasking) — next step re-guesses them with fuller context.
4. `s` → 0 linearly shrinks the masked count to 0; the last step fully unmasks.

> **Emergent behavior**: low-confidence remasking can let "reasoning that emerged mid-draft" get
> masked back — the model's draft reasoning may be re-masked until context is more complete. Visible
> in the `web_demo` streaming output.

---

## 3. Quick start

### 3.1 Install
```bash
pip install -r requirements.txt
```

### 3.2 Place tokenizer + data (reuse minimind)
Copy from [minimind](https://github.com/jingyaogong/minimind):
- `model/tokenizer.json` + `model/tokenizer_config.json` (vocab-6400 BPE)
- `dataset/pretrain_t2t_mini.jsonl` (~1.2GB)
- `dataset/sft_t2t_mini.jsonl` (~1.6GB)

(See `dataset/dataset.md`. Both are gitignored.)

### 3.3 Pretrain
```bash
python trainer/train_pretrain.py
# small profile (8GB GPU / CPU smoke):
python trainer/train_pretrain.py --hidden_size 512 --num_hidden_layers 6 --batch_size 4 --max_seq_len 128
```
→ `out/pretrain_768.pth` (or `_512.pth`).

### 3.4 SFT (conditional generation)
```bash
python trainer/train_sft.py
```
→ `out/sft_768.pth`. SFT masks only assistant tokens (prompt never masked), minimind chat template.

### 3.5 Inference
```bash
python eval_dlm.py --from_weight sft
```
Prints Chinese-prompt generations + tokens/s.

### 3.6 Web demo (streaming diffusion sampling)
```bash
streamlit run scripts/web_demo.py
```
Refreshes the unmasked tokens each step — watch "draft → refine" live.

---

## 4. Config

Default (aligned with minimind): `hidden=768, layers=8, heads=8, kv_heads=4 (GQA 2:1), vocab=6401,
intermediate_size=ceil(π·768/64)*64, rope_theta=1e6, tied embeddings`.
Small smoke profile: `hidden=512, layers=6`.

Sampling defaults: `steps=64, gen_length=128, temperature=0.0, low_confidence=True`.
- Paper-optimal is `steps ≈ gen_length` (slow); the fixed small T is a teaching compromise.
- `temperature=0` greedy (paper eval); >0 adds Gumbel noise.
- **Temperature-only sampling** (no top-k/top-p/repetition penalty): diffusion argmax-masks each
  step, so nucleus truncation adds little; "repetition" is ill-defined in non-AR.

---

## 5. Differences from minimind (quick reference)

| Axis | minimind | mind-diffusion |
|---|---|---|
| Generation | autoregressive | masked diffusion |
| Attention | causal | bidirectional |
| KV cache | yes | no |
| Timestep t | n/a | yes, not fed to model |
| Loss | next-token CE | 1/t-weighted masked CE |
| dataset returns | (input_ids, labels) | (input_ids, attention_mask[, response_mask]) |
| Where masking happens | n/a | in `forward`, sampled per step |
| generate | AR + KV cache | diffusion sampling + remasking |
| Vocab | 6400 | 6401 |
| Sampling knobs | temp/topk/topp/rep | temperature only |
| Stages | pretrain/sft/lora/dpo/ppo/grpo/agent/distill | **pretrain/sft** (this repo) |

> **Masking happens in `forward`**: minimind's dataset returns `labels`; diffusion's mask is random
> per step, so it can't live in the dataset — the dataset returns clean seq + mask ranges, and
> masking + loss happen in `DLMForMD.forward`.

---

## 6. Known issues / honest notes
1. **Small-model quality**: tiny model + mini corpus → generation quality not guaranteed; the goal
   is "principles clear + full pipeline runs".
2. **Sampling-step compromise**: fixed T=64 (paper-optimal T=L is slow).
3. **Sampling simplification**: temperature-only (see §4).
4. **`<mask>` init**: standard learnable init (no mean-averaging — matches LLaDA official code).
5. **Not in this repo**: DPO/LoRA/RL/distillation (LLaDA 2 block diffusion also not implemented) —
   future specs.

---

## 7. Structure
```
model/
  model_dlm.py        # DLMConfig + bidirectional transformer + DLMForMD (mask+loss+generate)
  tokenizer_loader.py  # reuse minimind BPE + <mask>
dataset/
  lm_dataset.py        # PretrainDataset + SFTDataset
trainer/
  trainer_utils.py     # get_lr / SkipBatchSampler / init_model / Logger
  train_pretrain.py    # full-seq random masking + 1/t loss
  train_sft.py         # mask response only
scripts/
  web_demo.py          # Streamlit streaming diffusion sampling
eval_dlm.py            # Chinese prompts + tokens/s
tests/                 # pytest: model / loss / sampling / dataset / trainer_utils
```

---

## 8. Acknowledgments
- [minimind](https://github.com/jingyaogong/minimind) — the source of the minimal style and "train from 0" philosophy.
- [LLaDA](https://github.com/ML-GSAI/LLaDA) (Nie et al., 2025, arXiv:2502.09992) — the masked-diffusion method and official implementation.

## License
Apache-2.0 (same as minimind).
