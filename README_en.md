# minimind-diffusion

[中文](README.md) · English

**Mask the text, then reveal it step by step. Learn masked diffusion with a small PyTorch model.**

Follow bidirectional attention and masked loss through training, sampling and visualization, then explore an experimental image-text extension. Built for readers who want to compare diffusion and autoregressive generation in code.

[Start with CPU tests](#first-run-small-cpu-tests) · [Core principles](#2-core-principles) · [Text training](#3-quick-start-text-dlm) · [Experiment sources](docs/experiment-evidence.en.md) · [Author's learning hub](https://adenzhou1350.github.io/learn/)

## A path through the code

| Step | What to inspect | Entry point |
|---|---|---|
| Small model mechanics | Bidirectional attention, shapes and masked loss | [Model tests](tests/test_model.py), [loss tests](tests/test_loss.py) |
| Text training | Random masking, pretraining and response-only SFT | [Model](model/model_dlm.py), [pretraining](trainer/train_pretrain.py) |
| Sampling | Predict, remask low-confidence tokens, repeat | [Sampling tests](tests/test_sampling.py), [Web demo](scripts/web_demo.py) |
| Image-text experiments | Frozen vision encoder, projector and diffusion | [Vision path](model/model_dlm_v.py), [limitations](#7-known-issues--honest-notes) |

```text
Training: clean text → random masks → bidirectional Transformer → masked loss
Sampling: all masks → predict → keep confident tokens → repeat until revealed
```

**Current scope:** a learning and reproduction project. The repository does not ship trained weights or datasets; text dialogue remains limited and the recorded VLM experiments do not reliably ground answers in images. Start with the small model tests, then prepare the tokenizer and data for training.

## First run: small CPU tests

```bash
git clone https://github.com/adenzhou1350/minimind-diffusion.git
cd minimind-diffusion
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest tests/test_model.py tests/test_loss.py -q
```

These tests use small random models and need no downloaded tokenizer, checkpoint or dataset. They check shapes, bidirectional attention and masked loss; passing them is not a measure of generation quality.

Next: [text training](#3-quick-start-text-dlm), [image-text training](#4-extension-multimodal-vlm), or [screening scripts and metric definitions](docs/experiment-evidence.en.md).

`minimind-diffusion` ports [minimind](https://github.com/jingyaogong/minimind)'s "train a small LLM from
scratch in PyTorch" minimalism onto a **Diffusion Language Model (DLM)**: instead of generating tokens
left-to-right, it starts from an all-`<mask>` sequence and **iteratively unmasks** it, like drafting and
then refining.

It builds on **full-sequence masked diffusion** from **LLaDA v1** (Nie et al., 2025), and adds a
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
| Loss | next-token CE | **uniform masked CE**, with mask ratio in [0.1, 0.5] (current implementation) |
| generate | AR loop + KV cache | diffusion sampling + remasking |
| Vocab | 6400 | 6401 (+`<mask>`) |

Style parity: single-file `model_dlm.py`, one `train_*.py` per stage, one-line cosine `get_lr`, JSONL data,
6400-BPE vocab, bilingual comments, `🌏🌎🌍` section markers. The multimodal side ([Part 2](#4-extension-multimodal-vlm))
likewise adds only a <50-line vision path on top of the text DLM, mirroring minimind-v's "inherit the LLM +
minimal vision" pattern.

**Who this is for**: anyone who wants to **verify from scratch that a diffusion model can actually be trained**,
and see the engineering differences between masked diffusion and autoregression. The training and sampling
code comes with historical notes on unstable dialogue and unreliable image grounding (§7).

---

## 2. Core principles

### 2.1 Diffusion vs autoregression
- **Autoregressive (AR)**: left to right, each step predicts the next token from what's already generated. `P(x) = ∏ P(x_i | x_{<i})`.
- **Masked diffusion (DLM)**: randomly **replace tokens with `<mask>`** (forward noising), train a
  **bidirectional** transformer to **predict the original tokens at masked positions**; at inference, start
  from all-`<mask>` and unmask over N steps.

### 2.2 Forward noising q(x_t | x_0)
The description below uses the original full range; the current `forward` samples `t ~ U(0.1, 0.5)`.
Each sequence samples a mask ratio `t ~ U(0,1)`; each real token position is **independently** masked to
`<mask>` with probability `t`:
- `t=0`: all clean; `t=1`: all masked.
- The mask ratio **grows linearly with t** — the key difference from BERT (fixed 15%).

### 2.3 Current training loss: uniformly weighted masked CE
```
Per sequence: sum(CE at masked positions) / max(number of masked tokens, 1)
Per batch: mean of the per-sequence losses
```
- Cross-entropy only at **masked positions**; padding is excluded, and SFT restricts masking to the response.
- The current code samples `t ~ U(0.1, 0.5)` with no additional `1/t` weight.
- Read the [implementation](model/model_dlm.py) alongside the [loss tests](tests/test_loss.py).

The old implementation multiplied the per-sequence mean above by `1/t`. This describes a change within
this repository, not evidence that the official LLaDA recipe fails on small models. See the
[experiment sources](docs/experiment-evidence.en.md) for the historical notes and missing evidence.

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
# smaller profile (memory needs depend on sequence length, batch and environment):
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
# smaller batch configuration: --max_seq_len 256 --batch_size 16 --accumulation_steps 8
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
Smaller training configuration: `hidden_size=512, num_hidden_layers=6`.

[`eval_dlm.py`](eval_dlm.py) defaults to `steps=64, gen_length=128, temperature=0.7, repetition_penalty=1.3`.
The model method has a different default temperature; check the entry point you use.
- Historical trial runs reported repetition at `temperature=0`; `0.6–0.9` and `repetition_penalty=1.2–1.5` were used as tuning ranges. These observations lack a common evaluation record and may not transfer to other weights.
- `block_length > 0` enables semi-autoregressive blocks and must divide `gen_length`. Earlier blocks stay fixed while tokens within a block are iteratively revealed; see [generate](model/model_dlm.py).

---

## 6. Differences from minimind

| Dimension | minimind | minimind-diffusion |
|---|---|---|
| Generation | autoregressive | masked diffusion |
| Attention | causal | bidirectional |
| KV cache | yes | no |
| Timestep t | none | yes, not fed to model |
| Loss | next-token CE | uniform-weighted masked CE (old implementation compared in §7.1) |
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

These are observations from historical runs. **The public repository and available master history do not include the matching raw logs, per-example results or checkpoints.** The numbers below are not reproduced benchmarks. Fixed source versions and missing evidence are listed in [experiment sources](docs/experiment-evidence.en.md). This documentation update checked source and history; it did not rerun training.

### 7.1 Loss recipe: current code and historical observations

The current code uses uniformly weighted masked CE with `t ~ U(0.1, 0.5)`. The [loss-change commit](https://github.com/adenzhou1350/minimind-diffusion/commit/fb96b344f5e58b7f9074de72e19473afdee67927) records this implementation change.

Two separate written records exist:

| Written source | Reported observation | Evidence available now |
|---|---|---|
| Loss-change commit (`fb96b34`) | 23M screening: about 5% versus 25.5% masked-reconstruction accuracy, the latter at 3,000 steps | Commit message only; no matching logs |
| [Early README](https://github.com/adenzhou1350/minimind-diffusion/blob/c08099625cbd36fec3bad8b2600d0f4036da03c2/README.md) | 0–5% for the old implementation; a later result labeled 64M reports 41.5% and loss 7.4→1.7 | No public record establishing identical data, training budget and evaluation settings |

**0–5% → 41.5% is not a controlled estimate of the gain from changing the loss alone.** Masked-reconstruction accuracy is also not question-answer accuracy. The [four-way screening script](screen_loss.py) can be inspected and rerun, but uses a 256×4 configuration; its presence does not substantiate the 64M result.

The extra `1/t` weight, mask range, model size and training steps may all matter. The old implementation divides by the number of masked tokens before multiplying by `1/t`, so the full normalization needs its own review. The earlier explanation that easy cases dominate gradients remains a hypothesis without gradient measurements or controlled ablations.

### 7.2 Sampling: repetition penalty and blocks

Historical trial runs produced repeated phrases; some examples became more coherent after adding repetition penalties and blocks. No common prompt set, saved outputs and seeds accompany that observation, so it does not establish a consistent quality gain.

The implementation down-weights revealed response tokens with `repetition_penalty` and fixes earlier blocks when `block_length > 0`. [Sampling tests](tests/test_sampling.py) check output length, removal of mask tokens and execution of sampling branches; they do not measure language quality.

### 7.3 Text: sentence generation with unreliable dialogue

The [early README](https://github.com/adenzhou1350/minimind-diffusion/blob/c08099625cbd36fec3bad8b2600d0f4036da03c2/README.md) describes a run as 64M, about 900k training examples and 2 epochs. Its sample observations were:

- Some coherent Chinese outputs and relevant answers to self-introduction or learning-to-program prompts.
- Continuations instead of answers, incorrect arithmetic and factual statements, and unreliable use of conversation context.
- A reported reconstruction change of roughly 41%→44% from epoch 1 to 2, without published evaluation data or logs.

Partial masking during training and all-mask starts at inference may create a distribution gap. Model size, data, training and sampling choices also deserve investigation. The available record does not identify the bottleneck or establish that a larger model would resolve it.

### 7.4 VLM: learned formatting, unreliable image grounding

Encoded images fill `<|image_pad|>` positions, which are never masked. Training first aligns the projector, then unfreezes selected language-model layers. See the [vision model](model/model_dlm_v.py), [alignment trainer](trainer/train_pretrain_vlm.py) and [SFT trainer](trainer/train_sft_vlm.py).

The [historical README](https://github.com/adenzhou1350/minimind-diffusion/blob/24a64b72f876fcc5e3aad2c7ffe0b08277f6914c/README.md) reports 22,691 Stage 2 steps, about 30 hours, and final loss 2.14 (roughly 1.8–2.4 during training). These lack matching logs and are not runtime estimates for the current configuration.

It records ChatML and image-description formatting, but similar hallucinations after changing the image; leaving image placeholders without supplying an image caused repetition. Per-image predictions and image-swap/no-image controls are not published, so these remain failure observations rather than a quantitative grounding evaluation.

The suggestion that visual tokens act as a format trigger is a hypothesis. Useful next checks are image-swap sensitivity, projector gradients and freezing settings before attributing the result to model or data scale.

### 7.5 Other implementation and experiment boundaries

- **Mask initialization:** [screen_init.py](screen_init.py) defines random / mean / zero configurations. Historical notes report no improvement, without raw outputs.
- **Data loading:** the text [JSONL index](dataset/lm_dataset.py) uses stdlib json and byte offsets; the [parquet loader](dataset/vlm_dataset.py) reads row groups.
- **Not included:** trained weights, datasets or DPO / LoRA / RL / distillation workflows. Start with the [tests requiring no model download](#first-run-small-cpu-tests), then prepare training assets.

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
