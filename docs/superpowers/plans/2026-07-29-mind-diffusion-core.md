# mind-diffusion 核心脊柱 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现一个 LLaDA v1 掩码扩散语言模型项目(mind-diffusion),从 tokenizer 复用、双向 transformer、1/t 加权掩码 loss、半自回归迭代采样,到 pretrain/SFT/eval/web_demo 全链路跑通。

**Architecture:** 镜像 minimind 的仓库结构(单文件 `model_dlm.py`、一 stage 一 `train_*.py`、一行余弦 `get_lr`、JSONL 数据),但内核换成扩散:双向注意力(去 causal mask)+ 无 KV cache + `1/t` 加权掩码 CE loss + 迭代 unmasking 采样。数据/tokenizer 复用 minimind,新增一个 `<mask>` token(vocab 6400→6401)。

**Tech Stack:** Python 3.10+,PyTorch(≥2.0,带 sdpa),HuggingFace `transformers`/`tokenizers`/`datasets`,Streamlit,pytest。

## Global Constraints

(来自 spec §1, §4.2, §7.1, §8 — 所有任务都隐式遵守)

- **项目根**:`D:\codes\mind-diffusion`,所有相对路径以此为根。
- **Python ≥ 3.10**,PyTorch ≥ 2.0(需 `scaled_dot_product_attention` 原生支持非因果)。
- **vocab_size = 6401**(minimind 6400 + `<mask>`),`mask_token_id = 6400`。
- **模型默认档**:`hidden=768, layers=8, heads=8, n_kv_heads=4(GQA 2:1)`,`intermediate_size = ceil(π·768/64)*64`,**tied embeddings**。
- **小档冒烟档**:`hidden=512, layers=6, heads=8, n_kv_heads=4`。
- **LR schedule** 一行余弦:`get_lr(current_step, total_steps, lr) = lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))`,无 warmup。
- **注释风格**:`model_dlm.py` 英文为主 + `🌏🌎🌍` 边框分段;`train_*.py` 用 `# ========== N. <中文标题> ==========`;`argparse help=` 用中文;`dataset/lm_dataset.py` 注释中文。
- **数据**:JSONL + HF `load_dataset('json')`(不是 .bin memmap)。复用 minimind `pretrain_t2t_mini.jsonl` / `sft_t2t_mini.jsonl` + `tokenizer.json`(用户自行放置,见 `dataset/dataset.md`)。
- **dtype**:训练 bf16(autocast + 仅 fp16 用 GradScaler)。
- **checkpoint**:`.pth`,保存时权重 halved + 移 CPU。
- **`generate` 装饰** `@torch.inference_mode()`。
- **采样默认**:`steps=64, gen_length=128, temperature=0.0, low_confidence=True`。
- **license**:Apache-2.0。
- **每完成一个 task 就 commit**(commit message 末尾加 `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`)。

---

## File Structure(本计划创建/修改的文件)

| 文件 | 责任 | 创建 task |
|---|---|---|
| `requirements.txt` | 依赖固定 | T1 |
| `.gitignore` | 忽略数据/checkpoint/venv | T1 |
| `model/model_dlm.py` | **一个文件**:DLMConfig + RMSNorm + Attention(双向) + FeedForward + DLMBlock + DLMModel + DLMForMD(掩码+loss+generate) | T4-T6 |
| `model/tokenizer_loader.py` | 加载 minimind tokenizer + resize 加 `<mask>` | T2 |
| `model/tokenizer.json`, `tokenizer_config.json` | 复用 minimind(用户放置);`.gitignore` 排除大文件 | — |
| `dataset/lm_dataset.py` | PretrainDataset + SFTDataset(返回干净序列 + mask 范围) | T7 |
| `dataset/dataset.md` | 5 行 stub:数据放置说明 | T7 |
| `trainer/trainer_utils.py` | get_lr / SkipBatchSampler / init_model / Logger / setup_seed / lm_checkpoint | T8 |
| `trainer/train_pretrain.py` | 全序列随机掩码 + 1/t loss 训练 | T9 |
| `trainer/train_sft.py` | [prompt;response] 只掩 response | T10 |
| `eval_dlm.py` | 中文 prompt + tokens/s + 生成样例 | T11 |
| `scripts/web_demo.py` | Streamlit 扩散采样流式 | T12 |
| `tests/test_model.py` | 形状/双向/`<mask>` id 测试 | T3 |
| `tests/test_loss.py` | 1/t 权重/pad/EOS 测试 | T5 |
| `tests/test_sampling.py` | generate 跑完/截断/两分支测试 | T6 |
| `README.md`, `README_en.md`, `LICENSE` | 文档 + 许可 | T13 |

**依赖链**:
- T1(脚手架) → 所有
- T2(tokenizer) → T4, T7, T9, T10, T11, T12
- T3+T4(模型骨架与测试) → T5(forward+loss) → T6(generate) → T9, T10, T11, T12
- T7(dataset) → T9, T10
- T8(trainer_utils) → T9, T10
- T9(pretrain) → T11, T12(有 checkpoint 才能 eval/demo)
- T10(SFT) → T11, T12(SFT checkpoint 才像对话)
- T11, T12 可并行
- T13(README) 最后,但可与 T11/T12 并行

---

## Task 1: 项目脚手架

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `model/__init__.py`(空)
- Create: `dataset/__init__.py`(空)
- Create: `trainer/__init__.py`(空)
- Create: `scripts/__init__.py`(空)
- Create: `tests/__init__.py`(空)

**Interfaces:**
- Consumes: 无
- Produces: 目录结构 + 依赖清单,后续任务直接 `from model.model_dlm import ...`

- [ ] **Step 1: 写 requirements.txt**

```
torch>=2.0
transformers>=4.40
tokenizers>=0.15
datasets>=2.14
streamlit>=1.30
pytest>=7.0
```

- [ ] **Step 2: 写 .gitignore**

```
__pycache__/
*.pyc
.venv/
venv/
# 大文件:语料与 tokenizer 不入库(用户从 minimind 自行放置)
dataset/*.jsonl
model/tokenizer.json
model/tokenizer_config.json
# checkpoint
out/
*.pth
# 训练曲线
images/*.png
```

- [ ] **Step 3: 建空 `__init__.py`**

5 个目录各一个空 `__init__.py`(`model/`、`dataset/`、`trainer/`、`scripts/`、`tests/`)。

- [ ] **Step 4: 验证可 import**

Run: `python -c "import model, dataset, trainer, scripts, tests"`
Expected: 无输出,退出码 0。

- [ ] **Step 5: Commit**

```bash
git add requirements.txt .gitignore model/__init__.py dataset/__init__.py trainer/__init__.py scripts/__init__.py tests/__init__.py
git commit -m "chore: project scaffold

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: tokenizer 加载器(复用 minimind + 加 `<mask>`)

**Files:**
- Create: `model/tokenizer_loader.py`
- Create: `dataset/dataset.md`

**Interfaces:**
- Consumes: minimind 的 `model/tokenizer.json` + `tokenizer_config.json`(用户从 minimind 仓库拷贝放到 `model/` 下;`.gitignore` 已排除)
- Produces: `load_tokenizer(path='model') -> PreTrainedTokenizerFast`,词表已 +1 `<mask>`,`mask_token_id` 固定为 `6400`。后续任务用 `mask_id = tokenizer.convert_tokens_to_ids('<mask>')` 或直接常量 `MASK_ID = 6400`。

- [ ] **Step 1: 写 dataset.md(5 行 stub,跟 minimind 一致)**

```markdown
# 数据集放置说明

将所有下载的数据集文件(pretrain_t2t_mini.jsonl、sft_t2t_mini.jsonl)放置到当前目录。
tokenizer.json、tokenizer_config.json 放置到 model/ 目录。

数据集与 tokenizer 均来自 [minimind](https://github.com/jingyaogong/minimind)。
```

- [ ] **Step 2: 写 tokenizer_loader.py**

```python
"""
🌏🌎🌍 tokenizer loader: 复用 minimind BPE tokenizer,新增 <mask> token 🌏🌎🌍
"""
from transformers import AutoTokenizer

MASK_TOKEN = '<mask>'
MASK_ID = 6400  # minimind vocab=6400, <mask> 追加为最后一个 id


def load_tokenizer(path='model'):
    """加载 minimind tokenizer 并 resize 词表 +1 <mask>。"""
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if MASK_TOKEN not in tok.get_vocab():
        tok.add_tokens([MASK_TOKEN])
    assert tok.convert_tokens_to_ids(MASK_TOKEN) == MASK_ID, \
        f'<mask> id 应为 {MASK_ID},实得 {tok.convert_tokens_to_ids(MASK_TOKEN)}'
    return tok


if __name__ == '__main__':
    tok = load_tokenizer('model')
    print(f'vocab_size={tok.vocab_size}, mask_id={tok.convert_tokens_to_ids(MASK_TOKEN)}')
```

- [ ] **Step 3: (用户放置 tokenizer 后)验证**

Run: `python -c "from model.tokenizer_loader import load_tokenizer; t=load_tokenizer('model'); print(t.vocab_size, t.convert_tokens_to_ids('<mask>'))"`
Expected: `6401 6400`

> **注:** 若用户尚未放 tokenizer,本步可跳过验证,直接 commit(代码逻辑已就位);实现阶段放好后跑一次冒烟即可。

- [ ] **Step 4: Commit**

```bash
git add model/tokenizer_loader.py dataset/dataset.md
git commit -m "feat: tokenizer loader reusing minimind BPE + <mask>

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: DLMConfig + 测试骨架

**Files:**
- Create: `model/model_dlm.py`(本任务只写 `DLMConfig` + 一个占位 `DLMForMD`,后续 task 填充)
- Create: `tests/test_model.py`

**Interfaces:**
- Consumes: `transformers.PretrainedConfig`
- Produces: `DLMConfig`(kwargs 式 `__init__`,默认见 Global Constraints);字段名 `hidden_size, num_hidden_layers, num_attention_heads, num_key_value_heads, vocab_size, intermediate_size, max_position_embeddings, rms_norm_eps, rope_theta, tie_word_embeddings, dropout, mask_token_id, bos_token_id, eos_token_id, use_moe`。

- [ ] **Step 1: 写失败测试 `tests/test_model.py`**

```python
import math
from model.model_dlm import DLMConfig


def test_config_defaults_align_minimind():
    c = DLMConfig()
    assert c.hidden_size == 768
    assert c.num_hidden_layers == 8
    assert c.num_attention_heads == 8
    assert c.num_key_value_heads == 4          # GQA 2:1
    assert c.vocab_size == 6401                  # 6400 + <mask>
    assert c.intermediate_size == math.ceil(768 * math.pi / 64) * 64
    assert c.tie_word_embeddings is True
    assert c.mask_token_id == 6400
    assert c.rope_theta == 1e6
    assert c.rms_norm_eps == 1e-6
    assert c.use_moe is False                    # 字段保留但不启用


def test_config_small_smoke_profile():
    c = DLMConfig(hidden_size=512, num_hidden_layers=6)
    assert c.hidden_size == 512
    assert c.num_hidden_layers == 6
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/test_model.py -v`
Expected: FAIL with `ImportError: cannot import name 'DLMConfig'` 或 `ModuleNotFoundError`。

- [ ] **Step 3: 写最小实现 `model/model_dlm.py`**

```python
"""
🌏🌎🌍 mind-diffusion model: LLaDA v1 masked-diffusion language model 🌏🌎🌍

与 minimind 的差异:
  1. Attention 去 causal mask -> 双向(is_causal=False)
  2. 去 KV cache(每次吃整条序列)
  3. 加 <mask> token(vocab 6400 -> 6401, tied embedding)
  4. 不喂时间步 t 给模型(time-free parameterization)
  5. generate 换扩散采样(迭代 unmasking + low-confidence remasking)
"""
import math
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_utils import PreTrainedModel as _PTM


class DLMConfig(PretrainedConfig):
    """kwargs 式 config,默认对齐 minimind(768/8/8 GQA)。"""

    model_type = 'mind_diffusion'

    def __init__(self,
                 hidden_size=768,
                 num_hidden_layers=8,
                 num_attention_heads=8,
                 num_key_value_heads=4,
                 vocab_size=6401,
                 intermediate_size=None,
                 max_position_embeddings=32768,
                 rms_norm_eps=1e-6,
                 rope_theta=1e6,
                 tie_word_embeddings=True,
                 dropout=0.0,
                 mask_token_id=6400,
                 bos_token_id=1,
                 eos_token_id=2,
                 use_moe=False,
                 **kwargs):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        # π-scaled + 64 对齐,跟 minimind 一致
        self.intermediate_size = intermediate_size or math.ceil(hidden_size * math.pi / 64) * 64
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.tie_word_embeddings = tie_word_embeddings
        self.dropout = dropout
        self.mask_token_id = mask_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.use_moe = use_moe  # 保留字段但不启用(跟 minimind 一致)
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
```

> 本 task 不实现 `DLMForMD`,下一个 task 实现。测试只覆盖 config。

- [ ] **Step 4: 跑测试确认通过**

Run: `pytest tests/test_model.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add model/model_dlm.py tests/test_model.py
git commit -m "feat: DLMConfig aligned with minimind defaults

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: 双向 transformer 骨架(RMSNorm + Attention + FeedForward + Block + DLMModel)

**Files:**
- Modify: `model/model_dlm.py`(追加层类,放在 `DLMConfig` 之后)
- Modify: `tests/test_model.py`(加形状 + 双向测试)

**Interfaces:**
- Consumes: `DLMConfig`(T3)
- Produces: `DLMModel.forward(input_ids, attention_mask) -> hidden_states [B,L,H]`。后续 T5 的 `DLMForMD` 包装它。Attention 必须是**非因果**的(无 causal mask)。

- [ ] **Step 1: 追加失败测试到 `tests/test_model.py`**

```python
import torch
from model.model_dlm import DLMConfig, DLMModel


def test_model_forward_shape():
    cfg = DLMConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                    num_key_value_heads=2, vocab_size=100, max_position_embeddings=128)
    m = DLMModel(cfg)
    ids = torch.randint(0, 100, (2, 10))
    attn = torch.ones(2, 10, dtype=torch.long)
    h = m(ids, attention_mask=attn)
    assert h.shape == (2, 10, 64)


def test_attention_is_bidirectional():
    # 双向:交换两个 token 位置,对应输出位置应跟着交换(非因果)
    cfg = DLMConfig(hidden_size=64, num_hidden_layers=1, num_attention_heads=4,
                    num_key_value_heads=2, vocab_size=100, max_position_embeddings=128)
    m = DLMModel(cfg).eval()
    ids = torch.tensor([[10, 20, 30, 40]])
    attn = torch.ones(1, 4, dtype=torch.long)
    with torch.inference_mode():
        h = m(ids, attention_mask=attn)
        swapped = m(torch.tensor([[10, 30, 20, 40]]), attention_mask=attn)
    # 位置 1 和 2 交换:输出也应交换(h 是双向,位置 i 只依赖输入位置 i + 全局)
    # 注意:双向注意力下位置 i 的输出会因"别处 token 变了"而变,但位置 i 看到的是"自己的 token"
    # 这里只断言 shape + 不崩,真正的双向断言用 attention 权重(下个 task 加 forward 返回权重更直接)
    assert h.shape == (1, 4, 64)
    assert swapped.shape == (1, 4, 64)


def test_attention_no_causal_mask_attribute():
    # 关键:Attention 必须是非因果的。构造一个 2-token 序列,
    # 第 0 个 token 的 hidden 应能"看到"第 1 个 token(因果的话看不到)。
    cfg = DLMConfig(hidden_size=64, num_hidden_layers=1, num_attention_heads=4,
                    num_key_value_heads=2, vocab_size=100, max_position_embeddings=128)
    m = DLMModel(cfg).eval()
    a = torch.tensor([[10, 20]])
    b = torch.tensor([[10, 99]])  # 只改第 1 位
    attn = torch.ones(1, 2, dtype=torch.long)
    with torch.inference_mode():
        h_a = m(a, attention_mask=attn)
        h_b = m(b, attention_mask=attn)
    # 若是因果:位置 0 看不到位置 1,h_a[0,0]==h_b[0,0];双向:应不等
    assert not torch.allclose(h_a[0, 0], h_b[0, 0], atol=1e-6), \
        "位置 0 的输出应依赖位置 1 的 token —— 双向注意力"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/test_model.py -v`
Expected: FAIL with `ImportError: cannot import name 'DLMModel'`

- [ ] **Step 3: 在 `model/model_dlm.py` 追加层实现**

```python
# 🌏🌎🌍 layers: RMSNorm + 双向 Attention(RoPE+GQA+QK-norm) + SwiGLU FFN + Block 🌏🌎🌍
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, rotate_half


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # upcast to fp32 then cast back(跟 minimind 一致)
        out = x.to(torch.float32).pow(2).mean(-1, keepdim=True).rsqrt()
        return (x * out) * self.weight


def precompute_freqs_cis(dim, end, theta=1e6):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))
    t = torch.arange(end)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64 [end, dim/2]


def apply_rotary(xq, xk, freqs_cis):
    def _apply(x, f):
        x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        f = f.unsqueeze(0).unsqueeze(2)  # [1, seq, 1, dim/2]
        return torch.view_as_real(x_ * f).flatten(-2).type_as(x)
    return _apply(xq, freqs_cis), _apply(xk, freqs_cis)


def repeat_kv(x, n_rep):
    B, S, H, D = x.shape
    return x[:, :, :, None, :].expand(B, S, H, n_rep, D).reshape(B, S, H * n_rep, D)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.num_attention_heads
        self.n_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.q_proj = nn.Linear(cfg.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, cfg.hidden_size, bias=False)
        # per-head QK-norm(Qwen 风格,跟 minimind 一致)
        self.q_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, cfg.rms_norm_eps)

    def forward(self, x, attn_mask, freqs_cis):
        B, S, H = x.shape
        q = self.q_norm(self.q_proj(x).view(B, S, self.n_heads, self.head_dim))
        k = self.k_norm(self.k_proj(x).view(B, S, self.n_kv_heads, self.head_dim))
        v = self.v_proj(x).view(B, S, self.n_kv_heads, self.head_dim)
        q, k = apply_rotary(q, k, freqs_cis)
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)
        q = q.transpose(1, 2)  # [B, H, S, D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # 🌏 KEY DIFF: is_causal=False -> 双向注意力(无 causal mask)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False)
        out = out.transpose(1, 2).reshape(B, S, self.n_heads * self.head_dim)
        return self.o_proj(out)


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DLMBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = Attention(cfg)
        self.ffn = FeedForward(cfg)
        self.norm1 = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.norm2 = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, x, attn_mask, freqs_cis):
        x = x + self.attn(self.norm1(x), attn_mask, freqs_cis)
        return x + self.ffn(self.norm2(x))


class DLMModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([DLMBlock(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        head_dim = cfg.hidden_size // cfg.num_attention_heads
        # RoPE 预计算(用 cfg.max_position_embeddings,够长)
        self.register_buffer(
            'freqs_cis',
            precompute_freqs_cis(head_dim, cfg.max_position_embeddings, cfg.rope_theta),
            persistent=False,
        )

    def forward(self, input_ids, attention_mask=None):
        B, S = input_ids.shape
        h = self.embed(input_ids)
        # 把 [B,S] 的 pad mask 转成 sdpa 要的 [B,1,1,S] additive mask
        if attention_mask is not None:
            am = attention_mask[:, None, None, :].to(h.dtype)  # 1=keep, 0=pad
            am = (1.0 - am) * torch.finfo(h.dtype).min  # pad 位 = -inf
        else:
            am = None
        freqs = self.freqs_cis[:S]
        for layer in self.layers:
            h = layer(h, am, freqs)
        return self.norm(h)
```

> 注:`attn_mask` 传给 sdpa 时是 `[B,1,1,S]` additive mask(pad 位 -inf),`is_causal=False` 保证无因果 mask。`repeat_kv` 用于 GQA。

- [ ] **Step 4: 跑测试确认通过**

Run: `pytest tests/test_model.py -v`
Expected: 全部 passed(含 T3 的 2 个 + 本 task 3 个 = 5)

- [ ] **Step 5: Commit**

```bash
git add model/model_dlm.py tests/test_model.py
git commit -m "feat: bidirectional transformer backbone (RMSNorm+GQA+RoPE+SwiGLU, no causal mask)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: DLMForMD(forward 掩码 + 1/t 加权 loss)

**Files:**
- Modify: `model/model_dlm.py`(追加 `DLMForMD` 类)
- Create: `tests/test_loss.py`

**Interfaces:**
- Consumes: `DLMModel`(T4),`DLMConfig.mask_token_id`
- Produces: `DLMForMD(cfg)` 继承 `PreTrainedModel`;`forward(input_ids, attention_mask, response_mask=None, labels=None) -> (loss, )` 或返回带 loss 的对象。loss = `1/t` 加权掩码 CE。后续 T6 的 `generate` 在此类里。

- [ ] **Step 1: 写失败测试 `tests/test_loss.py`**

```python
import torch
from model.model_dlm import DLMConfig, DLMForMD


def _small():
    return DLMConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                     num_key_value_heads=2, vocab_size=100, max_position_embeddings=128,
                     mask_token_id=99)


def test_forward_returns_scalar_loss():
    m = DLMForMD(_small())
    ids = torch.randint(0, 99, (2, 10))
    attn = torch.ones(2, 10, dtype=torch.long)
    out = m(input_ids=ids, attention_mask=attn, labels=ids)
    assert out.loss.dim() == 0  # scalar
    assert torch.isfinite(out.loss)


def test_pad_positions_excluded_from_loss():
    # 第 2 条后半是 pad,loss 不应因此爆炸;且换 pad 内容不影响 loss
    m = DLMForMD(_small())
    ids = torch.randint(0, 99, (2, 10))
    attn = torch.tensor([[1]*10, [1,1,1,1,1,0,0,0,0,0]])
    out1 = m(input_ids=ids, attention_mask=attn, labels=ids)
    ids2 = ids.clone()
    ids2[1, 5:] = torch.randint(0, 99, (5,))  # 改 pad 区内容
    out2 = m(input_ids=ids2, attention_mask=attn, labels=ids2)
    assert torch.allclose(out1.loss, out2.loss, atol=1e-5), "pad 位不应影响 loss"


def test_response_mask_restricts_masking_to_response():
    # 只有 response_mask=1 的位会被掩;prompt 区永不被掩 -> prompt 位 logits 不影响 loss
    m = DLMForMD(_small())
    ids = torch.randint(0, 99, (1, 6))
    attn = torch.ones(1, 6, dtype=torch.long)
    resp = torch.tensor([[0, 0, 1, 1, 1, 1]])  # 前 2 位 prompt
    torch.manual_seed(0)
    out1 = m(input_ids=ids, attention_mask=attn, response_mask=resp, labels=ids)
    # 改 prompt 位 token,prompt 不掩 -> 不进 loss -> loss 应几乎不变(掩采样随机,固定种子)
    ids2 = ids.clone()
    ids2[0, :2] = torch.randint(0, 99, (2,))
    torch.manual_seed(0)
    out2 = m(input_ids=ids2, attention_mask=attn, response_mask=resp, labels=ids2)
    assert torch.allclose(out1.loss, out2.loss, atol=1e-5), "prompt 位不应进 loss"


def test_no_nan_when_t_near_zero():
    # t 夹 [1e-4,1],t->0 时 1/t 不应爆
    m = DLMForMD(_small())
    ids = torch.randint(0, 99, (1, 8))
    attn = torch.ones(1, 8, dtype=torch.long)
    out = m(input_ids=ids, attention_mask=attn, labels=ids)
    assert torch.isfinite(out.loss)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/test_loss.py -v`
Expected: FAIL with `ImportError: cannot import name 'DLMForMD'`

- [ ] **Step 3: 在 `model/model_dlm.py` 追加 `DLMForMD`**

```python
# 🌏🌎🌍 DLMForMD: 排码 + 1/t 加权掩码 CE loss(time-free,不喂 t 给模型) 🌏🌎🌍
from dataclasses import dataclass
from transformers.modeling_outputs import ModelOutput
import random


@dataclass
class DLMOutput(ModelOutput):
    loss: torch.Tensor = None
    logits: torch.Tensor = None


class DLMForMD(PreTrainedModel):
    config_class = DLMConfig

    def __init__(self, cfg):
        super().__init__(cfg)
        self.model = DLMModel(cfg)
        # tied embedding: lm_head 与 embed 共享(跟 minimind 一致)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.tie_weights()
        self.post_init()

    def tie_weights(self):
        self.lm_head.weight = self.model.embed.weight

    def forward(self, input_ids, attention_mask=None, response_mask=None, labels=None):
        x_0 = labels if labels is not None else input_ids      # 干净目标
        B, L = x_0.shape
        device = x_0.device
        MASK_ID = self.config.mask_token_id
        V = self.config.vocab_size

        # 1. 每序列采掩码比例 t,夹 [1e-4, 1] 防除零
        t = torch.empty(B, device=device).uniform_(1e-4, 1.0)

        # 2. 可掩范围:真实 token(非 pad);SFT 时再 & response_mask
        maskable = attention_mask.bool() if attention_mask is not None else torch.ones(B, L, dtype=torch.bool, device=device)
        if response_mask is not None:
            maskable = maskable & response_mask.bool()

        # 3. 伯努利(t) 掩码:每个可掩位独立以概率 t[i] 被掩
        rand = torch.rand(B, L, device=device)
        mask = maskable & (rand < t[:, None])
        x_t = x_0.clone()
        x_t[mask] = MASK_ID

        # 4. 双向 transformer(不喂 t —— time-free)
        h = self.model(x_t, attention_mask=attention_mask)
        logits = self.lm_head(h)                              # [B, L, V]

        # 5. 1/t 加权的掩码 CE:只在被掩位算
        ce = F.cross_entropy(logits.view(-1, V), x_0.view(-1), reduction='none').view(B, L)
        ce = ce * mask
        n_masked = mask.sum(dim=1).clamp(min=1)               # [B]
        # 每序列: (1/t) * (sum_ce / n_masked);再 batch mean
        per_seq = (ce.sum(dim=1) / n_masked) * (1.0 / t)       # [B]
        loss = per_seq.mean()
        return DLMOutput(loss=loss, logits=logits)
```

> 关键:`mask` 是 `rand < t[:,None]` 的伯努利,`t` 每序列一个;loss 用 `1.0/t` 加权,只算 `mask` 位;pad 既不掩(`maskable` 屏掉)也不计 loss;EOS 是普通 token,在 `maskable` 里会被掩/算 loss。

- [ ] **Step 4: 跑测试确认通过**

Run: `pytest tests/test_loss.py -v`
Expected: 4 passed

- [ ] **Step 5: 跑全部测试**

Run: `pytest tests/ -v`
Expected: 全 passed

- [ ] **Step 6: Commit**

```bash
git add model/model_dlm.py tests/test_loss.py
git commit -m "feat: DLMForMD with 1/t-weighted masked CE loss

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: 扩散采样 `generate`

**Files:**
- Modify: `model/model_dlm.py`(在 `DLMForMD` 里加 `generate` 方法)
- Create: `tests/test_sampling.py`

**Interfaces:**
- Consumes: `DLMForMD.model`(双向 transformer),`config.mask_token_id`,`config.eos_token_id`
- Produces: `DLMForMD.generate(prompt_ids: LongTensor[1,P], gen_length=128, steps=64, temperature=0.0, low_confidence=True) -> LongTensor[1, gen_length]`(已 decode 截断后的 response token;截断由调用方做,这里返回原始 gen_length 个 token,decode 截断在 eval/demo 里)。实际返回 `[1, gen_length]`,遇 EOS 位保留 EOS(截断逻辑在 caller)。

- [ ] **Step 1: 写失败测试 `tests/test_sampling.py`**

```python
import torch
from model.model_dlm import DLMConfig, DLMForMD


def _small():
    return DLMConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                     num_key_value_heads=2, vocab_size=100, max_position_embeddings=128,
                     mask_token_id=99, eos_token_id=2)


def test_generate_returns_gen_length():
    m = DLMForMD(_small()).eval()
    prompt = torch.randint(0, 99, (1, 4))
    out = m.generate(prompt, gen_length=8, steps=4, temperature=0.0)
    assert out.shape == (1, 8)


def test_generate_no_mask_tokens_left():
    # 跑完后 response 区不应有 <mask> 残留
    m = DLMForMD(_small()).eval()
    prompt = torch.randint(0, 99, (1, 4))
    out = m.generate(prompt, gen_length=8, steps=4, temperature=0.0)
    assert (out != 99).all(), "所有 <mask> 应被揭开"


def test_generate_random_remasking_branch():
    m = DLMForMD(_small()).eval()
    prompt = torch.randint(0, 99, (1, 4))
    out = m.generate(prompt, gen_length=8, steps=4, low_confidence=False)
    assert out.shape == (1, 8)
    assert (out != 99).all()


def test_generate_temperature_positive_runs():
    m = DLMForMD(_small()).eval()
    prompt = torch.randint(0, 99, (1, 4))
    out = m.generate(prompt, gen_length=8, steps=4, temperature=1.0)
    assert out.shape == (1, 8)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `pytest tests/test_sampling.py -v`
Expected: FAIL with `AttributeError: 'DLMForMD' object has no attribute 'generate'`

- [ ] **Step 3: 在 `DLMForMD` 类里追加 `generate`**

```python
    # 🌏🌎🌍 扩散采样: 全 <mask> -> 迭代 unmasking + low-confidence remasking 🌏🌎🌍
    @torch.inference_mode()
    def generate(self, prompt_ids, gen_length=128, steps=64, temperature=0.0,
                 low_confidence=True):
        """
        prompt_ids: [1, P] 干净 prompt
        返回: [1, gen_length] response token(<mask> 全部揭开)
        """
        device = prompt_ids.device
        MASK_ID = self.config.mask_token_id
        P = prompt_ids.shape[1]
        # 1. 构造 prompt + 全 <mask> 的 response
        resp = torch.full((1, gen_length), MASK_ID, dtype=torch.long, device=device)
        x = torch.cat([prompt_ids, resp], dim=1)              # [1, P+L]
        attn = torch.ones_like(x)
        # prompt 位永不重掩
        is_prompt = torch.zeros_like(x, dtype=torch.bool)
        is_prompt[:, :P] = True

        T = steps
        for k in range(1, T + 1):
            s = 1.0 - k / T                                     # 目标"还剩多少比例被掩"
            # 跑双向 transformer(整条)
            h = self.model(x, attention_mask=attn)
            logits = self.lm_head(h)                            # [1, P+L, V]
            # 当前被掩的 response 位
            masked = (x == MASK_ID) & (~is_prompt)             # [1, P+L]
            idx = masked.nonzero(as_tuple=False)               # [N, 2]
            if idx.shape[0] == 0:
                break  # 全揭开了
            # 这些位的 logits
            lm_logits = logits[idx[:, 0], idx[:, 1]]           # [N, V]
            temp = max(temperature, 1e-4)
            prob = F.softmax(lm_logits / temp, dim=-1)
            pred = prob.argmax(dim=-1)                          # [N]
            if low_confidence:
                conf = prob.gather(1, pred[:, None]).squeeze(1)  # [N]
            else:
                conf = torch.rand(idx.shape[0], device=device)
            # 临时写回预测
            x[idx[:, 0], idx[:, 1]] = pred
            # 决定这轮固化多少个:期望到时间 s 时还剩 floor(gen_length*s) 个被掩
            n_remain = int(gen_length * s)                      # 应剩多少个 <mask>
            # 当前已揭开的 response 位(含刚写回的),按 conf 决定固化哪些
            # 低 conf 的 -> 重新掩回 <mask>(下轮重猜)
            if n_remain > 0 and n_remain < idx.shape[0]:
                # 把 conf 升序,n_remain 个最低置信的 -> 重掩
                order = torch.argsort(conf)                     # 升序
                remask_pos = idx[order[:n_remain]]             # [n_remain, 2]
                x[remask_pos[:, 0], remask_pos[:, 1]] = MASK_ID
            # 其余(idx 中非 remask 的)已固化(不动)
        # 4. 全部揭开,返回 response 区
        return x[:, P:]
```

> 关键逻辑:每步 argmax 预测所有 `<mask>` 位 → 算置信 → 临时写回 → 按置信升序把最低 `n_remain=floor(L·s)` 个重新掩回。`s` 从 1 降到 0,被掩数线性减到 0。`is_prompt` 保证 prompt 永不被重掩。temperature=0 时 `prob` 接近 onehot(argmax),>0 加softmax。

- [ ] **Step 4: 跑测试确认通过**

Run: `pytest tests/test_sampling.py -v`
Expected: 4 passed

- [ ] **Step 5: 跑全部测试**

Run: `pytest tests/ -v`
Expected: 全 passed(含 T3/T4/T5/T6)

- [ ] **Step 6: Commit**

```bash
git add model/model_dlm.py tests/test_sampling.py
git commit -m "feat: diffusion sampling (iterative unmasking + low-confidence remasking)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 7: 数据集(PretrainDataset + SFTDataset)

**Files:**
- Create: `dataset/lm_dataset.py`

**Interfaces:**
- Consumes: `load_tokenizer`(T2),`DLMConfig.mask_token_id`,`DLMConfig.bos/eos_token_id`
- Produces: `PretrainDataset(data_path, tokenizer, max_length=340) -> (input_ids[Tensor], attention_mask[Tensor])`;`SFTDataset(data_path, tokenizer, max_length=512) -> (input_ids, attention_mask, response_mask[Tensor])`。`response_mask` 只在 assistant 的 response+`<|im_end|>` 位为 1。

- [ ] **Step 1: 写 `dataset/lm_dataset.py`**

```python
"""
数据集类:返回干净序列 + 掩码范围(掩码本身在 model.forward 里现采,每步随机)
"""
import torch
from torch.utils.data import Dataset
from datasets import load_dataset


class PretrainDataset(Dataset):
    """预训练:返回 (input_ids, attention_mask)。labels=自己(forward 里当作 x_0)。"""

    def __init__(self, data_path, tokenizer, max_length=340):
        self.data = load_dataset('json', data_files=data_path, split='train')
        self.tok = tokenizer
        self.max_length = max_length
        self.bos = tokenizer.bos_token_id or 1
        self.eos = tokenizer.eos_token_id or 2

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        text = self.data[i].get('text', '') or self.data[i].get('content', '')
        ids = self.tok(text, add_special_tokens=False)['input_ids'][: self.max_length - 2]
        ids = [self.bos] + ids + [self.eos]
        attn = [1] * len(ids)
        # 右 pad 到 max_length
        pad = self.max_length - len(ids)
        if pad > 0:
            pad_id = self.tok.pad_token_id or 0
            ids = ids + [pad_id] * pad
            attn = attn + [0] * pad
        return torch.tensor(ids, dtype=torch.long), torch.tensor(attn, dtype=torch.long)


class SFTDataset(Dataset):
    """SFT:返回 (input_ids, attention_mask, response_mask)。response_mask 只标 assistant 回答位。"""

    # minimind chat template: <|im_start|>role\ncontent<|im_end|>
    IM_START = '<|im_start|>'
    IM_END = '<|im_end|>'

    def __init__(self, data_path, tokenizer, max_length=512):
        self.data = load_dataset('json', data_files=data_path, split='train')
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        row = self.data[i]
        messages = row.get('conversations') or row.get('messages') or []
        ids, resp_mask = [], []
        for msg in messages:
            role = msg.get('from', msg.get('role', 'user'))
            content = msg.get('value', msg.get('content', ''))
            chunk = f'{self.IM_START}{role}\n{content}{self.IM_END}'
            chunk_ids = self.tok(chunk, add_special_tokens=False)['input_ids']
            is_resp = role == 'assistant'
            ids += chunk_ids
            resp_mask += [1 if is_resp else 0] * len(chunk_ids)
        ids = ids[: self.max_length]
        resp_mask = resp_mask[: self.max_length]
        attn = [1] * len(ids)
        pad = self.max_length - len(ids)
        if pad > 0:
            pad_id = self.tok.pad_token_id or 0
            ids = ids + [pad_id] * pad
            attn = attn + [0] * pad
            resp_mask = resp_mask + [0] * pad
        return (torch.tensor(ids, dtype=torch.long),
                torch.tensor(attn, dtype=torch.long),
                torch.tensor(resp_mask, dtype=torch.long))
```

> 注:minimind SFT 用 conversations 格式(from/value)或 messages(role/content)。这里两种都兼容。`response_mask` 标记 assistant 段(含 `<|im_end|>`),forward 里只在 response_mask=1 的位随机掩。

- [ ] **Step 2: 写一个不依赖真实数据的形状冒烟测试,加到 `tests/test_loss.py` 或新建 `tests/test_dataset.py`**

```python
# tests/test_dataset.py
import torch
from unittest.mock import MagicMock
from dataset.lm_dataset import PretrainDataset, SFTDataset


def _fake_tok():
    tok = MagicMock()
    tok.bos_token_id = 1
    tok.eos_token_id = 2
    tok.pad_token_id = 0
    tok.return_value = {'input_ids': [10, 20, 30]}
    tok.side_effect = lambda text, **kw: {'input_ids': [10, 20, 30]}
    return tok


def test_pretrain_dataset_shape(monkeypatch, tmp_path):
    # 造一个临时 jsonl
    p = tmp_path / 't.jsonl'
    p.write_text('{"text": "hello"}\n{"text": "world"}\n')
    # mock load_dataset 返回简单列表
    import dataset.lm_dataset as mod
    monkeypatch.setattr(mod, 'load_dataset', lambda *a, **k: [{'text': 'hello'}, {'text': 'world'}])
    ds = PretrainDataset(str(p), _fake_tok(), max_length=16)
    ids, attn = ds[0]
    assert ids.shape == (16,)
    assert attn.shape == (16,)
    assert ids[0] == 1  # bos
    assert ids[1] == 2  # eos(mock tokenizer 只吐3个固定 token,但 bos+content+eos 至少 bos 对)


def test_sft_dataset_response_mask(monkeypatch, tmp_path):
    p = tmp_path / 's.jsonl'
    p.write_text('{}\n')
    import dataset.lm_dataset as mod
    monkeypatch.setattr(mod, 'load_dataset', lambda *a, **k: [{'conversations': [
        {'from': 'user', 'value': 'hi'},
        {'from': 'assistant', 'value': 'yo'},
    ]}])
    ds = SFTDataset(str(p), _fake_tok(), max_length=32)
    ids, attn, resp = ds[0]
    assert ids.shape == (32,) and attn.shape == (32,) and resp.shape == (32,)
    assert resp.sum() > 0, "assistant 位应被标"
```

- [ ] **Step 3: 跑测试**

Run: `pytest tests/test_dataset.py -v`
Expected: 2 passed

- [ ] **Step 4: Commit**

```bash
git add dataset/lm_dataset.py tests/test_dataset.py
git commit -m "feat: PretrainDataset + SFTDataset (return clean seq + mask ranges)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 8: trainer_utils.py

**Files:**
- Create: `trainer/trainer_utils.py`

**Interfaces:**
- Consumes: `DLMConfig`, `DLMForMD`(T3-T6),`load_tokenizer`(T2)
- Produces: `get_lr(current_step, total_steps, lr)`, `SkipBatchSampler`, `init_model(lm_config, from_weight, tokenizer_path, save_dir, device)`, `Logger`, `setup_seed(seed)`, `lm_checkpoint(model, path)`, `is_main_process()`, `init_distributed_mode()`。

- [ ] **Step 1: 写 `trainer/trainer_utils.py`**

```python
"""
训练工具:一行余弦 get_lr / SkipBatchSampler / init_model / Logger / setup_seed
"""
import math
import os
import time
import torch
import torch.distributed as dist
from torch.utils.data import Sampler
from model.model_dlm import DLMConfig, DLMForMD
from model.tokenizer_loader import load_tokenizer


def get_lr(current_step, total_steps, lr):
    """一行余弦,10% floor + 0.9x peak,无 warmup(跟 minimind 一致)"""
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))


def setup_seed(seed=1029):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


class Logger:
    def __init__(self, log_file=None):
        self.log_file = log_file

    def __call__(self, msg):
        t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        line = f'[{t}] {msg}'
        if is_main_process():
            print(line, flush=True)
        if self.log_file:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(line + '\n')


class SkipBatchSampler(Sampler):
    """跳过前 N 个 batch,用于 resume"""

    def __init__(self, dataset, batch_size, skip=0, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.skip = skip
        self.shuffle = shuffle

    def __iter__(self):
        n = len(self.dataset)
        idx = torch.randperm(n).tolist() if self.shuffle else list(range(n))
        for i in range(self.skip * self.batch_size, n, self.batch_size):
            yield idx[i: i + self.batch_size]

    def __len__(self):
        return (len(self.dataset) - self.skip * self.batch_size + self.batch_size - 1) // self.batch_size


def init_model(lm_config, from_weight='pretrain', tokenizer_path='model',
               save_dir='out', device='cuda'):
    """加载 tokenizer + 构建 DLMForMD + 加载 checkpoint"""
    tokenizer = load_tokenizer(tokenizer_path)
    model = DLMForMD(lm_config).to(device)
    weight_path = os.path.join(save_dir, f'{from_weight}_{lm_config.hidden_size}.pth')
    if os.path.exists(weight_path):
        state = torch.load(weight_path, map_location=device)
        model.load_state_dict(state, strict=False)
        print(f'loaded {weight_path}')
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'trainable params: {n_params / 1e6:.2f}M')
    return model, tokenizer


def lm_checkpoint(model, path):
    """checkpoint: 权重 halved + 移 CPU(跟 minimind 一致)"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {k: v.half().cpu() for k, v in model.state_dict().items()}
    torch.save(state, path)


def init_distributed_mode():
    """简易 DDP 初始化(minimind 风格)"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world = int(os.environ['WORLD_SIZE'])
        dist.init_process_group('nccl', rank=rank, world_size=world)
        torch.cuda.set_device(rank)
    return dist.is_initialized()
```

- [ ] **Step 2: 写测试 `tests/test_trainer_utils.py`**

```python
from trainer.trainer_utils import get_lr


def test_get_lr_cosine():
    # step=0: peak * 0.1 + 0.45 * 2 = 0.1 + 0.9 = 1.0 -> lr
    assert abs(get_lr(0, 100, 1e-3) - 1e-3) < 1e-9
    # step=total/2: cos(pi/2)=0 -> 0.1 + 0.45 = 0.55 * lr
    assert abs(get_lr(50, 100, 1e-3) - 0.55e-3) < 1e-9
    # step=total: cos(pi)=-1 -> 0.1 + 0 = 0.1 * lr
    assert abs(get_lr(100, 100, 1e-3) - 0.1e-3) < 1e-9


def test_get_lr_floor():
    # 不会低于 0.1 * lr
    assert get_lr(999, 1000, 1e-3) >= 0.1e-3
```

- [ ] **Step 3: 跑测试**

Run: `pytest tests/test_trainer_utils.py -v`
Expected: 2 passed

- [ ] **Step 4: Commit**

```bash
git add trainer/trainer_utils.py tests/test_trainer_utils.py
git commit -m "feat: trainer_utils (one-line cosine get_lr, SkipBatchSampler, init_model, Logger)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 9: train_pretrain.py

**Files:**
- Create: `trainer/train_pretrain.py`

**Interfaces:**
- Consumes: `DLMConfig`, `DLMForMD`, `PretrainDataset`(T7), `trainer_utils`(T8)
- Produces: 可执行训练脚本 `python trainer/train_pretrain.py`,产出 `out/pretrain_768.pth`。

- [ ] **Step 1: 写 `trainer/train_pretrain.py`**

```python
"""
预训练: 全序列随机掩码 + 1/t 加权 loss(在 DLMForMD.forward 里)
"""
import os
import argparse
import torch
from torch.utils.data import DataLoader
from model.model_dlm import DLMConfig, DLMForMD
from dataset.lm_dataset import PretrainDataset
from model.tokenizer_loader import load_tokenizer
from trainer.trainer_utils import (get_lr, SkipBatchSampler, init_model, lm_checkpoint,
                                   Logger, setup_seed, init_distributed_mode, is_main_process)


def main():
    # ========== 1. 初始化环境和随机种子 ==========
    parser = argparse.ArgumentParser(description='mind-diffusion 预训练')
    parser.add_argument('--epochs', type=int, default=2, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='学习率')
    parser.add_argument('--accumulation_steps', type=int, default=8, help='梯度累积')
    parser.add_argument('--max_seq_len', type=int, default=340, help='最大序列长度')
    parser.add_argument('--hidden_size', type=int, default=768, help='隐藏维度')
    parser.add_argument('--num_hidden_layers', type=int, default=8, help='层数')
    parser.add_argument('--data_path', type=str, default='dataset/pretrain_t2t_mini.jsonl', help='预训练数据')
    parser.add_argument('--tokenizer_path', type=str, default='model', help='tokenizer 目录')
    parser.add_argument('--save_dir', type=str, default='out', help='输出目录')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu', help='设备')
    args = parser.parse_args()

    setup_seed()
    init_distributed_mode()
    logger = Logger(os.path.join(args.save_dir, 'pretrain.log'))

    # ========== 2. 加载模型与 tokenizer ==========
    cfg = DLMConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers)
    model, tokenizer = init_model(cfg, from_weight=None,
                                 tokenizer_path=args.tokenizer_path,
                                 save_dir=args.save_dir, device=args.device)
    model = model.to(args.device)

    # ========== 3. 构造数据集与加载器 ==========
    ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    sampler = SkipBatchSampler(ds, args.batch_size)
    loader = DataLoader(ds, batch_size=1, sampler=sampler, num_workers=4, pin_memory=True)

    # ========== 4. 优化器 ==========
    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scaler = torch.cuda.amp.GradScaler(enabled=(torch.cuda.is_available()))

    total_steps = args.epochs * len(loader) // args.accumulation_steps
    step = 0
    os.makedirs(args.save_dir, exist_ok=True)

    # ========== 5. 训练循环 ==========
    for epoch in range(args.epochs):
        model.train()
        for i, (ids, attn) in enumerate(loader):
            ids, attn = ids.to(args.device), attn.to(args.device)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available(), dtype=torch.bfloat16):
                out = model(input_ids=ids, attention_mask=attn, labels=ids)
                loss = out.loss / args.accumulation_steps
            scaler.scale(loss).backward()
            if (i + 1) % args.accumulation_steps == 0:
                lr = get_lr(step, total_steps, args.learning_rate)
                for g in optim.param_groups:
                    g['lr'] = lr
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
                step += 1
                if step % 10 == 0 and is_main_process():
                    logger(f'epoch {epoch} step {step}/{total_steps} loss {loss.item()*args.accumulation_steps:.4f} lr {lr:.2e}')
        if is_main_process():
            lm_checkpoint(model, os.path.join(args.save_dir, f'pretrain_{args.hidden_size}.pth'))
            logger(f'saved pretrain_{args.hidden_size}.pth')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: 干跑语法检查**

Run: `python -c "import ast; ast.parse(open('trainer/train_pretrain.py').read()); print('ok')"`
Expected: `ok`

- [ ] **Step 3: (用户放置数据后)冒烟跑几步**

Run: `python trainer/train_pretrain.py --epochs 1 --batch_size 2 --max_seq_len 64 --hidden_size 64 --num_hidden_layers 2 --data_path dataset/pretrain_t2t_mini.jsonl`
Expected: 打印 `loss ...` 且不崩;Ctrl-C 可停。

> 若无数据,本步跳过,代码就绪即可。

- [ ] **Step 4: Commit**

```bash
git add trainer/train_pretrain.py
git commit -m "feat: train_pretrain (random masking + 1/t loss, minimind-style skeleton)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 10: train_sft.py

**Files:**
- Create: `trainer/train_sft.py`

**Interfaces:**
- Consumes: `DLMConfig`, `DLMForMD`, `SFTDataset`(T7), `trainer_utils`(T8),`out/pretrain_*.pth`(T9 产出)
- Produces: `python trainer/train_sft.py`,产出 `out/sft_768.pth`。

- [ ] **Step 1: 写 `trainer/train_sft.py`**

```python
"""
SFT: [prompt;response], 只掩 response(response_mask),1/t loss
"""
import os
import argparse
import torch
from torch.utils.data import DataLoader
from model.model_dlm import DLMConfig, DLMForMD
from dataset.lm_dataset import SFTDataset
from model.tokenizer_loader import load_tokenizer
from trainer.trainer_utils import (get_lr, SkipBatchSampler, init_model, lm_checkpoint,
                                   Logger, setup_seed, init_distributed_mode, is_main_process)


def main():
    # ========== 1. 初始化环境和随机种子 ==========
    parser = argparse.ArgumentParser(description='mind-diffusion SFT')
    parser.add_argument('--epochs', type=int, default=3, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=2.5e-5, help='学习率')
    parser.add_argument('--accumulation_steps', type=int, default=4, help='梯度累积')
    parser.add_argument('--max_seq_len', type=int, default=512, help='最大序列长度')
    parser.add_argument('--hidden_size', type=int, default=768, help='隐藏维度')
    parser.add_argument('--num_hidden_layers', type=int, default=8, help='层数')
    parser.add_argument('--data_path', type=str, default='dataset/sft_t2t_mini.jsonl', help='SFT 数据')
    parser.add_argument('--tokenizer_path', type=str, default='model', help='tokenizer 目录')
    parser.add_argument('--save_dir', type=str, default='out', help='输出目录')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu', help='设备')
    args = parser.parse_args()

    setup_seed()
    init_distributed_mode()
    logger = Logger(os.path.join(args.save_dir, 'sft.log'))

    # ========== 2. 加载模型(从 pretrain 续) ==========
    cfg = DLMConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers)
    model, tokenizer = init_model(cfg, from_weight='pretrain',
                                 tokenizer_path=args.tokenizer_path,
                                 save_dir=args.save_dir, device=args.device)
    model = model.to(args.device)

    # ========== 3. 构造数据集 ==========
    ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    sampler = SkipBatchSampler(ds, args.batch_size)
    loader = DataLoader(ds, batch_size=1, sampler=sampler, num_workers=4, pin_memory=True)

    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    total_steps = args.epochs * len(loader) // args.accumulation_steps
    step = 0
    os.makedirs(args.save_dir, exist_ok=True)

    # ========== 4. 训练循环 ==========
    for epoch in range(args.epochs):
        model.train()
        for i, (ids, attn, resp) in enumerate(loader):
            ids, attn, resp = ids.to(args.device), attn.to(args.device), resp.to(args.device)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available(), dtype=torch.bfloat16):
                out = model(input_ids=ids, attention_mask=attn,
                           response_mask=resp, labels=ids)
                loss = out.loss / args.accumulation_steps
            scaler.scale(loss).backward()
            if (i + 1) % args.accumulation_steps == 0:
                lr = get_lr(step, total_steps, args.learning_rate)
                for g in optim.param_groups:
                    g['lr'] = lr
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
                step += 1
                if step % 10 == 0 and is_main_process():
                    logger(f'epoch {epoch} step {step}/{total_steps} loss {loss.item()*args.accumulation_steps:.4f} lr {lr:.2e}')
        if is_main_process():
            lm_checkpoint(model, os.path.join(args.save_dir, f'sft_{args.hidden_size}.pth'))
            logger(f'saved sft_{args.hidden_size}.pth')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: 语法检查**

Run: `python -c "import ast; ast.parse(open('trainer/train_sft.py').read()); print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add trainer/train_sft.py
git commit -m "feat: train_sft (mask response only, conditional generation)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 11: eval_dlm.py

**Files:**
- Create: `eval_dlm.py`

**Interfaces:**
- Consumes: `DLMConfig`, `DLMForMD`, `load_tokenizer`, `out/sft_*.pth`(T10)
- Produces: `python eval_dlm.py` 打印一组中文 prompt 的生成结果 + tokens/s。

- [ ] **Step 1: 写 `eval_dlm.py`**

```python
"""
mind-diffusion 推理与对话(对照 minimind eval_llm.py)
"""
import os
import time
import argparse
import torch
from model.model_dlm import DLMConfig, DLMForMD
from model.tokenizer_loader import load_tokenizer
from trainer.trainer_utils import init_model

# 中文测试 prompt(对照 minimind)
PROMPTS = [
    '为什么天空是蓝色的?',
    '介绍一下你自己',
    '如何学习编程?',
]


def main():
    parser = argparse.ArgumentParser(description='mind-diffusion 推理与对话')
    parser.add_argument('--hidden_size', type=int, default=768, help='隐藏维度')
    parser.add_argument('--num_hidden_layers', type=int, default=8, help='层数')
    parser.add_argument('--from_weight', type=str, default='sft', help='加载哪个权重(pretrain/sft)')
    parser.add_argument('--steps', type=int, default=64, help='扩散采样步数')
    parser.add_argument('--gen_length', type=int, default=128, help='生成长度')
    parser.add_argument('--temperature', type=float, default=0.0, help='采样温度')
    parser.add_argument('--tokenizer_path', type=str, default='model', help='tokenizer 目录')
    parser.add_argument('--save_dir', type=str, default='out', help='权重目录')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu', help='设备')
    args = parser.parse_args()

    cfg = DLMConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers)
    model, tokenizer = init_model(cfg, from_weight=args.from_weight,
                                 tokenizer_path=args.tokenizer_path,
                                 save_dir=args.save_dir, device=args.device)
    model.eval()

    for p in PROMPTS:
        # chat template:<|im_start|>user\n{p}<|im_end|><|im_start|>assistant\n
        prompt_str = f'<|im_start|>user\n{p}<|im_end|><|im_start|>assistant\n'
        prompt_ids = tokenizer(prompt_str, return_tensors='pt')['input_ids'].to(args.device)
        t0 = time.time()
        out = model.generate(prompt_ids, gen_length=args.gen_length, steps=args.steps,
                           temperature=args.temperature)
        dt = time.time() - t0
        # 遇首个 eos 截断
        eos = tokenizer.eos_token_id
        ids = out[0].tolist()
        if eos in ids:
            ids = ids[:ids.index(eos)]
        text = tokenizer.decode(ids, skip_special_tokens=False)
        n_tok = len(ids)
        print(f'\n[Prompt] {p}')
        print(f'[Response] {text}')
        print(f'[Speed]: {n_tok / dt:.2f} tokens/s ({n_tok} tokens in {dt:.2f}s, {args.steps} diffusion steps)')


if __name__ == '__main__':
    main()
```

> 注:tokens/s = gen_length/总秒数(扩散定义,跟 AR 不同,README 注明)。EOS 截断在 caller 这里做。

- [ ] **Step 2: 语法检查**

Run: `python -c "import ast; ast.parse(open('eval_dlm.py').read()); print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add eval_dlm.py
git commit -m "feat: eval_dlm (chinese prompts, tokens/s, eos truncation)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 12: scripts/web_demo.py(Streamlit)

**Files:**
- Create: `scripts/web_demo.py`

**Interfaces:**
- Consumes: `DLMConfig`, `DLMForMD`, `load_tokenizer`, `out/sft_*.pth`
- Produces:`streamlit run scripts/web_demo.py` 起一个流式扩散采样界面。

- [ ] **Step 1: 写 `scripts/web_demo.py`**

```python
"""
Streamlit web demo: 流式展示扩散采样(每步刷新当前揭开的 token)
"""
import time
import torch
import streamlit as st
from model.model_dlm import DLMConfig, DLMForMD
from model.tokenizer_loader import load_tokenizer
from trainer.trainer_utils import init_model


@st.cache_resource
def load_model(hidden_size=768, num_hidden_layers=8, from_weight='sft', device='cuda:0'):
    cfg = DLMConfig(hidden_size=hidden_size, num_hidden_layers=num_hidden_layers)
    device = device if torch.cuda.is_available() else 'cpu'
    model, tokenizer = init_model(cfg, from_weight=from_weight, tokenizer_path='model',
                                 save_dir='out', device=device)
    model.eval()
    return model, tokenizer, cfg, device


def generate_stream(model, tokenizer, cfg, device, prompt, gen_length=128, steps=64, temperature=0.0):
    """流式:每步 yield 当前 response 的解码文本"""
    prompt_str = f'<|im_start|>user\n{prompt}<|im_end|><|im_start|>assistant\n'
    prompt_ids = tokenizer(prompt_str, return_tensors='pt')['input_ids'].to(device)
    MASK_ID = cfg.mask_token_id
    P = prompt_ids.shape[1]
    resp = torch.full((1, gen_length), MASK_ID, dtype=torch.long, device=device)
    x = torch.cat([prompt_ids, resp], dim=1)
    attn = torch.ones_like(x)
    is_prompt = torch.zeros_like(x, dtype=torch.bool)
    is_prompt[:, :P] = True
    for k in range(1, steps + 1):
        s = 1.0 - k / steps
        h = model.model(x, attention_mask=attn)
        logits = model.lm_head(h)
        masked = (x == MASK_ID) & (~is_prompt)
        idx = masked.nonzero(as_tuple=False)
        if idx.shape[0] == 0:
            break
        lm_logits = logits[idx[:, 0], idx[:, 1]]
        prob = torch.softmax(lm_logits / max(temperature, 1e-4), dim=-1)
        pred = prob.argmax(dim=-1)
        conf = prob.gather(1, pred[:, None]).squeeze(1)
        x[idx[:, 0], idx[:, 1]] = pred
        n_remain = int(gen_length * s)
        if 0 < n_remain < idx.shape[0]:
            order = torch.argsort(conf)
            remask = idx[order[:n_remain]]
            x[remask[:, 0], remask[:, 1]] = MASK_ID
        # 每步 yield 当前解码(含未揭开的 <mask> 显示为 ▍)
        ids = x[0, P:].tolist()
        eos = tokenizer.eos_token_id
        if eos in ids:
            ids = ids[:ids.index(eos)]
        text = tokenizer.decode(ids, skip_special_tokens=False)
        text = text.replace('<mask>', '▍').replace(tokenizer.eos_token or '<|im_end|>', '')
        yield text, k


def main():
    st.title('mind-diffusion')
    st.caption('LLaDA v1 掩码扩散语言模型 · 从全 <mask> 迭代揭开')
    model, tokenizer, cfg, device = load_model()
    prompt = st.text_input('Prompt', value='为什么天空是蓝色的?')
    gen_length = st.slider('生成长度', 32, 256, 128)
    steps = st.slider('扩散步数', 8, 256, 64)
    temperature = st.slider('温度', 0.0, 1.5, 0.0)
    if st.button('生成'):
        out_box = st.empty()
        for text, k in generate_stream(model, tokenizer, cfg, device, prompt,
                                        gen_length, steps, temperature):
            out_box.markdown(f'`step {k}/{steps}`\n\n{text}')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: 语法检查**

Run: `python -c "import ast; ast.parse(open('scripts/web_demo.py').read()); print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add scripts/web_demo.py
git commit -m "feat: streamlit web demo (streaming diffusion sampling)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 13: README + LICENSE

**Files:**
- Create: `README.md`
- Create: `README_en.md`
- Create: `LICENSE`(Apache-2.0 全文)

**Interfaces:**
- Consumes: 所有前置 task 的产物(描述用)
- Produces: 项目文档。

- [ ] **Step 1: 写 `README.md`(中文,讲清原理)**

> 大纲(用 markdown 写):
> 1. 标题 `# mind-diffusion` + tagline `大道至简 —— diffusion 版 minimind`
> 2. **它是什么**:一句话 + 跟 minimind 的关系(风格 parity,内核换成扩散)
> 3. **核心原理**(图文):
>    - 掩码扩散 vs 自回归(一张对照图/文字流程)
>    - 前向加噪:`t~U(0,1)` 随机掩码比例
>    - 训练 loss:`1/t` 加权掩码 CE(为什么是似然上界,区别 BERT)
>    - 架构:LLaMA 去 causal mask(双向)+ time-free
>    - 采样:半自回归迭代 unmasking + low-confidence remasking(配"草稿→精修"图)
> 4. **快速开始**:装依赖、放 tokenizer+数据、`train_pretrain`、`train_sft`、`eval_dlm`、`web_demo`
> 5. **与 minimind 的差异表**(附录速查)
> 6. **不确定性/已知问题**(诚实记:小模型质量、T 折中、采样简化)
> 7. **致谢** minimind + LLaDA

- [ ] **Step 2: 写 `README_en.md`(英文翻译)**

同结构翻译。

- [ ] **Step 3: 写 `LICENSE`(Apache-2.0 全文)**

从 https://www.apache.org/licenses/LICENSE-2.0.txt 拷贝全文。

- [ ] **Step 4: Commit**

```bash
git add README.md README_en.md LICENSE
git commit -m "docs: README (zh/en) + Apache-2.0 license

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 14: 端到端冒烟 + 全测试通过(验收)

**Files:**
- 无新文件,跑全链路验证

**Interfaces:**
- Consumes: T1-T13 全部产物

- [ ] **Step 1: 全测试**

Run: `pytest tests/ -v`
Expected: 全 passed(T3/4/5/6/7/8 的测试)

- [ ] **Step 2: 模型实例化(默认档 + 小档)**

Run: `python -c "from model.model_dlm import DLMConfig, DLMForMD; import torch; m=DLMForMD(DLMConfig()); print(sum(p.numel() for p in m.parameters())/1e6, 'M params')"`
Expected: 打印参数量(默认档 ~几十 M),不崩。

- [ ] **Step 3: 小模型端到端(随机初始化,CPU/GPU 都行)**

Run:
```
python -c "
import torch
from model.model_dlm import DLMConfig, DLMForMD
cfg = DLMConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, vocab_size=100, max_position_embeddings=128, mask_token_id=99, eos_token_id=2)
m = DLMForMD(cfg).eval()
# 训练一步
ids = torch.randint(0, 99, (2, 16))
attn = torch.ones(2, 16, dtype=torch.long)
out = m(input_ids=ids, attention_mask=attn, labels=ids)
out.loss.backward()
print('train step ok, loss', out.loss.item())
# 采样
prompt = torch.randint(0, 99, (1, 4))
g = m.generate(prompt, gen_length=16, steps=8)
print('generate ok, shape', g.shape, 'no mask', (g != 99).all().item())
"
```
Expected: `train step ok` + `generate ok` + `no mask True`。

- [ ] **Step 4: (用户放置数据+tokenizer 后)真训练冒烟**

> 这一步依赖用户把 minimind 的 tokenizer.json + tokenizer_config.json + mini 语料放好。
> 放好后跑:
> `python trainer/train_pretrain.py --epochs 1 --batch_size 4 --max_seq_len 128 --hidden_size 512 --num_hidden_layers 6`
> `python trainer/train_sft.py --epochs 1 --batch_size 4 --hidden_size 512 --num_hidden_layers 6`
> `python eval_dlm.py --hidden_size 512 --num_hidden_layers 6 --from_weight sft`
> Expected: loss 下降 + eval 出中文文本(质量不保证,可读是加分)。

- [ ] **Step 5: 标记 spec #1 完成**

在 spec 文件加一行 `**状态**:Spec #1 已实现并冒烟通过`,commit。

```bash
git add docs/superpowers/specs/2026-07-29-mind-diffusion-core-design.md
git commit -m "docs: mark spec #1 implemented + smoke-passed

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review(计划完成后自查)

**1. Spec 覆盖**(对照 spec §3 范围表 + §4-8):
- tokenizer 复用 + `<mask>` → T2 ✓
- `model_dlm.py`(config + 双向 transformer + loss + generate) → T3/T4/T5/T6 ✓
- pretrain → T9 ✓
- SFT → T10 ✓
- 采样(固定 T / low-conf / temperature-only / EOS 截断) → T6 + T11 ✓
- eval_dlm → T11 ✓
- web_demo → T12 ✓
- trainer_utils(get_lr 等) → T8 ✓
- 测试(形状/双向/loss/采样) → T3/T4/T5/T6 ✓
- README + LICENSE → T13 ✓
- 端到端验收 → T14 ✓
- 风格 parity(注释/分隔符/get_lr/JSONL) → Global Constraints + 各 task 代码 ✓

**2. 占位符扫描**:无 TBD/TODO;T13 README 用大纲而非占位(因文档内容,大纲式可接受,实现时填)。

**3. 类型/签名一致性**:
- `DLMConfig` 字段 T3 定义,T4/T5/T9/T10/T11/T12 引用一致 ✓
- `DLMModel.forward(input_ids, attention_mask)` T4 定义,T5/T6/T12 引用一致 ✓
- `DLMForMD.forward(input_ids, attention_mask, response_mask, labels)` T5 定义,T9(传 labels=ids)/T10(传 response_mask=resp, labels=ids)一致 ✓
- `generate(prompt_ids, gen_length, steps, temperature, low_confidence)` T6 定义,T11/T12 调用一致 ✓
- `load_tokenizer(path)` T2 定义,T8/T9/T10/T11 引用一致 ✓
- `init_model(cfg, from_weight, tokenizer_path, save_dir, device)` T8 定义,T9(from_weight=None)/T10(from_weight='pretrain')/T11(from_weight='sft')一致 ✓
- `MASK_ID = 6400` T2 定义,T5(用 config.mask_token_id)/T6/T12(用 config.mask_token_id)一致 ✓

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-29-mind-diffusion-core.md`. Two execution options:

1. **Subagent-Driven (recommended)** — 每 task 一个新 subagent,任务间两段 review,快速迭代
2. **Inline Execution** — 本会话内逐 task 执行,checkpoint 处暂停 review

选哪个?
