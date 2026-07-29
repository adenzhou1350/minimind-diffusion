# mind-diffusion — Diffusion 版 minimind 设计文档 (Spec #1: 核心脊柱)

> 大道至简 —— 把 minimind 的"从 0 用 PyTorch 训一个小 LLM"的极简哲学,移植到
> **扩散式语言模型 (Diffusion Language Model, DLM)** 上。
> 本 spec 实现 LLaDA v1 全序列掩码扩散的完整训练 + 采样脊柱。

- **状态**:设计已与用户确认,待实现
- **日期**:2026-07-29
- **作者**:Claude(与用户共同 brainstorm)
- **项目根**:`D:\codes\mind-diffusion`

---

## 1. 目标与非目标

### 目标
1. **讲明白原理**:README + `model_dlm.py` 注释把"掩码扩散语言模型"从加噪、训练 loss、到采样,
   讲到能跟 minimind 讲 AR-GPT 一样清楚。
2. **跑通**:单 GPU(8GB RTX 4060 起步,可换卡)上,tokenizer→pretrain→SFT→采样 全链路跑通,
   能生成可读中文、能做多轮对话。
3. **风格 parity**:文件结构、命名、注释风格、一行余弦 `get_lr`、双语段落分隔符,都跟 minimind 对齐,
   读起来"就是 minimind 的 diffusion 版"。

### 非目标(Spec #1 不做,留后续 spec)
- DPO / LoRA / PPO / GRPO / agent RL / 蒸馏(对齐 minimind 的 8 阶段里后 6 个)
- LLaDA 2 的 block diffusion、AR→MDLM 转换、α 带宽截断、CAP 训练
- MoE(`use_moe` 字段保留但不启用,跟 minimind 一致)
- top-k / top-p / repetition penalty 采样(扩散语义下意义不大,见 §6.4)
- GGUF 转换(minimind 也只做 HF 格式转换,本 spec 连转换都先不做)

---

## 2. 背景:LLaDA v1 原理(写给读 spec 的人)

LLaDA 是**离散掩码扩散 (masked diffusion)**:不往连续 embedding 加高斯噪声,而是把 token
**随机替换成一个 `<mask>` 占位符**,训练一个**双向 transformer 去预测被掩掉位置的原 token**。
推理时从一个**全 `<mask>`** 序列出发,迭代 N 步逐步揭开。

### 2.1 前向加噪 q(x_t | x_0)
每条序列采一个掩码比例 `t ~ U(0,1)`,每个真实 token 位独立以概率 `t` 被掩成 `<mask>`:
- `t=0`:全干净;`t=1`:全掩。
- 掩码比例**随 t 线性增长**——这是跟 BERT(固定 15%)的关键区别。

### 2.2 训练 loss(似然上界,带 1/t 权重)
```
L(θ) = - E_{t, x_0, x_t} [ (1/t) · Σ_i 1[x_t^i = MASK] · log p_θ(x_0^i | x_t) ]
```
- 只在**被掩位置**算交叉熵。
- **1/t 权重**是灵魂:t 小(少掩、易还原)权重大;t 大(多掩)权重小。
  这让 L(θ) 成为负对数似然的**上界**——区别于 BERT/MaskGIT 的均匀加权启发式。
  README 会画图讲为什么这个权重让"难样本(t 大)不被过度惩罚"。

### 2.3 time-free 参数化
扩散时间 `t` **不喂给模型**。transformer 只看"x_t 长什么样",不看"现在是第几步"。
同一套权重 pretrain 和 SFT 兼容(只差掩码范围)。

### 2.4 架构:LLaMA 去掉 causal mask
双向 transformer = 标准 LLaMA decoder,**关掉 causal mask**(`is_causal=False`)。
保留 RoPE + RMSNorm + SwiGLU。**没有 KV cache**(双向 + 非自回归,用不上)。

### 2.5 采样:半自回归迭代 unmasking
从全 `<mask>` 出发,N 步均匀时间步 `t_k = 1 - k/N`:
1. 跑双向 transformer → 每个 `<mask>` 位 argmax 预测原 token。
2. 算置信度(low-confidence 策略 = softmax 概率)。
3. 保留 top `floor(L·s)` 高置信的固化;**其余重新掩回 `<mask>`**(low-confidence remasking)。
4. 下一步基于更完整的上下文重猜低置信位。
最后全部固化,decode,遇首个 EOS 截断。

---

## 3. 范围(Spec #1)

| 组件 | 做 | 说明 |
|---|---|---|
| tokenizer | 复用 minimind `tokenizer.json` | vocab 6400、36 special token、含 `<|im_start|>`/`<|im_end|>`/思考标签。仅新增 1 个 `<mask>` token |
| model | `model_dlm.py` 单文件 | DLMConfig + 双向 transformer + 扩散 generate |
| pretrain | `train_pretrain.py` | 全序列随机掩码 + 1/t loss |
| SFT | `train_sft.py` | `[prompt;response]`,只掩 response,minimind chat template |
| 采样 | `model_dlm.generate` | 固定 T、low-confidence remasking、temperature-only |
| eval | `eval_dlm.py` | 一组中文 prompt,打印 tokens/s + 生成样例 |
| demo | `scripts/web_demo.py` | Streamlit,扩散采样流式输出 |
| 工具 | `trainer_utils.py` | get_lr / SkipBatchSampler / init_model / Logger / setup_seed(几乎照搬 minimind) |

---

## 4. 架构:`model/model_dlm.py`

单文件,镜像 minimind `model_minimind.py` 的结构,5 处关键改动。

### 4.1 类清单(对照 minimind)
| 类 | 基 | 改动 |
|---|---|---|
| `DLMConfig` | `PretrainedConfig` | 默认对齐 minimind(§4.2);`vocab_size=6401`(+`<mask>`) |
| `RMSNorm` | `nn.Module` | 照搬 |
| `Attention` | `nn.Module` | **去 causal mask** → `is_causal=False`、双向;**去 KV cache**;保留 GQA+RoPE+QK-norm |
| `FeedForward` | `nn.Module` | 照搬 SwiGLU |
| `DLMBlock` | `nn.Module` | 照搬 pre-norm block |
| `DLMModel` | `nn.Module` | embed + 堆叠 block + final norm |
| `DLMForDLM` | `PreTrainedModel` | LM head(tied);**`forward` 做掩码 + 1/t loss**;**`generate` 换扩散采样** |

> 命名:`DLMForDLM` 太绕,实际用 `DLMModel` + `DLMForMaskedDiffusion`(简写 `DLMForMD`)。
> 最终以实现为准,但**不**叫 `ForCausalLM`(它不是 causal 的)。

### 4.2 DLMConfig 默认(对齐 minimind)
```python
hidden_size = 768
num_hidden_layers = 8
num_attention_heads = 8
num_key_value_heads = 4          # GQA 2:1,跟 minimind 一致(LLaDA 用 MHA,但 GQA 在双向里照样工作且省参数)
vocab_size = 6401                 # minimind 6400 + <mask>
intermediate_size = ceil(π * 768 / 64) * 64   # minimind 的 π-scaled 对齐
max_position_embeddings = 32768
rms_norm_eps = 1e-6
rope_theta = 1e6
tie_word_embeddings = True
dropout = 0.0
mask_token_id = 6400             # 词表最后一个 = 新增的 <mask>
bos_token_id = 1                 # 复用 minimind <|im_start|>
eos_token_id = 2                 # 复用 minimind <|im_end|>
# use_moe 字段保留但不启用(跟 minimind 一致)
```
**小档冒烟配置**(8GB 4060 / CPU):`hidden=512, layers=6, heads=8`,~8M 参数。
README 注明两档切换。

### 4.3 五处关键改动(相对 minimind AR 版)
1. **Attention 去 causal mask**:`is_causal=False`,attention bias 全零(双向)。
   minimind 用 `scaled_dot_product_attention` + 手动 softmax fallback;DLM 版把 `is_causal=False`
   传进去即可,sdpa 原生支持非因果。
2. **去 KV cache**:`forward` 每次吃整条序列,无 `past_key_values` 参数、无 `use_cache`。
   (minimind 的手写 list-of-tuples cache 整段删除。)
3. **加 `<mask>` token**:vocab 6400→6401,tied embedding 所以 embed + lm_head 同步长一行。
   mask embedding 是普通可学习行(标准 init,**不做** mean-init——LLaDA 官方代码也不做,博客传的是 folk)。
4. **不喂时间步 `t`**:transformer 输入只有 `x_t` + attention_mask,无 `t` embedding。
5. **`generate` 换扩散采样**:删掉 AR 的 next-token 循环,换成 §6 的迭代 unmasking。

---

## 5. 数据流与训练 loss

### 5.1 数据集(`dataset/lm_dataset.py`)

| 类 | `__getitem__` 返回 | 掩码在哪做 |
|---|---|---|
| `PretrainDataset` | `(input_ids, attention_mask)` | **trainer/model forward 里现采**(每步随机) |
| `SFTDataset` | `(input_ids, attention_mask, response_mask)` | 同上,但只掩 response |

> **相对 minimind AR 版的刻意偏差**:minimind 的 dataset 返回 `labels`(目标 token);
> 扩散的掩码是每步随机的,**放数据集里没意义**,所以 dataset 只返回干净序列 + mask 范围,
> 掩码 + loss 在 `forward` 里做。这点会在代码注释里讲明白。

数据复用 minimind:`pretrain_t2t_mini.jsonl`(~1.2GB)、`sft_t2t_mini.jsonl`(~1.6GB),
放 `dataset/` 下(跟 minimind 的 `dataset.md` stub 一致)。SFT schema 复用 minimind 的
`role/content/reasoning_content` + chat template。

### 5.2 forward(掩码 + 1/t loss)
```python
def forward(self, input_ids, attention_mask, response_mask=None, labels=None):
    x_0 = labels if labels is not None else input_ids        # 干净目标
    B, L = x_0.shape
    # 1. 每序列采掩码比例 t,夹到 [1e-4, 1] 防除零
    t = uniform(1e-4, 1, shape=[B], device=x_0.device)
    # 2. 决定可掩范围:pretrain 全序列;SFT 只 response_mask 位
    maskable = attention_mask.bool()                          # 真实 token(非 pad)
    if response_mask is not None:
        maskable = maskable & response_mask.bool()
    # 3. 伯努利(t) 掩码
    mask = maskable & (rand(B, L) < t[:, None])
    x_t = x_0.clone(); x_t[mask] = self.config.mask_token_id
    # 4. 双向 transformer(不喂 t)
    logits = self.model(x_t, attention_mask)                  # [B, L, V]
    # 5. 1/t 加权的掩码 CE
    ce = F.cross_entropy(logits.view(-1, V), x_0.view(-1), reduction='none').view(B, L)
    ce = ce * mask                                              # 只算被掩位
    # 每序列: (1/t) * mean over masked;再 batch mean
    n_masked = mask.sum(dim=1).clamp(min=1)
    loss = ((ce.sum(dim=1) / n_masked) * (1.0 / t)).mean()
    return loss
```

**要点**:
- `t ~ U(1e-4, 1)`:论文 U(0,1),工程夹下界防除零(标准做法,注释注明)。
- **1/t 权重**:`(1.0/t)` 乘到每序列 loss 上。
- **pad 位不掩、不计 loss**:`maskable` 用 `attention_mask` 屏掉 pad。
- **EOS 当普通 token**:进 maskable 范围,被掩/被预测;推理遇首个截断。
- pretrain/SFT 同一 forward,只差 `response_mask` 是否传入。

### 5.3 训练脚本骨架(`train_pretrain.py` / `train_sft.py`)
照搬 minimind:`init_distributed_mode`→`setup_seed`→build `DLMConfig`→`init_model`→
dataset + `SkipBatchSampler` + `DataLoader(pin_memory=True)`→AdamW + 每步 `get_lr`→
autocast/`GradScaler` 循环→checkpoint `.pth`(权重减半 + 移 CPU)。
- pretrain 默认:epochs=2, batch=32, lr=5e-4, accum=8, max_seq=340, bf16, AdamW。
- LR 一行余弦:`get_lr = lr * (0.1 + 0.45 * (1 + cos(π * step/total)))`(无 warmup,跟 minimind 一致)。

---

## 6. 推理/采样(`DLMForMD.generate`)

### 6.1 算法
```python
@torch.inference_mode()
def generate(self, prompt_ids, gen_length=L, steps=T, temperature=0.0,
             low_confidence=True):
    # 1. 构造 prompt(干净) + response 段全 <mask>
    x   = concat([prompt_ids, full([MASK_ID] * gen_length)])      # [1, P+L]
    pm  = concat([ones(P), zeros(gen_length)])                    # prompt 段永不重掩
    # 2. 均匀时间步 t_k = 1 - k/T
    for k in 1..T:
        s     = 1 - k / T                                          # 目标"还剩多少比例被掩"
        logits = self.model(x, attention_mask=ones_like(x))       # 双向,整条
        # 每个被掩 response 位 argmax 预测
        rpos  = (x == MASK_ID) & ~pm                               # 当前被掩的 response 位
        prob  = softmax(logits[rpos] / max(temperature, 1e-4))
        pred  = argmax(prob, dim=-1)
        conf  = prob.gather(pred) if low_confidence else rand(prob.shape)
        x[rpos] = pred                                             # 临时写回
        # 3. 固化 top 高置信;重掩低置信回 <mask>
        n_keep = floor(gen_length * s)                             # 这轮该留多少被掩
        # 把当前已揭的 response 位按 conf 排序
        order = argsort(conf_of_unmasked_response_positions)      # 升序:低→高
        # 最低 n_keep 个 → 重新掩回 <mask>(下轮重猜)
        x[lowest_n_keep_positions] = MASK_ID
        # 其余 → 固化(不动,退出 rpos)
    # 4. 全固化,decode,遇首个 EOS 截断
    return decode(x[P:], stop_at_eos=True)
```

### 6.2 默认与可调
- `steps`(T):默认 **64**(够演示、又不太慢)。README 注明论文最优是 `T=L`(慢),给个 tradeoff 说明。
- `gen_length`(L):默认 128(对话够用)。
- `low_confidence`:默认 True。README 讲 low-conf vs random 的区别 + 涌现的"推理步骤被重掩"现象。
- `temperature`:默认 0(贪心,论文 eval 用)。>0 加 Gumbel 噪声增多样性。

### 6.3 EOS 处理
EOS 当普通 token 训练、被掩/被预测;推理 decode 后遇**首个** `<|im_end|>`(EOS)截断,后面的丢。
不实现论文附录 B.4 的 `logits_eos_inf` 抑制旗标(教学项目过重,注释注明可扩展)。

### 6.4 采样简化(取舍,README 注明)
minimind generate 有 temperature/top-k/top-p/repetition_penalty。DLM 版**只留 temperature**:
- top-k/top-p:扩散每步是 argmax 选 token(不是从分布采样),nucleus 截断意义不大。
- repetition penalty:非 AR 里"重复"语义不清(序列并行生成,没有"刚生成过"的时序)。
若用户坚持可加,但默认不开。这点诚实标在 README 的"与 minimind 的差异"节。

---

## 7. 项目结构
```
D:\codes\mind-diffusion\
├─ model/
│   ├─ model_dlm.py          # DLMConfig + RMSNorm + Attention(双向) + FeedForward + Block + DLMModel + DLMForMD + generate
│   ├─ tokenizer.json         # 复用 minimind + 1 个 <mask>
│   └─ tokenizer_config.json
├─ dataset/
│   ├─ lm_dataset.py          # PretrainDataset + SFTDataset(返回干净序列 + mask 范围)
│   ├─ pretrain_t2t_mini.jsonl  # 复用 minimind(用户自行放置,见 dataset.md stub)
│   ├─ sft_t2t_mini.jsonl
│   └─ dataset.md             # 5 行 stub:把数据放当前目录(跟 minimind 一致)
├─ trainer/
│   ├─ trainer_utils.py       # get_lr / SkipBatchSampler / init_model / Logger / setup_seed(几乎照搬)
│   ├─ train_pretrain.py      # 全序列随机掩码 + 1/t loss
│   └─ train_sft.py           # [prompt;response] 只掩 response
├─ scripts/
│   └─ web_demo.py            # Streamlit,扩散采样流式输出(每步把当前揭开的 token 刷出来)
├─ eval_dlm.py                # 一组中文 prompt,打印 tokens/s + 生成样例(对照 minimind eval_llm.py)
├─ README.md / README_en.md / requirements.txt / LICENSE(Apache-2.0)
└─ docs/superpowers/specs/2026-07-29-mind-diffusion-core-design.md  # 本文件
```

### 7.1 风格 parity 清单(对齐 minimind)
- [ ] `model_dlm.py` 注释**英文为主** + `🌏🌎🌍` 边框分段
- [ ] `train_*.py` 用 `# ========== N. <中文标题> ==========` 双语分隔符
- [ ] argparse `help=` 用中文
- [ ] `dataset/lm_dataset.py` 注释中文
- [ ] 一行余弦 `get_lr`
- [ ] JSONL + HF `load_dataset('json')`(非 .bin memmap)
- [ ] checkpoint `.pth` 权重减半 + 移 CPU
- [ ] `@torch.inference_mode()` 装饰 generate

---

## 8. 测试与"跑通"验收标准

### 8.1 单元 / 冒烟(写进 repo,可 `pytest`)
- `test_model.py`:形状测试(forward 出 loss 是标量、logits 形状对)、双向注意力测试
  (交换两 token 位置输出对应交换,验证非因果)、`<mask>` id 正确。
- `test_sampling.py`:小模型随机初始化,`generate` 能跑完 T 步、输出长度 = gen_length、
  遇 EOS 截断、low-conf 与 random 两个分支都能跑。
- `test_loss.py`:1/t 权重在 t→0 时 loss 不爆(夹 1e-4)、t=1 时全掩 loss 合理、
  pad 位不计入 loss。

### 8.2 端到端"跑通"验收(用户在 4060 或更强卡上)
1. 复用 minimind tokenizer + mini 语料,`train_pretrain.py` 跑 ≥1 epoch,**loss 下降**。
2. `train_sft.py` 跑 ≥1 epoch,**loss 下降**。
3. `eval_dlm.py`:给几个中文 prompt(如"为什么天空是蓝色的"),SFT 后模型能生成**可读、相关**的中文回复。
4. `web_demo.py`:Streamlit 能流式展示扩散采样(每步刷新当前揭开的 token,直观看到"草稿→精修")。
5. tokens/s 打印(注意:扩散的 tokens/s 定义是 `gen_length / 采样总秒数`,跟 AR 不同,README 注明)。

**"跑通"的最低门槛**:步骤 1-3 全绿 + 步骤 4 能跑出界面。质量(可读性)是加分项,
不作为 spec #1 的硬验收——模型太小、语料太 mini,不追求质量,追求"原理讲清 + 全链路通"。

### 8.3 验证手段
- 训练 loss 曲线(tensorboard / 简单 matplot 保存 png,跟 minimind images/ 一致)
- 生成样例对比:pretrain-only vs SFT(看 SFT 后是否更像对话)
- 固定 prompt + 不同 T(16/64/128)对比,验证"T 越大质量越好但越慢"

---

## 9. 已标记的不确定项(诚实记录)
1. **LLaDA 1B 的确切 d_model/n_layers/n_heads**:HF 封了直连,只确认了 8B(4096/32/32 MHA)。
   本 spec 不照搬 1B,而是对齐 minimind(768/8/8 GQA),所以不依赖这个不确定项。
2. **`<mask>` embedding 初始化**:LLaDA 官方代码是标准可学习 init,**不做** mean-averaging
   (博客传的是 folk)。本 spec 跟官方一致。如果实测 mean-init 更稳,再调(注释留 TODO)。
3. **采样步数预算 `n_keep = floor(L·s)` 的精确形式**:论文 Algorithm 5 是"期望到时间 s 还剩
   floor(L·s) 个被掩"。本 spec 用这个线性版本;LLaDA 2 的 block 版本更复杂,留后续。
4. **temperature 在扩散里的语义**:LLaDA 的 temperature 是加到 logits 后再 softmax+argmax
   (temperature=0 纯 argmax,>0 加 Gumbel)。本 spec 用这个;不照搬 AR 的"除 logits"语义。

---

## 10. 后续 spec(本 spec 不做,记录用)
- **Spec #2**:`train_dpo.py`(扩散 DPO,套 BBDLM ELBO + 单样本 MC)+ `train_lora.py`(直接套 backbone)
- **Spec #3**:LLaDA 2 block diffusion(块内扩散 + 块间 AR + KV cache 复用 + 变长)
- **Spec #4**:RL(GRPO/PPO for diffusion,研究前沿,骨架 + 实验性)

---

## 附录:与 minimind 的差异速查表
| 维度 | minimind | mind-diffusion |
|---|---|---|
| 生成范式 | 自回归(左→右 next token) | 掩码扩散(全 mask→迭代揭) |
| 注意力 | causal(因果) | 双向(去 causal mask) |
| KV cache | 有(手写 list-of-tuples) | 无(每次吃整条) |
| 时间步 t | 无 | 有(掩码比例),但**不喂模型** |
| loss | next-token CE | 1/t 加权掩码 CE |
| dataset 返回 | (input_ids, labels) | (input_ids, attention_mask[, response_mask]) |
| 掩码在哪 | 无 | forward 里现采(每步随机) |
| generate | AR 循环 + KV cache | 扩散采样循环 + remasking |
| 词表 | 6400 | 6401(+`<mask>`) |
| 采样开关 | temp/topk/topp/rep_penalty | 仅 temperature |
| 训练阶段 | pretrain/sft/lora/dpo/ppo/grpo/agent/distill | pretrain/sft(本 spec) |
