# mind-diffusion

> **大道至简 —— diffusion 版 minimind**

`mind-diffusion` 把 [minimind](https://github.com/jingyaogong/minimind) 的"从 0 用 PyTorch 训一个小 LLM"的极简哲学,
移植到**扩散式语言模型 (Diffusion Language Model, DLM)** 上:不左到右一个个生成 token,
而是从一个**全 `<mask>`** 的序列出发,像"打草稿再精修"一样**迭代揭开**。

实现的是 **LLaDA v1**(Nie et al., 2025)的**全序列掩码扩散**配方。

---

## 1. 它是什么

一个**最小、能跑通、讲明白原理**的扩散语言模型项目,和 minimind 风格对齐:

| 维度 | minimind(AR) | mind-diffusion(DLM) |
|---|---|---|
| 生成范式 | 自回归(左→右 next token) | 掩码扩散(全 mask→迭代揭) |
| 注意力 | causal(因果) | **双向**(去 causal mask) |
| KV cache | 有 | 无(每次吃整条) |
| 时间步 t | 无 | 有(掩码比例),但**不喂模型** |
| loss | next-token CE | **1/t 加权掩码 CE**(似然上界) |
| generate | AR 循环 + KV cache | 扩散采样 + remasking |
| 词表 | 6400 | 6401(+`<mask>`) |

风格 parity:单文件 `model_dlm.py`、一 stage 一 `train_*.py`、一行余弦 `get_lr`、
JSONL 数据、vocab 6400 BPE、双语注释、`🌏🌎🌍` 分段。

---

## 2. 核心原理

### 2.1 扩散 vs 自回归
- **自回归 (AR)**:从左到右,每步基于"已生成的"预测下一个 token。`P(x) = ∏ P(x_i | x_{<i})`。
- **掩码扩散 (DLM)**:把 token **随机替换成 `<mask>`**(前向加噪),训练一个**双向** transformer
  去**预测被掩掉位置的原 token**;推理时从**全 `<mask>`** 出发,迭代 N 步逐步揭开。

### 2.2 前向加噪 q(x_t | x_0)
每条序列采一个掩码比例 `t ~ U(0,1)`,每个真实 token 位**独立**以概率 `t` 被掩成 `<mask>`:
- `t=0`:全干净;`t=1`:全掩。
- 掩码比例**随 t 线性增长**——这是跟 BERT(固定 15%)的关键区别。

### 2.3 训练 loss(带 1/t 权重,似然上界)
```
L(θ) = - E[ (1/t) · Σ_{被掩位 i} CE(p_θ(x_0^i | x_t), x_0^i) ]
```
- 只在**被掩位置**算交叉熵。
- **1/t 权重是灵魂**:t 小(少掩、易还原)权重大;t 大(多掩、难)权重小。
  这让 L(θ) 成为负对数似然的**上界**——区别于 BERT/MaskGIT 的均匀加权启发式。
  直觉:t 大时模型几乎在猜数据边缘分布,不该被过度惩罚;t 小时模型该把"简单还原"学好,权重给大。

### 2.4 架构:LLaMA 去掉 causal mask
双向 transformer = 标准 LLaMA decoder,**关掉 causal mask**(`is_causal=False`)。
保留 RoPE + RMSNorm + SwiGLU + GQA + per-head QK-norm。**没有 KV cache**
(双向 + 非自回归,用不上)。**不喂时间步 t 给模型**(time-free):transformer
只看"x_t 长什么样",不看"现在是第几步"——同一套权重 pretrain 和 SFT 兼容。

### 2.5 采样:半自回归迭代 unmasking
从全 `<mask>` 出发,N 步均匀时间步 `t_k = 1 - k/N`(`s` 从 1 降到 0):
1. 跑双向 transformer → 每个 `<mask>` 位 **argmax** 预测原 token。
2. 算**置信度**(low-confidence 策略 = softmax 概率)。
3. 临时写回所有预测;再按置信度**升序**,把最低 `floor(L·s)` 个**重新掩回 `<mask>`**
   (low-confidence remasking)——下步基于更完整的上下文重猜它们。
4. `s` 从 1 降到 0,被掩数线性减到 0;最后一步全揭开。

> **涌现现象**:low-confidence remasking 会让"中间冒出来的推理步骤又被掩掉"——
> 模型在草稿阶段想出来的推理,可能因为置信度不够被重新掩回,等上下文更完整了再重猜。
> 这是 LLaDA 论文里提到的一个有意思的现象,在 `web_demo` 的流式输出里能直观看到。

---

## 3. 快速开始

### 3.1 装依赖
```bash
pip install -r requirements.txt
```

### 3.2 放 tokenizer + 数据(复用 minimind)
从 [minimind](https://github.com/jingyaogong/minimind) 仓库拷贝:
- `model/tokenizer.json` + `model/tokenizer_config.json`(vocab 6400 BPE)
- `dataset/pretrain_t2t_mini.jsonl`(~1.2GB)
- `dataset/sft_t2t_mini.jsonl`(~1.6GB)

(详见 `dataset/dataset.md`。这两个文件已 gitignore,不入库。)

### 3.3 预训练
```bash
python trainer/train_pretrain.py
# 小档(8GB 显卡 / CPU 冒烟):
python trainer/train_pretrain.py --hidden_size 512 --num_hidden_layers 6 --batch_size 4 --max_seq_len 128
```
产出 `out/pretrain_768.pth`(或 `_512.pth`)。

### 3.4 SFT(条件生成)
```bash
python trainer/train_sft.py
# 小档:
python trainer/train_sft.py --hidden_size 512 --num_hidden_layers 6 --batch_size 4
```
产出 `out/sft_768.pth`。SFT 只掩 assistant 回答位(prompt 永不掩),用 minimind chat template。

### 3.5 推理
```bash
python eval_dlm.py --from_weight sft
# 小档:
python eval_dlm.py --hidden_size 512 --num_hidden_layers 6 --from_weight sft
```
打印中文 prompt 的生成结果 + tokens/s。

### 3.6 Web demo(流式看扩散采样)
```bash
streamlit run scripts/web_demo.py
```
每步刷新当前揭开的 token,直观看到"草稿→精修"。

---

## 4. 配置

默认档(对齐 minimind):
```
hidden_size=768, num_hidden_layers=8, num_attention_heads=8,
num_key_value_heads=4 (GQA 2:1), vocab_size=6401,
intermediate_size = ceil(π·768/64)*64, rope_theta=1e6, tied embeddings
```
小档冒烟(8GB / CPU):`hidden_size=512, num_hidden_layers=6`。

采样默认:`steps=128, gen_length=128, temperature=0.7, repetition_penalty=1.3, low_confidence=True`。
- 论文最优是 `steps ≈ gen_length`(慢);教学版 128 够演示。
- `temperature=0` 会塌缩成重复高频 token(扩散特性,与 AR 相反);建议 **0.6–0.9**,>0.9 开始乱说。
- `repetition_penalty` 1.2–1.5 打散重复(扩散双向无生成历史,易循环);1.0 关闭会死循环,2.0 太狠逼出乱码。
- `block_length` >0 启用半自回归块生成(LLaDA 2 思路),需整除 gen_length;更防重复但更短促。

---

## 5. 与 minimind 的差异速查

| 维度 | minimind | mind-diffusion |
|---|---|---|
| 生成范式 | 自回归 | 掩码扩散 |
| 注意力 | causal | 双向 |
| KV cache | 有 | 无 |
| 时间步 t | 无 | 有,不喂模型 |
| loss | next-token CE | 均匀加权掩码 CE(原 1/t 在小模型失效,见 §6.1) |
| dataset 返回 | (input_ids, labels) | (input_ids, attention_mask[, response_mask]) |
| 掩码在哪 | 无 | forward 里现采(每步随机) |
| generate | AR + KV cache | 扩散采样 + low-conf remasking |
| 词表 | 6400 | 6401 |
| 采样开关 | temp/topk/topp/rep | temperature + repetition_penalty + block |
| 训练阶段 | pretrain/sft/lora/dpo/ppo/grpo/agent/distill | **pretrain/sft**(本 repo) |

> **掩码在 forward 里现采**:minimind 的 dataset 返回 `labels`(目标 token);
> 扩散的掩码是每步随机的,**放数据集里没意义**,所以 dataset 只返回干净序列 + mask 范围,
> 掩码 + loss 在 `DLMForMD.forward` 里做。

---

## 6. 已知问题 / 诚实记录

### 6.1 loss 配方:从 LLaDA 的 1/t 改成均匀权重(关键发现)
原版 LLaDA 用 `1/t` 加权掩码 CE(数学上是似然上界,严谨),`t~U(0,1)`。但**在小模型 + 少数据上不工作**:held-out 掩码重建准确率只有 **0–5%**。

诊断:1/t 权重让梯度被"少掩 easy case"主导(小 t 时 1/t 爆炸),模型只学了 token 边缘分布,没学"根据上下文还原"。用 4 个对照组筛选后,**均匀权重 + `t~U(0.1,0.5)`** 把准确率拉到 **41.5%**(64M),收敛曲线也健康(loss 7.4→1.7)。

代价:不再是严格的扩散似然上界(NELBO),更像 BERT-style MLM + 扩散采样。**去噪匹配目标仍在做,采样数学没动**,只是权衡的权重不最优了。LLaDA 2 自己也截断 mask 比例到 `[α_min,α_max]`,本改动踩在同一路径上。诚实标:偏离原版,换小模型可学性。

### 6.2 采样:repetition penalty + block(防重复循环)
小扩散 LM 会无限重复高频模式("天空是蓝色的,天空是蓝色的...")——双向采样无生成历史(每个位置独立预测),不像 AR 有 causal 约束。
- `repetition_penalty`(默认 1.3):对已揭开的 response token 降权。1.0 死循环,1.2-1.5 甜区,2.0 逼出乱码。
- `block_length`(默认 0):>0 启用半自回归块生成(LLaDA 2 思路),块间自回归、块内扩散,后面块能看到前面已生成的干净块。

效果:从"死循环重复"→"通顺连贯多句中文"。

### 6.3 规模墙:能说不能对话(目标1 现状)
64M + mini 语料(90万条)训到 2 epoch,base 模型:
- ✅ 能产出**通顺连贯的中文**(语言建模学会了)
- ✅ **部分问题能真回答**("你叫什么名字"→自我介绍;"如何学习编程"→给 Python/Codecademy 建议)
- ❌ **不稳**:仍会滑回续写模式("1+1等于几"开始列 "-2的平方是..." 数学题)
- ❌ **知识幻觉**("1+1=0.2133"、"水星反射太阳")
- ❌ **多轮不看上文**

诊断:**训练/推理分布 gap**(训练只掩部分,推理全掩开头,小模型跨不过)+ **规模不够**(LLaDA 原版 8B+2.3T 压住,64M 压不住)。多 epoch 边际递减(1→2 epoch,acc 41→44,+3 个点),撞到规模墙。要质变到"稳定对话",需更大模型 / 更多数据——超出 minimind 同档范围。

### 6.4 其他
- **`<mask>` 初始化**:标准可学习 init(跟 LLaDA 官方一致;mean/zero init 筛选后无改善,见 screen_init.py)
- **本 repo 不含**:DPO/LoRA/RL/蒸馏(LLaDA 2 block diffusion 部分实现)— 见后续
- **数据加载**:用 stdlib json + 字节偏移索引,不依赖 datasets/pandas/pyarrow(pyarrow 在 Python 3.14 本环境 import 崩溃)

---

## 7. 项目结构
```
model/
  model_dlm.py        # DLMConfig + 双向 transformer + DLMForMD(掩码+loss+generate)
  tokenizer_loader.py  # 复用 minimind BPE + <mask>
dataset/
  lm_dataset.py        # PretrainDataset + SFTDataset
  dataset.md           # 数据放置说明
trainer/
  trainer_utils.py     # get_lr / SkipBatchSampler / init_model / Logger
  train_pretrain.py    # 全序列随机掩码 + 1/t loss
  train_sft.py         # [prompt;response] 只掩 response
scripts/
  web_demo.py          # Streamlit 流式扩散采样
eval_dlm.py            # 中文 prompt + tokens/s(对照 minimind eval_llm.py)
chat_dlm.py            # 交互式多轮对话
tests/                 # pytest: model / loss / sampling / dataset / trainer_utils
```

---

## 8. 致谢
- [minimind](https://github.com/jingyaogong/minimind) — 极简风格与"从 0 训练"哲学的本源。
- [LLaDA](https://github.com/ML-GSAI/LLaDA)(Nie et al., 2025, arXiv:2502.09992)— 掩码扩散语言模型的方法与官方实现。

## License
Apache-2.0(同 minimind)。
