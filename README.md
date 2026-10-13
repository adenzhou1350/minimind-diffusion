# minimind-diffusion

中文 · [English](README_en.md)

**把文字遮住，再逐步补回来。用一个小模型看懂掩码扩散。**

从 PyTorch 的双向注意力和掩码损失开始，串起训练、采样与可视化，再尝试加入图像条件。适合想亲手比较扩散生成与自回归、而不只停留在论文示意图的读者。

[先跑小模型测试](#先跑一个不用下载模型的例子) · [核心原理](#2-核心原理) · [文本训练](#3-快速开始文本-dlm) · [图文实验](#4-扩展多模态-vlm) · [实验出处](docs/experiment-evidence.md) · [作者的教学专区](https://adenzhou1350.github.io/learn/)

## 你会沿着哪条路径学习

| 阶段 | 看什么 | 动手入口 |
|---|---|---|
| 理解最小机制 | 双向注意力、张量形状、哪些 token 参与 loss | [model 测试](tests/test_model.py)、[loss 测试](tests/test_loss.py) |
| 训练文本模型 | 随机掩码、预训练、仅回答部分计算 SFT loss | [模型实现](model/model_dlm.py)、[训练入口](trainer/train_pretrain.py) |
| 观察采样 | 预测、按置信度重新遮盖、逐步揭开 token | [采样测试](tests/test_sampling.py)、[Web demo](scripts/web_demo.py) |
| 继续做图文实验 | 冻结视觉编码器、projector 与文本扩散的连接 | [视觉路径](model/model_dlm_v.py)、[失败与局限](#74-vlm学会格式学不会看图多模态侧现状) |

```text
训练：原始文本 → 随机遮盖 → 双向 Transformer → 被遮盖位置的损失
采样：全 mask → 预测 → 保留高置信度 token → 重复，直到揭开
```

**当前定位：学习与复现实验。** 仓库不附带训练好的权重或数据；文本对话仍不稳定，已记录的图文实验也没有可靠地根据图片回答。可以先运行小模型测试，再准备 tokenizer 和数据开始训练。

## 先跑一个不用下载模型的例子

```bash
git clone https://github.com/adenzhou1350/minimind-diffusion.git
cd minimind-diffusion
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest tests/test_model.py tests/test_loss.py -q
```

这组测试使用随机初始化的小模型，检查张量形状、双向注意力和掩码损失，不需要 tokenizer、数据集或 checkpoint。测试通过只代表这些基础行为符合预期，生成效果需要另行训练和评估。

想继续训练：[文本流程](#3-快速开始文本-dlm) · [图文流程](#4-扩展多模态-vlm) · [已有实验记录与局限](#7-已知问题--诚实记录) · [筛选脚本与数据口径](docs/experiment-evidence.md)

`minimind-diffusion` 把 [minimind](https://github.com/jingyaogong/minimind) 的"从 0 用 PyTorch 训一个小 LLM"的极简哲学,
移植到**扩散式语言模型 (Diffusion Language Model, DLM)** 上:不左到右一个个生成 token,
而是从一个**全 `<mask>`** 的序列出发,像"打草稿再精修"一样**迭代揭开**。

参考 **LLaDA v1**(Nie et al., 2025)的**全序列掩码扩散**思路；当前训练采用均匀加权与截断掩码比例（见 §7.1）,并在其上加一条
视觉路径([minimind-v](https://github.com/jingyaogong/minimind-v) 风格),从**文本 DLM** 一路走到**图文多模态 VLM**。

一个仓库,两条线:

- **Part 1 · 文本扩散 LM** —— 预训练 → SFT → 推理 → Web demo(§3)
- **Part 2 · 多模态扩展 VLM** —— 冻结 SigLIP2 + projector,两阶段对齐 → SFT(§4)

---

## 目录

- [1. 它是什么](#1-它是什么)
- [2. 核心原理](#2-核心原理)
- [3. 快速开始:文本 DLM](#3-快速开始文本-dlm)
- [4. 扩展:多模态 VLM](#4-扩展多模态-vlm)
- [5. 配置](#5-配置)
- [6. 与 minimind 的差异速查](#6-与-minimind-的差异速查)
- [7. 已知问题 / 诚实记录](#7-已知问题--诚实记录)
- [8. 项目结构](#8-项目结构)
- [9. 致谢](#9-致谢)

---

## 1. 它是什么

一个**最小、能跑通、讲明白原理**的扩散语言模型项目,和 minimind 风格对齐:

| 维度 | minimind(AR) | minimind-diffusion(DLM) |
|---|---|---|
| 生成范式 | 自回归(左→右 next token) | 掩码扩散(全 mask→迭代揭) |
| 注意力 | causal(因果) | **双向**(去 causal mask) |
| KV cache | 有 | 无(每次吃整条) |
| 时间步 t | 无 | 有(掩码比例),但**不喂模型** |
| loss | next-token CE | **均匀加权掩码 CE**，掩码比例 [0.1, 0.5]（当前代码） |
| generate | AR 循环 + KV cache | 扩散采样 + remasking |
| 词表 | 6400 | 6401(+`<mask>`) |

风格 parity:单文件 `model_dlm.py`、一 stage 一 `train_*.py`、一行余弦 `get_lr`、
JSONL 数据、vocab 6400 BPE、双语注释、`🌏🌎🌍` 分段。多模态侧([Part 2](#4-扩展多模态-vlm))
同样只在文本 DLM 上加一条 <50 行的视觉路径,对照 minimind-v 的"LLM 继承 + 极简 vision"范式。

**这个仓库适合谁**:想**从 0 验证一个扩散模型能不能训起来**、看清"掩码扩散 vs 自回归"
每一处工程差别的人。仓库提供训练和采样代码，也保留了未能稳定对话、未能可靠看图的历史记录（§7）。

---

## 2. 核心原理

### 2.1 扩散 vs 自回归
- **自回归 (AR)**:从左到右,每步基于"已生成的"预测下一个 token。`P(x) = ∏ P(x_i | x_{<i})`。
- **掩码扩散 (DLM)**:把 token **随机替换成 `<mask>`**(前向加噪),训练一个**双向** transformer
  去**预测被掩掉位置的原 token**;推理时从**全 `<mask>`** 出发,迭代 N 步逐步揭开。

### 2.2 前向加噪 q(x_t | x_0)

以下描述原始配方；当前 `forward` 采样 `t ~ U(0.1, 0.5)`。
每条序列采一个掩码比例 `t ~ U(0,1)`,每个真实 token 位**独立**以概率 `t` 被掩成 `<mask>`:
- `t=0`:全干净;`t=1`:全掩。
- 掩码比例**随 t 线性增长**——这是跟 BERT(固定 15%)的关键区别。

### 2.3 当前训练 loss：均匀加权的掩码 CE
```
每条序列：sum(被掩位置的 CE) / max(被掩 token 数, 1)
一个 batch：对各序列的 loss 取平均
```
- 只在**被掩位置**算交叉熵；padding 不参与，SFT 进一步限制在回答部分。
- 当前采样 `t ~ U(0.1, 0.5)`，没有额外的 `1/t` 权重。
- 对照阅读：[实际实现](model/model_dlm.py)与[掩码损失测试](tests/test_loss.py)。

旧实现会在上述每序列平均 loss 上再乘 `1/t`。这是本仓库早期实现与当前实现的差别，不能据此判断 LLaDA 官方配方是否适合小模型；历史记录和缺失的证据见[实验出处](docs/experiment-evidence.md)。

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

## 3. 快速开始:文本 DLM

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
# 较小配置（显存需求取决于序列长度、batch 和环境）:
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
打印中文 prompt 的生成结果 + tokens/s。交互式多轮对话用 `python chat_dlm.py`。

### 3.6 Web demo(流式看扩散采样)
```bash
streamlit run scripts/web_demo.py
```
每步刷新当前揭开的 token,直观看到"草稿→精修"。

---

## 4. 扩展:多模态 VLM

文本 DLM 之上加一条 vision 路径,对照 [minimind-v](https://github.com/jingyaogong/minimind-v):
冻结 SigLIP2(95M,256px,8×8=64 token,768-dim)+ MLP projector,
vision token 填到 `<|image_pad|>` 占位符(观测条件,**永不掩**),扩散 loss 只掩文本。
LLaVA 式前缀注入(非 cross-attn)。`vocab_size=6401` 不变(image_pad 复用 minimind 预留 id 12,不新增 token)。

> **只加了视觉路径,没动文本扩散内核**:`DLMForVLM` 继承 `DLMForMD`,
> vision 侧是薄薄一层 encoder + projector。所以它和 Part 1 共享同一套采样 / loss / tokenizer。

### 4.1 放数据 + 视觉编码器(自行下载,已 gitignore)
- `dataset/pretrain_i2t.parquet`(~4.1GB,图-文对,127 万行)+ `dataset/sft_i2t.parquet`(~4.6GB,多轮指令带图,290 万行)
- `model/siglip2-base-p32-256-ve/`(从 [jingyaogong/siglip2-base-p32-256-ve](https://huggingface.co/jingyaogong/siglip2-base-p32-256-ve) 下载)

### 4.2 两阶段训练
```bash
# Stage 1 对齐:LLM 全冻结,只训 projector(lr 4e-4, freeze_mode=2)
python trainer/train_pretrain_vlm.py
# Stage 2 SFT:LLM 首尾层 + final norm + lm_head + projector(lr 5e-6, freeze_mode=1)
python trainer/train_sft_vlm.py
# 较小 batch 配置:--max_seq_len 256 --batch_size 16 --accumulation_steps 8
```
Stage1 产出 `out/vlm_align_768.pth`,Stage2 产出 `out/vlm_sft_768.pth`。
Stage2 每 1000 步中途存盘(单 epoch ~47h,崩了不丢);`--skip N --from_weight vlm_sft` 可从第 N 个 accum step 续训。

### 4.3 VLM 推理
```bash
python eval_dlm_vlm.py --from_weight vlm_sft --sample_idx 0
```
从 `dataset/sft_i2t.parquet` 取第 `--sample_idx` 行的图,跑 3 个中文 prompt(描述/有什么/主色调),
打印生成 + tokens/s,参考 prompt/answer 也从 parquet 取出对照。样本图存 `out/eval_sample.jpg`。

### 4.4 VLM Web demo
```bash
streamlit run scripts/web_demo_vlm.py
```
上传图 + 文本 prompt,流式看扩散揭开(未揭开的 `<mask>` 显示为 ▍)。带 gen_length/steps/温度/重复惩罚滑块。

---

## 5. 配置

默认档(对齐 minimind):
```
hidden_size=768, num_hidden_layers=8, num_attention_heads=8,
num_key_value_heads=4 (GQA 2:1), vocab_size=6401,
intermediate_size = ceil(π·768/64)*64, rope_theta=1e6, tied embeddings
```
较小训练配置:`hidden_size=512, num_hidden_layers=6`。

[`eval_dlm.py`](eval_dlm.py) 的采样默认：`steps=64, gen_length=128, temperature=0.7, repetition_penalty=1.3`。模型方法本身的默认温度不同，请以所用入口为准。
- 历史试跑中，`temperature=0` 容易重复；`0.6–0.9` 和 `repetition_penalty=1.2–1.5` 曾被用作调参范围。这些观察没有附统一评测，不能保证对其他权重有效。
- `block_length > 0` 启用半自回归块生成，需整除 `gen_length`；块间固定已生成内容，块内迭代揭开 token。实现见 [generate](model/model_dlm.py)。

---

## 6. 与 minimind 的差异速查

| 维度 | minimind | minimind-diffusion |
|---|---|---|
| 生成范式 | 自回归 | 掩码扩散 |
| 注意力 | causal | 双向 |
| KV cache | 有 | 无 |
| 时间步 t | 无 | 有,不喂模型 |
| loss | next-token CE | 均匀加权掩码 CE（旧实现对比见 §7.1） |
| dataset 返回 | (input_ids, labels) | (input_ids, attention_mask[, response_mask]) |
| 掩码在哪 | 无 | forward 里现采(每步随机) |
| generate | AR + KV cache | 扩散采样 + low-conf remasking |
| 词表 | 6400 | 6401 |
| 采样开关 | temp/topk/topp/rep | temperature + repetition_penalty + block |
| 训练阶段 | pretrain/sft/lora/dpo/ppo/grpo/agent/distill | **pretrain/sft**(文本) + **align/sft**(多模态) |

> **掩码在 forward 里现采**:minimind 的 dataset 返回 `labels`(目标 token);
> 扩散的掩码是每步随机的,**放数据集里没意义**,所以 dataset 只返回干净序列 + mask 范围,
> 掩码 + loss 在 `DLMForMD.forward` 里做。

---

## 7. 已知问题 / 诚实记录

这一节保留历史试跑的观察。2026-10-13 从旧本机实验目录找回了一份 [VLM 训练日志](docs/evidence/README.md)，现已原样附上。**文本重建指标仍缺对应原始日志，逐样本评测和 checkpoint 也未发布**；不能把下面的历史数字都当作已复现的 benchmark。固定版本、找回的产物和缺失项集中在[实验出处](docs/experiment-evidence.md)，本次没有重跑历史训练。

### 7.1 loss 配方：当前实现与历史记录

当前代码使用均匀加权的掩码 CE，`t ~ U(0.1, 0.5)`。这项改动可以从 [loss 修改提交](https://github.com/adenzhou1350/minimind-diffusion/commit/fb96b344f5e58b7f9074de72e19473afdee67927)核对。

历史文字中有两组不同记录：

| 文字出处 | 当时记下的结果 | 现在能确认什么 |
|---|---|---|
| loss 修改提交（`fb96b34`） | 23M 筛选，约 5% 与 25.5% 掩码重建准确率，后者为 3,000 步 | 提交说明中的观察，没有随附日志 |
| [早期 README](https://github.com/adenzhou1350/minimind-diffusion/blob/c08099625cbd36fec3bad8b2600d0f4036da03c2/README.md) | 旧实现 0–5%；后续标为 64M 的记录为 41.5%，loss 7.4→1.7 | 无法从公开材料确认同数据、同训练预算和同评测设置 |

因此不能把 **0–5% → 41.5%** 当成只改 loss 得到的受控提升，也不能把掩码重建准确率当作问答准确率。[四组筛选脚本](screen_loss.py)仍可阅读和重新运行，但它使用 256×4 的配置；脚本本身不是 64M 结果的原始证据。

额外的 `1/t` 权重、掩码范围、模型大小和训练步数都可能影响结果。旧实现先除以被掩 token 数，再乘 `1/t`，完整归一化方式也需要单独核对。此前“容易样本主导梯度”的解释只是假设，尚无梯度统计和单变量对照支持。

### 7.2 采样：重复惩罚与分块

历史试跑中出现过“天空是蓝色的”一类重复输出；加入重复惩罚和分块后，部分样例更通顺。没有保存统一提示集、输出和随机种子，不能据此报告稳定的质量提升。

当前实现可以核对两件事：`repetition_penalty` 会对已揭开的回答 token 降权；`block_length > 0` 会固定前面的块，再生成后面的块。[采样测试](tests/test_sampling.py)检查输出长度、mask 是否清除及不同分支能否运行，不评估语言质量。

### 7.3 文本：能生成句子，对话仍不稳定

[早期 README](https://github.com/adenzhou1350/minimind-diffusion/blob/c08099625cbd36fec3bad8b2600d0f4036da03c2/README.md)将一次训练记为 64M、约 90 万条语料、2 epoch，并记录了以下样例现象：

- 有些输出是通顺的中文，也能回答部分自我介绍、学习编程类问题。
- 有时会把问题续写成别的内容，出现错误算术和事实陈述，多轮上下文利用不稳定。
- 曾记下 1→2 epoch 的重建准确率约 41%→44%，但对应的评测数据和日志没有发布。

训练时只遮住部分 token、推理时从全 mask 开始，可能带来分布差异。模型规模、数据、训练和采样设置也都值得检查。现有记录不足以确定瓶颈，更不能证明扩大模型就能解决对话问题。

### 7.4 VLM：学会格式，学不会看图（多模态侧现状）

视觉路径将编码后的图片填入 `<|image_pad|>`，这些位置不参与掩码；训练分为 projector 对齐和部分语言模型层解冻两阶段。实现见[视觉模型](model/model_dlm_v.py)、[对齐训练](trainer/train_pretrain_vlm.py)和 [SFT](trainer/train_sft_vlm.py)。

[找回的原始训练日志](docs/evidence/vlm-sft-20260811-20260814.log)最后记录的是 `step 22690/22691, loss 2.1442`，随后打印了保存 `vlm_sft_768.pth` 的消息。首末训练指标时间为 `2026-08-11 20:54:07` 与 `2026-08-14 02:40:07`，相隔 **53 小时 46 分钟**。[旧 README](https://github.com/adenzhou1350/minimind-diffusion/blob/24a64b72f876fcc5e3aad2c7ffe0b08277f6914c/README.md)中的“约 30 小时”不能由此支持；日志跨度也不是 GPU 利用时间或当前配置的耗时承诺。

这份日志没有 checkpoint 哈希、完整命令、数据指纹或源码版本；末条训练指标和保存消息不能证明逐步核验了全部 22,691 步，也不能确认权重身份或可取得性。[文件校验与保留的原始残行](docs/evidence/README.md)另有说明。

当时观察到：输出具备 ChatML 和描述图片的格式，但换图后仍出现相似幻觉；保留图片占位符却不传图时，会出现重复输出。公开材料没有逐图预测或换图、去图对照集，所以这里保留的是失败观察，不能量化 grounding 能力。

“视觉 token 更像格式触发器”是待验证的解释。后续应先检查图片替换是否改变输出、projector 梯度和冻结设置，再讨论模型与数据规模。

### 7.5 其他实现与实验边界

- **mask 初始化**：[screen_init.py](screen_init.py)包含 random / mean / zero 三组设置；历史上记为未见改善，但没有对应原始输出。
- **文本数据加载**：[JSONL 索引](dataset/lm_dataset.py)使用 stdlib json 与字节偏移；图文的 [parquet 加载](dataset/vlm_dataset.py)按行组读取。
- **仓库不包含**：训练权重、数据集、DPO / LoRA / RL / 蒸馏流程。先从[无需下载模型的测试](#先跑一个不用下载模型的例子)读懂机制，再准备训练资产。

---

## 8. 项目结构
```
model/
  model_dlm.py         # DLMConfig + 双向 transformer + DLMForMD(掩码+loss+generate)
  model_dlm_v.py       # DLMVLMConfig + MMVisionProjector + DLMForVLM(继承 DLMForMD 加 vision 路径)
  tokenizer_loader.py  # 复用 minimind BPE + <mask>(<|image_pad|> 用预留 id 12)
dataset/
  lm_dataset.py        # PretrainDataset + SFTDataset(文本)
  vlm_dataset.py       # PretrainVLMDataset + SFTVLMDataset(parquet+PIL,行组流式读)
  dataset.md           # 数据放置说明
trainer/
  trainer_utils.py     # get_lr / SkipBatchSampler / init_model / init_vlm_model / freeze_vlm / vlm_checkpoint
  train_pretrain.py    # 全序列随机掩码 + 均匀权重 loss
  train_sft.py         # [prompt;response] 只掩 response
  train_pretrain_vlm.py # Stage1 对齐:LLM 全冻结,只训 projector
  train_sft_vlm.py     # Stage2 SFT:LLM 首尾层 + projector,每 1000 步中途存盘
scripts/
  web_demo.py          # Streamlit 流式扩散采样(文本)
  web_demo_vlm.py      # Streamlit 流式扩散采样(图+文)
eval_dlm.py            # 中文 prompt + tokens/s(对照 minimind eval_llm.py)
eval_dlm_vlm.py        # 图+中文 prompt 扩散采样(对照 minimind-v eval)
chat_dlm.py            # 交互式多轮对话
screen_*.py            # 训练配方筛选脚本(loss 权重 / mask 初始化 / 规模,见 §7.1)
tests/                 # pytest: model / loss / sampling / dataset / vlm / trainer_utils
```

---

## 9. 致谢
- [minimind](https://github.com/jingyaogong/minimind) — 极简风格与"从 0 训练"哲学的本源。
- [minimind-v](https://github.com/jingyaogong/minimind-v) — 多模态扩展的"LLM 继承 + 极简 vision 路径"范式。
- [LLaDA](https://github.com/ML-GSAI/LLaDA)(Nie et al., 2025, arXiv:2502.09992)— 掩码扩散语言模型的方法与官方实现。

## License
Apache-2.0(同 minimind)。
