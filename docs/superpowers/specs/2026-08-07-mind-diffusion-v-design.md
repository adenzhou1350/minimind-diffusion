# mind-diffusion-v — Diffusion 版 minimind-v 设计文档

> 把 minimind-v（多模态 minimind）改成 diffusion 版,跟之前 mind-diffusion
> 把 minimind 改成 diffusion 版是同一个精神。文本侧复用已有的掩码扩散 LM
> (LLaDA v1 风格,均匀权重 + t~U(0.1,0.5)),vision 侧加一个冻结的 SigLIP2
> encoder + MLP projector,vision token 以"观测条件"形式拼进文本序列(永不掩)。

- **状态**:设计阶段,待实现
- **日期**:2026-08-07
- **项目根**:`D:\codes\mind-diffusion`(在现有 mind-diffusion repo 上扩展)
- **前置**:mind-diffusion spec #1 已完成(pretrain/SFT/采样跑通,base "能说不能对话")

---

## 1. 目标与非目标

### 目标
1. **讲明白原理**:README + `model_dlm_v.py` 注释把"多模态掩码扩散"讲到能跟 minimind-v
   讲 VLM 一样清楚——vision token 是观测条件(永不掩/永不去噪),扩散只生成文本。
2. **跑通**:单 GPU(8GB RTX 4060)上,vision encoder → projector 对齐 → SFT →
   给图采样出**跟图相关的中文 token** 全链路通。
3. **风格 parity**:镜像 minimind-v 的"继承文本 LLM + <50 行 vision 路径"极简风格,
   单文件 `model_dlm_v.py`,一 stage 一 `train_*_vlm.py`,复用现有 DLM 的 forward/loss/generate。

### 非目标
- 真正"看图对话"质量——文本 base 已经"能说不能对话"(规模墙,见 mind-diffusion spec §6.3),
  多模态 SFT 大概率也是"能出跟图相关的中文、不能稳定看图对话"。不追质量,追原理+跑通。
- cross-attention 注入(列为备选,见 §10);本 spec 用 LLaVA 前缀拼接。
- DPO/LoRA/RL(留后续)。

---

## 2. 背景:minimind-v 怎么做多模态(写给读 spec 的人)

minimind-v(`jingyaogong/minimind-v`)**继承自文本 minimind LLM**,<50 行改动:
- **vision encoder**:SigLIP2 ViT-B/32(`jingyaogong/siglip2-base-p32-256-ve`,~95M,冻结),
  256px 输入,patch 32 → **8×8=64 个 image token**,768-dim 输出。
- **projector**:LLaVA-1.5 式 MLP(LayerNorm → Linear(768→768) → GELU → Linear(768→768),~1M)。
- **注入**:LLaVA 前缀拼接(无 cross-attn)。文本 prompt 里有 64 个 `<|image_pad|>` 占位符;
  forward 里 vision encoder 出 64 token → projector → **替换占位符位置的 embedding**。
  下游(RoPE/transformer/lm_head)不变。vision token 永不掩。
- **训练**:Stage 1 对齐(LLM 全冻结,只训 projector,lr 4e-4)→ Stage 2 SFT(解冻 LLM 首尾层 +
  projector,lr 5e-6)。vision encoder 全程冻结,checkpoint 里 strip 掉 vision 权重。
- **数据**:`jingyaogong/minimind-v_dataset`(parquet,image_bytes 内联 256×256 JPEG,
  pretrain_i2t 1.27M 对 + sft_i2t 2.9M 对,中英对照)。

**对扩散 LM 的好处**:LLaVA 前缀拼接对掩码扩散比对 AR 更干净——64 个 vision token 是
**观测条件**(像 LLaDA 的 prompt),永不掩、永不去噪,扩散 loss 只掩文本段。直接复用现有 DLM。

---

## 3. 架构:`model/model_dlm_v.py`

单文件,镜像 minimind-v 的"继承 + <50 行"。

### 3.1 类清单
| 类 | 基 | 作用 |
|---|---|---|
| `DLMVLMConfig` | `DLMConfig` | 加 vision 字段(image_hidden_size=768,image_token_len=64,image_pad_token_id,freeze_vision=True) |
| `MMVisionProjector` | `nn.Module` | LayerNorm → Linear(768→768) → GELU → Linear(768→768),~1M |
| `DLMForVLM` | `DLMForMD` | 包装父类;加 vision_encoder + projector;forward 里往 `<|image_pad|>` 位塞 vision embedding;generate 继承 |

> 命名对齐 minimind-v(`MiniMindVLM` 继承 `MiniMindForCausalLM`),但因为是扩散,叫 `DLMForVLM`。

### 3.2 DLMVLMConfig 默认(对齐 mind-diffusion)
```python
hidden_size = 768            # 继承
num_hidden_layers = 8        # 继承
# ... 其余继承 DLMConfig(768/8/8 GQA, vocab 6401, mask_token_id=6400 ...)
image_hidden_size = 768      # SigLIP2 输出维度
image_token_len = 64          # 8×8 patch
image_pad_token_id = 6402    # <|image_pad|>,vocab 6401 → 6402(resize embedding)
freeze_vision = True          # vision encoder 全程冻结
projector_hidden = 768        # MLP 中间维度
# vision_encoder_name = 'jingyaogong/siglip2-base-p32-256-ve'
```
> vocab 6400(minimind)→ 6401(<mask>,mind-diffusion)→ **6402(<|image_pad|>)**。
> tied embedding 同步 resize。

### 3.3 vision 路径(forward 关键改动,仅此一段)
```python
def forward(self, input_ids, attention_mask=None, response_mask=None, labels=None,
            pixel_values=None, image_pad_token_id=None):
    # pixel_values: [B, 3, 256, 256] 或 None(纯文本 batch)
    if pixel_values is not None:
        # 1. vision encoder 出 64 token(冻结,no_grad)
        with torch.no_grad():
            vis = self.vision_encoder(pixel_values).last_hidden_state  # [B, 64, 768]
        # 2. projector
        vis = self.projector(vis)  # [B, 64, 768]
    # 3. 取 input_ids embedding(手动 embed,因为要替换占位符)
    x = self.model.embed(input_ids)  # [B, L, H]
    if pixel_values is not None:
        # 4. 找 <|image_pad|> 占位符位,替换 embedding
        pad_mask = (input_ids == image_pad_token_id)  # [B, L]
        # 每条序列的 64 个占位符位,用 vis 替换
        x[pad_mask] = vis.reshape(-1, vis.shape[-1])
    # 5. 后续完全不变:双向 transformer + 掩码 + 均匀权重 CE
    # (掩码只作用在文本位;vision 占位符位若被标 maskable 也要排除——见 §4)
    ...  # 调父类逻辑,或重写 embed 之后的步骤
```

**关键**:vision 占位符位(`image_pad_token_id`)在 `maskable` 里要**排除**(永不掩)——
它是观测条件,不是生成目标。跟 SFT 的 prompt 位一样处理。

### 3.4 generate 继承
prompt 里带 64 个 `<|image_pad|>`(由 tokenizer/dataset 填好),forward 时这些位被
vision embedding 替换。`generate` 直接继承 `DLMForMD.generate`,因为采样循环里:
- `is_prompt` 标记 prompt + vision 占位符位永不重掩(已含 `<|image_pad|>`)
- 扩散只作用在 response 的 `<mask>` 位

> 小坑:`generate` 里 `is_prompt = x[:, :P]` 需要包含 vision 占位符位——P 是 prompt
> 总长(含 64 个 image_pad),这部分逻辑继承即可,因为 image_pad 在 prompt 段。

---

## 4. 数据流与训练 loss

### 4.1 数据集(`dataset/vlm_dataset.py`)
parquet + image_bytes 内联(不依赖 datasets 库的 image decode,用 pyarrow + PIL)。

| 类 | 返回 | 说明 |
|---|---|---|
| `PretrainVLMDataset` | `(input_ids, attention_mask, image_pad_mask, pixel_values)` | pretrain_i2t,prompt="<image>\n请描述这张图片",response=caption |
| `SFTVLMDataset` | `(input_ids, attention_mask, response_mask, image_pad_mask, pixel_values)` | sft_i2t,multi-turn,image_pad 在 user turn |

- `input_ids`:含 64 个 `<|image_pad|>` 占位符(在 `<image>` 标记位置展开)
- `image_pad_mask`:bool,标哪些位是 vision 占位符(forward 里排除出 maskable)
- `pixel_values`:[3,256,256] 归一化张量(SigLIP processor)
- 纯文本 turn(image=8×8 黑图占位,跟 minimind-v 一致)保证 projector 在无图时也稳

### 4.2 forward(掩码 + loss,改 §3.3 的 forward)
```python
# maskable 排除 vision 占位符位
maskable = attention_mask.bool() & (~image_pad_mask)
if response_mask is not None:
    maskable = maskable & response_mask.bool()
# 其余完全继承 DLMForMD:t~U(0.1,0.5) 伯努利掩码 + 均匀权重 CE
# vision 占位符位的 embedding 已是 vis(观测条件),不掩、不进 loss
```

### 4.3 两阶段训练(镜像 minimind-v)

**Stage 1 对齐**(`train_pretrain_vlm.py`):
- vision encoder:冻结
- LLM(DLMModel + lm_head):**全冻结**(`requires_grad=False`)
- projector:**训**
- lr 4e-4,pretrain_i2t(1.27M 对),1 epoch(或子集冒烟)

**Stage 2 SFT**(`train_sft_vlm.py`):
- vision encoder:冻结
- LLM:**解冻首尾层**(第 0 层 + 最后一层 + final norm),其余冻结
- projector:训
- lr 5e-6,sft_i2t(2.9M 对),1 epoch(或子集冒烟)

> 冻结策略实现:遍历 `model.named_parameters()`,按名字/层级设 `requires_grad`。
> minimind-v 的 `freeze_llm` 模式:2=全冻结(对齐),1=首尾层,0=全解冻。

---

## 5. 推理/采样

`eval_dlm_vlm.py` + `scripts/web_demo_vlm.py`:
- 加载图 → SigLIP processor → pixel_values
- prompt = `<|im_start|>user\n<image>\n请描述这张图片<|im_end|><|im_start|>assistant\n`
  (`<image>` 展开成 64 个 `<|image_pad|>`)
- forward 填 vision embedding → 扩散采样(继承 `DLMForMD.generate`,vision 位永不重掩)
- decode response,遇 EOS 截断

采样参数继承 mind-diffusion:`temp=0.7, rep=1.3, steps=128`。

---

## 6. 项目结构(新增/修改)
```
model/
  model_dlm_v.py        # 新:DLMVLMConfig + MMVisionProjector + DLMForVLM
  tokenizer_loader.py  # 改:加 <|image_pad|> token(vocab 6402)
  model_dlm.py         # 不改(DLMForMD.forward 支持 image_pad_mask 排除)
dataset/
  vlm_dataset.py       # 新:PretrainVLMDataset + SFTVLMDataset(parquet + PIL)
trainer/
  train_pretrain_vlm.py # 新:Stage 1 对齐
  train_sft_vlm.py     # 新:Stage 2 SFT
  trainer_utils.py     # 改:init_model 支持 vlm 权重 + 冻结策略
scripts/
  web_demo_vlm.py      # 新:传图 + 文本,流式扩散
eval_dlm_vlm.py        # 新:给图 + prompt,生成描述
tests/
  test_vlm.py          # 新:vision 路径 + image_pad_mask 排除 + forward shape
```

### 6.1 风格 parity(对齐 minimind-v + mind-diffusion)
- [ ] `model_dlm_v.py` 注释英文为主 + 🌏 边框
- [ ] `train_*_vlm.py` 用 `# ========== N. <中文标题> ==========` 双语分隔符
- [ ] argparse `help=` 中文
- [ ] 一行余弦 `get_lr` 复用
- [ ] parquet 读取用 pyarrow + PIL(不依赖 datasets 库,避免 py3.14 pyarrow 崩溃——
      实测 mind-diffusion 已踩过,直接用 stdlib + pyarrow 直读)
- [ ] checkpoint strip vision encoder 权重(只存 LLM + projector,跟 minimind-v 一致)

---

## 7. 测试与"跑通"验收

### 7.1 单元/冒烟(`tests/test_vlm.py`)
- `test_vlm_forward_shape`:带 pixel_values 的 forward 出 loss 标量 + logits 形状对
- `test_image_pad_mask_excludes_vision`:vision 占位符位不被掩、不进 loss
  (固定种子,改 vision embedding 不影响文本 loss——不,会影响,因为双向;
   改为:vision 占位符位不在 mask 里,且 loss 不计入)
- `test_projector_trains_stage1`:Stage 1 只 projector 有 grad,LLM 无 grad
- `test_generate_with_image`:带图的 generate 跑完,vision 位保持(未被重掩)

### 7.2 端到端"跑通"
1. vision encoder 加载 + 图预处理 OK
2. Stage 1 对齐跑 ≥N 步,loss 下降,projector 学到(检查:给同一图,projector 输出稳定)
3. Stage 2 SFT 跑 ≥N 步,loss 下降
4. `eval_dlm_vlm.py`:给几张图,生成**跟图相关的中文 token**(不强求准确 caption,
   只求"图变了,生成内容也变"——证明 vision 真在起作用)
5. `web_demo_vlm.py`:传图 + 文本,流式扩散采样

**"跑通"最低门槛**:1-4 步全绿。质量(准确 caption)是加分项,不作为硬验收
(同 mind-diffusion,文本 base 规模墙会带到 VLM)。

### 7.3 验证 vision 是否真起作用(关键诊断)
对照实验:给同一 prompt,换不同图,看生成的 response 是否不同。
- 若不同 → vision 在起作用 ✓
- 若相同 → vision 没接进去(projector 没训好 / forward 没填对)✗

---

## 8. 已知风险 / 诚实标注
1. **Python 3.14 + SigLIP 加载**:transformers 对 3.14 支持未验证。若 `SiglipVisionModel`
   import 失败,回退 CLIP-B/32(`openai/clip-vit-base-patch32`,50 token,生态最稳),
   projector 改 768→768 不变(CLIPTVisionModel 也是 768-dim)。
2. **vision encoder 权重下载**:~189MB,可能撞 DLPPlus/限速(同 torch wheel 的坑)。
   若卡,用户手动下 + 本地路径加载。
3. **规模墙**:文本 base "能说不能对话"(mind-diffusion §6.3),多模态 SFT 大概率
   "能出跟图相关的中文、不能稳定看图对话"。诚实记,不假装能对话。
4. **parquet/pyarrow on py3.14**:mind-diffusion 已踩过 pyarrow import 崩溃,
   vlm_dataset 用 pyarrow 直读 parquet 也可能撞。备选:转 jsonl + 独立图片文件。
5. **序列长度**:64 vision token 吃 512 序列的 1/8。若 Stage 2 SFT 多轮对话超长,
   截断或减 vision token(用 stride pooling projector 64→32)。

---

## 9. 决策点(已拍板)
- **vision token 注入:LLaVA 前缀拼接**(非 cross-attn)。理由:贴 minimind-v、
  数学干净(vision 位不掩=观测条件)、复用现有 DLM、最快跑通。cross-attn 列为备选(§10)。
- **vision encoder:`jingyaogong/siglip2-base-p32-256-ve`**(minimind-v 同款)。
  CLIP-B/32 作 3.14/加载失败的后盾。
- **数据:minimind-v_dataset**(pretrain_i2t + sft_i2t)。
- **两阶段训练**:Stage 1 全冻结 LLM 训 projector;Stage 2 解冻 LLM 首尾层 + projector。
- **跑通标准:原理+跑通**,不追质量。

---

## 10. 备选(本 spec 不做,记录用)
- **cross-attention 注入**:给双向 transformer 加 cross-attn block(文本 query,
  vision key/value)。好处:vision 不占文本序列长度。代价:架构改动大、偏离 minimind-v。
  仅当 §8.5 序列长度成瓶颈时考虑。
- **vision token pooling**:projector 把 64 token 池化到 32/16,省序列长度。
- **LLaDA 2 block diffusion 采样**:块间自回归,vision 作为首块条件。留后续。

---

## 附录:与 minimind-v 的差异
| 维度 | minimind-v | mind-diffusion-v |
|---|---|---|
| 文本生成 | 自回归 | 掩码扩散(均匀权重) |
| vision 注入 | LLaVA 前缀 | LLaVA 前缀(同) |
| vision encoder | SigLIP2(冻结) | SigLIP2(冻结,同) |
| projector | MLP 768→768 | MLP 768→768(同) |
| 训练阶段 | 对齐 + SFT | 对齐 + SFT(同) |
| vision token 是否掩 | 否(观测) | 否(观测,同) |
| 文本 loss | next-token CE | 均匀权重掩码 CE |
| base 能力 | 能对话 | 能说不能对话(规模墙) |
