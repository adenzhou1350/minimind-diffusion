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
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig


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
                 image_pad_token_id=12,
                 bos_token_id=1,
                 eos_token_id=2,
                 use_moe=False,
                 **kwargs):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        # π-scaled + 64 对齐,跟 minimind 一致
        self.intermediate_size = intermediate_size or math.ceil(hidden_size * math.pi / 64) * 64
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.tie_word_embeddings = tie_word_embeddings
        self.dropout = dropout
        self.mask_token_id = mask_token_id
        self.image_pad_token_id = image_pad_token_id  # VLM 视觉占位符(mind-diffusion-v 用)
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.use_moe = use_moe  # 保留字段但不启用(跟 minimind 一致)
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


# 🌏🌎🌍 layers: RMSNorm + 双向 Attention(RoPE+GQA+QK-norm) + SwiGLU FFN + Block 🌏🌎🌍

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
        B, S, _ = x.shape
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


# 🌏🌎🌍 DLMForMD: 掩码 + 1/t 加权掩码 CE loss(time-free,不喂 t 给模型) 🌏🌎🌍
from dataclasses import dataclass
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput


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
            self._tie_or_clone_weights()
        self.post_init()

    def _tie_or_clone_weights(self):
        # transformers v5: PreTrainedModel.tie_weights() 接受 recompute_mapping 等参数,
        # 直接覆盖会破坏父类调用契约。这里只用最小方式绑定 lm_head.weight = embed.weight。
        self.lm_head.weight = self.model.embed.weight

    def forward(self, input_ids, attention_mask=None, response_mask=None, labels=None):
        x_0 = labels if labels is not None else input_ids      # 干净目标
        B, L = x_0.shape
        device = x_0.device
        MASK_ID = self.config.mask_token_id
        V = self.config.vocab_size

        # 1. 每序列采掩码比例 t。小模型上 U(0,1)+1/t 权重会让梯度被"少掩 easy case"主导、
        #    且极端 t(近 0 / 近 1)方差大难学;截断到 [0.1, 0.5] 稳定收敛
        #    (筛选实验:512/6 3k 步 U(0.1,0.5) 均匀权重 acc 25.5%, vs 1/t U(0,1) 仅 ~5%)
        t = torch.empty(B, device=device).uniform_(0.1, 0.5)

        # 2. 可掩范围:真实 token(非 pad);SFT 时再 & response_mask
        maskable = attention_mask.bool() if attention_mask is not None \
            else torch.ones(B, L, dtype=torch.bool, device=device)
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

        # 5. 均匀加权的掩码 CE(只在被掩位算)。
        #    原始 LLaDA 用 1/t 权重(似然上界),但小模型+少数据压不住其方差;
        #    均匀权重(BERT-style MLM)在小模型上收敛更稳,筛选实验证实。
        ce = F.cross_entropy(logits.view(-1, V), x_0.view(-1), reduction='none').view(B, L)
        ce = ce * mask
        n_masked = mask.sum(dim=1).clamp(min=1)               # [B]
        loss = (ce.sum(dim=1) / n_masked).mean()
        return DLMOutput(loss=loss, logits=logits)

    # 🌏🌎🌍 扩散采样: 全 <mask> -> 迭代 unmasking + low-confidence remasking 🌏🌎🌍
    @torch.inference_mode()
    def generate(self, prompt_ids, gen_length=128, steps=64, temperature=0.0,
                 low_confidence=True, repetition_penalty=1.2,
                 block_length=0, block_steps=None):
        """
        prompt_ids: [1, P] 干净 prompt
        返回: [1, gen_length] response token(<mask> 全部揭开)

        改进(针对小模型扩散采样的重复循环):
        - repetition_penalty: 对已在 response 揭开过的 token 降权,打散重复
          (扩散是双向、无生成历史,易重复高频模式;AR 靠 causal 自然不重复)
        - block_length: >0 时启用半自回归 block 生成(LLaDA 2 思路)——
          把 response 分成 gen_length/block_length 块,块内扩散,块间自回归:
          每块生成时能看到前面已揭开的块(作为干净上下文),后面就不会重复前面。
          block_steps: 每块的扩散步数(默认 = steps / num_blocks)。
        """
        device = prompt_ids.device
        MASK_ID = self.config.mask_token_id
        P = prompt_ids.shape[1]
        V = self.config.vocab_size

        # ---------- 半自回归 block 生成路径 ----------
        if block_length and block_length > 0:
            assert gen_length % block_length == 0, f'gen_length({gen_length}) 必须被 block_length({block_length}) 整除'
            num_blocks = gen_length // block_length
            if block_steps is None:
                assert steps % num_blocks == 0, f'steps({steps}) 必须被 num_blocks({num_blocks}) 整除'
                block_steps = steps // num_blocks
            # 已生成的块(干净 token),逐块累加;prompt 始终在前
            committed = prompt_ids                      # [1, P]
            for b in range(num_blocks):
                # 本块:全 <mask> 起步,在 [committed | 本块全mask] 上扩散采样
                blk = torch.full((1, block_length), MASK_ID, dtype=torch.long, device=device)
                x = torch.cat([committed, blk], dim=1)  # [1, P + b*L_b + L_b]
                attn = torch.ones_like(x)
                is_prompt = torch.zeros_like(x, dtype=torch.bool)
                is_prompt[:, :x.shape[1] - block_length] = True  # prompt + 前面已生成块 = 永不重掩
                self._diffuse_block(x, attn, is_prompt, block_length, block_steps,
                                    temperature, low_confidence, repetition_penalty, P)
                committed = x                            # 本块已揭开,并入"已生成"
            return committed[:, P:]                      # response 区

        # ---------- 全序列生成路径(原 LLaDA v1 语义) ----------
        resp = torch.full((1, gen_length), MASK_ID, dtype=torch.long, device=device)
        x = torch.cat([prompt_ids, resp], dim=1)
        attn = torch.ones_like(x)
        is_prompt = torch.zeros_like(x, dtype=torch.bool)
        is_prompt[:, :P] = True
        self._diffuse_block(x, attn, is_prompt, gen_length, steps,
                            temperature, low_confidence, repetition_penalty, P)
        return x[:, P:]

    def _diffuse_block(self, x, attn, is_prompt, block_len, T,
                       temperature, low_confidence, repetition_penalty, prompt_len):
        """对 x 里最后 block_len 个 <mask> 位做迭代 unmasking + low-conf remasking。
        is_prompt 标记永不重掩的位(prompt + 已生成的块);其余在 block 区。
        repetition_penalty: 对当前 response 区已揭开的 token 降权。
        """
        device = x.device
        MASK_ID = self.config.mask_token_id
        V = self.config.vocab_size
        for k in range(1, T + 1):
            s = 1.0 - k / T
            h = self.model(x, attention_mask=attn)
            logits = self.lm_head(h)                     # [1, S, V]
            masked = (x == MASK_ID) & (~is_prompt)
            idx = masked.nonzero(as_tuple=False)         # [N, 2]
            if idx.shape[0] == 0:
                break
            lm_logits = logits[idx[:, 0], idx[:, 1]]    # [N, V]

            # 重复惩罚:对当前已揭开的 response token 降权
            # (用整个 response 区——含前面已生成块——里出现过的 token id)
            resp_ids = x[0, prompt_len:]                 # [S_resp]
            seen = resp_ids[resp_ids != MASK_ID].unique()
            if repetition_penalty != 1.0 and seen.numel() > 0:
                # seen 里的 logit 除以 penalty(>1 降权;<0 的乘,保持符号)
                seen_logits = lm_logits[:, seen]
                seen_logits = torch.where(seen_logits > 0,
                                          seen_logits / repetition_penalty,
                                          seen_logits * repetition_penalty)
                lm_logits = lm_logits.scatter(1, seen.unsqueeze(0).expand(idx.shape[0], -1),
                                              seen_logits)

            temp = max(temperature, 1e-4)
            prob = F.softmax(lm_logits / temp, dim=-1)
            pred = prob.argmax(dim=-1)
            if low_confidence:
                conf = prob.gather(1, pred[:, None]).squeeze(1)
            else:
                conf = torch.rand(idx.shape[0], device=device)
            x[idx[:, 0], idx[:, 1]] = pred
            n_remain = int(block_len * s)
            n_remask = min(n_remain, idx.shape[0])
            if n_remask > 0:
                order = torch.argsort(conf)
                remask_pos = idx[order[:n_remask]]
                x[remask_pos[:, 0], remask_pos[:, 1]] = MASK_ID

