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
