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
