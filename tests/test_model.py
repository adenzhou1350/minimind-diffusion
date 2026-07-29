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
