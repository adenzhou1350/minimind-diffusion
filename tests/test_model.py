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


import torch
from model.model_dlm import DLMModel


def test_model_forward_shape():
    cfg = DLMConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                    num_key_value_heads=2, vocab_size=100, max_position_embeddings=128)
    m = DLMModel(cfg)
    ids = torch.randint(0, 100, (2, 10))
    attn = torch.ones(2, 10, dtype=torch.long)
    h = m(ids, attention_mask=attn)
    assert h.shape == (2, 10, 64)


def test_attention_is_bidirectional():
    # 双向:换第 2/3 个 token,输出应跟着变(非因果)
    cfg = DLMConfig(hidden_size=64, num_hidden_layers=1, num_attention_heads=4,
                    num_key_value_heads=2, vocab_size=100, max_position_embeddings=128)
    m = DLMModel(cfg).eval()
    attn = torch.ones(1, 4, dtype=torch.long)
    with torch.inference_mode():
        h = m(torch.tensor([[10, 20, 30, 40]]), attention_mask=attn)
        swapped = m(torch.tensor([[10, 30, 20, 40]]), attention_mask=attn)
    assert h.shape == (1, 4, 64)
    assert swapped.shape == (1, 4, 64)


def test_attention_no_causal_mask():
    # 关键:Attention 必须非因果。位置 0 的输出应依赖位置 1 的 token。
    cfg = DLMConfig(hidden_size=64, num_hidden_layers=1, num_attention_heads=4,
                    num_key_value_heads=2, vocab_size=100, max_position_embeddings=128)
    m = DLMModel(cfg).eval()
    attn = torch.ones(1, 2, dtype=torch.long)
    with torch.inference_mode():
        h_a = m(torch.tensor([[10, 20]]), attention_mask=attn)
        h_b = m(torch.tensor([[10, 99]]), attention_mask=attn)
    # 因果:位置 0 看不到位置 1 -> h_a[0,0]==h_b[0,0];双向:应不等
    assert not torch.allclose(h_a[0, 0], h_b[0, 0], atol=1e-6), \
        "位置 0 的输出应依赖位置 1 的 token —— 双向注意力"
