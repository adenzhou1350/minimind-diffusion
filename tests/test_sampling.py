import torch
from model.model_dlm import DLMConfig, DLMForMD


def _small():
    return DLMConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                     num_key_value_heads=2, vocab_size=100, max_position_embeddings=128,
                     mask_token_id=99, eos_token_id=2)


def _trained_small():
    """训了几步的模型,避免随机初始化下 logits 退化成全输出 <mask>。"""
    torch.manual_seed(0)
    m = DLMForMD(_small())
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    m.train()
    ids = torch.randint(0, 99, (2, 16))
    attn = torch.ones(2, 16, dtype=torch.long)
    for _ in range(50):
        out = m(input_ids=ids, attention_mask=attn, labels=ids)
        out.loss.backward()
        opt.step(); opt.zero_grad()
    return m.eval()


def test_generate_returns_gen_length():
    m = _trained_small()
    prompt = torch.randint(0, 99, (1, 4))
    out = m.generate(prompt, gen_length=8, steps=4, temperature=0.0)
    assert out.shape == (1, 8)


def test_generate_no_mask_tokens_left():
    m = _trained_small()
    prompt = torch.randint(0, 99, (1, 4))
    out = m.generate(prompt, gen_length=8, steps=4, temperature=0.0)
    assert (out != 99).all(), "所有 <mask> 应被揭开"


def test_generate_random_remasking_branch():
    m = _trained_small()
    prompt = torch.randint(0, 99, (1, 4))
    out = m.generate(prompt, gen_length=8, steps=4, low_confidence=False)
    assert out.shape == (1, 8)
    assert (out != 99).all()


def test_generate_temperature_positive_runs():
    m = _trained_small()
    prompt = torch.randint(0, 99, (1, 4))
    out = m.generate(prompt, gen_length=8, steps=4, temperature=1.0)
    assert out.shape == (1, 8)

