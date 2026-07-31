import torch
from model.model_dlm import DLMConfig, DLMForMD


def _small():
    return DLMConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                     num_key_value_heads=2, vocab_size=100, max_position_embeddings=128,
                     mask_token_id=99)


def test_forward_returns_scalar_loss():
    m = DLMForMD(_small())
    ids = torch.randint(0, 99, (2, 10))
    attn = torch.ones(2, 10, dtype=torch.long)
    out = m(input_ids=ids, attention_mask=attn, labels=ids)
    assert out.loss.dim() == 0  # scalar
    assert torch.isfinite(out.loss)


def test_pad_positions_excluded_from_loss():
    # 第 2 条后半是 pad。固定种子使两次 forward 的随机掩码完全一致,
    # 这样只有 pad 内容不同 -> 若 pad 不进 loss,两次 loss 必须相等。
    m = DLMForMD(_small())
    ids = torch.randint(0, 99, (2, 10))
    attn = torch.tensor([[1]*10, [1,1,1,1,1,0,0,0,0,0]])
    torch.manual_seed(0)
    out1 = m(input_ids=ids, attention_mask=attn, labels=ids)
    ids2 = ids.clone()
    ids2[1, 5:] = torch.randint(0, 99, (5,))  # 改 pad 区内容
    torch.manual_seed(0)
    out2 = m(input_ids=ids2, attention_mask=attn, labels=ids2)
    assert torch.allclose(out1.loss, out2.loss, atol=1e-5), "pad 位不应影响 loss"


def test_response_mask_restricts_masking_to_response():
    # response_mask 的契约:只有 response_mask=1 的位可被掩。
    # 验证:response_mask 全 0 -> 没有任何位可掩 -> mask 全 False -> loss 恰为 0。
    m = DLMForMD(_small())
    ids = torch.randint(0, 99, (1, 6))
    attn = torch.ones(1, 6, dtype=torch.long)
    resp_zero = torch.zeros(1, 6, dtype=torch.long)  # 无可掩位
    out_zero = m(input_ids=ids, attention_mask=attn, response_mask=resp_zero, labels=ids)
    assert out_zero.loss.item() == 0.0, "response_mask 全 0 时不应有掩码 -> loss 必为 0"
    # 对照:response_mask 全 1(等同 pretrain) -> 有掩码 -> loss 为正
    resp_one = torch.ones(1, 6, dtype=torch.long)
    torch.manual_seed(0)
    out_one = m(input_ids=ids, attention_mask=attn, response_mask=resp_one, labels=ids)
    assert out_one.loss.item() > 0.0, "response_mask 全 1 时应有掩码 -> loss > 0"



def test_no_nan_when_t_near_zero():
    # t 截断到 [0.1,0.5],无极端值,loss 始终有限
    m = DLMForMD(_small())
    ids = torch.randint(0, 99, (1, 8))
    attn = torch.ones(1, 8, dtype=torch.long)
    out = m(input_ids=ids, attention_mask=attn, labels=ids)
    assert torch.isfinite(out.loss)
