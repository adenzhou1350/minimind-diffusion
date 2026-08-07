# tests/test_vlm.py
import torch
from unittest.mock import MagicMock
from model.tokenizer_loader import IMAGE_PAD_TOKEN, IMAGE_PAD_ID
from model.model_dlm_v import DLMVLMConfig, MMVisionProjector, DLMForVLM


def test_image_pad_constants():
    assert IMAGE_PAD_TOKEN == '<|image_pad|>'
    assert IMAGE_PAD_ID == 6401  # minimind 6400 + <mask>(6400) + <image_pad>(6401)


def test_vlm_config_defaults():
    c = DLMVLMConfig()
    assert c.image_hidden_size == 768
    assert c.image_token_len == 64
    assert c.image_pad_token_id == 6401
    assert c.freeze_vision is True
    assert c.projector_hidden == 768
    assert c.vision_encoder_name == 'jingyaogong/siglip2-base-p32-256-ve'
    # vocab 应是 6402(minimind 6400 + <mask> + <image_pad>)
    assert c.vocab_size == 6402


def test_projector_shape():
    proj = MMVisionProjector(in_dim=768, out_dim=768, mid=768)
    x = torch.randn(2, 64, 768)  # [B, 64 tokens, 768]
    y = proj(x)
    assert y.shape == (2, 64, 768)


def _small_vlm():
    return DLMVLMConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                        num_key_value_heads=2, vocab_size=100, max_position_embeddings=128,
                        mask_token_id=99, image_pad_token_id=98, image_token_len=8,
                        image_hidden_size=64)


def test_dlmforvlm_forward_with_mock_vision():
    cfg = _small_vlm()
    m = DLMForVLM(cfg)
    # mock vision encoder(不真下权重)
    feat = torch.randn(1, 8, 64)  # [B, image_token_len, image_hidden_size]
    m.vision_encoder = MagicMock()
    m.vision_encoder.return_value = MagicMock(last_hidden_state=feat)

    # input_ids 含 8 个 image_pad(98)+ 文本
    ids = torch.tensor([[1, 98, 98, 98, 98, 98, 98, 98, 98, 5, 6, 7, 2]])
    attn = torch.ones_like(ids)
    pixel = torch.randn(1, 3, 256, 256)
    out = m(input_ids=ids, attention_mask=attn, labels=ids, pixel_values=pixel)
    assert out.loss.dim() == 0
    assert torch.isfinite(out.loss)


def test_image_pad_positions_never_masked():
    """vision 占位符位不在 mask 里 -> 改变 vision embedding 不应让 vision 位进 loss。
    用固定种子验证:换图后 loss 会变(双向注意力,vision 影响文本 hidden),
    但 vision 位本身从未被掩(无法直接断言,这里只验证 forward 稳定)。"""
    cfg = _small_vlm()
    m = DLMForVLM(cfg)
    feat1 = torch.randn(1, 8, 64)
    feat2 = torch.randn(1, 8, 64)
    m.vision_encoder = MagicMock()
    ids = torch.tensor([[1, 98, 98, 98, 98, 98, 98, 98, 98, 5, 6, 7, 2]])
    attn = torch.ones_like(ids)
    pixel = torch.randn(1, 3, 256, 256)

    torch.manual_seed(0)
    m.vision_encoder.return_value = MagicMock(last_hidden_state=feat1)
    out1 = m(input_ids=ids, attention_mask=attn, labels=ids, pixel_values=pixel)

    torch.manual_seed(0)
    m.vision_encoder.return_value = MagicMock(last_hidden_state=feat2)
    out2 = m(input_ids=ids, attention_mask=attn, labels=ids, pixel_values=pixel)
    # 换图后 loss 应变(vision 通过双向注意力影响文本 hidden state)——证明 vision 真接进去了
    assert not torch.allclose(out1.loss, out2.loss, atol=1e-5), "vision embedding 应影响 loss(双向注意力)"


def test_generate_with_image_runs():
    """带图的 generate 跑完,response 全揭开(无 <mask> 残留)。"""
    torch.manual_seed(0)
    cfg = _small_vlm()
    m = DLMForVLM(cfg)
    m.vision_encoder = MagicMock()
    m.vision_encoder.return_value = MagicMock(last_hidden_state=torch.randn(1, 8, 64))
    # 训几步让 argmax 不塌缩到 <mask>(同 mind-diffusion test_sampling 经验)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=3e-3)
    m.train()
    ids = torch.randint(0, 99, (2, 16))
    attn = torch.ones(2, 16, dtype=torch.long)
    for _ in range(50):
        out = m(input_ids=ids, attention_mask=attn, labels=ids)  # 无图
        out.loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
    m.eval()
    prompt = torch.tensor([[1, 98, 98, 98, 98, 98, 98, 98, 98, 5, 6, 7]])  # 含 8 个 image_pad
    pixel = torch.randn(1, 3, 256, 256)
    out = m.generate(prompt, gen_length=8, steps=4, pixel_values=pixel)
    assert out.shape == (1, 8)
    assert (out != 99).all(), "response 应全揭开(无 <mask> 残留)"

