# tests/test_vlm.py
from model.tokenizer_loader import IMAGE_PAD_TOKEN, IMAGE_PAD_ID
from model.model_dlm_v import DLMVLMConfig


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
