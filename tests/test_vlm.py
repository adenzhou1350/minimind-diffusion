# tests/test_vlm.py
from model.tokenizer_loader import IMAGE_PAD_TOKEN, IMAGE_PAD_ID


def test_image_pad_constants():
    assert IMAGE_PAD_TOKEN == '<|image_pad|>'
    assert IMAGE_PAD_ID == 6401  # minimind 6400 + <mask>(6400) + <image_pad>(6401)
