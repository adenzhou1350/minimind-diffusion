# tests/test_vlm_dataset.py
import io, json
import torch
from PIL import Image
from dataset.vlm_dataset import PretrainVLMDataset, SFTVLMDataset


def _fake_tok():
    """假 tokenizer:<image> 展开成 N 个 image_pad token(98),其余每字一 token。"""
    class FakeTok:
        bos_token_id = 1
        eos_token_id = 2
        pad_token_id = 0
        image_pad_token = '<|image_pad|>'
        def __call__(self, text, **kw):
            # <|image_pad|> 是 13 字符,作单 token(id 98);其余每字一 token(10)
            PAD = '<|image_pad|>'  # len 13
            tokens = []
            i = 0
            while i < len(text):
                if text[i:i+len(PAD)] == PAD:
                    tokens.append(98); i += len(PAD)
                else:
                    tokens.append(10); i += 1
            return {'input_ids': tokens}
        def convert_tokens_to_ids(self, t):
            return 98 if 'image_pad' in t else 10
    return FakeTok()


def _make_parquet(path, rows):
    import pyarrow as pa, pyarrow.parquet as pq
    tbl = pa.table({
        'conversations': [r['conversations'] for r in rows],
        'image_bytes': [r['image_bytes'] for r in rows],
    })
    pq.write_table(tbl, path)


def _img_bytes():
    img = Image.new('RGB', (256, 256), 'red')
    buf = io.BytesIO(); img.save(buf, 'JPEG'); return buf.getvalue()


def test_pretrain_vlm_dataset(tmp_path):
    convs = json.dumps([
        {'role': 'user', 'content': '<image>\n请描述这张图片'},
        {'role': 'assistant', 'content': '一只红色的猫'},
    ])
    rows = [{'conversations': convs, 'image_bytes': _img_bytes()}]
    pq_path = str(tmp_path / 't.parquet')
    _make_parquet(pq_path, rows)
    ds = PretrainVLMDataset(pq_path, _fake_tok(), max_length=128, image_token_len=4)
    item = ds[0]
    assert len(item) == 4  # (ids, attn, img_pad_mask, pixel)
    ids, attn, imgpad, pixel = item
    assert ids.shape == (128,)
    assert attn.shape == (128,)
    assert imgpad.shape == (128,)
    assert pixel.shape == (3, 256, 256)
    assert imgpad.sum() == 4, f"应有 4 个 image_pad 位,实得 {imgpad.sum()}"


def test_sft_vlm_dataset_response_mask(tmp_path):
    convs = json.dumps([
        {'role': 'user', 'content': '<image>\n这张图是什么'},
        {'role': 'assistant', 'content': '是一只猫'},
    ])
    rows = [{'conversations': convs, 'image_bytes': _img_bytes()}]
    pq_path = str(tmp_path / 's.parquet')
    _make_parquet(pq_path, rows)
    ds = SFTVLMDataset(pq_path, _fake_tok(), max_length=128, image_token_len=4)
    item = ds[0]
    assert len(item) == 5  # (ids, attn, resp, img_pad_mask, pixel)
    ids, attn, resp, imgpad, pixel = item
    assert ids.shape == (128,) and resp.shape == (128,) and imgpad.shape == (128,)
    assert resp.sum() > 0, "assistant 位应被标"
    assert imgpad.sum() == 4
    assert pixel.shape == (3, 256, 256)
