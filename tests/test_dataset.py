# tests/test_dataset.py
import json
import torch
from dataset.lm_dataset import PretrainDataset, SFTDataset, _JsonlIndex


def _fake_tok():
    class FakeTok:
        bos_token_id = 1
        eos_token_id = 2
        pad_token_id = 0
        def __call__(self, text, **kw):
            return {'input_ids': [10, 20, 30]}
    return FakeTok()


def test_pretrain_dataset_shape(tmp_path):
    p = tmp_path / 't.jsonl'
    p.write_text('{"text": "hello"}\n{"text": "world"}\n', encoding='utf-8')
    ds = PretrainDataset(str(p), _fake_tok(), max_length=16)
    ids, attn = ds[0]
    assert ids.shape == (16,)
    assert attn.shape == (16,)
    assert ids[0] == 1  # bos
    assert ids[1] == 10  # content first token (mock tok returns [10,20,30])
    assert ids[4] == 2  # eos (bos + 3 content + eos)
    assert attn[5] == 0  # pad


def test_sft_dataset_response_mask(tmp_path):
    p = tmp_path / 's.jsonl'
    p.write_text(json.dumps({'conversations': [
        {'from': 'user', 'value': 'hi'},
        {'from': 'assistant', 'value': 'yo'},
    ]}) + '\n', encoding='utf-8')
    ds = SFTDataset(str(p), _fake_tok(), max_length=32)
    ids, attn, resp = ds[0]
    assert ids.shape == (32,) and attn.shape == (32,) and resp.shape == (32,)
    assert resp.sum() > 0, "assistant 位应被标"


def test_jsonl_index_random_access(tmp_path):
    p = tmp_path / 'm.jsonl'
    p.write_text('{"i":0}\n{"i":1}\n{"i":2}\n', encoding='utf-8')
    idx = _JsonlIndex(str(p))
    assert len(idx) == 3
    assert idx.get(0)['i'] == 0
    assert idx.get(2)['i'] == 2
