# tests/test_dataset.py
import torch
from unittest.mock import MagicMock
from dataset.lm_dataset import PretrainDataset, SFTDataset


def _fake_tok():
    tok = MagicMock()
    tok.bos_token_id = 1
    tok.eos_token_id = 2
    tok.pad_token_id = 0
    tok.return_value = {'input_ids': [10, 20, 30]}
    tok.side_effect = lambda text, **kw: {'input_ids': [10, 20, 30]}
    return tok


def test_pretrain_dataset_shape(monkeypatch, tmp_path):
    # 造一个临时 jsonl
    p = tmp_path / 't.jsonl'
    p.write_text('{"text": "hello"}\n{"text": "world"}\n')
    # mock load_dataset 返回简单列表
    import dataset.lm_dataset as mod
    monkeypatch.setattr(mod, 'load_dataset', lambda *a, **k: [{'text': 'hello'}, {'text': 'world'}])
    ds = PretrainDataset(str(p), _fake_tok(), max_length=16)
    ids, attn = ds[0]
    assert ids.shape == (16,)
    assert attn.shape == (16,)
    assert ids[0] == 1  # bos
    # mock tokenizer 对任意文本只吐 [10,20,30];bos + content(10,20,30) + eos
    assert ids[1] == 10  # content 首个 token
    assert ids[4] == 2  # eos(位置 1=bos, 2-4=content, 5=eos)
    assert attn[0] == 1 and attn[4] == 1 and attn[5] == 0  # pad 位 0



def test_sft_dataset_response_mask(monkeypatch, tmp_path):
    p = tmp_path / 's.jsonl'
    p.write_text('{}\n')
    import dataset.lm_dataset as mod
    monkeypatch.setattr(mod, 'load_dataset', lambda *a, **k: [{'conversations': [
        {'from': 'user', 'value': 'hi'},
        {'from': 'assistant', 'value': 'yo'},
    ]}])
    ds = SFTDataset(str(p), _fake_tok(), max_length=32)
    ids, attn, resp = ds[0]
    assert ids.shape == (32,) and attn.shape == (32,) and resp.shape == (32,)
    assert resp.sum() > 0, "assistant 位应被标"
