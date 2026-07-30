"""
数据集类:返回干净序列 + 掩码范围(掩码本身在 model.forward 里现采,每步随机)

用 stdlib json 直接读 jsonl(字节偏移索引,流式),不依赖 datasets/pandas/pyarrow
—— 更轻、零 C 依赖、避免 pyarrow 在某些 Python 版本上的 import 崩溃,也更贴 minimind 最小实现精神。
"""
import json
import torch
from torch.utils.data import Dataset


class _JsonlIndex:
    """建一次性的行字节偏移索引,支持 O(1) 随机访问、O(1) 内存。"""

    def __init__(self, path):
        self.path = path
        self.offsets = []  # 每行起始字节偏移
        with open(path, 'rb') as f:
            off = 0
            for line in f:
                self.offsets.append(off)
                off += len(line)
        self._f = None

    def _file(self):
        if self._f is None:
            self._f = open(self.path, 'r', encoding='utf-8')
        return self._f

    def get(self, i):
        f = self._file()
        f.seek(self.offsets[i])
        return json.loads(f.readline())

    def __len__(self):
        return len(self.offsets)


class PretrainDataset(Dataset):
    """预训练:返回 (input_ids, attention_mask)。labels=自己(forward 里当作 x_0)。"""

    def __init__(self, data_path, tokenizer, max_length=340):
        self.data = _JsonlIndex(data_path)
        self.tok = tokenizer
        self.max_length = max_length
        self.bos = tokenizer.bos_token_id or 1
        self.eos = tokenizer.eos_token_id or 2

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        row = self.data.get(i)
        text = row.get('text', '') or row.get('content', '')
        ids = self.tok(text, add_special_tokens=False)['input_ids'][: self.max_length - 2]
        ids = [self.bos] + ids + [self.eos]
        attn = [1] * len(ids)
        # 右 pad 到 max_length
        pad = self.max_length - len(ids)
        if pad > 0:
            pad_id = self.tok.pad_token_id or 0
            ids = ids + [pad_id] * pad
            attn = attn + [0] * pad
        return torch.tensor(ids, dtype=torch.long), torch.tensor(attn, dtype=torch.long)


class SFTDataset(Dataset):
    """SFT:返回 (input_ids, attention_mask, response_mask)。response_mask 只标 assistant 回答位。"""

    # minimind chat template: <|im_start|>role\ncontent<|im_end|>
    IM_START = '<|im_start|>'
    IM_END = '<|im_end|>'

    def __init__(self, data_path, tokenizer, max_length=512):
        self.data = _JsonlIndex(data_path)
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        row = self.data.get(i)
        messages = row.get('conversations') or row.get('messages') or []
        ids, resp_mask = [], []
        for msg in messages:
            role = msg.get('from', msg.get('role', 'user'))
            content = msg.get('value', msg.get('content', ''))
            chunk = f'{self.IM_START}{role}\n{content}{self.IM_END}'
            chunk_ids = self.tok(chunk, add_special_tokens=False)['input_ids']
            is_resp = role == 'assistant'
            ids += chunk_ids
            resp_mask += [1 if is_resp else 0] * len(chunk_ids)
        ids = ids[: self.max_length]
        resp_mask = resp_mask[: self.max_length]
        attn = [1] * len(ids)
        pad = self.max_length - len(ids)
        if pad > 0:
            pad_id = self.tok.pad_token_id or 0
            ids = ids + [pad_id] * pad
            attn = attn + [0] * pad
            resp_mask = resp_mask + [0] * pad
        return (torch.tensor(ids, dtype=torch.long),
                torch.tensor(attn, dtype=torch.long),
                torch.tensor(resp_mask, dtype=torch.long))
