"""
数据集类:返回干净序列 + 掩码范围(掩码本身在 model.forward 里现采,每步随机)
"""
import torch
from torch.utils.data import Dataset
from datasets import load_dataset


class PretrainDataset(Dataset):
    """预训练:返回 (input_ids, attention_mask)。labels=自己(forward 里当作 x_0)。"""

    def __init__(self, data_path, tokenizer, max_length=340):
        self.data = load_dataset('json', data_files=data_path, split='train')
        self.tok = tokenizer
        self.max_length = max_length
        self.bos = tokenizer.bos_token_id or 1
        self.eos = tokenizer.eos_token_id or 2

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        text = self.data[i].get('text', '') or self.data[i].get('content', '')
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
        self.data = load_dataset('json', data_files=data_path, split='train')
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        row = self.data[i]
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
