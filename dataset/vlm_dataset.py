"""
VLM 数据集:parquet + image_bytes 内联。不依赖 datasets 库(py3.14 pyarrow import 崩溃)。
用 pyarrow 直读 + PIL 解码。input_ids 含 <|image_pad|> 占位符(image_token_len 个)。
"""
import io
import json
import torch
from torch.utils.data import Dataset
import pyarrow.parquet as pq
from PIL import Image


class _ParquetIndex:
    """pyarrow 直读 parquet,一次性读入内存(数据已压缩,可接受)。"""

    def __init__(self, path):
        self._table = pq.ParquetFile(path).read()

    def __len__(self):
        return self._table.num_rows

    def get(self, i):
        return {c: self._table.column(c)[i].as_py() for c in self._table.column_names}


def _decode_image(image_bytes, size=256):
    """JPEG bytes -> [3, H, W] tensor,SigLIP 归一化(mean/std=0.5)。"""
    if not image_bytes:
        return torch.zeros(3, size, size)
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB').resize((size, size))
    t = torch.tensor(list(img.getdata()), dtype=torch.float32).reshape(size, size, 3)
    t = t.permute(2, 0, 1) / 255.0
    return (t - 0.5) / 0.5


class _VLMBase(Dataset):
    """Pretrain/SFT 共用:渲染 conversations + 解码图。"""

    def __init__(self, path, tokenizer, max_length=512, image_token_len=64,
                 image_size=256, image_pad_token='<|image_pad|>'):
        self.data = _ParquetIndex(path)
        self.tok = tokenizer
        self.max_length = max_length
        self.image_token_len = image_token_len
        self.image_size = image_size
        self.image_pad = image_pad_token
        self.bos = tokenizer.bos_token_id or 1
        self.eos = tokenizer.eos_token_id or 2
        self.pad = tokenizer.pad_token_id or 0
        self._pad_id = None  # 惰性缓存 image_pad 的 token id

    @property
    def pad_id(self):
        if self._pad_id is None:
            self._pad_id = self.tok.convert_tokens_to_ids(self.image_pad)
        return self._pad_id

    def _expand_image(self, content):
        """把 <image> 替换成 image_token_len 个 <|image_pad|>。"""
        if '<image>' in content:
            content = content.replace('<image>', self.image_pad * self.image_token_len)
        return content

    def _render_chunks(self, convs):
        """渲染对话 -> [(chunk_ids, is_resp), ...]。"""
        if isinstance(convs, str):
            convs = json.loads(convs)
        chunks = []
        for c in convs:
            role = c.get('role', c.get('from', 'user'))
            content = self._expand_image(c.get('content', c.get('value', '')))
            chunk = f'<|im_start|>{role}\n{content}<|im_end|>'
            chunk_ids = self.tok(chunk, add_special_tokens=False)['input_ids']
            chunks.append((chunk_ids, role == 'assistant'))
        return chunks


class PretrainVLMDataset(_VLMBase):
    """pretrain_i2t:返回 (input_ids, attention_mask, image_pad_mask, pixel_values)。
    全序列可掩(prompt 不分出 response_mask);pretrain 是无条件掩码补全。"""

    def __getitem__(self, i):
        row = self.data.get(i)
        chunks = self._render_chunks(row['conversations'])
        ids = []
        for chunk_ids, _ in chunks:
            ids += chunk_ids
        ids = [self.bos] + ids[: self.max_length - 2] + [self.eos]
        attn = [1] * len(ids)
        img_pad = [1 if t == self.pad_id else 0 for t in ids]
        while len(ids) < self.max_length:
            ids.append(self.pad); attn.append(0); img_pad.append(0)
        ids = ids[: self.max_length]; attn = attn[: self.max_length]; img_pad = img_pad[: self.max_length]
        pixel = _decode_image(row.get('image_bytes'), self.image_size)
        return (torch.tensor(ids, dtype=torch.long),
                torch.tensor(attn, dtype=torch.long),
                torch.tensor(img_pad, dtype=torch.long),
                pixel)


class SFTVLMDataset(_VLMBase):
    """sft_i2t:返回 (input_ids, attention_mask, response_mask, image_pad_mask, pixel_values)。
    response_mask 只标 assistant 段(含其 <|im_end|>);prompt + vision 占位符不掩。"""

    def __getitem__(self, i):
        row = self.data.get(i)
        chunks = self._render_chunks(row['conversations'])
        ids, resp = [], []
        for chunk_ids, is_resp in chunks:
            ids += chunk_ids
            resp += [1 if is_resp else 0] * len(chunk_ids)
        ids = [self.bos] + ids[: self.max_length - 2] + [self.eos]
        resp = [0] + resp[: self.max_length - 2] + [0]
        attn = [1] * len(ids)
        img_pad = [1 if t == self.pad_id else 0 for t in ids]
        while len(ids) < self.max_length:
            ids.append(self.pad); attn.append(0); resp.append(0); img_pad.append(0)
        ids = ids[: self.max_length]; attn = attn[: self.max_length]
        resp = resp[: self.max_length]; img_pad = img_pad[: self.max_length]
        pixel = _decode_image(row.get('image_bytes'), self.image_size)
        return (torch.tensor(ids, dtype=torch.long),
                torch.tensor(attn, dtype=torch.long),
                torch.tensor(resp, dtype=torch.long),
                torch.tensor(img_pad, dtype=torch.long),
                pixel)
