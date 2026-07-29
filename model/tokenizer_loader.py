"""
tokenizer loader: 复用 minimind BPE tokenizer,新增 <mask> token
"""
from transformers import AutoTokenizer

MASK_TOKEN = '<mask>'
MASK_ID = 6400  # minimind vocab=6400, <mask> 追加为最后一个 id


def load_tokenizer(path='model'):
    """加载 minimind tokenizer 并 resize 词表 +1 <mask>。"""
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if MASK_TOKEN not in tok.get_vocab():
        tok.add_tokens([MASK_TOKEN])
    assert tok.convert_tokens_to_ids(MASK_TOKEN) == MASK_ID, \
        f'<mask> id 应为 {MASK_ID},实得 {tok.convert_tokens_to_ids(MASK_TOKEN)}'
    return tok


if __name__ == '__main__':
    tok = load_tokenizer('model')
    print(f'vocab_size={tok.vocab_size}, mask_id={tok.convert_tokens_to_ids(MASK_TOKEN)}')
