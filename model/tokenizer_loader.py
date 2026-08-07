"""
🌏🌎🌍 tokenizer loader: 复用 minimind BPE tokenizer,新增 <mask> + <|image_pad|> token 🌏🌎🌍
"""
from transformers import AutoTokenizer

MASK_TOKEN = '<mask>'
MASK_ID = 6400  # minimind vocab=6400, <mask> 追加为最后一个 id
IMAGE_PAD_TOKEN = '<|image_pad|>'
IMAGE_PAD_ID = 6401  # minimind 6400 + <mask>(6400) + <image_pad>(6401)


def load_tokenizer(path='model'):
    """加载 minimind tokenizer 并 resize 词表 +<mask> +<|image_pad|>。"""
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    new_tokens = []
    if MASK_TOKEN not in tok.get_vocab():
        new_tokens.append(MASK_TOKEN)
    if IMAGE_PAD_TOKEN not in tok.get_vocab():
        new_tokens.append(IMAGE_PAD_TOKEN)
    if new_tokens:
        tok.add_tokens(new_tokens)
    assert tok.convert_tokens_to_ids(MASK_TOKEN) == MASK_ID, \
        f'<mask> id 应为 {MASK_ID},实得 {tok.convert_tokens_to_ids(MASK_TOKEN)}'
    assert tok.convert_tokens_to_ids(IMAGE_PAD_TOKEN) == IMAGE_PAD_ID, \
        f'<|image_pad|> id 应为 {IMAGE_PAD_ID},实得 {tok.convert_tokens_to_ids(IMAGE_PAD_TOKEN)}'
    return tok


if __name__ == '__main__':
    tok = load_tokenizer('model')
    print(f'vocab_size={tok.vocab_size}, mask_id={tok.convert_tokens_to_ids(MASK_TOKEN)}, '
          f'image_pad_id={tok.convert_tokens_to_ids(IMAGE_PAD_TOKEN)}')
