"""
mind-diffusion-v 推理与对话:给图 + prompt,扩散采样出跟图相关的中文。
对照 eval_dlm.py,差别只在:(1) 从 vlm_sft/vlm_align 加载 VLM;(2) prompt 里展开
<image> 成 64 个 <|image_pad|> 占位符;(3) generate 传 pixel_values。
"""
import os
import sys
import io
import time
import argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Windows 控制台默认 GBK,生成中文含 U+FFFD 会 UnicodeEncodeError;统一 UTF-8 输出
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import torch
import pyarrow.parquet as pq
from PIL import Image
from model.model_dlm_v import DLMVLMConfig
from model.tokenizer_loader import load_tokenizer, IMAGE_PAD_TOKEN
from trainer.trainer_utils import init_vlm_model


def _load_sample_image(parquet_path, idx=0):
    """从 parquet 取第 idx 行的 image_bytes + conversations(默认 user 第一句)。"""
    pf = pq.ParquetFile(parquet_path)
    # 每 row_group 5000 行;idx 映射到对应 row_group 的局部行
    rg = idx // 5000
    local = idx % 5000
    tbl = pf.read_row_group(rg, columns=['image_bytes', 'conversations'])
    import json
    row_img = tbl.column('image_bytes')[local].as_py()
    row_conv = tbl.column('conversations')[local].as_py()
    if isinstance(row_conv, str):
        row_conv = json.loads(row_conv)
    return row_img, row_conv


def _decode_image(image_bytes, size=256):
    """JPEG bytes -> [3, H, W] tensor,SigLIP 归一化(mean/std=0.5)。同 vlm_dataset。"""
    if not image_bytes:
        return torch.zeros(3, size, size)
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB').resize((size, size))
    t = torch.tensor(list(img.getdata()), dtype=torch.float32).reshape(size, size, 3)
    t = t.permute(2, 0, 1) / 255.0
    return (t - 0.5) / 0.5


# 测试:图 + 中文 prompt(对照 minimind-v 的图描述场景)
TEST_PROMPTS = [
    '请描述这张图片。',
    '图片里有什么?',
    '这张图的主色调是什么?',
]


def main():
    parser = argparse.ArgumentParser(description='mind-diffusion-v 推理与对话')
    parser.add_argument('--hidden_size', type=int, default=768, help='隐藏维度')
    parser.add_argument('--num_hidden_layers', type=int, default=8, help='层数')
    parser.add_argument('--from_weight', type=str, default='vlm_sft',
                        help='加载哪个权重(vlm_align=Stage1, vlm_sft=Stage2)')
    parser.add_argument('--data_path', type=str, default='dataset/sft_i2t.parquet',
                        help='取样本图的 parquet')
    parser.add_argument('--sample_idx', type=int, default=0, help='取第几行的图')
    parser.add_argument('--steps', type=int, default=64, help='扩散采样步数')
    parser.add_argument('--gen_length', type=int, default=128, help='生成长度')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='采样温度(扩散建议 0.6-0.9;0 会塌缩重复)')
    parser.add_argument('--repetition_penalty', type=float, default=1.3, help='重复惩罚')
    parser.add_argument('--tokenizer_path', type=str, default='model', help='tokenizer 目录')
    parser.add_argument('--save_dir', type=str, default='out', help='权重目录')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu', help='设备')
    args = parser.parse_args()

    cfg = DLMVLMConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers)
    # eval 用 freeze_mode=0(不训,freeze 无所谓;但确保 vision encoder 加载)
    model, tokenizer = init_vlm_model(cfg, from_weight=args.from_weight,
                                      tokenizer_path=args.tokenizer_path,
                                      save_dir=args.save_dir, device=args.device,
                                      freeze_mode=0)
    model = model.to(args.device)
    model.eval()

    # 取一张样本图
    img_bytes, conv = _load_sample_image(args.data_path, args.sample_idx)
    pixel = _decode_image(img_bytes).unsqueeze(0).to(args.device)  # [1, 3, 256, 256]
    # 保存一份原图到 out 方便人眼看
    try:
        Image.open(io.BytesIO(img_bytes)).save(os.path.join(args.save_dir, 'eval_sample.jpg'))
    except Exception:
        pass
    print(f'[Image] sample_idx={args.sample_idx} saved to {args.save_dir}/eval_sample.jpg')
    if conv:
        first_user = next((c.get('content', c.get('value', '')) for c in conv
                           if c.get('role', c.get('from')) == 'user'), '')
        ref = next((c.get('content', c.get('value', '')) for c in conv
                    if c.get('role', c.get('from')) == 'assistant'), '')
        print(f'[Ref prompt] {first_user[:80]}')
        print(f'[Ref answer] {ref[:80]}')

    image_pad_str = IMAGE_PAD_TOKEN * cfg.image_token_len  # 64 个 <|image_pad|>
    for p in TEST_PROMPTS:
        # chat template:<|im_start|>user\n<image>{问}<|im_end|><|im_start|>assistant\n
        prompt_str = f'<|im_start|>user\n{image_pad_str}{p}<|im_end|><|im_start|>assistant\n'
        prompt_ids = tokenizer(prompt_str, return_tensors='pt', add_special_tokens=False)['input_ids'].to(args.device)
        t0 = time.time()
        out = model.generate(prompt_ids, gen_length=args.gen_length, steps=args.steps,
                             temperature=args.temperature,
                             repetition_penalty=args.repetition_penalty,
                             pixel_values=pixel)
        dt = time.time() - t0
        eos = tokenizer.eos_token_id
        ids = out[0].tolist()
        if eos in ids:
            ids = ids[:ids.index(eos)]
        text = tokenizer.decode(ids, skip_special_tokens=False)
        n_tok = len(ids)
        print(f'\n[Prompt] {p}')
        print(f'[Response] {text}')
        print(f'[Speed]: {n_tok / dt:.2f} tokens/s ({n_tok} tokens in {dt:.2f}s, {args.steps} diffusion steps)')


if __name__ == '__main__':
    main()
