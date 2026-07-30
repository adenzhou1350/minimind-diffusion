"""
mind-diffusion 推理与对话(对照 minimind eval_llm.py)
"""
import os
import sys
import time
import argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from model.model_dlm import DLMConfig, DLMForMD
from model.tokenizer_loader import load_tokenizer
from trainer.trainer_utils import init_model

# 中文测试 prompt(对照 minimind)
PROMPTS = [
    '为什么天空是蓝色的?',
    '介绍一下你自己',
    '如何学习编程?',
]


def main():
    parser = argparse.ArgumentParser(description='mind-diffusion 推理与对话')
    parser.add_argument('--hidden_size', type=int, default=768, help='隐藏维度')
    parser.add_argument('--num_hidden_layers', type=int, default=8, help='层数')
    parser.add_argument('--from_weight', type=str, default='sft', help='加载哪个权重(pretrain/sft)')
    parser.add_argument('--steps', type=int, default=64, help='扩散采样步数')
    parser.add_argument('--gen_length', type=int, default=128, help='生成长度')
    parser.add_argument('--temperature', type=float, default=0.0, help='采样温度')
    parser.add_argument('--tokenizer_path', type=str, default='model', help='tokenizer 目录')
    parser.add_argument('--save_dir', type=str, default='out', help='权重目录')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu', help='设备')
    args = parser.parse_args()

    cfg = DLMConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers)
    model, tokenizer = init_model(cfg, from_weight=args.from_weight,
                                 tokenizer_path=args.tokenizer_path,
                                 save_dir=args.save_dir, device=args.device)
    model.eval()

    for p in PROMPTS:
        # chat template:<|im_start|>user\n{p}<|im_end|><|im_start|>assistant\n
        prompt_str = f'<|im_start|>user\n{p}<|im_end|><|im_start|>assistant\n'
        prompt_ids = tokenizer(prompt_str, return_tensors='pt')['input_ids'].to(args.device)
        t0 = time.time()
        out = model.generate(prompt_ids, gen_length=args.gen_length, steps=args.steps,
                           temperature=args.temperature)
        dt = time.time() - t0
        # 遇首个 eos 截断
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
