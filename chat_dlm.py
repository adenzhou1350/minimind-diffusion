"""
minimind-diffusion 交互式对话(对照 minimind 的对话脚本)
用法: python chat_dlm.py [--hidden_size 768 --num_hidden_layers 8 --from_weight sft]
输 'exit' 或 'quit' 退出。多轮对话(带历史)。
"""
import os
import sys
import argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from model.model_dlm import DLMConfig, DLMForMD
from model.tokenizer_loader import load_tokenizer
from trainer.trainer_utils import init_model


def main():
    parser = argparse.ArgumentParser(description='minimind-diffusion 交互式对话')
    parser.add_argument('--hidden_size', type=int, default=768, help='隐藏维度')
    parser.add_argument('--num_hidden_layers', type=int, default=8, help='层数')
    parser.add_argument('--from_weight', type=str, default='sft', help='加载哪个权重(pretrain/sft)')
    parser.add_argument('--steps', type=int, default=128, help='扩散采样步数')
    parser.add_argument('--gen_length', type=int, default=128, help='单轮生成长度')
    parser.add_argument('--temperature', type=float, default=0.7, help='采样温度(建议 0.6-0.9)')
    parser.add_argument('--repetition_penalty', type=float, default=1.3, help='重复惩罚(建议 1.2-1.5)')
    parser.add_argument('--max_turns', type=int, default=6, help='保留的最大对话轮数(防超长)')
    parser.add_argument('--tokenizer_path', type=str, default='model', help='tokenizer 目录')
    parser.add_argument('--save_dir', type=str, default='out', help='权重目录')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu', help='设备')
    args = parser.parse_args()

    cfg = DLMConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers)
    model, tokenizer = init_model(cfg, from_weight=args.from_weight,
                                 tokenizer_path=args.tokenizer_path,
                                 save_dir=args.save_dir, device=args.device)
    model.eval()

    # 对话历史:每轮 [user, assistant] 交替,渲染成 minimind chat template
    history = []  # list of (role, content)
    print('minimind-diffusion 对话(输入 exit 退出)')
    print(f'  [模型 {args.from_weight}_{args.hidden_size} | steps={args.steps} temp={args.temperature} rep={args.repetition_penalty}]')
    print()

    while True:
        try:
            user = input('你: ').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user.lower() in ('exit', 'quit', '退出'):
            break

        history.append(('user', user))
        # 只保留最近 max_turns 轮,防序列过长
        if len(history) > args.max_turns * 2:
            history = history[-args.max_turns * 2:]

        # 渲染 chat template:历史 + assistant 开头(让模型续写)
        prompt_str = ''
        for role, content in history:
            prompt_str += f'<|im_start|>{role}\n{content}<|im_end|>'
        prompt_str += '<|im_start|>assistant\n'
        prompt_ids = tokenizer(prompt_str, return_tensors='pt')['input_ids'].to(args.device)

        # 如果 prompt + 历史超长,截断到能容纳生成的长度(简单策略:从开头砍)
        max_prompt = 384  # 留点给生成(512 - gen_length 余量)
        if prompt_ids.shape[1] > max_prompt:
            prompt_ids = prompt_ids[:, -max_prompt:]

        with torch.inference_mode():
            out = model.generate(prompt_ids, gen_length=args.gen_length, steps=args.steps,
                               temperature=args.temperature,
                               repetition_penalty=args.repetition_penalty)
        ids = out[0].tolist()
        eos = tokenizer.eos_token_id
        if eos in ids:
            ids = ids[:ids.index(eos)]
        reply = tokenizer.decode(ids, skip_special_tokens=True).strip()
        print(f'AI: {reply}')
        print()
        history.append(('assistant', reply))


if __name__ == '__main__':
    main()
