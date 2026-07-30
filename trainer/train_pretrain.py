"""
预训练: 全序列随机掩码 + 1/t 加权 loss(在 DLMForMD.forward 里)
"""
import os
import sys
import argparse
# 让从项目根目录运行时(无论 cwd)能 import model/dataset/trainer
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch.utils.data import DataLoader
from model.model_dlm import DLMConfig, DLMForMD
from dataset.lm_dataset import PretrainDataset
from model.tokenizer_loader import load_tokenizer
from trainer.trainer_utils import (get_lr, SkipBatchSampler, init_model, lm_checkpoint,
                                   Logger, setup_seed, init_distributed_mode, is_main_process)


def main():
    # ========== 1. 初始化环境和随机种子 ==========
    parser = argparse.ArgumentParser(description='mind-diffusion 预训练')
    parser.add_argument('--epochs', type=int, default=2, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='学习率')
    parser.add_argument('--accumulation_steps', type=int, default=8, help='梯度累积')
    parser.add_argument('--max_seq_len', type=int, default=340, help='最大序列长度')
    parser.add_argument('--hidden_size', type=int, default=768, help='隐藏维度')
    parser.add_argument('--num_hidden_layers', type=int, default=8, help='层数')
    parser.add_argument('--data_path', type=str, default='dataset/pretrain_t2t_mini.jsonl', help='预训练数据')
    parser.add_argument('--tokenizer_path', type=str, default='model', help='tokenizer 目录')
    parser.add_argument('--save_dir', type=str, default='out', help='输出目录')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu', help='设备')
    args = parser.parse_args()

    setup_seed()
    init_distributed_mode()
    logger = Logger(os.path.join(args.save_dir, 'pretrain.log'))

    # ========== 2. 加载模型与 tokenizer ==========
    cfg = DLMConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers)
    model, tokenizer = init_model(cfg, from_weight=None,
                                 tokenizer_path=args.tokenizer_path,
                                 save_dir=args.save_dir, device=args.device)
    model = model.to(args.device)

    # ========== 3. 构造数据集与加载器 ==========
    ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    sampler = SkipBatchSampler(ds, args.batch_size)
    loader = DataLoader(ds, batch_sampler=sampler, num_workers=4, pin_memory=True)

    # ========== 4. 优化器 ==========
    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    total_steps = args.epochs * len(loader) // args.accumulation_steps
    step = 0
    os.makedirs(args.save_dir, exist_ok=True)

    # ========== 5. 训练循环 ==========
    for epoch in range(args.epochs):
        model.train()
        for i, (ids, attn) in enumerate(loader):
            ids, attn = ids.to(args.device), attn.to(args.device)
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available(), dtype=torch.bfloat16):
                out = model(input_ids=ids, attention_mask=attn, labels=ids)
                loss = out.loss / args.accumulation_steps
            scaler.scale(loss).backward()
            if (i + 1) % args.accumulation_steps == 0:
                lr = get_lr(step, total_steps, args.learning_rate)
                for g in optim.param_groups:
                    g['lr'] = lr
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
                step += 1
                if step % 10 == 0 and is_main_process():
                    logger(f'epoch {epoch} step {step}/{total_steps} loss {loss.item()*args.accumulation_steps:.4f} lr {lr:.2e}')
        if is_main_process():
            lm_checkpoint(model, os.path.join(args.save_dir, f'pretrain_{args.hidden_size}.pth'))
            logger(f'saved pretrain_{args.hidden_size}.pth')


if __name__ == '__main__':
    main()
