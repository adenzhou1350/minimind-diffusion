"""
Stage 2 SFT:vision 冻结 + LLM 首尾层 + projector 训。
数据:sft_i2t.parquet(多轮指令,带图)。
"""
import os
import sys
import argparse
# 让从项目根目录运行时(无论 cwd)能 import model/dataset/trainer
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch.utils.data import DataLoader
from model.model_dlm_v import DLMVLMConfig, DLMForVLM
from dataset.vlm_dataset import SFTVLMDataset
from trainer.trainer_utils import (get_lr, SkipBatchSampler, init_vlm_model, vlm_checkpoint,
                                   Logger, setup_seed, init_distributed_mode, is_main_process)


def collate(batch):
    """batch: list of (ids, attn, resp, img_pad, pixel) -> stacked tensors。"""
    ids = torch.stack([b[0] for b in batch])
    attn = torch.stack([b[1] for b in batch])
    resp = torch.stack([b[2] for b in batch])
    imgpad = torch.stack([b[3] for b in batch])
    pixel = torch.stack([b[4] for b in batch])
    return ids, attn, resp, imgpad, pixel


def main():
    # ========== 1. 初始化环境和随机种子 ==========
    parser = argparse.ArgumentParser(description='mind-diffusion-v Stage 2 SFT')
    parser.add_argument('--epochs', type=int, default=1, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=4, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=5e-6, help='学习率')
    parser.add_argument('--accumulation_steps', type=int, default=4, help='梯度累积')
    parser.add_argument('--max_seq_len', type=int, default=512, help='最大序列长度')
    parser.add_argument('--hidden_size', type=int, default=768, help='隐藏维度')
    parser.add_argument('--num_hidden_layers', type=int, default=8, help='层数')
    parser.add_argument('--data_path', type=str, default='dataset/sft_i2t.parquet', help='SFT 数据')
    parser.add_argument('--tokenizer_path', type=str, default='model', help='tokenizer 目录')
    parser.add_argument('--save_dir', type=str, default='out', help='输出目录')
    parser.add_argument('--from_weight', type=str, default='vlm_align', help='从 Stage1 对齐权重续')
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu', help='设备')
    args = parser.parse_args()

    setup_seed()
    init_distributed_mode()
    logger = Logger(os.path.join(args.save_dir, 'sft_vlm.log'))

    # ========== 2. 加载模型(Stage2: LLM 首尾层 + projector 训) ==========
    cfg = DLMVLMConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers)
    model, tokenizer = init_vlm_model(cfg, from_weight=args.from_weight,
                                       tokenizer_path=args.tokenizer_path,
                                       save_dir=args.save_dir, device=args.device,
                                       freeze_mode=1)
    model = model.to(args.device)

    # ========== 3. 构造数据集与加载器 ==========
    ds = SFTVLMDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    sampler = SkipBatchSampler(ds, args.batch_size)
    loader = DataLoader(ds, batch_sampler=sampler, num_workers=0,
                        collate_fn=collate, pin_memory=True)

    # ========== 4. 优化器(首尾层 + final norm + lm_head + projector) ==========
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate)
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    total_steps = args.epochs * len(loader) // args.accumulation_steps
    step = 0
    os.makedirs(args.save_dir, exist_ok=True)

    # ========== 5. 训练循环 ==========
    for epoch in range(args.epochs):
        model.train()
        for i, (ids, attn, resp, imgpad, pixel) in enumerate(loader):
            ids, attn, resp, pixel = ids.to(args.device), attn.to(args.device), resp.to(args.device), pixel.to(args.device)
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available(), dtype=torch.bfloat16):
                out = model(input_ids=ids, attention_mask=attn, response_mask=resp,
                           labels=ids, pixel_values=pixel)
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
            vlm_checkpoint(model, os.path.join(args.save_dir, f'vlm_sft_{args.hidden_size}.pth'))
            logger(f'saved vlm_sft_{args.hidden_size}.pth')


if __name__ == '__main__':
    main()
