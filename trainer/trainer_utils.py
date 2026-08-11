"""
训练工具:一行余弦 get_lr / SkipBatchSampler / init_model / Logger / setup_seed
"""
import math
import os
import time
import torch
import torch.distributed as dist
from torch.utils.data import Sampler
from model.model_dlm import DLMConfig, DLMForMD
from model.tokenizer_loader import load_tokenizer


def get_lr(current_step, total_steps, lr):
    """一行余弦,10% floor + 0.9x peak,无 warmup(跟 minimind 一致)"""
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))


def setup_seed(seed=1029):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


class Logger:
    def __init__(self, log_file=None):
        self.log_file = log_file

    def __call__(self, msg):
        t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        line = f'[{t}] {msg}'
        if is_main_process():
            print(line, flush=True)
        if self.log_file:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(line + '\n')


class SkipBatchSampler(Sampler):
    """跳过前 N 个 batch,用于 resume"""

    def __init__(self, dataset, batch_size, skip=0, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.skip = skip
        self.shuffle = shuffle

    def __iter__(self):
        n = len(self.dataset)
        idx = torch.randperm(n).tolist() if self.shuffle else list(range(n))
        for i in range(self.skip * self.batch_size, n, self.batch_size):
            yield idx[i: i + self.batch_size]

    def __len__(self):
        return (len(self.dataset) - self.skip * self.batch_size + self.batch_size - 1) // self.batch_size


def init_model(lm_config, from_weight='pretrain', tokenizer_path='model',
               save_dir='out', device='cuda'):
    """加载 tokenizer + 构建 DLMForMD + 加载 checkpoint"""
    tokenizer = load_tokenizer(tokenizer_path)
    model = DLMForMD(lm_config).to(device)
    weight_path = os.path.join(save_dir, f'{from_weight}_{lm_config.hidden_size}.pth')
    if from_weight is not None and os.path.exists(weight_path):
        state = torch.load(weight_path, map_location=device)
        model.load_state_dict(state, strict=False)
        print(f'loaded {weight_path}')
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'trainable params: {n_params / 1e6:.2f}M')
    return model, tokenizer


def lm_checkpoint(model, path):
    """checkpoint: 权重 halved + 移 CPU(跟 minimind 一致)"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {k: v.half().cpu() for k, v in model.state_dict().items()}
    torch.save(state, path)


def init_distributed_mode():
    """简易 DDP 初始化(minimind 风格)"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world = int(os.environ['WORLD_SIZE'])
        dist.init_process_group('nccl', rank=rank, world_size=world)
        torch.cuda.set_device(rank)
    return dist.is_initialized()


# 🌏🌎🌍 VLM 工具:冻结策略 + init_vlm_model + vlm_checkpoint(剥离 vision encoder) 🌏🌎🌍
def freeze_vlm(model, mode=2):
    """冻结策略:minimind-v 风格。vision encoder 始终冻结。
    mode=2: LLM 全冻结,只 projector 训(Stage 1 对齐)
    mode=1: LLM 首尾层(第0层+最后层)+ final norm + lm_head + projector 训(Stage 2 SFT)
    mode=0: 全解冻
    """
    if getattr(model, 'vision_encoder', None) is not None:
        for p in model.vision_encoder.parameters():
            p.requires_grad = False
    if mode == 2:
        for p in model.model.parameters():
            p.requires_grad = False
        for p in model.lm_head.parameters():
            p.requires_grad = False
        # projector 保持可训
    elif mode == 1:
        n = len(model.model.layers)
        for i, layer in enumerate(model.model.layers):
            for p in layer.parameters():
                p.requires_grad = (i == 0 or i == n - 1)
        for p in model.model.embed.parameters():
            p.requires_grad = False
        for p in model.model.norm.parameters():
            p.requires_grad = True
        for p in model.lm_head.parameters():
            p.requires_grad = True
    # mode 0: 不动


def init_vlm_model(cfg, from_weight='pretrain', tokenizer_path='model', save_dir='out',
                   device='cuda', freeze_mode=2):
    """加载 tokenizer + 构建 DLMForVLM + 加载 LLM 权重 + 冻结策略。
    vision encoder 惰性加载(forward 首次调用时触发)。"""
    from model.model_dlm_v import DLMForVLM  # 函数内 import 避免循环
    tokenizer = load_tokenizer(tokenizer_path)
    model = DLMForVLM(cfg).to(device)
    # 加载 LLM 权重(LLM 部分的 key 不带 vision_encoder./projector. 前缀)
    weight_path = os.path.join(save_dir, f'{from_weight}_{cfg.hidden_size}.pth')
    if from_weight is not None and os.path.exists(weight_path):
        state = torch.load(weight_path, map_location=device)
        own = model.state_dict()
        loaded = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
        model.load_state_dict(loaded, strict=False)
        print(f'loaded {len(loaded)}/{len(state)} keys from {weight_path}')
    freeze_vlm(model, freeze_mode)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'trainable params: {n / 1e6:.2f}M (freeze_mode={freeze_mode})')
    return model, tokenizer


def vlm_checkpoint(model, path):
    """strip vision encoder 权重,只存 LLM + projector(跟 minimind-v 一致)。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {k: v.half().cpu() for k, v in model.state_dict().items()
             if not k.startswith('vision_encoder.')}
    torch.save(state, path)
