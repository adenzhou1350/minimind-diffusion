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
