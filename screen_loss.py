"""
快速筛选:不同 loss 配方各训几百步,量"掩码重建准确率"(训练任务本身)。
挑出能学会"根据上下文预测被掩 token"的配置,再上 64M 全量训。
"""
import sys, os, io, json, time, math
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn.functional as F
from model.model_dlm import DLMConfig, DLMModel
from model.tokenizer_loader import load_tokenizer
from dataset.lm_dataset import _JsonlIndex

torch.manual_seed(0)
dev = 'cuda'
tok = load_tokenizer('model')
MASK = tok.convert_tokens_to_ids('<mask>')
V = len(tok)  # 6401
BOS, EOS, PAD = tok.bos_token_id, tok.eos_token_id, (tok.pad_token_id or 0)

# 小模型用于筛选(快)
def make_model():
    cfg = DLMConfig(hidden_size=256, num_hidden_layers=4, num_attention_heads=4,
                    num_key_value_heads=2, vocab_size=V, max_position_embeddings=512,
                    mask_token_id=MASK)
    m = DLMModel(cfg).to(dev)
    lm_head = torch.nn.Linear(256, V, bias=False).to(dev)
    lm_head.weight = m.embed.weight  # tie
    return m, lm_head

def encode(text, L=128):
    ids = tok(text, add_special_tokens=False)['input_ids'][:L-2]
    ids = [BOS] + ids + [EOS]
    if len(ids) > L: ids = ids[:L]
    while len(ids) < L: ids.append(PAD)
    attn = [1 if t != PAD else 0 for t in ids]
    return ids, attn

# 加载一批训练数据 + 一批 held-out
idx = _JsonlIndex('dataset/pretrain_t2t_mini.jsonl')
N = 4000
train_texts = [idx.get(i)['text'] for i in range(N)]
held = [idx.get(j)['text'] for j in range(N, N+50)]
train_batch = [encode(t) for t in train_texts]
held_ids = [encode(t)[0] for t in held]

def make_batch(slice_):
    ids = torch.tensor([train_batch[i][0] for i in slice_], device=dev)
    attn = torch.tensor([train_batch[i][1] for i in slice_], device=dev)
    return ids, attn

def masked_recon_acc(model, lm_head, t_eval=0.3):
    """在 held-out 上:固定掩 30%,预测被掩位,量准确率。"""
    model.eval()
    correct = 0; total = 0
    with torch.inference_mode():
        for x0 in held_ids:
            x0 = torch.tensor([x0], device=dev)
            attn = (x0 != PAD).long()
            B, L = x0.shape
            t = torch.full((B,), t_eval, device=dev)
            maskable = attn.bool()
            mask = maskable & (torch.rand(B, L, device=dev) < t[:, None])
            xt = x0.clone(); xt[mask] = MASK
            h = model(xt, attention_mask=attn)
            logits = lm_head(h)
            pred = logits.argmax(-1)
            mpos = mask.nonzero(as_tuple=False)
            pp = pred[mpos[:,0], mpos[:,1]]
            tp = x0[mpos[:,0], mpos[:,1]]
            correct += (pp == tp).sum().item(); total += mpos.shape[0]
    model.train()
    return correct / max(total, 1)

def train_config(name, weight_scheme, t_low, t_high, steps=500, lr=5e-4):
    torch.manual_seed(0)
    model, lm_head = make_model()
    opt = torch.optim.AdamW(list(model.parameters()) + list(lm_head.parameters()) if not lm_head.weight.requires_grad else list(model.parameters()), lr=lr)
    # note: lm_head tied to embed, so just model params
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    t0 = time.time()
    acc0 = masked_recon_acc(model, lm_head)
    for step in range(steps):
        i = (step*16) % len(train_batch)
        sl = [(i+k) % len(train_batch) for k in range(16)]
        ids, attn = make_batch(sl)
        B, L = ids.shape
        t = torch.empty(B, device=dev).uniform_(t_low, t_high)
        maskable = attn.bool()
        mask = maskable & (torch.rand(B, L, device=dev) < t[:, None])
        xt = ids.clone(); xt[mask] = MASK
        h = model(xt, attention_mask=attn)
        logits = lm_head(h)
        ce = F.cross_entropy(logits.view(-1, V), ids.view(-1), reduction='none').view(B, L)
        ce = ce * mask
        n_masked = mask.sum(1).clamp(min=1)
        per_seq = (ce.sum(1) / n_masked)
        if weight_scheme == '1_over_t':
            per_seq = per_seq * (1.0 / t)
        # elif 'uniform': per_seq as is
        loss = per_seq.mean()
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        if step % 100 == 0:
            acc = masked_recon_acc(model, lm_head)
            print(f'  [{name}] step {step:4d} loss {loss.item():.3f} recon_acc {acc*100:.1f}%', flush=True)
    acc = masked_recon_acc(model, lm_head)
    print(f'>>> [{name}] DONE {time.time()-t0:.0f}s  final recon_acc={acc*100:.1f}%  (start {acc0*100:.1f}%)', flush=True)
    return acc

print(f'start screening (V={V} MASK={MASK})', flush=True)
configs = [
    ('A_baseline_1over_t_U01', '1_over_t', 1e-4, 1.0),
    ('B_uniform_U01',          'uniform',  1e-4, 1.0),
    ('C_1over_t_clip_0.1_0.7',  '1_over_t', 0.1,  0.7),
    ('D_uniform_clip_0.1_0.5',  'uniform',  0.1,  0.5),
]
results = {}
for name, ws, lo, hi in configs:
    acc = train_config(name, ws, lo, hi, steps=500)
    results[name] = acc
print()
print('=== SUMMARY (masked recon acc @ t=0.3) ===')
for name, acc in results.items():
    print(f'  {name:30s} {acc*100:5.1f}%')
