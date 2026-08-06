"""
筛选 mask embedding 初始化策略,量 held-out 掩码重建准确率。
"""
import sys, os, io, time
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
V = len(tok)
BOS, EOS, PAD = tok.bos_token_id, tok.eos_token_id, (tok.pad_token_id or 0)

idx = _JsonlIndex('dataset/pretrain_t2t_mini.jsonl')
N_TR = 8000
train_texts = [idx.get(i)['text'] for i in range(N_TR)]
held = [idx.get(j)['text'] for j in range(N_TR, N_TR+100)]

def encode(text, L=128):
    ids = tok(text, add_special_tokens=False)['input_ids'][:L-2]
    ids = [BOS] + ids + [EOS]
    if len(ids) > L: ids = ids[:L]
    while len(ids) < L: ids.append(PAD)
    attn = [1 if t != PAD else 0 for t in ids]
    return ids, attn

train_batch = [encode(t) for t in train_texts]
held_enc = [encode(t)[0] for t in held]

def recon_acc(model, lm_head, t_eval=0.2):
    model.eval()
    correct = 0; total = 0
    with torch.inference_mode():
        for x0 in held_enc:
            x0 = torch.tensor([x0], device=dev)
            attn = (x0 != PAD).long()
            B, L = x0.shape
            mask = (x0 != PAD) & (torch.rand(B, L, device=dev) < t_eval)
            xt = x0.clone(); xt[mask] = MASK
            h = model(xt, attention_mask=attn)
            pred = lm_head(h).argmax(-1)
            mp = mask.nonzero(as_tuple=False)
            correct += (pred[mp[:,0], mp[:,1]] == x0[mp[:,0], mp[:,1]]).sum().item()
            total += mp.shape[0]
    model.train()
    return correct / max(total, 1)

def make_model(mask_init):
    torch.manual_seed(0)
    cfg = DLMConfig(hidden_size=256, num_hidden_layers=4, num_attention_heads=4,
                    num_key_value_heads=2, vocab_size=V, max_position_embeddings=512,
                    mask_token_id=MASK)
    m = DLMModel(cfg).to(dev)
    lm_head = torch.nn.Linear(256, V, bias=False).to(dev)
    lm_head.weight = m.embed.weight
    with torch.no_grad():
        emb = m.embed.weight
        if mask_init == 'mean':
            emb[MASK] = emb[:MASK].mean(0)
        elif mask_init == 'zero':
            emb[MASK] = 0
        # 'random' = 默认不动
    return m, lm_head

def run(name, mask_init, t_lo, t_hi, steps=800, lr=1e-3, weight='uniform'):
    m, lm_head = make_model(mask_init)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    m.train()
    t0 = time.time()
    for step in range(steps):
        i = (step*64) % len(train_batch)
        sl = [(i+k) % len(train_batch) for k in range(64)]
        ids = torch.tensor([train_batch[k][0] for k in sl], device=dev)
        attn = torch.tensor([train_batch[k][1] for k in sl], device=dev)
        B, L = ids.shape
        t = torch.empty(B, device=dev).uniform_(t_lo, t_hi)
        mask = attn.bool() & (torch.rand(B, L, device=dev) < t[:, None])
        xt = ids.clone(); xt[mask] = MASK
        h = m(xt, attention_mask=attn)
        logits = lm_head(h)
        ce = F.cross_entropy(logits.view(-1, V), ids.view(-1), reduction='none').view(B, L)
        ce = ce * mask
        n = mask.sum(1).clamp(min=1)
        per_seq = ce.sum(1) / n
        if weight == '1/t':
            per_seq = per_seq * (1.0 / t)
        loss = per_seq.mean()
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        if step % 200 == 0 or step == steps-1:
            acc = recon_acc(m, lm_head)
            print(f'  [{name}] step {step:4d} loss {loss.item():.2f} acc {acc*100:.1f}%', flush=True)
    final = recon_acc(m, lm_head)
    print(f'>>> [{name}] {time.time()-t0:.0f}s  final={final*100:.1f}%', flush=True)
    return final

print(f'start (V={V} MASK={MASK})', flush=True)
results = {}
for mi in ['random', 'mean', 'zero']:
    results[f'init={mi}'] = run(f'init={mi}', mi, 0.1, 0.5, steps=800)

print()
print('=== SUMMARY (held-out recon acc @ t=0.2, 800 steps) ===')
for k, v in results.items():
    print(f'  {k:20s} {v*100:5.1f}%')
