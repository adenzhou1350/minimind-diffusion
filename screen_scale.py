"""
扩大筛选:更长训练(3000步)、更大模型(512/6)、不同 lr + mask 比例,
看小模型能学到多高。找能上 20%+ 的配置。
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
N_TR = 20000
train_texts = [idx.get(i)['text'] for i in range(N_TR)]
held = [idx.get(j)['text'] for j in range(N_TR, N_TR+200)]

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

def make_model(hid, layers, heads, kv):
    torch.manual_seed(0)
    cfg = DLMConfig(hidden_size=hid, num_hidden_layers=layers, num_attention_heads=heads,
                    num_key_value_heads=kv, vocab_size=V, max_position_embeddings=512,
                    mask_token_id=MASK)
    m = DLMModel(cfg).to(dev)
    lm_head = torch.nn.Linear(hid, V, bias=False).to(dev)
    lm_head.weight = m.embed.weight
    return m, lm_head

def run(name, hid, layers, heads, kv, t_lo, t_hi, steps, lr, bs=64):
    m, lm_head = make_model(hid, layers, heads, kv)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    m.train()
    t0 = time.time()
    nparams = sum(p.numel() for p in m.parameters())/1e6
    for step in range(steps):
        i = (step*bs) % len(train_batch)
        sl = [(i+k) % len(train_batch) for k in range(bs)]
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
        loss = (ce.sum(1) / n).mean()
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        if step % 500 == 0 or step == steps-1:
            acc = recon_acc(m, lm_head)
            print(f'  [{name}] step {step:5d} loss {loss.item():.2f} acc {acc*100:.1f}%  ({nparams:.1f}M)', flush=True)
    final = recon_acc(m, lm_head)
    print(f'>>> [{name}] {time.time()-t0:.0f}s final={final*100:.1f}%\n', flush=True)
    return final

print(f'start (V={V})\n', flush=True)
R = {}
# 更长训 + 更大模型
R['512x6_3k_lr1e-3'] = run('512x6_3k', 512, 6, 8, 4, 0.1, 0.5, 3000, 1e-3)
R['512x6_3k_lr3e-4'] = run('512x6_3k_lr3e-4', 512, 6, 8, 4, 0.1, 0.5, 3000, 3e-4)
# 小模型长训看是否数据/步数问题
R['256x4_5k'] = run('256x4_5k', 256, 4, 4, 2, 0.1, 0.5, 5000, 1e-3)

print('=== SUMMARY ===')
for k,v in R.items():
    print(f'  {k:25s} {v*100:5.1f}%')
