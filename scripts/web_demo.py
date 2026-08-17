"""
Streamlit web demo: 流式展示扩散采样(每步刷新当前揭开的 token)
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import streamlit as st
from model.model_dlm import DLMConfig, DLMForMD
from model.tokenizer_loader import load_tokenizer
from trainer.trainer_utils import init_model


@st.cache_resource
def load_model(hidden_size=768, num_hidden_layers=8, from_weight='sft', device='cuda:0'):
    cfg = DLMConfig(hidden_size=hidden_size, num_hidden_layers=num_hidden_layers)
    device = device if torch.cuda.is_available() else 'cpu'
    model, tokenizer = init_model(cfg, from_weight=from_weight, tokenizer_path='model',
                                 save_dir='out', device=device)
    model.eval()
    return model, tokenizer, cfg, device


def generate_stream(model, tokenizer, cfg, device, prompt, gen_length=128, steps=64,
                    temperature=0.7, repetition_penalty=1.3, block_length=0):
    """流式:每步 yield 当前 response 的解码文本

    全序列路径(非 block)。流式展示每步揭开过程,含未揭开的 <mask> 显示为 ▍。
    block 模式无法逐步流式展示(块间自回归),故流式只支持全序列。
    """
    prompt_str = f'<|im_start|>user\n{prompt}<|im_end|><|im_start|>assistant\n'
    prompt_ids = tokenizer(prompt_str, return_tensors='pt')['input_ids'].to(device)
    MASK_ID = cfg.mask_token_id
    P = prompt_ids.shape[1]
    resp = torch.full((1, gen_length), MASK_ID, dtype=torch.long, device=device)
    x = torch.cat([prompt_ids, resp], dim=1)
    attn = torch.ones_like(x)
    is_prompt = torch.zeros_like(x, dtype=torch.bool)
    is_prompt[:, :P] = True
    for k in range(1, steps + 1):
        s = 1.0 - k / steps
        h = model.model(x, attention_mask=attn)
        logits = model.lm_head(h)
        masked = (x == MASK_ID) & (~is_prompt)
        idx = masked.nonzero(as_tuple=False)
        if idx.shape[0] == 0:
            break
        lm_logits = logits[idx[:, 0], idx[:, 1]]
        # 重复惩罚
        if repetition_penalty != 1.0:
            resp_ids = x[0, P:]
            seen = resp_ids[resp_ids != MASK_ID].unique()
            if seen.numel() > 0:
                seen_logits = lm_logits[:, seen]
                seen_logits = torch.where(seen_logits > 0,
                                          seen_logits / repetition_penalty,
                                          seen_logits * repetition_penalty)
                lm_logits = lm_logits.scatter(1, seen.unsqueeze(0).expand(idx.shape[0], -1),
                                              seen_logits)
        prob = torch.softmax(lm_logits / max(temperature, 1e-4), dim=-1)
        pred = prob.argmax(dim=-1)
        conf = prob.gather(1, pred[:, None]).squeeze(1)
        x[idx[:, 0], idx[:, 1]] = pred
        n_remain = int(gen_length * s)
        n_remask = min(n_remain, idx.shape[0])
        if n_remask > 0:
            order = torch.argsort(conf)
            remask = idx[order[:n_remask]]
            x[remask[:, 0], remask[:, 1]] = MASK_ID
        # 每步 yield 当前解码(含未揭开的 <mask> 显示为 ▍)
        ids = x[0, P:].tolist()
        eos = tokenizer.eos_token_id
        if eos in ids:
            ids = ids[:ids.index(eos)]
        text = tokenizer.decode(ids, skip_special_tokens=False)
        text = text.replace('<mask>', '▍').replace(tokenizer.eos_token or '<|im_end|>', '')
        yield text, k


def main():
    st.title('minimind-diffusion')
    st.caption('LLaDA v1 掩码扩散语言模型 · 从全 <mask> 迭代揭开')
    model, tokenizer, cfg, device = load_model()
    prompt = st.text_input('Prompt', value='为什么天空是蓝色的?')
    col1, col2 = st.columns(2)
    with col1:
        gen_length = st.slider('生成长度', 32, 256, 128)
        steps = st.slider('扩散步数', 8, 256, 128)
    with col2:
        temperature = st.slider('温度(0 会塌缩重复,建议 0.6-0.9)', 0.0, 1.5, 0.7, 0.05)
        repetition_penalty = st.slider('重复惩罚(1.0 关闭,建议 1.2-1.5)', 1.0, 2.0, 1.3, 0.05)
    if st.button('生成'):
        out_box = st.empty()
        for text, k in generate_stream(model, tokenizer, cfg, device, prompt,
                                        gen_length, steps, temperature,
                                        repetition_penalty):
            out_box.markdown(f'`step {k}/{steps}`\n\n{text}')


if __name__ == '__main__':
    main()
