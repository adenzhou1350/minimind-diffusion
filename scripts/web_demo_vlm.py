"""
Streamlit web demo (VLM): 上传图 + 文本 prompt,流式扩散采样。

每步刷新当前揭开的 token(含未揭开的 <mask> 显示为 ▍),
对照 scripts/web_demo.py,差别只在:vision 路径(pixel_values + <|image_pad|> 占位符)。
"""
import os
import sys
import io
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from PIL import Image
import streamlit as st
from model.model_dlm_v import DLMVLMConfig
from model.tokenizer_loader import load_tokenizer, IMAGE_PAD_TOKEN
from trainer.trainer_utils import init_vlm_model


@st.cache_resource
def load_model(hidden_size=768, num_hidden_layers=8, from_weight='vlm_sft', device='cuda:0'):
    cfg = DLMVLMConfig(hidden_size=hidden_size, num_hidden_layers=num_hidden_layers)
    device = device if torch.cuda.is_available() else 'cpu'
    model, tokenizer = init_vlm_model(cfg, from_weight=from_weight, tokenizer_path='model',
                                      save_dir='out', device=device, freeze_mode=0)
    model = model.to(device)
    model.eval()
    return model, tokenizer, cfg, device


def _decode_pil(img, size=256):
    """PIL Image -> [3, H, W] tensor,SigLIP 归一化(mean/std=0.5)。"""
    img = img.convert('RGB').resize((size, size))
    t = torch.tensor(list(img.getdata()), dtype=torch.float32).reshape(size, size, 3)
    t = t.permute(2, 0, 1) / 255.0
    return (t - 0.5) / 0.5


def generate_stream(model, tokenizer, cfg, device, prompt, pixel,
                    gen_length=128, steps=64, temperature=0.7, repetition_penalty=1.3):
    """流式:每步 yield 当前 response 的解码文本(含未揭开的 <mask> 显示为 ▍)。

    全序列路径(非 block)。每步重新跑 vision + transformer,揭开置信度最高的位,
    把低置信度的位重掩回去(s 衰减)。vision 占位符位永不重掩。
    """
    image_pad_str = IMAGE_PAD_TOKEN * cfg.image_token_len  # 64 个 <|image_pad|>
    prompt_str = f'<|im_start|>user\n{image_pad_str}{prompt}<|im_end|><|im_start|>assistant\n'
    prompt_ids = tokenizer(prompt_str, return_tensors='pt', add_special_tokens=False)['input_ids'].to(device)
    MASK_ID = cfg.mask_token_id
    IMG_PAD = cfg.image_pad_token_id
    P = prompt_ids.shape[1]

    # vision 特征预计算一次(采样循环里反复用)
    vis = model._compute_vision(pixel, device)

    resp = torch.full((1, gen_length), MASK_ID, dtype=torch.long, device=device)
    x = torch.cat([prompt_ids, resp], dim=1)
    attn = torch.ones_like(x)
    is_prompt = torch.zeros_like(x, dtype=torch.bool)
    is_prompt[:, :P] = True
    is_prompt |= (x == IMG_PAD)  # vision 占位符位永不重掩

    for k in range(1, steps + 1):
        s = 1.0 - k / steps
        h = model._embed_with_vision(x, vis)
        h = model._run_transformer(h, attn, x.shape[1])
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
    st.title('minimind-diffusion-v')
    st.caption('LLaDA v1 掩码扩散 VLM · SigLIP2 + projector · 给图迭代揭开文本')
    model, tokenizer, cfg, device = load_model()

    uploaded = st.file_uploader('上传图片', type=['jpg', 'jpeg', 'png', 'webp'])
    prompt = st.text_input('Prompt', value='请描述这张图片。')
    col1, col2 = st.columns(2)
    with col1:
        gen_length = st.slider('生成长度', 32, 256, 128)
        steps = st.slider('扩散步数', 8, 256, 64)
    with col2:
        temperature = st.slider('温度(0 会塌缩重复,建议 0.6-0.9)', 0.0, 1.5, 0.7, 0.05)
        repetition_penalty = st.slider('重复惩罚(1.0 关闭,建议 1.2-1.5)', 1.0, 2.0, 1.3, 0.05)

    pixel = None
    if uploaded is not None:
        img = Image.open(uploaded)
        st.image(img, caption='输入图', width=256)
        pixel = _decode_pil(img).unsqueeze(0).to(device)

    if st.button('生成') and pixel is not None:
        out_box = st.empty()
        for text, k in generate_stream(model, tokenizer, cfg, device, prompt, pixel,
                                       gen_length, steps, temperature, repetition_penalty):
            out_box.markdown(f'`step {k}/{steps}`\n\n{text}')
    elif st.button('生成') and pixel is None:
        st.warning('请先上传图片')


if __name__ == '__main__':
    main()
