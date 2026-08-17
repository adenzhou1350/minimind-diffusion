"""
🌏🌎🌍 minimind-diffusion-v: diffusion 版 minimind-v (多模态掩码扩散) 🌏🌎🌍

文本侧复用 minimind-diffusion 的 DLMForMD(双向 transformer + 均匀权重掩码 CE)。
vision 侧:冻结 SigLIP2 + MLP projector,vision token 填到 <|image_pad|> 占位符
(观测条件,永不掩),扩散 loss 只掩文本。LLaVA 前缀注入(非 cross-attn)。
"""
from model.model_dlm import DLMConfig


class DLMVLMConfig(DLMConfig):
    """多模态 config,继承 DLMConfig + vision 字段。"""

    model_type = 'mind_diffusion_vlm'

    def __init__(self,
                 image_hidden_size=768,
                 image_token_len=64,
                 image_pad_token_id=12,
                 freeze_vision=True,
                 projector_hidden=768,
                 vision_encoder_name='model/siglip2-base-p32-256-ve',  # 本地路径(已下载)
                 **kwargs):
        self.image_hidden_size = image_hidden_size
        self.image_token_len = image_token_len
        self.image_pad_token_id = image_pad_token_id
        self.freeze_vision = freeze_vision
        self.projector_hidden = projector_hidden
        self.vision_encoder_name = vision_encoder_name
        super().__init__(image_pad_token_id=image_pad_token_id, **kwargs)


# 🌏🌎🌍 vision path: projector + encoder + DLMForVLM 🌏🌎🌍
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.model_dlm import DLMForMD, DLMOutput


class MMVisionProjector(nn.Module):
    """LLaVA-1.5 式 MLP:LayerNorm -> Linear -> GELU -> Linear。"""

    def __init__(self, in_dim, out_dim, mid=None):
        super().__init__()
        mid = mid or out_dim
        self.norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, mid)
        self.fc2 = nn.Linear(mid, out_dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(self.norm(x))))


class DLMForVLM(DLMForMD):
    """包装 DLMForMD,加 vision 路径。vision token 填占位符,永不掩。"""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.projector = MMVisionProjector(cfg.image_hidden_size, cfg.hidden_size,
                                          cfg.projector_hidden)
        # vision encoder 惰性加载(测试可 mock);真实加载见 _load_vision_encoder
        self.vision_encoder = None

    def _load_vision_encoder(self, name=None):
        """加载冻结的 vision encoder(SigLIP2 或 CLIP fallback)。"""
        from transformers import SiglipVisionModel
        name = name or self.config.vision_encoder_name
        try:
            enc = SiglipVisionModel.from_pretrained(name)
        except Exception as e:
            # fallback: CLIP-B/32(768-dim,生态最稳)
            from transformers import CLIPVisionModel
            print(f'SigLIP 加载失败({e}),回退 CLIP-B/32')
            enc = CLIPVisionModel.from_pretrained('openai/clip-vit-base-patch32')
        if self.config.freeze_vision:
            for p in enc.parameters():
                p.requires_grad = False
            enc.eval()
        return enc

    def _compute_vision(self, pixel_values, device):
        """vision encoder + projector,返回 [B, T_img, hidden] 或 None。
        vision encoder 是 fp16,pixel 转 fp16 喂进去;projector 输出转回 fp32
        (后续 _embed_with_vision 再对齐到 h 的 dtype)。"""
        if pixel_values is None:
            return None
        if self.vision_encoder is None:
            self.vision_encoder = self._load_vision_encoder().to(device)
        with torch.no_grad():
            # vision encoder 权重 fp16,输入也要 fp16。mock 时无 parameters,直接用原 dtype
            try:
                ve_params = list(self.vision_encoder.parameters())
                ve_dtype = ve_params[0].dtype if ve_params else pixel_values.dtype
            except Exception:
                ve_dtype = pixel_values.dtype
            pix = pixel_values.to(ve_dtype) if ve_dtype != pixel_values.dtype else pixel_values
            vis = self.vision_encoder(pix)
            vis = vis.last_hidden_state if hasattr(vis, 'last_hidden_state') else vis
        # projector 输出 fp32(mock 也兼容)
        vis = vis.float() if vis.dtype != torch.float32 else vis
        return self.projector(vis)

    def _embed_with_vision(self, input_ids, vis):
        """embed input_ids,把 vision 占位符位替换成 vis。
        防御性对齐:按 vis 的实际 token 数填(数据里某些样本的占位符数 ≠ image_token_len,
        导致 img_mask 数 > vis token 数时 shape mismatch)。"""
        h = self.model.embed(input_ids)  # [B, L, H]
        if vis is not None:
            img_mask = (input_ids == self.config.image_pad_token_id)  # [B, L]
            vis_flat = vis.reshape(-1, vis.shape[-1]).to(h.dtype)  # [B*T_img, H]
            h = h.clone()
            n_vis = vis_flat.shape[0]
            img_pos = img_mask.nonzero(as_tuple=False)  # [N_mask, 2]
            n_mask = img_pos.shape[0]
            if n_mask == n_vis:
                h[img_mask] = vis_flat
            elif n_mask > n_vis:
                # 占位符多于 vis token:只填前 n_vis 个(截断多余占位符,保留其原 embedding)
                h[img_pos[:n_vis, 0], img_pos[:n_vis, 1]] = vis_flat
            else:
                # 占位符少于 vis token:只用前 n_mask 个 vis token
                h[img_mask] = vis_flat[:n_mask]
        return h

    def _run_transformer(self, h, attention_mask, L):
        am = None
        if attention_mask is not None:
            am = attention_mask[:, None, None, :].to(h.dtype)
            am = (1.0 - am) * torch.finfo(h.dtype).min
        freqs = self.model.freqs_cis[:L]
        for layer in self.model.layers:
            h = layer(h, am, freqs)
        return self.model.norm(h)

    def forward(self, input_ids, attention_mask=None, response_mask=None, labels=None,
                pixel_values=None):
        x_0 = labels if labels is not None else input_ids
        B, L = x_0.shape
        device = x_0.device
        MASK_ID = self.config.mask_token_id
        V = self.config.vocab_size
        IMG_PAD = self.config.image_pad_token_id

        # 1. vision 特征(若有图)
        vis = self._compute_vision(pixel_values, device)

        # 2. 采掩码比例 t + maskable(排除 vision 占位符位)
        t = torch.empty(B, device=device).uniform_(0.1, 0.5)
        maskable = attention_mask.bool() if attention_mask is not None \
            else torch.ones(B, L, dtype=torch.bool, device=device)
        if pixel_values is not None:
            maskable = maskable & (input_ids != IMG_PAD)  # vision 位永不掩
        if response_mask is not None:
            maskable = maskable & response_mask.bool()

        rand = torch.rand(B, L, device=device)
        mask = maskable & (rand < t[:, None])
        x_t = x_0.clone()
        x_t[mask] = MASK_ID

        # 3. embed + 填 vision
        h = self._embed_with_vision(x_t, vis)

        # 4. transformer + lm_head
        h = self._run_transformer(h, attention_mask, L)
        logits = self.lm_head(h)

        # 5. 均匀权重掩码 CE
        ce = F.cross_entropy(logits.view(-1, V), x_0.view(-1), reduction='none').view(B, L)
        ce = ce * mask
        n_masked = mask.sum(dim=1).clamp(min=1)
        loss = (ce.sum(dim=1) / n_masked).mean()
        return DLMOutput(loss=loss, logits=logits)

    # 🌏🌎🌍 扩散采样:带 vision 的生成(继承父类采样循环,vision 位永不重掩) 🌏🌎🌍
    @torch.inference_mode()
    def generate(self, prompt_ids, gen_length=128, steps=64, temperature=0.0,
                 low_confidence=True, repetition_penalty=1.2, block_length=0,
                 pixel_values=None):
        """继承父类采样,vision 占位符位永不重掩。pixel_values 给图。"""
        device = prompt_ids.device
        MASK_ID = self.config.mask_token_id
        P = prompt_ids.shape[1]
        IMG_PAD = self.config.image_pad_token_id

        # vision 特征预计算一次(采样循环里反复用)
        vis = self._compute_vision(pixel_values, device)

        resp = torch.full((1, gen_length), MASK_ID, dtype=torch.long, device=device)
        x = torch.cat([prompt_ids, resp], dim=1)
        attn = torch.ones_like(x)
        is_prompt = torch.zeros_like(x, dtype=torch.bool)
        is_prompt[:, :P] = True
        # vision 占位符位也永不重掩
        is_prompt |= (x == IMG_PAD)

        T = steps
        for k in range(1, T + 1):
            s = 1.0 - k / T
            # embed + 填 vision
            h = self._embed_with_vision(x, vis)
            h = self._run_transformer(h, attn, x.shape[1])
            logits = self.lm_head(h)

            masked = (x == MASK_ID) & (~is_prompt)
            idx = masked.nonzero(as_tuple=False)
            if idx.shape[0] == 0:
                break
            lm_logits = logits[idx[:, 0], idx[:, 1]]
            # 重复惩罚
            resp_ids = x[0, P:]
            seen = resp_ids[resp_ids != MASK_ID].unique()
            if repetition_penalty != 1.0 and seen.numel() > 0:
                seen_logits = lm_logits[:, seen]
                seen_logits = torch.where(seen_logits > 0,
                                          seen_logits / repetition_penalty,
                                          seen_logits * repetition_penalty)
                lm_logits = lm_logits.scatter(1, seen.unsqueeze(0).expand(idx.shape[0], -1),
                                               seen_logits)
            temp = max(temperature, 1e-4)
            prob = F.softmax(lm_logits / temp, dim=-1)
            pred = prob.argmax(dim=-1)
            conf = prob.gather(1, pred[:, None]).squeeze(1) if low_confidence \
                else torch.rand(idx.shape[0], device=device)
            x[idx[:, 0], idx[:, 1]] = pred
            n_remain = int(gen_length * s)
            n_remask = min(n_remain, idx.shape[0])
            if n_remask > 0:
                order = torch.argsort(conf)
                remask_pos = idx[order[:n_remask]]
                x[remask_pos[:, 0], remask_pos[:, 1]] = MASK_ID
        return x[:, P:]
