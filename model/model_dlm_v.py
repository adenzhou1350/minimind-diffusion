"""
🌏🌎🌍 mind-diffusion-v: diffusion 版 minimind-v (多模态掩码扩散) 🌏🌎🌍

文本侧复用 mind-diffusion 的 DLMForMD(双向 transformer + 均匀权重掩码 CE)。
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
                 image_pad_token_id=6401,
                 freeze_vision=True,
                 projector_hidden=768,
                 vision_encoder_name='jingyaogong/siglip2-base-p32-256-ve',
                 **kwargs):
        self.image_hidden_size = image_hidden_size
        self.image_token_len = image_token_len
        self.image_pad_token_id = image_pad_token_id
        self.freeze_vision = freeze_vision
        self.projector_hidden = projector_hidden
        self.vision_encoder_name = vision_encoder_name
        super().__init__(image_pad_token_id=image_pad_token_id, **kwargs)
