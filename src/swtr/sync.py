from network import SwinTransformer
from config import get_config
import timm
import torch

MODEL_CONFIGS = {
    'swin_tiny_patch4_window7_224.ms_in22k_ft_in1k': {
        'embed_dim': 96,
        'depths': [2, 2, 6, 2],
        'num_heads': [3, 6, 12, 24]
    },
    'swin_base_patch4_window7_224.ms_in22k_ft_in1k': {
        'embed_dim': 128,
        'depths': [2, 2, 18, 2],
        'num_heads': [4, 8, 16, 32]
    },
    'swin_large_patch4_window7_224.ms_in22k_ft_in1k': {
        'embed_dim': 192,
        'depths': [2, 2, 18, 2],
        'num_heads': [6, 12, 24, 48]
    }
}


def load_pretrained_weights(model, model_name, num_classes):
    print(f"Loading pretrained weights for {model_name}")
    pretrained_model = timm.create_model(model_name, pretrained=True)
    pretrained_dict = pretrained_model.state_dict()

    model_dict = model.state_dict()

    pretrained_dict.pop('head.weight', None)
    pretrained_dict.pop('head.bias', None)

    model_dict.update(
        {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].size() == v.size()})

    model.load_state_dict(model_dict, strict=False)

    model.head = torch.nn.Linear(model.num_features, num_classes)
    torch.nn.init.trunc_normal_(model.head.weight, std=.02)
    if model.head.bias is not None:
        torch.nn.init.constant_(model.head.bias, 0)

    return model


def build_swtr(model_name, num_classes=30, pretrained=True):
    config = get_config()

    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Model name {model_name} not supported yet")

    model_config = MODEL_CONFIGS[model_name]

    model = SwinTransformer(
        img_size=224,
        patch_size=config.MODEL.SWIN.PATCH_SIZE,
        in_chans=config.MODEL.SWIN.IN_CHANS,
        num_classes=num_classes,
        embed_dim=model_config['embed_dim'],
        depths=model_config['depths'],
        num_heads=model_config['num_heads'],
        window_size=config.MODEL.SWIN.WINDOW_SIZE,
        mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
        qkv_bias=config.MODEL.SWIN.QKV_BIAS,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        ape=config.MODEL.SWIN.APE,
        patch_norm=config.MODEL.SWIN.PATCH_NORM
    )

    if pretrained:
        model = load_pretrained_weights(model, model_name, num_classes)

    return model
