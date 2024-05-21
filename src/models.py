import timm
import torch.nn as nn


def get_model(model_name, num_classes, pretrained=True):
    if model_name == 'vit_small_patch16_224':
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    elif model_name == 'swin_tiny_patch4_window7_224':
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    elif model_name == 'resnet18':
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    else:
        raise ValueError(f"Model {model_name} not recognized.")

    return model
