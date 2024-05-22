import timm
import torch.nn as nn


class Model:
    def __init__(self, model_id, name, num_classes, pretrained=True):
        self.model_id = model_id
        self.name = name
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.model = None

    def _create_model(self):
        if self.model is None:
            self.model = timm.create_model(self.name, pretrained=self.pretrained, num_classes=self.num_classes)
        return self.model

    def get_model(self):
        return self._create_model()

    def get_name(self):
        return self.name


available_models = [
    Model(1, 'vit_small_patch16_224', num_classes=30),
    Model(2, 'swin_tiny_patch4_window7_224', num_classes=30),
    Model(3, 'swin_base_patch4_window7_224', num_classes=30),
    Model(4, 'convnext_tiny', num_classes=30),
    Model(5, 'resnet50', num_classes=30),
    Model(6, 'nfnet_f0', num_classes=30, pretrained=False),
    Model(7, 'deit_small_patch16_224', num_classes=30),
    Model(8, 'regnety_040', num_classes=30),
    Model(9, 'cait_s24_224', num_classes=30),
    Model(10, 'coat_lite_small', num_classes=30),
    Model(11, 'levit_128s', num_classes=30),
    Model(12, 'pvt_v2_b2', num_classes=30),
    Model(13, 'tnt_s_patch16_224', num_classes=30),
    Model(14, 'twins_svt_small', num_classes=30),
    Model(15, 'efficientformer_l1', num_classes=30),
    Model(16, 'vit_small_patch16_224.augreg_in21k_ft_in1k', num_classes=30),
]


def get_model_by_id(model_id):
    for model in available_models:
        if model.model_id == model_id:
            return model
    raise ValueError(f"Model ID {model_id} not recognized.")
