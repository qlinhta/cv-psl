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
    Model(2, 'vit_small_patch16_224.augreg_in1k', num_classes=30),
    Model(3, 'vit_small_patch16_224.augreg_in21k_ft_in1k', num_classes=30),
    Model(4, 'swin_tiny_patch4_window7_224', num_classes=30),
    Model(5, 'swin_tiny_patch4_window7_224.ms_in1k', num_classes=30),
    Model(6, 'swin_tiny_patch4_window7_224.ms_in22k', num_classes=30),
    Model(7, 'swin_tiny_patch4_window7_224.ms_in22k_ft_in1k', num_classes=30),
    Model(8, 'swin_base_patch4_window7_224', num_classes=30),
    Model(9, 'swin_base_patch4_window7_224.ms_in1k', num_classes=30),
    Model(10, 'swin_base_patch4_window7_224.ms_in21k', num_classes=30),
    Model(11, 'swin_base_patch4_window7_224.ms_in21k_ft_in1k', num_classes=30),
]


# parameters: {'batch_size': 128, 'lr': 5.728983638103915e-05, 'weight_decay': 0.0005535766560991741}, lr=1e-4


def get_model_by_id(model_id):
    for model in available_models:
        if model.model_id == model_id:
            return model
    raise ValueError(f"Model ID {model_id} not recognized.")
