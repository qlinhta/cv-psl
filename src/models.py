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


def get_model_by_id(model_id):
    for model in available_models:
        if model.model_id == model_id:
            return model
    raise ValueError(f"Model ID {model_id} not recognized.")


available_models = [
    Model(1, 'swin_tiny_patch4_window7_224.ms_in22k_ft_in1k', num_classes=30),
    Model(2, 'swin_small_patch4_window7_224.ms_in22k_ft_in1k', num_classes=30),
    Model(3, 'swin_base_patch4_window7_224.ms_in22k_ft_in1k', num_classes=30),
    Model(4, 'swin_large_patch4_window7_224.ms_in22k_ft_in1k', num_classes=30)
]
