import torch
import torch.nn as nn
import timm
from transformers import CLIPModel


class SwinC(nn.Module):
    def __init__(self, model_id, num_classes=30, pretrained=True):
        super(SwinC, self).__init__()
        model_names = [
            'swin_tiny_patch4_window7_224.ms_in22k_ft_in1k',
            'swin_small_patch4_window7_224.ms_in22k_ft_in1k',
            'swin_base_patch4_window7_224.ms_in22k_ft_in1k',
            'swin_large_patch4_window7_224.ms_in22k_ft_in1k'
        ]
        model_name = model_names[model_id - 1]
        self.swin_model = timm.create_model(model_name, pretrained=pretrained, features_only=True)
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        swin_feature_dim = self.swin_model.feature_info[-1]['num_chs']
        clip_feature_dim = self.clip_model.config.text_config.hidden_size
        self.proj_swin = nn.Linear(swin_feature_dim * 7 * 7, clip_feature_dim)
        self.dropout = nn.Dropout(p=0.6)
        self.bn = nn.BatchNorm1d(clip_feature_dim)

        self.gmu = GatedMultimodalUnit(image_dim=clip_feature_dim, text_dim=clip_feature_dim, hidden_dim=512)
        combined_dim = 512

        self.fc = nn.Linear(combined_dim, num_classes)

    def forward(self, image, text_inputs):
        swin_features = self.swin_model(image)[-1]  # Get the last layer features (batch_size, 7, 7, 768)
        batch_size = swin_features.size(0)
        swin_features = swin_features.view(batch_size, -1)  # Flatten to (batch_size, 7*7*768)
        swin_features = self.proj_swin(swin_features)
        swin_features = self.bn(self.dropout(swin_features))
        text_features = self.clip_model.get_text_features(**text_inputs)  # (batch_size, 512)
        combined_features = self.gmu(swin_features, text_features)
        logits = self.fc(combined_features)
        return logits


class GatedMultimodalUnit(nn.Module):
    def __init__(self, image_dim, text_dim, hidden_dim):
        super(GatedMultimodalUnit, self).__init__()
        self.fc_image = nn.Linear(image_dim, hidden_dim)
        self.fc_text = nn.Linear(text_dim, hidden_dim)
        self.gate = nn.Linear(image_dim + text_dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.6)
        self.bn = nn.BatchNorm1d(hidden_dim)

    def forward(self, image_features, text_features):
        text_features = text_features.expand_as(image_features)
        image_proj = torch.relu(self.fc_image(image_features))
        image_proj = self.bn(self.dropout(image_proj))
        text_proj = torch.relu(self.fc_text(text_features))
        text_proj = self.bn(self.dropout(text_proj))
        gate = torch.sigmoid(self.gate(torch.cat((image_features, text_features), dim=1)))
        return gate * image_proj + (1 - gate) * text_proj
