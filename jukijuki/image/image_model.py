import timm
import torch.nn as nn


class SimpleCustomModel(nn.Module):
    def __init__(self, config, pretrained=False):
        super().__init__()
        self.config = config
        self.model = timm.create_model(self.config.model_name, pretrained=pretrained)
        if "resnet" in self.config.model_name:
            self.n_features = self.model.fc.in_features
            self.model.reset_classifier(0)
        elif "efficient" in self.config.model_name:
            self.n_features = self.model.classifier.in_features
            self.model.reset_classifier(0)
        elif 'vit' in self.config.model_name or "swin" in self.config.model_name:
            self.n_features = self.model.head.in_features
            self.model.reset_classifier(0)
        elif "conv" in self.config.model_name:
            self.n_features = self.model.get_classifier().in_features
            self.model.reset_classifier(0)
        self.fc = nn.Linear(self.n_features, self.config.target_size)

    def feature(self, image):
        return self.model(image)

    def forward(self, image):
        feature = self.feature(image)
        output = self.fc(feature)
        return output