import cv2
import numpy as np
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset


class SimSiam(nn.Module):
    def __init__(self, base_encoder, out_dim=2048, pred_dim=512):
        """
        parameter:
            base_encoder: (exp: timm.model(model_name), pretrain=True)
            backbone: (exp: nn.Sequential(*list(base_encoder.children())[:-1])
            out_dim(int): model.num_feature
            pred_dim(int): num_feature / 4

        default (resnet50):
            out_dim = 2048
            pred_dim = 512

        ref:
           https://github.com/facebookresearch/simsiam/blob/main/simsiam/builder.py
           https://github.com/lightly-ai/lightly

        """
        super(SimSiam, self).__init__()
        self.encoder = base_encoder
        self.backbone = nn.Sequential(*list(self.encoder.children())[:-1])
        # resnet50d ->2048, resnet34d, resnet18 -> 512
        prev_dim = self.encoder.fc.weight.shape[1]
        "3-layer predictor"
        self.encoder.fc = nn.Sequential(
            nn.Linear(prev_dim, prev_dim, bias=False),
            nn.BatchNorm1d(prev_dim),
            nn.ReLU(inplace=True),
            nn.Linear(prev_dim, prev_dim, bias=False),
            nn.BatchNorm1d(prev_dim),
            nn.ReLU(inplace=True),
            self.encoder.fc,
            nn.BatchNorm1d(out_dim, affine=False)
        )
        self.encoder.fc[6].bias.requires_grad = False

        "2-layer predictor"
        self.predictor = nn.Sequential(nn.Linear(out_dim, pred_dim, bias=False),
                                       nn.BatchNorm1d(pred_dim),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(pred_dim, out_dim)
                                       )

    def forward(self, x1, x2):
        z1, z2 = self.encoder(x1), self.encoder(x2)
        p1, p2 = self.predictor(z1), self.predictor(z2)

        return p1, p2, z1.detach(), z2.detach()


class SimsiamDataset(Dataset):
    """
    ref:
       https://github.com/tawatawara/atmaCup-11/blob/main/src/train_simsiam.py
    """

    def __init__(self, df: pd.DataFrame, filepath_name: str, target_name: str, transform=None):
        self.transform = transform
        self.filepath = df[filepath_name].values
        self.labels = df[target_name].values
        self.len = len(self.filepath)

    def __len__(self):
        return self.len

    def _read_input_file(self, path):
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _apply_transform(self, image: np.ndarray):
        image = self.transform(image=image)["image"]
        return image

    def __getitem__(self, idx):
        path, label = self.filepath[idx], self.labels[idx]
        data = self._read_input_file(path)
        data0 = self._apply_transform(data)
        data1 = self._apply_transform(data)
        return {"data0": data0, "data1": data1, "target": label}


criterion = nn.CosineSimilarity(dim=1)


def loss_function(criterion, z1, z2, p1, p2):
    loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5
    return loss
