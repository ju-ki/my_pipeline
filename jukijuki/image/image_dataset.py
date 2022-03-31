import cv2
import torch
from torch.utils.data import Dataset


class SimpleTrainDataset(Dataset):
    def __init__(self, df, config, transform=None):
        self.df = df
        self.config = config
        self.file_path = self.df["file_path"].values
        self.target_col = self.df[self.config.target_col].values
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_image = self.file_path[idx]
        image = cv2.imread(file_image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)["image"]
        label = torch.tensor(self.target_col[idx], dtype=torch.float)
        return image, label


class SimpleTestDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.file_names = df['file_path'].values
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.file_names[idx]
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image=image)["image"]
        return image