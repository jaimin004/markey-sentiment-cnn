import os
import torch
import pandas as pd
from torch.utils.data import Dataset

LABEL_MAP = {
    "positive": 0,
    "neutral": 1,
    "negative": 2
}

class TextSentimentDataset(Dataset):
    """
    Used for training FinBERT sentiment model
    using financial text CSV.
    """

    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        text = row["text"]
        label = LABEL_MAP[row["label"].strip().lower()]

        return text, torch.tensor(label, dtype=torch.long)

class FusionDataset(Dataset):
    """
    Used for:
    - Fusion-based inference
    - Screenshot-based evaluation
    """

    def __init__(self, csv_path, visual_dir, text_dir):
        self.df = pd.read_csv(csv_path)
        self.visual_dir = visual_dir
        self.text_dir = text_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        base = os.path.splitext(row["image_name"])[0]

        visual = torch.load(os.path.join(self.visual_dir, base + ".pt"))
        text = torch.load(os.path.join(self.text_dir, base + ".pt"))

        label = LABEL_MAP[row["sentiment"].strip().lower()]

        return visual, text, torch.tensor(label, dtype=torch.long)
