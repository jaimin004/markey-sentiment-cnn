import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("🔥 Loading FinBERT model...")
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
finbert = AutoModel.from_pretrained("ProsusAI/finbert").to(DEVICE)
print("✔ FinBERT loaded.")

class TextDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.df = self.df.dropna()
        self.texts = self.df["text"].tolist()
        self.labels = self.df["label"].str.lower().tolist()

        self.label_map = {
            "positive": 0,
            "neutral": 1,
            "negative": 2
        }

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.label_map[self.labels[idx]]

class SentimentClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(768, 3)

    def forward(self, x):
        return self.fc(x)

classifier = SentimentClassifier().to(DEVICE)

dataset = TextDataset("data/train.csv")

loader = DataLoader(
    dataset,
    batch_size=16,      # 🔥 increased batch
    shuffle=True
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    list(finbert.parameters()) + list(classifier.parameters()),
    lr=2e-5
)

EPOCHS = 2

for epoch in range(EPOCHS):
    total_loss = 0
    print(f"\n🚀 Epoch {epoch+1}/{EPOCHS}")

    for texts, labels in tqdm(loader):
        enc = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=128,   # 🔥 reduced length
            return_tensors="pt"
        ).to(DEVICE)

        labels = labels.to(DEVICE)

        outputs = finbert(**enc)
        cls_embed = outputs.last_hidden_state[:, 0, :]

        logits = classifier(cls_embed)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"✅ Epoch {epoch+1} Loss: {total_loss:.4f}")

torch.save({
    "finbert": finbert.state_dict(),
    "classifier": classifier.state_dict()
}, "finbert_text_sentiment_model.pt")

print("\n🎉 Training complete & model saved!")
