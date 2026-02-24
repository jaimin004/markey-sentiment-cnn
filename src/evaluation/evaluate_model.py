import torch
import pandas as pd
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, mean_squared_error
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
finbert = AutoModel.from_pretrained("ProsusAI/finbert").to(DEVICE)

class SentimentClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(768, 3)

    def forward(self, x):
        return self.fc(x)

classifier = SentimentClassifier().to(DEVICE)

checkpoint = torch.load("finbert_text_sentiment_model.pt", map_location=DEVICE)
finbert.load_state_dict(checkpoint["finbert"])
classifier.load_state_dict(checkpoint["classifier"])

finbert.eval()
classifier.eval()

class TextDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path).dropna()
        self.texts = df["text"].tolist()
        self.labels = df["label"].str.lower().map({
            "positive": 0,
            "neutral": 1,
            "negative": 2
        }).tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

dataset = TextDataset("data/train.csv")

loader = DataLoader(dataset, batch_size=16, shuffle=False)

y_true = []
y_pred = []

with torch.no_grad():
    for texts, labels in tqdm(loader, desc="Evaluating"):
        enc = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(DEVICE)

        outputs = finbert(**enc)
        cls_embed = outputs.last_hidden_state[:, 0, :]
        logits = classifier(cls_embed)

        preds = torch.argmax(logits, dim=1).cpu().numpy()

        y_pred.extend(preds)
        y_true.extend(labels)


accuracy = accuracy_score(y_true, y_pred)
f1_macro = f1_score(y_true, y_pred, average="macro")
f1_weighted = f1_score(y_true, y_pred, average="weighted")
mse = mean_squared_error(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)

print("\n📊 MODEL EVALUATION METRICS")
print("--------------------------------")
print(f"Accuracy       : {accuracy:.4f}")
print(f"F1 Score (Macro): {f1_macro:.4f}")
print(f"F1 Score (Weighted): {f1_weighted:.4f}")
print(f"MSE            : {mse:.4f}")

print("\n🧮 Confusion Matrix (rows=true, cols=pred):")
print(cm)
