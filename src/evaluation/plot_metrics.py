import torch
import pandas as pd
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
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

y_true, y_pred = [], []

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
cm = confusion_matrix(y_true, y_pred)

report = classification_report(
    y_true, y_pred,
    target_names=["Positive", "Neutral", "Negative"],
    output_dict=True
)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["Positive", "Neutral", "Negative"],
    yticklabels=["Positive", "Neutral", "Negative"]
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()

metrics_df = pd.DataFrame(report).transpose().iloc[:3]
metrics_df = metrics_df[["precision", "recall", "f1-score"]]

metrics_df.plot(
    kind="bar",
    figsize=(7, 5),
    ylim=(0, 1),
    title="Precision, Recall, and F1-score per Class"
)
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.grid(axis="y")
plt.tight_layout()
plt.savefig("classwise_metrics.png", dpi=300)
plt.show()

plt.figure(figsize=(5, 4))
plt.bar(
    ["Accuracy", "F1 (Macro)", "F1 (Weighted)"],
    [accuracy, f1_macro, f1_weighted]
)
plt.ylim(0, 1)
plt.title("Overall Model Performance")
plt.ylabel("Score")
plt.grid(axis="y")
plt.tight_layout()
plt.savefig("overall_performance.png", dpi=300)
plt.show()

print("\n✅ All plots saved successfully!")
