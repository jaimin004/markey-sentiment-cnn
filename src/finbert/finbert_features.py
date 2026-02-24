import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

print("🔥 Loading FinBERT model...")
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
finbert = AutoModel.from_pretrained("ProsusAI/finbert").to(device)
print("✔ FinBERT loaded.")


def extract_finbert_embeddings(text):
    """
    Used for:
    - OCR text (inference)
    - Optional fusion experiments
    """

    encoded = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        output = finbert(**encoded)

    cls_embedding = output.last_hidden_state[:, 0, :]
    return cls_embedding.squeeze(0)   # shape: (768,)


class FinBERTSentimentModel(nn.Module):
    """
    Used for:
    - Training on financial text CSV
    - Inference on OCR text from screenshots
    """

    def __init__(self, num_classes=3):
        super().__init__()

        self.finbert = finbert
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.finbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)

        logits = self.classifier(cls_output)
        return logits


def tokenize_text(text):
    """
    Shared tokenizer for:
    - CSV training text
    - OCR extracted text
    """

    encoded = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=256
    )

    return {
        "input_ids": encoded["input_ids"].to(device),
        "attention_mask": encoded["attention_mask"].to(device)
    }


def process_all_text(
    input_folder="data/processed_text",
    output_folder="data/text_features"
):
    """
    Keeps backward compatibility with your
    earlier fusion-based pipeline.
    """

    input_folder = os.path.abspath(input_folder)
    output_folder = os.path.abspath(output_folder)

    os.makedirs(output_folder, exist_ok=True)

    txt_files = [f for f in os.listdir(input_folder) if f.endswith(".txt")]

    if not txt_files:
        print("❌ No .txt files found.")
        return

    for txt_file in tqdm(txt_files, desc="Extracting FinBERT features"):
        path = os.path.join(input_folder, txt_file)

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        embedding = extract_finbert_embeddings(text)

        out_path = os.path.join(
            output_folder, txt_file.replace(".txt", ".pt")
        )
        torch.save(embedding.cpu(), out_path)

    print("🎉 FinBERT embedding extraction complete!")


if __name__ == "__main__":
    process_all_text()
