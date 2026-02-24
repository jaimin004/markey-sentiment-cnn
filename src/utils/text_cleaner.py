import re
import unicodedata

def clean_text(text):
    # 1) Normalize unicode characters
    text = unicodedata.normalize("NFKD", text)

    # 2) Convert to lowercase
    text = text.lower()

    # 3) Remove line-breaks and extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # 4) Remove unwanted characters except punctuation .,!?
    text = re.sub(r"[^a-zA-Z0-9.,!? ]+", "", text)

    # 5) Remove isolated single characters
    text = re.sub(r"\b[a-zA-Z]\b", "", text)

    # 6) Clean extra spaces
    text = re.sub(' +', ' ', text)

    return text
