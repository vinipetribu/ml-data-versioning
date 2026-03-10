import pandas as pd
import re

df = pd.read_csv("data/raw/pii_dataset.csv")

def clean_text(text):

    text = text.lower()
    text = re.sub(r'[^a-z0-9\s@.]', '', text)
    text = re.sub(r'\s+', ' ', text)

    return text

df["text"] = df["text"].apply(clean_text)

df.to_csv("data/processed/clean_text.csv", index=False)