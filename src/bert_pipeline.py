import pandas as pd
import pickle
from transformers import AutoTokenizer

df = pd.read_csv("data/processed/clean_text.csv")

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

tokens = tokenizer(
    df["text"].tolist(),
    padding=True,
    truncation=True
)

with open("data/features/bert_tokens.pkl", "wb") as f:
    pickle.dump(tokens, f)

print("BERT tokens salvos em data/features/bert_tokens.pkl")