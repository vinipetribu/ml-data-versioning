import pandas as pd
import os

url = "hf://datasets/blendacesarguedes/pii-detection-dataset/pii_dataset.csv"

output_path = "data/raw/pii_dataset.csv"

os.makedirs("data/raw", exist_ok=True)

df = pd.read_csv(url)

df.to_csv(output_path, index=False)

print("Dataset baixado para data/raw/pii_dataset.csv")