import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# carregar dataset limpo
df = pd.read_csv("data/processed/clean_text.csv")

# criar vetor TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(df["text"])

# salvar features
with open("data/features/tfidf_features.pkl", "wb") as f:
    pickle.dump(X, f)

print("TF-IDF features salvas em data/features/tfidf_features.pkl")