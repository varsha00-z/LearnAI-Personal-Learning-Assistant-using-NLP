"""
train_model.py
One-time script to train the topic classifier and save it to models/.
Run this once before launching the Streamlit app (or the app auto-trains).

Usage:
    python train_model.py
"""

import os
import sys

# Allow imports from project root
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
from src.nlp_engine import TopicClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
import numpy as np

KB_PATH = os.path.join("data", "knowledge_base.csv")
MODEL_PATH = os.path.join("models", "topic_classifier.joblib")


def train():
    print("=" * 55)
    print("  LearnAI — Topic Classifier Training")
    print("=" * 55)

    # 1. Load knowledge base
    if not os.path.exists(KB_PATH):
        print(f"[ERROR] Knowledge base not found at {KB_PATH}")
        sys.exit(1)

    df = pd.read_csv(KB_PATH)
    print(f"[INFO] Loaded {len(df)} rows from knowledge base.")
    print(f"[INFO] Topics: {df['topic'].unique().tolist()}")

    # 2. Train classifier
    clf = TopicClassifier()
    clf.train(df, text_col="question", label_col="topic")
    print("[INFO] Classifier trained.")

    # 3. Quick cross-validation
    from src.nlp_engine import TextPreprocessor
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.preprocessing import LabelEncoder
    from sklearn.pipeline import Pipeline

    preprocessor = TextPreprocessor()
    texts = df["question"].apply(preprocessor.preprocess).tolist()
    le = LabelEncoder()
    labels = le.fit_transform(df["topic"])

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=3000, ngram_range=(1, 2),
                                  stop_words="english", sublinear_tf=True)),
        ("clf", MultinomialNB(alpha=0.5)),
    ])

    cv_scores = cross_val_score(pipeline, texts, labels, cv=3, scoring="accuracy")
    print(f"[INFO] Cross-validation accuracy: {np.mean(cv_scores):.3f} "
          f"(± {np.std(cv_scores):.3f})")

    # 4. Save
    os.makedirs("models", exist_ok=True)
    clf.save(MODEL_PATH)
    print(f"[INFO] Model saved to: {MODEL_PATH}")
    print("[DONE] Training complete.")


if __name__ == "__main__":
    train()
