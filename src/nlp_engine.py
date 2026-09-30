"""
nlp_engine.py
Core NLP preprocessing, TF-IDF vectorisation, cosine similarity, and topic classification.
"""

import re
import string
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# ---------------------------------------------------------------------------
# NLTK resource bootstrap
# ---------------------------------------------------------------------------
def download_nltk_resources():
    resources = ["punkt", "stopwords", "wordnet", "averaged_perceptron_tagger",
                 "punkt_tab", "omw-1.4"]
    for r in resources:
        try:
            nltk.download(r, quiet=True)
        except Exception:
            pass

download_nltk_resources()

# ---------------------------------------------------------------------------
# Text Preprocessing
# ---------------------------------------------------------------------------
class TextPreprocessor:
    """Clean, tokenise, lemmatise and remove stop-words from text."""

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words("english"))

    def clean(self, text: str) -> str:
        """Lowercase, remove punctuation and extra whitespace."""
        text = text.lower()
        text = re.sub(r"http\S+|www\S+", "", text)          # URLs
        text = re.sub(r"[^a-z0-9\s]", " ", text)            # non-alphanumeric
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize(self, text: str) -> list:
        return word_tokenize(self.clean(text))

    def remove_stopwords(self, tokens: list) -> list:
        return [t for t in tokens if t not in self.stop_words and len(t) > 2]

    def lemmatize(self, tokens: list) -> list:
        return [self.lemmatizer.lemmatize(t) for t in tokens]

    def stem(self, tokens: list) -> list:
        return [self.stemmer.stem(t) for t in tokens]

    def preprocess(self, text: str, use_stemming: bool = False) -> str:
        """Full pipeline → returns a cleaned string ready for vectorisation."""
        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)
        if use_stemming:
            tokens = self.stem(tokens)
        else:
            tokens = self.lemmatize(tokens)
        return " ".join(tokens)

    def get_sentences(self, text: str) -> list:
        return sent_tokenize(text)

    def get_token_stats(self, text: str) -> dict:
        raw_tokens = word_tokenize(text)
        clean_tokens = self.remove_stopwords(self.tokenize(text))
        return {
            "total_tokens": len(raw_tokens),
            "unique_tokens": len(set(raw_tokens)),
            "clean_tokens": len(clean_tokens),
            "sentences": len(self.get_sentences(text)),
            "avg_word_length": np.mean([len(t) for t in clean_tokens]) if clean_tokens else 0,
        }


# ---------------------------------------------------------------------------
# TF-IDF Engine
# ---------------------------------------------------------------------------
class TFIDFEngine:
    """TF-IDF vectoriser with cosine-similarity search over a document corpus."""

    def __init__(self, max_features: int = 5000, ngram_range: tuple = (1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words="english",
            sublinear_tf=True,
        )
        self.matrix = None
        self.documents = []
        self.preprocessor = TextPreprocessor()

    def fit(self, documents: list):
        self.documents = documents
        cleaned = [self.preprocessor.preprocess(d) for d in documents]
        self.matrix = self.vectorizer.fit_transform(cleaned)
        return self

    def search(self, query: str, top_k: int = 5) -> list:
        """Return top-k (index, score) tuples most similar to the query."""
        if self.matrix is None:
            raise ValueError("TFIDFEngine must be fitted before searching.")
        q_vec = self.vectorizer.transform([self.preprocessor.preprocess(query)])
        scores = cosine_similarity(q_vec, self.matrix).flatten()
        top_indices = scores.argsort()[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_indices if scores[i] > 0]

    def get_top_keywords(self, text: str, top_k: int = 10) -> list:
        """Return the top-k TF-IDF keywords from a piece of text."""
        vec = self.vectorizer.transform([self.preprocessor.preprocess(text)])
        feature_names = self.vectorizer.get_feature_names_out()
        scores = vec.toarray().flatten()
        top_idx = scores.argsort()[::-1][:top_k]
        return [(feature_names[i], float(scores[i])) for i in top_idx if scores[i] > 0]

    def similarity(self, text_a: str, text_b: str) -> float:
        """Cosine similarity between two texts."""
        vecs = self.vectorizer.transform([
            self.preprocessor.preprocess(text_a),
            self.preprocessor.preprocess(text_b),
        ])
        return float(cosine_similarity(vecs[0], vecs[1])[0][0])


# ---------------------------------------------------------------------------
# Topic / Intent Classifier
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "topic_classifier.joblib")

class TopicClassifier:
    """
    Multinomial Naive Bayes topic classifier trained on the knowledge base.
    Saved/loaded via joblib so the Streamlit app never re-trains at runtime.
    """

    def __init__(self):
        self.pipeline: Pipeline | None = None
        self.label_encoder = LabelEncoder()
        self.preprocessor = TextPreprocessor()
        self.classes_: list = []

    # -- Training -----------------------------------------------------------
    def train(self, df: pd.DataFrame, text_col: str = "question", label_col: str = "topic"):
        texts = df[text_col].apply(self.preprocessor.preprocess).tolist()
        labels = df[label_col].tolist()
        encoded = self.label_encoder.fit_transform(labels)
        self.classes_ = list(self.label_encoder.classes_)
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=3000, ngram_range=(1, 2),
                                      stop_words="english", sublinear_tf=True)),
            ("clf", MultinomialNB(alpha=0.5)),
        ])
        self.pipeline.fit(texts, encoded)
        return self

    def save(self, path: str = MODEL_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"pipeline": self.pipeline,
                     "label_encoder": self.label_encoder,
                     "classes": self.classes_}, path)

    def load(self, path: str = MODEL_PATH):
        if not os.path.exists(path):
            return False
        data = joblib.load(path)
        self.pipeline = data["pipeline"]
        self.label_encoder = data["label_encoder"]
        self.classes_ = data["classes"]
        return True

    # -- Inference ----------------------------------------------------------
    def predict(self, text: str) -> str:
        if self.pipeline is None:
            return "Unknown"
        cleaned = self.preprocessor.preprocess(text)
        encoded = self.pipeline.predict([cleaned])[0]
        return self.label_encoder.inverse_transform([encoded])[0]

    def predict_proba(self, text: str) -> dict:
        """Return a dict of {topic: probability}."""
        if self.pipeline is None:
            return {}
        cleaned = self.preprocessor.preprocess(text)
        probs = self.pipeline.predict_proba([cleaned])[0]
        return dict(zip(self.classes_, probs.tolist()))
