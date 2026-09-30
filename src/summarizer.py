"""
summarizer.py
Extractive text summariser using TF-IDF sentence scoring.
No external AI APIs — pure NLP with NLTK + scikit-learn.
"""

import re
import numpy as np
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)


class Summarizer:
    """
    Extractive summariser.
    Scores each sentence by the sum of TF-IDF weights of its tokens,
    then returns the top-N sentences in original document order.
    """

    def __init__(self):
        self.stop_words = list(stopwords.words("english"))

    # ------------------------------------------------------------------
    def summarize(self, text: str, ratio: float = 0.3, max_sentences: int = 8) -> str:
        """
        Summarise *text* and return a bullet-point string.

        Parameters
        ----------
        text          : raw input text
        ratio         : fraction of sentences to keep (0 < ratio <= 1)
        max_sentences : hard ceiling on output sentences
        """
        sentences = sent_tokenize(text)
        if len(sentences) <= 3:
            return text  # too short to summarise

        n_keep = max(2, min(max_sentences, int(len(sentences) * ratio)))

        # Build TF-IDF matrix over sentences
        vectorizer = TfidfVectorizer(stop_words="english")
        try:
            tfidf_matrix = vectorizer.fit_transform(sentences)
        except ValueError:
            return text

        # Score each sentence = sum of its TF-IDF values
        sentence_scores = np.asarray(tfidf_matrix.sum(axis=1)).flatten()

        # Pick top-n indices, preserve order
        top_indices = sorted(
            np.argsort(sentence_scores)[::-1][:n_keep]
        )
        summary_sentences = [sentences[i] for i in top_indices]
        return "\n\n".join(f"• {s.strip()}" for s in summary_sentences)

    # ------------------------------------------------------------------
    def key_points(self, text: str, top_k: int = 5) -> list:
        """Return a list of the most informative sentences."""
        sentences = sent_tokenize(text)
        if not sentences:
            return []

        vectorizer = TfidfVectorizer(stop_words="english")
        try:
            tfidf_matrix = vectorizer.fit_transform(sentences)
        except ValueError:
            return sentences[:top_k]

        scores = np.asarray(tfidf_matrix.sum(axis=1)).flatten()
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [sentences[i].strip() for i in sorted(top_idx)]

    # ------------------------------------------------------------------
    def extract_topics_from_text(self, text: str, top_k: int = 10) -> list:
        """Return the top-k TF-IDF keywords from the text."""
        vectorizer = TfidfVectorizer(
            stop_words="english", max_features=200, ngram_range=(1, 1)
        )
        try:
            tfidf_matrix = vectorizer.fit_transform([text])
        except ValueError:
            return []
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray().flatten()
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [feature_names[i] for i in top_idx if scores[i] > 0]

    # ------------------------------------------------------------------
    def word_frequency(self, text: str) -> dict:
        """Return word → count dict (stop-words excluded)."""
        from collections import Counter
        import string
        words = re.findall(r"\b[a-z]{3,}\b", text.lower())
        stop = set(self.stop_words)
        words = [w for w in words if w not in stop]
        return dict(Counter(words).most_common(30))
