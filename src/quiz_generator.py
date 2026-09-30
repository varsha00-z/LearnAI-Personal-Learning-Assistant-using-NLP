"""
quiz_generator.py
Generates multiple-choice and fill-in-the-blank questions from study notes
using NLP heuristics — no external AI APIs.
"""

import re
import random
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)


class QuizGenerator:
    """
    Generates quiz questions from:
      1. A pre-built knowledge-base CSV  (structured Q&A)
      2. Free-form text (heuristic sentence-based MCQ)
    """

    def __init__(self):
        self.stop_words = set(stopwords.words("english"))

    # ------------------------------------------------------------------
    # Knowledge-base quiz
    # ------------------------------------------------------------------
    def from_knowledge_base(self, df: pd.DataFrame, topic: str = "All",
                             n: int = 5) -> list:
        """
        Sample `n` Q&A rows from the knowledge base DataFrame.
        Returns list of dicts: {question, answer, topic, subtopic, type}.
        """
        if topic != "All":
            subset = df[df["topic"] == topic]
        else:
            subset = df

        subset = subset.dropna(subset=["question", "answer"])
        n = min(n, len(subset))
        if n == 0:
            return []

        sampled = subset.sample(n=n, random_state=random.randint(0, 9999))
        questions = []
        for _, row in sampled.iterrows():
            questions.append({
                "question": row["question"],
                "answer": row["answer"],
                "topic": row["topic"],
                "subtopic": row.get("subtopic", ""),
                "type": "short_answer",
                "options": None,
            })
        return questions

    # ------------------------------------------------------------------
    # MCQ from free text
    # ------------------------------------------------------------------
    def from_text(self, text: str, n: int = 5) -> list:
        """
        Generate n fill-in-the-blank MCQs from free-form text.
        Blanks are created by masking the most meaningful noun/keyword
        in key sentences, with three random distractors.
        """
        sentences = sent_tokenize(text)
        # Filter sentences that are long enough and meaningful
        sentences = [s.strip() for s in sentences
                     if len(s.split()) >= 8 and len(s) < 400]
        if not sentences:
            return []

        random.shuffle(sentences)
        questions = []

        for sent in sentences:
            if len(questions) >= n:
                break
            q = self._make_mcq(sent)
            if q:
                questions.append(q)

        return questions

    def _make_mcq(self, sentence: str) -> dict | None:
        """Create one MCQ by blanking the most important noun in the sentence."""
        tokens = word_tokenize(sentence)
        tagged = nltk.pos_tag(tokens)

        # Prefer proper nouns > nouns > adjectives
        candidates = [(w, t) for w, t in tagged
                      if t in ("NNP", "NNPS", "NN", "NNS", "JJ")
                      and w.lower() not in self.stop_words
                      and len(w) > 3
                      and w.isalpha()]

        if not candidates:
            return None

        # Pick the candidate that appears least often (more unique → better blank)
        word_counts = {w: sentence.lower().count(w.lower()) for w, _ in candidates}
        target_word = min(word_counts, key=word_counts.get)

        # Build question by replacing first occurrence
        question_text = re.sub(
            r"\b" + re.escape(target_word) + r"\b",
            "_______",
            sentence,
            count=1,
        )

        # Generate distractors: other noun-like words from the sentence
        distractors = [w for w, _ in candidates if w != target_word]
        # Pad with generic distractors if needed
        generic = ["algorithm", "function", "variable", "method", "object",
                   "module", "library", "parameter", "instance", "attribute"]
        random.shuffle(generic)
        distractors = (distractors + generic)[:3]

        options = [target_word] + distractors[:3]
        random.shuffle(options)

        return {
            "question": f"Fill in the blank: {question_text}",
            "answer": target_word,
            "topic": "Text-based",
            "subtopic": "",
            "type": "mcq",
            "options": options,
        }

    # ------------------------------------------------------------------
    # Mixed quiz (KB + text)
    # ------------------------------------------------------------------
    def generate_mixed_quiz(self, df: pd.DataFrame, text: str,
                            topic: str = "All", n: int = 8) -> list:
        """Combine KB questions and text-based MCQs for a richer quiz."""
        kb_n = max(1, n // 2)
        text_n = n - kb_n
        kb_qs = self.from_knowledge_base(df, topic=topic, n=kb_n)
        text_qs = self.from_text(text, n=text_n) if text.strip() else []
        combined = kb_qs + text_qs
        random.shuffle(combined)
        return combined[:n]

    # ------------------------------------------------------------------
    # Score a quiz attempt
    # ------------------------------------------------------------------
    @staticmethod
    def score_attempt(questions: list, user_answers: list) -> dict:
        """
        Compare user answers against correct answers.

        Parameters
        ----------
        questions    : list returned by any generate_* method
        user_answers : list of strings (one per question)

        Returns dict with score, total, percentage, and per-question breakdown.
        """
        total = len(questions)
        correct = 0
        breakdown = []

        for q, ua in zip(questions, user_answers):
            ua_clean = (ua or "").strip().lower()
            correct_clean = q["answer"].strip().lower()

            # Exact or contained match (generous for short-answer)
            is_correct = (
                ua_clean == correct_clean
                or correct_clean in ua_clean
                or ua_clean in correct_clean
            )
            if is_correct:
                correct += 1

            breakdown.append({
                "question": q["question"],
                "your_answer": ua,
                "correct_answer": q["answer"],
                "is_correct": is_correct,
                "topic": q["topic"],
            })

        return {
            "score": correct,
            "total": total,
            "percentage": round(100 * correct / total, 1) if total else 0,
            "breakdown": breakdown,
        }
