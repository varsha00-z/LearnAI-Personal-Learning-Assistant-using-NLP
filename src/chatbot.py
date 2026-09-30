"""
chatbot.py
Rule-based AI study chatbot that answers student questions
using TF-IDF cosine similarity against the knowledge base.
No external AI APIs — fully offline.
"""

import pandas as pd
import numpy as np
from src.nlp_engine import TFIDFEngine, TextPreprocessor, TopicClassifier

GREETING_TRIGGERS = {"hi", "hello", "hey", "good morning", "good evening", "howdy"}
FAREWELL_TRIGGERS = {"bye", "goodbye", "quit", "exit", "see you", "thanks"}


class StudyChatbot:
    """
    Retrieval-based chatbot that:
      - Detects greeting / farewell intents
      - Classifies the question topic (Naive Bayes)
      - Retrieves the best-matching answer via TF-IDF cosine similarity
      - Falls back gracefully when no good match is found
    """

    def __init__(self, knowledge_base: pd.DataFrame):
        self.kb = knowledge_base.dropna(subset=["question", "answer"]).copy()
        self.preprocessor = TextPreprocessor()
        self.classifier = TopicClassifier()
        self.tfidf = TFIDFEngine(max_features=4000, ngram_range=(1, 2))

        # Fit TF-IDF on all questions in the KB
        self._questions = self.kb["question"].tolist()
        self._answers = self.kb["answer"].tolist()
        self._topics = self.kb["topic"].tolist()
        self.tfidf.fit(self._questions)

        # Train or load classifier
        if not self.classifier.load():
            self.classifier.train(self.kb)
            self.classifier.save()

        self.chat_history: list = []   # [(role, text), ...]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def respond(self, user_input: str) -> dict:
        """
        Process a user message and return a response dict:
          {reply, topic, confidence, matched_question, source}
        """
        user_input = user_input.strip()
        if not user_input:
            return self._reply("Please type a question!", "—", 0.0)

        lower = user_input.lower()

        # --- Intent: greeting ---
        if any(g in lower for g in GREETING_TRIGGERS):
            return self._reply(
                "Hello! 👋 I'm your AI study assistant. Ask me anything about "
                "Python, Machine Learning, NLP, Data Science, or Algorithms!",
                "Greeting", 1.0
            )

        # --- Intent: farewell ---
        if any(f in lower for f in FAREWELL_TRIGGERS):
            return self._reply(
                "Goodbye! Keep studying hard. You've got this! 🎓", "Farewell", 1.0
            )

        # --- Topic classification ---
        topic = self.classifier.predict(user_input)
        topic_probs = self.classifier.predict_proba(user_input)
        confidence = max(topic_probs.values()) if topic_probs else 0.0

        # --- Retrieval ---
        results = self.tfidf.search(user_input, top_k=3)

        if not results or results[0][1] < 0.05:
            return self._reply(
                f"I'm not sure about that one. Try rephrasing your question, "
                f"or upload study notes so I can learn from them. "
                f"(Detected topic: **{topic}**)",
                topic, confidence
            )

        best_idx, best_score = results[0]
        answer = self._answers[best_idx]
        matched_q = self._questions[best_idx]
        matched_topic = self._topics[best_idx]

        # Build a friendly reply
        reply = (
            f"**{answer}**\n\n"
            f"*(Matched topic: {matched_topic} · Similarity: {best_score:.2f})*"
        )

        # Log
        self.chat_history.append(("user", user_input))
        self.chat_history.append(("bot", reply))

        return {
            "reply": reply,
            "topic": matched_topic,
            "confidence": round(best_score, 3),
            "matched_question": matched_q,
            "source": "knowledge_base",
        }

    def add_document(self, text: str, topic: str = "User Notes"):
        """
        Extend the chatbot's knowledge with uploaded text.
        Each sentence becomes a searchable document.
        """
        from nltk.tokenize import sent_tokenize
        sentences = [s.strip() for s in sent_tokenize(text) if len(s.split()) >= 6]
        new_rows = [{"question": s, "answer": s, "topic": topic, "subtopic": "Notes"}
                    for s in sentences]
        if new_rows:
            new_df = pd.DataFrame(new_rows)
            self.kb = pd.concat([self.kb, new_df], ignore_index=True)
            self._questions = self.kb["question"].tolist()
            self._answers = self.kb["answer"].tolist()
            self._topics = self.kb["topic"].tolist()
            self.tfidf.fit(self._questions)

    def clear_history(self):
        self.chat_history = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _reply(text: str, topic: str, confidence: float) -> dict:
        return {
            "reply": text,
            "topic": topic,
            "confidence": confidence,
            "matched_question": "",
            "source": "rule",
        }
