"""
progress_tracker.py
Manages quiz history, computes learning progress, identifies weak topics,
and produces chart-ready data — all in-memory using pandas/numpy.
"""

import json
import os
from datetime import datetime
import numpy as np
import pandas as pd

PROGRESS_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "progress.json")


class ProgressTracker:
    """
    Persists quiz results in a JSON file and exposes analytics methods.
    Each record:  {timestamp, topic, score, total, percentage, session_id}
    """

    def __init__(self, filepath: str = PROGRESS_FILE):
        self.filepath = filepath
        self.records: list = self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _load(self) -> list:
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return []

    def save(self):
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        with open(self.filepath, "w") as f:
            json.dump(self.records, f, indent=2)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------
    def record_quiz(self, topic: str, score: int, total: int,
                    breakdown: list | None = None):
        """Append a quiz result to the history."""
        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "topic": topic,
            "score": score,
            "total": total,
            "percentage": round(100 * score / total, 1) if total else 0,
            "breakdown": breakdown or [],
        }
        self.records.append(record)
        self.save()

    def clear(self):
        self.records = []
        self.save()

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------
    def to_dataframe(self) -> pd.DataFrame:
        if not self.records:
            return pd.DataFrame(columns=["timestamp", "topic", "score",
                                         "total", "percentage"])
        df = pd.DataFrame(self.records)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    def overall_stats(self) -> dict:
        df = self.to_dataframe()
        if df.empty:
            return {"quizzes_taken": 0, "avg_score": 0,
                    "best_score": 0, "total_questions": 0}
        return {
            "quizzes_taken": len(df),
            "avg_score": round(df["percentage"].mean(), 1),
            "best_score": round(df["percentage"].max(), 1),
            "total_questions": int(df["total"].sum()),
        }

    def topic_stats(self) -> pd.DataFrame:
        """Return per-topic average score, attempts, and trend."""
        df = self.to_dataframe()
        if df.empty:
            return pd.DataFrame(columns=["topic", "avg_score",
                                         "attempts", "trend"])
        grouped = (
            df.groupby("topic")
            .agg(
                avg_score=("percentage", "mean"),
                attempts=("percentage", "count"),
                last_score=("percentage", "last"),
                first_score=("percentage", "first"),
            )
            .reset_index()
        )
        grouped["avg_score"] = grouped["avg_score"].round(1)
        grouped["trend"] = (grouped["last_score"] - grouped["first_score"]).round(1)
        return grouped

    def weak_topics(self, threshold: float = 60.0) -> list:
        """Return topics where the average score is below threshold."""
        stats = self.topic_stats()
        if stats.empty:
            return []
        weak = stats[stats["avg_score"] < threshold]["topic"].tolist()
        return weak

    def strong_topics(self, threshold: float = 80.0) -> list:
        stats = self.topic_stats()
        if stats.empty:
            return []
        return stats[stats["avg_score"] >= threshold]["topic"].tolist()

    def score_over_time(self) -> pd.DataFrame:
        """Return timestamp + percentage for line-chart plotting."""
        df = self.to_dataframe()
        if df.empty:
            return pd.DataFrame(columns=["timestamp", "percentage", "topic"])
        return df[["timestamp", "percentage", "topic"]].sort_values("timestamp")

    def topic_score_distribution(self) -> dict:
        """Return {topic: [list of percentage scores]} for box-plot data."""
        df = self.to_dataframe()
        if df.empty:
            return {}
        result = {}
        for topic, grp in df.groupby("topic"):
            result[topic] = grp["percentage"].tolist()
        return result

    def revision_recommendations(self) -> list:
        """
        Return a ranked list of topics to revise, ordered by:
          1. Below-threshold topics (ascending score)
          2. Topics with declining trend
        """
        stats = self.topic_stats()
        if stats.empty:
            return []
        stats_sorted = stats.sort_values(["avg_score", "trend"])
        return stats_sorted[["topic", "avg_score", "trend", "attempts"]].to_dict("records")
