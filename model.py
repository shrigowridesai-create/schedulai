import os
from typing import Dict, Optional, List
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .feature_extractor import FEATURE_COLUMNS

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "meeting_ranker.joblib")


class MeetingPreferenceModel:
    """
    Scikit-learn based Machine Learning model to predict participant preference
    and acceptance suitability scores for candidate meeting slots.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.pipeline: Optional[Pipeline] = None
        self._load_model_if_exists()

    def _load_model_if_exists(self):
        if os.path.exists(self.model_path):
            try:
                self.pipeline = joblib.load(self.model_path)
            except Exception:
                self.pipeline = None

    def train(self, df: pd.DataFrame, target_col: str = "preference_score") -> Dict[str, float]:
        """
        Trains a RandomForestRegressor pipeline with StandardScaler on the provided dataset.
        Returns training performance metrics.
        """
        X = df[FEATURE_COLUMNS]
        y = df[target_col]

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", RandomForestRegressor(
                n_estimators=100,
                max_depth=8,
                random_state=42,
                n_jobs=-1
            ))
        ])

        pipeline.fit(X, y)
        self.pipeline = pipeline
        self.save(self.model_path)

        train_score = float(pipeline.score(X, y))
        return {"r2_score": train_score, "samples": len(df)}

    def predict_score(self, features: Dict[str, float]) -> float:
        """
        Predicts suitability score between 0.0 (highly undesirable) and 1.0 (ideal).
        If the trained model is not yet loaded, uses an intelligent heuristic fallback.
        """
        if self.pipeline is not None:
            # Prepare single-row DataFrame with correct column order
            df_row = pd.DataFrame([{col: features.get(col, 0.0) for col in FEATURE_COLUMNS}])
            raw_pred = float(self.pipeline.predict(df_row)[0])
            # Clamp strictly between 0.0 and 1.0
            return max(0.0, min(1.0, round(raw_pred, 3)))

        return self._heuristic_fallback_score(features)

    def _heuristic_fallback_score(self, f: Dict[str, float]) -> float:
        """
        Rule-based heuristic that mimics human preferences if ML model is cold.
        """
        score = 0.85 # baseline favorable score

        # Penalty 1: Lunch overlap ratio
        lunch_ratio = f.get("lunch_overlap_ratio", 0.0)
        if lunch_ratio > 0.0:
            score -= 0.35 * lunch_ratio

        # Penalty 2: High calendar load ratio (> 0.5 of day already booked)
        load = f.get("calendar_load_ratio", 0.0)
        if load > 0.5:
            score -= 0.15 * (load - 0.5)

        # Penalty 3: Back-to-back meetings without rest
        consec = f.get("consecutive_meetings", 0.0)
        if consec > 0:
            score -= min(0.30, consec * 0.10)

        # Penalty 4: Tiny buffer before/after (< 15 mins)
        if f.get("min_buffer_before_min", 180.0) < 15.0:
            score -= 0.08
        if f.get("min_buffer_after_min", 180.0) < 15.0:
            score -= 0.08

        # Penalty 5: Large timezone difference fatigue (> 4 hours)
        tz_diff = f.get("max_tz_difference_hrs", 0.0)
        if tz_diff > 4.0:
            score -= 0.05

        return max(0.05, min(0.98, round(score, 3)))

    def save(self, path: Optional[str] = None):
        """Serializes the model using joblib."""
        save_path = path or self.model_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(self.pipeline, save_path)

    def is_trained(self) -> bool:
        return self.pipeline is not None
