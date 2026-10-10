import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from .model import MeetingPreferenceModel, DEFAULT_MODEL_PATH
from .feature_extractor import FEATURE_COLUMNS


def generate_synthetic_training_data(n_samples: int = 6000) -> pd.DataFrame:
    """
    Generates realistic historical meeting data strictly matching the 11 grounded features:
    - duration_minutes
    - priority_level
    - num_participants
    - day_of_week
    - start_hour_utc
    - max_tz_difference_hrs
    - calendar_load_ratio
    - min_buffer_before_min
    - min_buffer_after_min
    - consecutive_meetings
    - lunch_overlap_ratio
    
    Target: preference_score in [0.0, 1.0]
    """
    random.seed(42)
    np.random.seed(42)

    rows = []
    for _ in range(n_samples):
        duration = random.choice([30.0, 45.0, 60.0, 90.0])
        priority = random.choice([1.0, 2.0, 3.0, 4.0]) # LOW=1, CRITICAL=4
        num_participants = random.randint(2, 8)
        day_of_week = random.choice([0, 1, 2, 3, 4]) # Mon - Fri
        start_hour_utc = round(random.uniform(8.0, 18.0), 2)
        max_tz_diff = random.choice([0.0, 1.0, 3.0, 5.0, 8.0]) # hours
        calendar_load = round(random.uniform(0.1, 0.9), 3) # fraction of day booked
        
        # Lunch overlap ratio
        if 11.5 <= start_hour_utc <= 13.5:
            lunch_ratio = round(random.uniform(0.2, 1.0), 3)
        else:
            lunch_ratio = 0.0

        min_buffer_before = random.choice([0.0, 5.0, 15.0, 30.0, 60.0, 120.0])
        min_buffer_after = random.choice([0.0, 5.0, 15.0, 30.0, 60.0, 120.0])
        consecutive = random.choice([0.0, 1.0, 2.0, 3.0, 4.0])

        # Base human satisfaction score
        score = 0.85

        # 1. Lunch penalty
        if lunch_ratio > 0.0:
            score -= 0.35 * lunch_ratio

        # 2. Calendar load penalty
        if calendar_load > 0.5:
            score -= 0.15 * (calendar_load - 0.5)

        # 3. Back-to-back meeting fatigue
        if consecutive >= 3.0:
            score -= 0.25
        elif consecutive >= 1.0:
            score -= 0.10

        # 4. Low buffer penalty (< 15 mins)
        if min_buffer_before < 15.0:
            score -= 0.08
        if min_buffer_after < 15.0:
            score -= 0.08

        # 5. Timezone spread penalty
        if max_tz_diff > 4.0:
            score -= 0.06

        # 6. Priority damping (urgent meetings are more readily accepted)
        score += (priority - 2.0) * 0.04

        # 7. Preferred core hours (10 AM to 12 PM, 2 PM to 4 PM UTC)
        if 10.0 <= start_hour_utc <= 12.0 or 14.0 <= start_hour_utc <= 16.0:
            score += 0.08
        elif start_hour_utc < 9.0 or start_hour_utc > 17.0:
            score -= 0.20

        # Random human variance
        noise = np.random.normal(0, 0.03)
        final_score = max(0.05, min(0.98, score + noise))

        row = {
            "duration_minutes": duration,
            "priority_level": priority,
            "num_participants": float(num_participants),
            "day_of_week": float(day_of_week),
            "start_hour_utc": start_hour_utc,
            "max_tz_difference_hrs": max_tz_diff,
            "calendar_load_ratio": calendar_load,
            "min_buffer_before_min": min_buffer_before,
            "min_buffer_after_min": min_buffer_after,
            "consecutive_meetings": consecutive,
            "lunch_overlap_ratio": lunch_ratio,
            "preference_score": round(float(final_score), 3)
        }
        rows.append(row)

    return pd.DataFrame(rows)


def run_training():
    print("=" * 60)
    print("TRAINING SCIKIT-LEARN MODEL ON GROUNDED FEATURES")
    print("=" * 60)
    print(f"Features: {FEATURE_COLUMNS}")

    df = generate_synthetic_training_data(n_samples=6000)
    print(f"Generated {len(df)} samples.")

    X = df[FEATURE_COLUMNS]
    y = df["preference_score"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_df = pd.concat([X_train, y_train], axis=1)

    model = MeetingPreferenceModel(model_path=DEFAULT_MODEL_PATH)
    metrics = model.train(train_df, target_col="preference_score")
    print(f"Train R² Score: {metrics['r2_score']:.4f}")

    y_pred = [model.predict_score(row.to_dict()) for _, row in X_test.iterrows()]
    test_mse = mean_squared_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)

    print(f"Test MSE: {test_mse:.5f}")
    print(f"Test R² Score: {test_r2:.4f}")
    print(f"Model saved to: {DEFAULT_MODEL_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    run_training()
