#!/usr/bin/env python3
"""
Train a baseline impact classifier using walk-forward validation.

Target:
 - impact_2h

Features:
 - surprise (if available)
 - sigma_2h
 - regime_1d (one-hot)
 - pre_event_return_1h

Outputs:
 - Prints per-year AUC + Brier score
 - Saves final model to outputs/model.pkl
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def main():
    base = Path(".")
    data_path = base / "outputs" / "dataset.parquet"
    out_model = base / "outputs" / "model.pkl"

    df = pd.read_parquet(data_path)

    # Target
    y = df["impact_2h"].astype(int)

    # Features
    numeric_features = [
        "sigma_2h",
        "pre_event_return_1h",
    ]

    categorical_features = ["regime_1d"]

    if "surprise" in df.columns and df["surprise"].notna().any():
        numeric_features.append("surprise")

    X = df[numeric_features + categorical_features]

    # Year for walk-forward split
    df["year"] = df["event_time_et"].dt.year
    years = sorted(df["year"].unique())

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
    )

    pipe = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", model),
        ]
    )

    aucs = []
    briers = []

    print("WALK-FORWARD RESULTS")
    print("--------------------")

    for test_year in years[1:]:
        train_idx = df["year"] < test_year
        test_idx = df["year"] == test_year

        X_train, y_train = X.loc[train_idx], y.loc[train_idx]
        X_test, y_test = X.loc[test_idx], y.loc[test_idx]

        if len(y_test) < 5:
            continue

        pipe.fit(X_train, y_train)
        probs = pipe.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, probs)
        brier = brier_score_loss(y_test, probs)

        aucs.append(auc)
        briers.append(brier)

        print(f"Year {test_year}: AUC={auc:.3f}, Brier={brier:.3f}")

    print("\nSUMMARY")
    print("-------")
    print(f"Mean AUC:    {np.mean(aucs):.3f}")
    print(f"Mean Brier: {np.mean(briers):.3f}")

    # Fit final model on all data
    pipe.fit(X, y)
    joblib.dump(pipe, out_model)

    print(f"\nSaved final model to {out_model}")


if __name__ == "__main__":
    main()
