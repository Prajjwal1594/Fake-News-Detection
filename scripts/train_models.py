"""
Offline training script for FakeShield.

Run this LOCALLY (not on Vercel) whenever you want to retrain on new data:

    python scripts/train_models.py

It trains all 5 pipelines on data/fakenews_clean.csv, prints accuracy/AUC,
and writes the pickled pipelines + accuracies.json into app/models/.
Commit those files before deploying — the deployed app only ever loads
them, it never trains.
"""

import json
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app.model_manager import TextPreprocessor  # noqa: E402

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

CSV_PATH = os.path.join(ROOT, "data", "fakenews_clean.csv")
MODELS_DIR = os.path.join(ROOT, "app", "models")
TEXT_COL = "text"
LABEL_COL = "label"


def load_dataset_from_csv(csv_path: str, text_col: str, label_col: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset not found at '{csv_path}'.\n"
            f"Please provide a CSV with '{text_col}' and '{label_col}' columns."
        )

    df = pd.read_csv(csv_path)
    missing = [c for c in [text_col, label_col] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Column(s) {missing} not found in CSV. Available: {list(df.columns)}"
        )

    df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})

    before = len(df)
    df = df.dropna(subset=["text", "label"])
    if before - len(df):
        print(f"  [CSV] Dropped {before - len(df)} rows with missing values.")

    if df["label"].dtype == object:
        str_map = {
            "real": 0, "true": 0, "legitimate": 0, "0": 0,
            "fake": 1, "false": 1, "conspiracy": 1, "1": 1,
        }
        df["label"] = df["label"].str.strip().str.lower().map(str_map)
        bad_rows = df["label"].isna().sum()
        if bad_rows:
            print(f"  [CSV] Dropped {bad_rows} rows with unrecognised labels.")
            df = df.dropna(subset=["label"])

    df["label"] = df["label"].astype(int)

    invalid = set(df["label"].unique()) - {0, 1}
    if invalid:
        raise ValueError(f"Labels must be 0/1. Found: {invalid}.")

    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    real_count = (df["label"] == 0).sum()
    fake_count = (df["label"] == 1).sum()
    print(f"  [CSV] Loaded {len(df)} samples — Real: {real_count}, Fake: {fake_count}")
    return df


def build_pipelines() -> dict:
    tfidf_params = dict(max_features=20_000, ngram_range=(1, 3), sublinear_tf=True, min_df=2)
    return {
        "logistic_regression": Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf_params)),
            ("clf", LogisticRegression(max_iter=1000, C=5, random_state=RANDOM_STATE)),
        ]),
        "linear_svm": Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf_params)),
            ("clf", LinearSVC(C=1.0, max_iter=2000, random_state=RANDOM_STATE)),
        ]),
        "random_forest": Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf_params)),
            ("clf", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)),
        ]),
        "naive_bayes": Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf_params, use_idf=False)),
            ("clf", MultinomialNB(alpha=0.1)),
        ]),
    }


def build_ensemble(pipelines: dict) -> VotingClassifier:
    estimators = [(name, pipe) for name, pipe in pipelines.items() if name != "linear_svm"]
    return VotingClassifier(estimators=estimators, voting="soft")


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    df = load_dataset_from_csv(CSV_PATH, TEXT_COL, LABEL_COL)
    preprocessor = TextPreprocessor()
    df["clean"] = preprocessor.transform(df["text"])

    X, y = df["clean"], df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    pipelines = build_pipelines()
    ensemble = build_ensemble(pipelines)
    all_models = {**pipelines, "ensemble": ensemble}

    accuracies = {}
    for model_id, pipe in all_models.items():
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        try:
            auc = roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])
        except Exception:
            auc = None
        accuracies[model_id] = {"accuracy": round(acc, 4), "roc_auc": round(auc, 4) if auc else None}
        print(f"  [{model_id}] Acc={acc:.4f}" + (f"  AUC={auc:.4f}" if auc else ""))

        out_path = os.path.join(MODELS_DIR, f"{model_id}.joblib")
        joblib.dump(pipe, out_path, compress=3)
        size_kb = os.path.getsize(out_path) / 1024
        print(f"    -> saved {out_path} ({size_kb:.0f} KB)")

    with open(os.path.join(MODELS_DIR, "accuracies.json"), "w") as f:
        json.dump(accuracies, f, indent=2)

    print("\nDone. Commit app/models/*.joblib before deploying.")


if __name__ == "__main__":
    main()
