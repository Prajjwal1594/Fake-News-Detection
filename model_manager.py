"""
ModelManager — trains, stores, and serves all ML pipelines.
This is the bridge between the original ML script and the FastAPI layer.
"""

import re
import string
import warnings
import numpy as np
import pandas as pd
import threading

warnings.filterwarnings("ignore")

# NLTK (downloaded lazily)
import nltk
for pkg in ["stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score

from app.schemas import PredictResponse, StatsResponse

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# ── Text Preprocessor ─────────────────────────────────────────────────────────
class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))

    def clean(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"https?://\S+|www\.\S+", "", text)
        text = re.sub(r"<.*?>", "", text)
        text = re.sub(r"\d+", "", text)
        text = text.translate(str.maketrans("", "", string.punctuation))
        tokens = text.split()
        tokens = [
            self.lemmatizer.lemmatize(t)
            for t in tokens
            if t not in self.stop_words and len(t) > 2
        ]
        return " ".join(tokens)

    def transform(self, texts):
        return [self.clean(t) for t in texts]


# ── CSV Dataset Loader ────────────────────────────────────────────────────────
def load_dataset_from_csv(
    csv_path: str,
    text_col: str = "text",
    label_col: str = "label",
) -> pd.DataFrame:
    """
    Loads a fake/real news dataset from a CSV file.

    Expected CSV format:
        - A column containing the news text  (default name: 'text')
        - A column containing the label      (default name: 'label')
          where 0 = real, 1 = fake

    Common real-world datasets and their column names:
        LIAR dataset     → text_col="statement",  label_col="label"
        FakeNewsNet      → text_col="title",       label_col="label"
        WELFake          → text_col="text",        label_col="label"
        Kaggle Fake News → text_col="text",        label_col="label"

    Label encoding:
        Numeric  0 / 1         → used directly  (0=real, 1=fake)
        Strings "real"/"fake"  → auto-remapped
        Strings "true"/"false" → auto-remapped
        Other strings          → raises ValueError with instructions

    Args:
        csv_path:  Path to the CSV file, e.g. "data/fakenews.csv"
        text_col:  Name of the column containing article text or headline
        label_col: Name of the column containing the binary label

    Returns:
        pd.DataFrame with columns ['text', 'label'], shuffled.

    Raises:
        FileNotFoundError: if the CSV path does not exist.
        ValueError:        if required columns are missing or labels are invalid.
    """
    import os
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset not found at '{csv_path}'.\n"
            f"Please provide a CSV with '{text_col}' and '{label_col}' columns.\n"
            f"Supported datasets: LIAR, FakeNewsNet, WELFake, Kaggle Fake News."
        )

    df = pd.read_csv(csv_path)

    # Validate required columns exist
    missing = [c for c in [text_col, label_col] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Column(s) {missing} not found in CSV.\n"
            f"Available columns: {list(df.columns)}\n"
            f"Pass correct names via text_col= and label_col= arguments."
        )

    df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})

    # Drop rows with missing values
    before = len(df)
    df = df.dropna(subset=["text", "label"])
    dropped = before - len(df)
    if dropped:
        print(f"  [CSV] Dropped {dropped} rows with missing values.")

    # Auto-remap string labels → 0 / 1
    if df["label"].dtype == object:
        str_map = {
            "real": 0, "true": 0, "legitimate": 0, "0": 0,
            "fake": 1, "false": 1, "conspiracy": 1, "1": 1,
        }
        df["label"] = df["label"].str.strip().str.lower().map(str_map)
        bad_rows = df["label"].isna().sum()
        if bad_rows > 0:
            print(f"  [CSV] Dropped {bad_rows} rows with unrecognised label values (malformed CSV rows).")
            df = df.dropna(subset=["label"])

    df["label"] = df["label"].astype(int)

    # Final sanity check
    invalid = set(df["label"].unique()) - {0, 1}
    if invalid:
        raise ValueError(
            f"Labels must be 0 (real) or 1 (fake). Found unexpected values: {invalid}."
        )

    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    real_count = (df["label"] == 0).sum()
    fake_count = (df["label"] == 1).sum()
    print(f"  [CSV] Loaded {len(df)} samples — Real: {real_count}, Fake: {fake_count}")

    if len(df) > 0 and abs(real_count - fake_count) / len(df) > 0.3:
        print(
            f"  [CSV] ⚠ Imbalanced dataset detected "
            f"({real_count} real vs {fake_count} fake). "
            f"Consider resampling or using class_weight='balanced'."
        )

    return df


# ── Pipeline builder ──────────────────────────────────────────────────────────
def build_pipelines() -> dict:
    tfidf_params = dict(max_features=20_000, ngram_range=(1, 3), sublinear_tf=True, min_df=2)
    return {
        "logistic_regression": Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf_params)),
            ("clf",   LogisticRegression(max_iter=1000, C=5, random_state=RANDOM_STATE)),
        ]),
        "linear_svm": Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf_params)),
            ("clf",   LinearSVC(C=1.0, max_iter=2000, random_state=RANDOM_STATE)),
        ]),
        "random_forest": Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf_params)),
            ("clf",   RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)),
        ]),
        "naive_bayes": Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf_params, use_idf=False)),
            ("clf",   MultinomialNB(alpha=0.1)),
        ]),
    }


def build_ensemble(pipelines: dict) -> VotingClassifier:
    estimators = [
        (name, pipe)
        for name, pipe in pipelines.items()
        if name != "linear_svm"
    ]
    return VotingClassifier(estimators=estimators, voting="soft")


# ── Model Manager ─────────────────────────────────────────────────────────────
class ModelManager:
    """Thread-safe manager for all ML models and prediction state."""

    MODEL_DISPLAY_NAMES = {
        "ensemble":            "Ensemble (Voting)",
        "logistic_regression": "Logistic Regression",
        "linear_svm":          "Linear SVM",
        "random_forest":       "Random Forest",
        "naive_bayes":         "Naive Bayes",
    }

    def __init__(self):
        self._ready = False
        self._lock  = threading.Lock()
        self._pipelines: dict   = {}
        self._accuracies: dict  = {}
        self._preprocessor      = TextPreprocessor()
        self._stats = {"total": 0, "real": 0, "fake": 0}

    # ── Training ──────────────────────────────────────────────────────────────
    def train(
        self,
        csv_path: str = "data/fakenews.csv",
        text_col: str = "text",
        label_col: str = "label",
    ):
        """
        Train all models from a CSV dataset.

        Args:
            csv_path:  Path to your CSV file.
            text_col:  Column name for the news text.
            label_col: Column name for the label (0=real, 1=fake).

        Example usage:
            manager.train("data/WELFake.csv")
            manager.train("data/liar.csv", text_col="statement", label_col="label")
        """
        df = load_dataset_from_csv(csv_path, text_col=text_col, label_col=label_col)
        df["clean"] = self._preprocessor.transform(df["text"])

        X, y = df["clean"], df["label"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )

        pipelines = build_pipelines()
        ensemble  = build_ensemble(pipelines)

        all_models = {**pipelines, "ensemble": ensemble}

        trained = {}
        accuracies = {}

        for model_id, pipe in all_models.items():
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            try:
                auc = roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])
            except Exception:
                auc = None
            trained[model_id]    = pipe
            accuracies[model_id] = {"accuracy": round(acc, 4), "roc_auc": round(auc, 4) if auc else None}
            print(f"  [{model_id}] Acc={acc:.4f}" + (f"  AUC={auc:.4f}" if auc else ""))

        with self._lock:
            self._pipelines  = trained
            self._accuracies = accuracies
            self._ready      = True

    # ── Inference ─────────────────────────────────────────────────────────────
    def predict(self, text: str, model_id: str = "ensemble") -> PredictResponse:
        with self._lock:
            if model_id not in self._pipelines:
                raise ValueError(f"Unknown model: {model_id}")
            pipe = self._pipelines[model_id]

        clean = self._preprocessor.clean(text)
        pred  = pipe.predict([clean])[0]

        try:
            proba     = pipe.predict_proba([clean])[0]
            prob_fake = float(proba[1])
            prob_real = float(proba[0])
            confidence = float(max(proba))
        except AttributeError:
            # LinearSVC has no predict_proba
            prob_fake  = 1.0 if pred == 1 else 0.0
            prob_real  = 1.0 - prob_fake
            confidence = 1.0

        label = "FAKE" if pred == 1 else "REAL"

        # Update session stats
        with self._lock:
            self._stats["total"] += 1
            self._stats["fake" if pred == 1 else "real"] += 1

        return PredictResponse(
            text=text,
            prediction=label,
            confidence=round(confidence * 100, 2),
            probability_real=round(prob_real * 100, 2),
            probability_fake=round(prob_fake * 100, 2),
            model_used=self.MODEL_DISPLAY_NAMES.get(model_id, model_id),
        )

    # ── Accessors ─────────────────────────────────────────────────────────────
    def is_ready(self) -> bool:
        return self._ready

    def available_models(self) -> list:
        return list(self._pipelines.keys())

    def get_stats(self) -> StatsResponse:
        with self._lock:
            return StatsResponse(
                total_predictions=self._stats["total"],
                real_count=self._stats["real"],
                fake_count=self._stats["fake"],
                model_accuracies=self._accuracies,
            )
