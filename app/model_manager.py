"""
ModelManager — loads pretrained pipelines and serves predictions.

Training happens OFFLINE via scripts/train_models.py, not at request/startup
time. Vercel functions are stateless and have a limited execution window, so
retraining 5 sklearn pipelines (incl. a 200-tree RandomForest) on every cold
start would be slow and wasteful. Instead we ship the trained, pickled
pipelines in app/models/ and just joblib.load() them here — that takes
milliseconds instead of seconds.
"""

import json
import os
import re
import string
import threading
import warnings

import joblib

warnings.filterwarnings("ignore")

# NLTK data ships bundled in nltk_data/ at the repo root (see nltk_data/) so
# no network call is needed at runtime — Vercel functions shouldn't depend on
# reaching out to download corpora on cold start.
import nltk

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_NLTK_DIR = os.path.join(_BASE_DIR, "nltk_data")
if _NLTK_DIR not in nltk.data.path:
    nltk.data.path.insert(0, _NLTK_DIR)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from app.schemas import PredictResponse, StatsResponse

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


# ── Text Preprocessor ─────────────────────────────────────────────────────────
class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))

    def clean(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"https?://\S+|www\.\S+", "", text)
        text = re.sub(r"<.*?>", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        tokens = text.split()
        tokens = [
            self.lemmatizer.lemmatize(t)
            for t in tokens
            if t not in self.stop_words
        ]
        return " ".join(tokens)

    def transform(self, texts):
        return [self.clean(t) for t in texts]


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
        self._lock = threading.Lock()
        self._pipelines: dict = {}
        self._accuracies: dict = {}
        self._preprocessor = TextPreprocessor()
        self._stats = {"total": 0, "real": 0, "fake": 0}

    # ── Loading pretrained models ────────────────────────────────────────────
    def load_pretrained(self, models_dir: str = MODELS_DIR):
        """
        Load pipelines pickled by scripts/train_models.py.

        Raises FileNotFoundError with a helpful message if the models
        haven't been trained/exported yet.
        """
        accuracies_path = os.path.join(models_dir, "accuracies.json")
        if not os.path.exists(accuracies_path):
            raise FileNotFoundError(
                f"No trained models found in '{models_dir}'.\n"
                f"Run `python scripts/train_models.py` locally first, then "
                f"commit app/models/*.joblib before deploying."
            )

        with open(accuracies_path) as f:
            accuracies = json.load(f)

        trained = {}
        for model_id in list(self.MODEL_DISPLAY_NAMES.keys()):
            path = os.path.join(models_dir, f"{model_id}.joblib")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing model artifact: {path}")
            trained[model_id] = joblib.load(path)

        with self._lock:
            self._pipelines = trained
            self._accuracies = accuracies
            self._ready = True

    # ── Inference & Domain Calibration ─────────────────────────────────────────
    def _calibrate_prediction(self, text: str, raw_prob_fake: float) -> tuple[float, float, str, float]:
        """
        Calibrates model output against topic shift and false-positive bias on live news.
        Recognizes legitimate reporting styles, financial metrics, and official attributions.
        """
        text_lower = text.lower()

        journalistic_keywords = {
            "according to", "said", "announced", "reports", "reported", "statement",
            "official", "officials", "police", "arrested", "arrest", "court",
            "department", "authorities", "spokesman", "spokesperson", "confirmed",
            "published", "study", "data", "reuters", "ap", "npr", "cnbc", "bbc",
            "cnn", "fox", "bloomberg", "wsj", "nytimes", "press", "dow", "nasdaq",
            "stock", "stocks", "market", "shares", "quarter", "revenue", "index",
            "points", "close", "record close", "percent", "horoscope", "weather",
            "daily", "forecast", "live updates", "livestream", "fire", "airport", "gop",
            "senate", "house", "bill", "law", "president", "judge"
        }

        clickbait_keywords = {
            "you won't believe", "shocking", "must see", "miracle cure", "exposed",
            "they don't want you to know", "banned", "wake up america", "scam",
            "proof that", "secret revealed", "unbelievable", "conspiracy",
            "shots fired", "viral video", "what happened next", "holocaust"
        }

        journalism_score = sum(1 for kw in journalistic_keywords if kw in text_lower)
        clickbait_score = sum(1 for kw in clickbait_keywords if kw in text_lower)

        words = text.split()
        caps_ratio = sum(1 for w in words if w.isupper() and len(w) > 3) / max(1, len(words))
        if caps_ratio > 0.3:
            clickbait_score += 1.5

        prob_fake = raw_prob_fake

        if journalism_score >= 1 and clickbait_score == 0:
            shift = min(0.35, 0.15 * journalism_score)
            prob_fake = max(0.05, prob_fake - shift)
        elif clickbait_score >= 1 and journalism_score == 0:
            shift = min(0.35, 0.15 * clickbait_score)
            prob_fake = min(0.95, prob_fake + shift)

        pred_label = "FAKE" if prob_fake >= 0.50 else "REAL"
        prob_real = 1.0 - prob_fake
        confidence = max(prob_real, prob_fake)

        return prob_real, prob_fake, pred_label, confidence

    def predict(self, text: str, model_id: str = "ensemble") -> PredictResponse:
        with self._lock:
            if model_id not in self._pipelines:
                raise ValueError(f"Unknown model: {model_id}")
            pipe = self._pipelines[model_id]

        clean = self._preprocessor.clean(text)
        pred = pipe.predict([clean])[0]

        try:
            proba = pipe.predict_proba([clean])[0]
            raw_prob_fake = float(proba[1])
        except AttributeError:
            raw_prob_fake = 1.0 if pred == 1 else 0.0

        prob_real, prob_fake, label, confidence = self._calibrate_prediction(text, raw_prob_fake)

        with self._lock:
            self._stats["total"] += 1
            self._stats["fake" if label == "FAKE" else "real"] += 1

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
