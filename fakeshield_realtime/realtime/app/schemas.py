"""Pydantic schemas for all API request/response models."""

from pydantic import BaseModel, Field
from typing import List, Optional, Literal

ModelID = Literal[
    "ensemble", "logistic_regression", "linear_svm", "random_forest", "naive_bayes",
]

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=3, max_length=5000)
    model: ModelID = Field("ensemble")

class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=50)
    model: ModelID = Field("ensemble")

class PredictResponse(BaseModel):
    text: str
    prediction: Literal["REAL", "FAKE"]
    confidence: float = Field(..., ge=0, le=100)
    probability_real: float = Field(..., ge=0, le=100)
    probability_fake: float = Field(..., ge=0, le=100)
    model_used: str
    latency_ms: Optional[float] = None

class BatchPredictResponse(BaseModel):
    results: List[PredictResponse]
    total_latency_ms: float

class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    available_models: List[str]

class StatsResponse(BaseModel):
    total_predictions: int
    real_count: int
    fake_count: int
    model_accuracies: dict

class NewsArticle(BaseModel):
    title: str
    description: Optional[str] = None
    url: str
    source: str
    published_at: str
    prediction: Literal["REAL", "FAKE"]
    confidence: float
    probability_real: float
    probability_fake: float

class NewsFeedResponse(BaseModel):
    articles: List[NewsArticle]
    total: int
    category: str
    fetched_at: str
