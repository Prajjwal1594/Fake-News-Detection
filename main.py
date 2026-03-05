"""
FakeShield — Real-Time News Detector
Run with: uvicorn app.main:app --reload
"""

import os, time
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from typing import Optional

load_dotenv()  # loads .env automatically

from app.schemas import (
    PredictRequest, PredictResponse,
    BatchPredictRequest, BatchPredictResponse,
    HealthResponse, StatsResponse, NewsFeedResponse,
)
from app.model_manager import ModelManager
from app.news_service import NewsService, CATEGORIES

manager      = ModelManager()
news_service = None   # initialized after models are ready

@asynccontextmanager
async def lifespan(app: FastAPI):
    global news_service
    print("\n FakeShield Real-Time starting up...")
    manager.train(
        csv_path  = "data/fakenews_clean.csv",
        text_col  = "text",
        label_col = "label",
    )
    news_service = NewsService(manager)
    print("Models ready. NewsAPI key loaded:", bool(os.getenv("NEWS_API_KEY")))
    yield

app = FastAPI(title="FakeShield Real-Time API", version="2.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

FRONTEND = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html")

@app.get("/", include_in_schema=False)
async def root():
    return FileResponse(FRONTEND) if os.path.exists(FRONTEND) else {"message": "FakeShield API running. See /docs"}

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    return HealthResponse(
        status          = "ok",
        models_loaded   = manager.is_ready(),
        available_models= manager.available_models(),
    )

@app.get("/stats", response_model=StatsResponse, tags=["System"])
async def stats():
    return manager.get_stats()

@app.get("/models", tags=["System"])
async def list_models():
    return {"models": [
        {"id": "ensemble",            "name": "Ensemble (Voting)",   "description": "Best accuracy."},
        {"id": "logistic_regression", "name": "Logistic Regression", "description": "Fast, interpretable."},
        {"id": "linear_svm",          "name": "Linear SVM",          "description": "Strong text classifier."},
        {"id": "random_forest",       "name": "Random Forest",       "description": "200 decision trees."},
        {"id": "naive_bayes",         "name": "Naive Bayes",         "description": "Probabilistic baseline."},
    ]}

@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict(req: PredictRequest):
    if not manager.is_ready():
        raise HTTPException(503, "Models not yet loaded.")
    if not req.text.strip():
        raise HTTPException(422, "Text must not be empty.")
    start  = time.perf_counter()
    result = manager.predict(req.text, req.model)
    result.latency_ms = round((time.perf_counter() - start) * 1000, 2)
    return result

@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Prediction"])
async def predict_batch(req: BatchPredictRequest):
    if not manager.is_ready():
        raise HTTPException(503, "Models not yet loaded.")
    if len(req.texts) > 50:
        raise HTTPException(422, "Maximum 50 texts per batch.")
    start   = time.perf_counter()
    results = [manager.predict(t, req.model) for t in req.texts]
    return BatchPredictResponse(
        results          = results,
        total_latency_ms = round((time.perf_counter() - start) * 1000, 2),
    )

# ── Real-Time News endpoints ───────────────────────────────────────────────────

@app.get("/news", response_model=NewsFeedResponse, tags=["Live News"])
async def get_news(
    category:  str           = Query("general", enum=CATEGORIES),
    country:   str           = Query("us"),
    page_size: int           = Query(20, ge=1, le=50),
    model:     str           = Query("ensemble"),
    query:     Optional[str] = Query(None, description="Search specific topic"),
):
    """
    Fetch live news headlines from NewsAPI and classify each as REAL or FAKE.
    Requires NEWS_API_KEY in your .env file.
    """
    if not manager.is_ready():
        raise HTTPException(503, "Models not yet loaded.")
    try:
        return await news_service.fetch_and_analyze(
            category  = category,
            country   = country,
            page_size = page_size,
            model_id  = model,
            query     = query,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"News fetch failed: {str(e)}")

@app.get("/news/categories", tags=["Live News"])
async def get_categories():
    """List all available news categories."""
    return {"categories": CATEGORIES}
