"""
NewsService — fetches live headlines from NewsAPI and runs them
through the FakeShield ML models.
"""

import os
import httpx
from datetime import datetime, timezone
from typing import Optional
from app.schemas import NewsArticle, NewsFeedResponse

NEWS_API_BASE = "https://newsapi.org/v2"

CATEGORIES = ["general", "technology", "science", "health", "business", "entertainment", "sports"]

class NewsService:
    def __init__(self, model_manager):
        self.manager  = model_manager
        self.api_key  = os.getenv("NEWS_API_KEY", "")

    def _api_key_ok(self) -> bool:
        return bool(self.api_key and self.api_key != "your_newsapi_key_here")

    async def fetch_and_analyze(
        self,
        category: str = "general",
        country: str = "us",
        page_size: int = 20,
        model_id: str = "ensemble",
        query: Optional[str] = None,
    ) -> NewsFeedResponse:

        if not self._api_key_ok():
            raise ValueError(
                "NEWS_API_KEY not set. Add it to your .env file."
            )

        # Build request
        if query:
            url    = f"{NEWS_API_BASE}/everything"
            params = {
                "q":        query,
                "pageSize": page_size,
                "language": "en",
                "sortBy":   "publishedAt",
                "apiKey":   self.api_key,
            }
        else:
            url    = f"{NEWS_API_BASE}/top-headlines"
            params = {
                "category": category,
                "country":  country,
                "pageSize": page_size,
                "apiKey":   self.api_key,
            }

        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url, params=params)

        if resp.status_code == 401:
            raise ValueError("Invalid NewsAPI key. Check your .env file.")
        if resp.status_code == 429:
            raise ValueError("NewsAPI rate limit reached. Try again later.")
        if resp.status_code != 200:
            raise ValueError(f"NewsAPI error: {resp.status_code} — {resp.text[:200]}")

        data     = resp.json()
        articles = data.get("articles", [])

        results = []
        for art in articles:
            title = (art.get("title") or "").strip()
            if not title or title == "[Removed]":
                continue

            # Combine title + description for better signal
            desc  = (art.get("description") or "").strip()
            text  = f"{title}. {desc}" if desc else title

            try:
                pred = self.manager.predict(text, model_id)
            except Exception:
                continue

            results.append(NewsArticle(
                title          = title,
                description    = desc or None,
                url            = art.get("url", ""),
                source         = art.get("source", {}).get("name", "Unknown"),
                published_at   = art.get("publishedAt", ""),
                prediction     = pred.prediction,
                confidence     = pred.confidence,
                probability_real = pred.probability_real,
                probability_fake = pred.probability_fake,
            ))

        return NewsFeedResponse(
            articles   = results,
            total      = len(results),
            category   = query if query else category,
            fetched_at = datetime.now(timezone.utc).isoformat(),
        )
