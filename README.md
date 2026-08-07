# FakeShield — Real-Time News Detector (Vercel-ready)

Ensemble ML fake news detector (TF-IDF + Logistic Regression / Linear SVM /
Random Forest / Naive Bayes, soft-voting ensemble) with a live news feed
classifier powered by NewsAPI.

This is a restructured version of the original FastAPI app, adapted to
deploy cleanly on **Vercel Functions**. The main change: models are
**trained once, offline, and shipped as pickled artifacts** instead of being
retrained on every server start. Vercel functions are stateless and
short-lived, so training 5 sklearn pipelines (including a 200-tree Random
Forest) on every cold start would be slow, wasteful, and would repeat on
every new container.

## What changed vs. the original repo

| | Original | This version |
|---|---|---|
| Models | Trained at FastAPI `lifespan` startup | Pretrained offline, loaded from `app/models/*.joblib` |
| NLTK data | Downloaded from network on first run | Bundled in `nltk_data/` (no network call needed) |
| Frontend | Separate `frontend/index.html`, hits `http://localhost:8000` | Moved to `public/index.html`, same-origin relative API calls |
| Entrypoint | `app/main.py` run via `uvicorn` | Same file, auto-detected by Vercel's Python/FastAPI runtime |
| requirements.txt | Included `pandas`/`pytest` (training-only) | Split into `requirements.txt` (runtime) and `requirements-dev.txt` (training/tests) |

Cold start (loading models + first prediction, which lazily loads the
WordNet corpus) is absorbed once at container startup via a warm-up call —
after that, predictions run in ~30-40ms.

## Project structure

```
app/
  main.py            FastAPI app (Vercel entrypoint — exports `app`)
  model_manager.py    Loads pretrained pipelines, runs predictions
  news_service.py      NewsAPI integration
  schemas.py            Pydantic request/response models
  models/                 Pretrained pipelines (*.joblib) + accuracies.json
nltk_data/            Bundled WordNet + English stopwords (no runtime download)
public/
  index.html            Frontend (served at "/")
data/
  fakenews_clean.csv    Training data (not deployed — see vercel.json excludeFiles)
scripts/
  train_models.py      Offline training script — run locally, commit the output
tests/
  test_api.py
requirements.txt        Runtime dependencies
requirements-dev.txt   + pandas/pytest for training & tests
vercel.json
```

## Deploying to Vercel

1. **Push this repo to GitHub** (or your Git provider of choice).
2. **Import it in Vercel** (vercel.com → New Project → import the repo).
   Vercel auto-detects the FastAPI app at `app/main.py` — no extra config
   needed beyond what's in `vercel.json`.
3. **Add environment variables** in the Vercel project settings:
   - `NEWS_API_KEY` — get a free key at https://newsapi.org (only needed for
     the `/news` live-feed endpoint; `/predict` works without it).
4. **Deploy.** Vercel installs `requirements.txt` and bundles everything
   reachable from the project root (models, nltk_data, public/ included
   automatically).

### Retraining the models

If you want to retrain on new/updated data:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt
python scripts/train_models.py
```

This overwrites `app/models/*.joblib` and `app/models/accuracies.json`.
**Commit those files** before redeploying — the deployed app only loads
them, it never trains.

### Local development

```bash
pip install -r requirements-dev.txt
uvicorn app.main:app --reload
```

Open http://localhost:8000. Or use the Vercel CLI to run it exactly as it
would run in production:

```bash
npm i -g vercel
vercel dev
```

### Running tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Frontend UI |
| POST | `/predict` | Analyze a single headline |
| POST | `/predict/batch` | Analyze up to 50 headlines |
| GET | `/news` | Fetch & classify live headlines (requires `NEWS_API_KEY`) |
| GET | `/news?query=climate` | Search a specific topic |
| GET | `/news?category=technology` | Filter by category |
| GET | `/health` | Server status |
| GET | `/stats` | Session statistics |
| GET | `/models` | List available models |

## Known limitations on Vercel

- **Stats are per-container, not persistent.** `/stats` counts predictions
  made against the current warm container only — a new cold start (or a
  request routed to a different container under load) resets it. For
  durable stats you'd need external storage (e.g. a database or Vercel KV).
- **No online learning.** The original architecture supports retraining;
  on Vercel, retraining has to happen offline (`scripts/train_models.py`)
  since the filesystem is ephemeral and functions are stateless.
- **Model bundle size:** ~36MB (mostly the Random Forest and ensemble
  pipelines), well under Vercel's 500MB Python function bundle limit.
