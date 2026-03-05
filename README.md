# FakeShield v2 — Real-Time News Detector

Analyzes live news headlines from NewsAPI in real time using ML classifiers.

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add your NewsAPI key
Create a file called `.env` in the `fakeshield/` folder:
```
NEWS_API_KEY=your_key_here
```
Get a free key at https://newsapi.org

### 3. Add your cleaned CSV
Make sure `data/fakenews_clean.csv` exists.
Run the fix_csv.py script if needed.

### 4. Run
```bash
uvicorn app.main:app --reload
```

Open http://localhost:8000

## Features
- 📡 Live news feed — fetches & classifies top headlines by category
- 🔍 Search any topic (e.g. "climate", "election", "AI")
- 🤖 Switch between 5 ML models
- ✍️ Manual headline checker
- 📊 Real-time stats

## API Endpoints
| Method | Path | Description |
|--------|------|-------------|
| GET | `/news` | Fetch & analyze live headlines |
| GET | `/news?query=climate` | Search specific topic |
| GET | `/news?category=technology` | Filter by category |
| POST | `/predict` | Analyze single headline |
| POST | `/predict/batch` | Analyze up to 50 headlines |
| GET | `/health` | Server status |
| GET | `/stats` | Session statistics |
