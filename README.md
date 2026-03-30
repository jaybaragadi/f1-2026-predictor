# F1 2026 Race Predictor

An AI-powered Formula 1 race prediction system that uses a 5-algorithm stacking ensemble trained on leak-free historical data to predict 2026 Grand Prix finishing positions.

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-3.0-lightgrey?style=for-the-badge&logo=flask)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?style=for-the-badge&logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-green?style=for-the-badge)

---

## Features

- **Leak-free ML model** — all rolling features use `.shift(1)` so the current race result never leaks into its own prediction
- **5-algorithm stacking ensemble** — Ridge, Lasso, XGBoost, GradientBoosting, RandomForest with Ridge meta-learner
- **50 engineered features** including rolling form, track-specific performance, qualifying form, teammate gaps, championship pressure, reliability, and career stats
- **Dynamic feature cache** — app.py adapts automatically to any feature set; no hardcoded lists
- **Track-aware predictions** — circuit history for each driver and team is looked up per-race
- **Auto-retrain pipeline** — after each race, `auto_retrain.py --round N` fetches FastF1 data, appends to the dataset, rebuilds features, and retrains
- **GitHub Actions cron** — model auto-updates every Monday at 2AM UTC

---

## Model Performance

| Metric | Training | Test |
|--------|----------|------|
| Accuracy within ±2 positions | 50.4% | **40.9%** |
| Accuracy within ±3 positions | 68.6% | **59.9%** |
| MAE | 2.59 positions | 3.23 positions |
| R² | 0.630 | 0.451 |
| Overfitting gap (Train R² − Test R²) | | **0.179** |

> **Why ~41% and not 80%+?**
> Earlier versions reported 82% accuracy, but that was inflated by data leakage — rolling features included the current race's result when predicting that same race, and `train_test_split()` randomly mixed future races into training. After fixing both issues with `.shift(1)` throughout and switching to a time-based split, the honest test accuracy is ~41% within ±2 positions. This is genuine generalisation on unseen future races, which is the only number that matters for real-world use.

### Top Features (by XGBoost importance)

| Feature | Importance | Description |
|---------|-----------|-------------|
| Points_Rolling_10 | 34.1% | 10-race points average (prior races only) |
| ChampionshipPosition | 11.2% | Current standings position |
| TeamYearBest | 9.7% | Best result for driver's team this year |
| Points_Rolling_5 | 6.4% | 5-race points form |
| GridPosition | 3.6% | Qualifying result for this race |
| GridPosition_Rolling_5 | 1.5% | Recent qualifying trend |
| Track features (7 total) | 4.0% | Circuit-specific history |

---

## Quick Start

### Prerequisites

- Python 3.11+
- 4 GB RAM (for model training with hyperparameter tuning)
- Internet connection (for FastF1 caching on first run)

### 1. Clone and install

```bash
git clone https://github.com/jaybaragadi/f1-2026-predictor.git
cd f1-2026-predictor

python -m venv .venv

# Windows
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Build features

```bash
# Windows (PowerShell / Git Bash)
PYTHONIOENCODING=utf-8 python feature_engineering/build_features.py

# macOS / Linux
python feature_engineering/build_features.py
```

Expected output:
```
Output samples: 1395
Output features: 62
All rolling/expanding features are leak-free (shift(1) applied)
Features saved to: data/processed/f1_race_features.csv
```

### 3. Train the model

```bash
PYTHONIOENCODING=utf-8 python model/train_model.py
```

Expected output:
```
Features: 50
Train/test split: 80% / 20% (TIME-BASED)
Test accuracy (±2): 40.9%
Test R²: 0.451
Model saved to: model/saved_models/
```

> Training takes ~10–15 minutes with hyperparameter tuning enabled (`TUNE_HYPERPARAMETERS = True` in `train_model.py`). Set it to `False` for a ~30-second run at slightly lower quality.

### 4. Run the app

```bash
PYTHONIOENCODING=utf-8 python app/app.py
```

Open: **http://127.0.0.1:5001**

---

## How to Use

1. **Select a race** from the 2026 calendar dropdown
2. **Enter grid positions** — manually type qualifying results, or click **Load Default Grid** for a championship-order baseline
3. **Click Predict** — results appear instantly with predicted finishing order, grid position, and positions gained/lost
4. **Compare circuits** — run the same grid through Monaco vs Monza to see how track history shapes predictions

---

## Project Structure

```
f1-2026-predictor/
├── app/
│   ├── app.py                        # Flask backend, F1RacePredictor class
│   ├── templates/
│   │   └── index.html                # Single-page UI
│   └── static/
│       ├── css/style.css             # F1-themed dark UI
│       └── js/main.js                # API calls, grid validation, toast notifications
│
├── feature_engineering/
│   └── build_features.py             # 50-feature leak-free pipeline
│
├── model/
│   ├── train_model.py                # 5-algorithm stacking ensemble
│   └── saved_models/
│       ├── f1_race_predictor_model.pkl
│       ├── scaler.pkl
│       ├── feature_columns.pkl
│       └── model_metadata.json       # accuracy, R², feature importance
│
├── data/
│   ├── raw/
│   │   ├── historical_race_results.csv   # 1,395 rows (2023–2025)
│   │   ├── historical_quali_results.csv
│   │   └── historical_standings.csv
│   ├── processed/
│   │   └── f1_race_features.csv          # output of build_features.py
│   └── reference/
│       ├── 2026_drivers.csv              # 22 drivers, numbers, teams
│       └── 2026_teams.csv               # 11 teams, engine suppliers
│
├── auto_retrain.py                   # Post-race retrain pipeline
├── grid_manager.py                   # Default grid logic
├── config.py                         # All paths, driver/team/race lists
├── requirements.txt
└── .github/
    └── workflows/
        └── auto-retrain.yml          # Monday 2AM UTC cron job
```

---

## Feature Engineering

All 50 model features are computed in `feature_engineering/build_features.py` with strict leak-prevention:

| Group | Features | Leak-prevention |
|-------|----------|-----------------|
| Rolling form (3/5/10 race) | Position, Points, PositionsGained, Podiums, DNFs | `.rolling().shift(1)` |
| Track-specific | DriverTrackAvg, DriverTrackBest, TeamTrackAvg, TeamCircuitBest, CircuitAvgPosition, CircuitBestPosition, CircuitRacesCount | `expanding().shift(1)` |
| Momentum | PointsMomentum, WinStreak, PodiumStreak, PointsFinishStreak, RecentReliability, GapToLeader | `.shift(1)` |
| Team | TeamYearPoints, TeamYearAvgPosition, TeamRaceAvgPosition, TeamYearReliability, TeamYearBest, TeamRecentForm | `expanding().shift(1)` |
| Qualifying/race pace | GridPosition (raw), AvgRacePaceVsQuali, QualifyingAdvantage | GridPosition is current race; rolling averages `.shift(1)` |
| Championship | ChampionshipPoints, ChampionshipPosition, FightingForTitle, GapToLeader | Already correct in source data |
| Experience | CareerRaceCount, CareerWins, CareerPodiums, CareerPoints, RacesWithTeam | `expanding().shift(1)` |
| Qualifying form | GridPosition_Rolling_3, GridPosition_Rolling_5, TeamMateGridGap | `.rolling().shift(1)` |

---

## API Reference

All endpoints are served from `http://127.0.0.1:5001`.

### `GET /api/model-info`

Returns model load status and metadata.

```json
{
  "loaded": true,
  "metadata": {
    "features": 50,
    "drivers": 22,
    "teams": 11,
    "races": 24,
    "season": 2026,
    "accuracy": "40.9%",
    "model_type": "5-Algorithm Ensemble",
    "track_aware": true,
    "defending_champion": "Lando Norris (#1)"
  }
}
```

### `GET /api/races`

Returns all 24 races with status (`completed` / `today` / `upcoming`) and sprint flags.

```json
[
  {
    "round": 3,
    "name": "Japanese Grand Prix",
    "location": "Suzuka",
    "date": "2026-03-29",
    "formatted_date": "Mar 29, 2026",
    "status": "completed",
    "has_sprint": false,
    "is_sprint_race": false
  }
]
```

### `GET /api/drivers`

Returns all 22 drivers with number, code, name, team, championships.

### `GET /api/default-grid`

Returns a championship-order grid (suitable as a starting point for predictions).

```json
{
  "status": "success",
  "grid_positions": {
    "VER": 1, "NOR": 2, "HAM": 3, "ALO": 4, "RUS": 5
  }
}
```

### `POST /api/predict`

**Request:**
```json
{
  "race": "Japanese Grand Prix",
  "grid_positions": {
    "VER": 1, "HAM": 2, "LEC": 3, "NOR": 4, "RUS": 5,
    "PIA": 6, "ANT": 7, "ALO": 8, "STR": 9, "HAD": 10,
    "HUL": 11, "BOR": 12, "PER": 13, "BOT": 14, "GAS": 15,
    "COL": 16, "ALB": 17, "SAI": 18, "LAW": 19, "LIN": 20,
    "OCO": 21, "BEA": 22
  }
}
```

**Response:**
```json
{
  "status": "success",
  "race": "Japanese Grand Prix",
  "predictions": [
    {
      "position": 1,
      "predictedPosition": 2.09,
      "driverCode": "VER",
      "driverName": "Max Verstappen",
      "team": "Red Bull Racing",
      "driverNumber": 3,
      "gridPosition": 1,
      "positionsGained": 0
    }
  ]
}
```

**Validation rules:**
- Exactly 22 driver codes required
- Grid positions must be 1–22, all unique
- `race` must exactly match a name from `GET /api/races`

### `GET /health`

```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2026-03-30T05:30:00"
}
```

---

## Auto-Retrain Pipeline

After each race weekend, run:

```bash
# Retrain after a specific round
python auto_retrain.py --round 3

# Auto-detect latest completed race by date
python auto_retrain.py --auto

# Process all completed 2026 races in order
python auto_retrain.py --all
```

**What it does:**
1. Fetches race results from FastF1 for the given round
2. Appends new rows to `data/raw/historical_race_results.csv` (skip-if-exists safe)
3. Calls `build_features.py` on the updated dataset → saves `f1_race_features.csv`
4. Calls `train_model.py` → overwrites saved model artifacts

**GitHub Actions** runs `python auto_retrain.py --auto` every Monday at 2:00 AM UTC. If the model files change, it commits and pushes automatically. Requires a `PAT_TOKEN` secret in the repository settings (Settings → Secrets → Actions).

---

## Deployment

### Render (recommended)

1. Push to GitHub
2. Create a new **Web Service** on [render.com](https://render.com), connect the repository
3. Set:
   - **Build command**: `pip install -r requirements.txt && python feature_engineering/build_features.py && python model/train_model.py`
   - **Start command**: `python app/app.py`
   - **Environment variable**: `SECRET_KEY` = any random string
4. The app listens on `0.0.0.0:$PORT` (Render injects `PORT` automatically)

> The `HOST = '0.0.0.0'` and `PORT = int(os.getenv('PORT', 5001))` in `config.py` handle this automatically.

---

## 2026 Season

**22 drivers, 11 teams, 24 races — 6 sprint weekends**

| Team | Drivers | Engine |
|------|---------|--------|
| McLaren | Lando Norris (#1), Oscar Piastri (#81) | Mercedes |
| Red Bull Racing | Max Verstappen (#3), Isack Hadjar (#6) | Red Bull Ford |
| Mercedes | George Russell (#63), Andrea Kimi Antonelli (#12) | Mercedes |
| Ferrari | Charles Leclerc (#16), Lewis Hamilton (#44) | Ferrari |
| Aston Martin | Fernando Alonso (#14), Lance Stroll (#18) | Honda |
| Audi | Nico Hulkenberg (#27), Gabriel Bortoleto (#5) | Audi |
| Cadillac | Sergio Perez (#11), Valtteri Bottas (#77) | Ferrari |
| Alpine | Pierre Gasly (#10), Franco Colapinto (#43) | Mercedes |
| Williams | Alexander Albon (#23), Carlos Sainz (#55) | Mercedes |
| RB | Liam Lawson (#30), Arvid Lindblad (#41) | Red Bull Ford |
| Haas | Esteban Ocon (#31), Oliver Bearman (#87) | Ferrari |

**Sprint weekends:** Chinese GP, Miami GP, Canadian GP, British GP, Dutch GP, Singapore GP

---

## Technical Stack

| Layer | Technology |
|-------|-----------|
| ML | scikit-learn (Ridge, Lasso, RF, GB, Stacking), XGBoost |
| Data | pandas, numpy, FastF1 |
| Backend | Python 3.11, Flask 3.0, Flask-CORS |
| Frontend | HTML5, CSS3 (F1 dark theme), vanilla JavaScript |
| CI/CD | GitHub Actions |

---

## Disclaimer

This project is built for F1 fans and data science enthusiasts. Predictions are based on historical race data and machine learning models. They are for entertainment purposes only and should not be used for betting or commercial purposes.

---

## Acknowledgements

- [FastF1](https://github.com/theOehrly/Fast-F1) — F1 timing and telemetry data
- scikit-learn, XGBoost, pandas, Flask — the foundation
- The F1 community for feedback and circuit knowledge
