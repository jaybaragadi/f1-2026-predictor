"""
Flask Backend for F1 2026 Race Predictor
Dynamic feature cache — adapts automatically to any trained feature set.
"""

import sys
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS

# Add parent directory to path so feature_engineering imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    MODEL_DIR,
    RACES_2026,
    DRIVERS_2026,
    TEAMS_2026,
    REFERENCE_DATA_DIR,
    HOST,
    PORT,
    SECRET_KEY,
)

# -------------------- Flask App --------------------
app = Flask(__name__)
app.secret_key = SECRET_KEY
CORS(app)

predictor = None
model_loaded = False
model_metadata = {}

# Feature groups used to route lookups to the right cache
_TRACK_DRIVER_FEATS = {
    'DriverTrackAvg', 'DriverTrackBest', 'CircuitBestPosition', 'CircuitRacesCount',
}
_TRACK_TEAM_FEATS = {
    'TeamTrackAvg', 'TeamCircuitBest',
}
_CIRCUIT_FEATS = {
    'CircuitAvgPosition',
}
_TEAM_FEATS = {
    'TeamYearPoints', 'TeamYearAvgPosition', 'TeamRaceAvgPosition',
    'TeamYearReliability', 'TeamYearBest', 'TeamRecentForm',
}

# Sensible defaults for features where 11.0 is wrong
_FEATURE_DEFAULTS = {
    'CircuitRacesCount': 0.0,
    'CareerRaceCount':   0.0,
    'RacesWithTeam':     0.0,
    'CareerPodiums':     0.0,
    'CareerWins':        0.0,
    'CareerPoints':      0.0,
    'ChampionshipPoints':0.0,
    'GapToLeader':       0.0,
    'FightingForTitle':  0.0,
    'PointsMomentum':    1.0,
    'WinStreak':         0.0,
    'PodiumStreak':      0.0,
    'PointsFinishStreak':0.0,
    'RecentReliability': 5.0,
    'ChampionshipPosition': 11.0,
}


# -------------------- Predictor --------------------
class F1RacePredictor:
    """Race predictor — builds features dynamically from the historical dataset."""

    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.drivers_2026 = None
        self.teams_2026 = None
        self.feature_cache = None   # populated in load_model

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def load_model(self):
        try:
            self.model          = joblib.load(MODEL_DIR / "f1_race_predictor_model.pkl")
            self.scaler         = joblib.load(MODEL_DIR / "scaler.pkl")
            self.feature_columns = joblib.load(MODEL_DIR / "feature_columns.pkl")

            self.drivers_2026 = pd.read_csv(REFERENCE_DATA_DIR / "2026_drivers.csv")
            self.teams_2026   = pd.read_csv(REFERENCE_DATA_DIR / "2026_teams.csv")
            self.drivers_2026["DriverNumber"] = self.drivers_2026["DriverNumber"].astype(int)

            # Build the feature cache from the clean feature CSV (or raw data)
            self.feature_cache = self._build_feature_cache()

            print("Model loaded successfully")
            print(f"  Features : {len(self.feature_columns)}")
            print(f"  Cache    : {len(self.feature_cache['driver'])} drivers, "
                  f"{len(self.feature_cache['circuit'])} circuits")
            return True

        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            return False

    # ------------------------------------------------------------------
    # Feature cache construction
    # ------------------------------------------------------------------
    def _build_feature_cache(self):
        """
        Load the clean feature CSV and build fast lookup caches.
        Falls back to building features from raw data if the CSV is absent.
        """
        project_root  = Path(__file__).parent.parent
        features_csv  = project_root / "data" / "processed" / "f1_race_features.csv"
        raw_csv       = project_root / "data" / "raw"       / "historical_race_results.csv"

        features_df = None

        if features_csv.exists():
            features_df = pd.read_csv(features_csv)
        elif raw_csv.exists():
            from feature_engineering.build_features import build_features
            features_df = build_features(pd.read_csv(raw_csv))

        if features_df is None or features_df.empty:
            print("WARNING: No feature data found — using neutral defaults for all drivers")
            return {'driver': {}, 'team': {}, 'track_driver': {}, 'track_team': {}, 'circuit': {}}

        # Ensure chronological order
        if 'Date' in features_df.columns:
            features_df['Date'] = pd.to_datetime(features_df['Date'])
            features_df = features_df.sort_values('Date')

        circuit_col = 'Circuit' if 'Circuit' in features_df.columns else 'RaceName'

        # ---------- per-driver latest row ----------
        driver_cache = {
            code: grp.iloc[-1].to_dict()
            for code, grp in features_df.groupby('DriverCode')
        }

        # ---------- per-team latest row ----------
        team_cache = {
            team: grp.iloc[-1].to_dict()
            for team, grp in features_df.groupby('Team')
        }

        # ---------- per-driver+circuit latest row ----------
        track_driver_cache = {
            (code, circuit): grp.iloc[-1].to_dict()
            for (code, circuit), grp in features_df.groupby(['DriverCode', circuit_col])
        }

        # ---------- per-team+circuit latest row ----------
        track_team_cache = {
            (team, circuit): grp.iloc[-1].to_dict()
            for (team, circuit), grp in features_df.groupby(['Team', circuit_col])
        }

        # ---------- per-circuit latest row (for CircuitAvgPosition) ----------
        circuit_cache = {
            circuit: grp.iloc[-1].to_dict()
            for circuit, grp in features_df.groupby(circuit_col)
        }

        return {
            'driver':       driver_cache,
            'team':         team_cache,
            'track_driver': track_driver_cache,
            'track_team':   track_team_cache,
            'circuit':      circuit_cache,
            'circuit_col':  circuit_col,
        }

    # ------------------------------------------------------------------
    # Circuit name matching
    # ------------------------------------------------------------------
    def _find_circuit_name(self, race_name: str):
        """Return the circuit key used in the training data that best matches race_name."""
        if not self.feature_cache or not self.feature_cache['circuit']:
            return None

        known = set(self.feature_cache['circuit'].keys())
        race_lower = race_name.lower()

        # 1. Exact
        if race_name in known:
            return race_name

        # 2. Case-insensitive
        for c in known:
            if c.lower() == race_lower:
                return c

        # 3. Substring
        for c in known:
            if c.lower() in race_lower or race_lower in c.lower():
                return c

        # 4. Keyword (e.g. "Australian" in "Australian Grand Prix")
        keywords = [w for w in race_lower.replace(' grand prix', '').split() if len(w) > 3]
        for c in known:
            if any(kw in c.lower() for kw in keywords):
                return c

        return None

    # ------------------------------------------------------------------
    # Feature vector construction
    # ------------------------------------------------------------------
    def _get_cached(self, cache_dict, key, feature):
        """Safe feature lookup from a cache dict with NaN guard."""
        row = cache_dict.get(key, {})
        val = row.get(feature)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        return float(val)

    def prepare_features(self, grid_positions_by_number, race_info):
        """
        Build one feature row per driver using the historical feature cache.
        Works with any model feature set — no hardcoded feature list.
        """
        rows = []
        race_name    = race_info.get("name", "")
        circuit_name = self._find_circuit_name(race_name)

        for driver_num, grid_pos in grid_positions_by_number.items():
            driver_num_int = int(driver_num)

            driver_df = self.drivers_2026[self.drivers_2026["DriverNumber"] == driver_num_int]
            if driver_df.empty:
                raise ValueError(f"Driver number {driver_num_int} not found in 2026 drivers")

            driver      = driver_df.iloc[0]
            driver_code = driver["DriverCode"]
            driver_team = driver["Team"]

            if self.teams_2026[self.teams_2026["Team"] == driver_team].empty:
                raise ValueError(f"Team '{driver_team}' not found")

            row = {}

            for feat in self.feature_columns:
                val = None

                if feat == 'GridPosition':
                    val = float(grid_pos)

                # --- Circuit + driver features ---
                elif feat in _TRACK_DRIVER_FEATS and circuit_name:
                    val = self._get_cached(
                        self.feature_cache['track_driver'],
                        (driver_code, circuit_name), feat
                    )

                # --- Circuit + team features ---
                elif feat in _TRACK_TEAM_FEATS and circuit_name:
                    val = self._get_cached(
                        self.feature_cache['track_team'],
                        (driver_team, circuit_name), feat
                    )

                # --- Circuit-wide features ---
                elif feat in _CIRCUIT_FEATS and circuit_name:
                    val = self._get_cached(
                        self.feature_cache['circuit'],
                        circuit_name, feat
                    )

                # --- Team-wide features ---
                elif feat in _TEAM_FEATS:
                    val = self._get_cached(
                        self.feature_cache['team'],
                        driver_team, feat
                    )

                # --- Everything else: driver's latest historical value ---
                if val is None:
                    val = self._get_cached(
                        self.feature_cache['driver'],
                        driver_code, feat
                    )

                # --- Final fallback: feature-specific sensible default ---
                if val is None:
                    val = _FEATURE_DEFAULTS.get(feat, 11.0)

                row[feat] = val

            # Metadata carried alongside features (not used by model)
            row['DriverNumber'] = driver_num_int
            row['DriverName']   = driver.get("DriverName", str(driver_num_int))
            row['DriverCode']   = driver_code
            row['Team']         = driver_team

            rows.append(row)

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, grid_positions_by_number, race_info):
        features_df = self.prepare_features(grid_positions_by_number, race_info)

        X = features_df[self.feature_columns]
        # Preserve DataFrame so sub-models keep feature names (no sklearn warnings)
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=self.feature_columns,
            index=X.index,
        )

        preds = self.model.predict(X_scaled)
        preds = np.clip(preds, 1, 22)

        features_df["PredictedPosition"] = preds
        features_df = features_df.sort_values("PredictedPosition")

        results = []
        for idx, (_, row) in enumerate(features_df.iterrows(), 1):
            results.append({
                "position":           idx,
                "predicted_position": float(row["PredictedPosition"]),
                "driver_number":      int(row["DriverNumber"]),
                "driver_code":        row["DriverCode"],
                "driver_name":        row["DriverName"],
                "team":               row["Team"],
                "grid_position":      int(row["GridPosition"]),
                "position_change":    int(row["GridPosition"]) - idx,
            })
        return results


# -------------------- Helpers --------------------
def init_predictor():
    global predictor, model_loaded, model_metadata

    print("\n" + "=" * 70)
    print("F1 2026 RACE PREDICTOR - FLASK SERVER")
    print("=" * 70)

    predictor    = F1RacePredictor()
    model_loaded = predictor.load_model()

    if model_loaded:
        # Read accuracy from metadata file if available
        meta_file = MODEL_DIR / "model_metadata.json"
        saved_accuracy = "N/A"
        if meta_file.exists():
            import json
            try:
                with open(meta_file) as f:
                    saved_meta = json.load(f)
                saved_accuracy = saved_meta.get("test_accuracy", saved_accuracy)
            except Exception:
                pass

        model_metadata = {
            "features":           len(predictor.feature_columns),
            "drivers":            22,
            "teams":              11,
            "races":              len(RACES_2026),
            "season":             2026,
            "defending_champion": "Lando Norris (#1)",
            "track_aware":        any('Track' in f or 'Circuit' in f
                                      for f in predictor.feature_columns),
            "accuracy":           saved_accuracy,
            "model_type":         "5-Algorithm Ensemble",
        }
        print(f"\nModel Status  : LOADED")
        print(f"Features      : {model_metadata['features']}")
        print(f"Accuracy      : {model_metadata['accuracy']} (within +/-2 positions)")
        print(f"Track-aware   : {'Yes' if model_metadata['track_aware'] else 'No'}")
        print("=" * 70 + "\n")
    else:
        print("\nModel failed to load\n")


def race_by_name(race_name: str):
    if not race_name:
        return None
    name_l = race_name.strip().lower()
    for r in RACES_2026:
        if r["name"].strip().lower() == name_l:
            return r
    return None


def build_code_to_number_map():
    return {d["code"]: int(d["number"]) for d in DRIVERS_2026}


def build_number_to_driver_map():
    return {int(d["number"]): d for d in DRIVERS_2026}


# -------------------- Routes --------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/model-info", methods=["GET"])
def get_model_info():
    return jsonify({"loaded": model_loaded, "metadata": model_metadata})


@app.route("/api/races", methods=["GET"])
def get_races():
    races_with_status = []
    today = datetime.now().date()

    for race in RACES_2026:
        race_date = datetime.strptime(race["date"], "%Y-%m-%d").date()
        if race_date < today:
            status = "completed"
        elif race_date == today:
            status = "today"
        else:
            status = "upcoming"

        races_with_status.append({
            **race,
            "status":         status,
            "formatted_date": race_date.strftime("%b %d, %Y"),
            "full_date":      race_date.strftime("%B %d, %Y"),
            "has_sprint":     bool(race.get("has_sprint", False)),
            "is_sprint_race": bool(race.get("is_sprint_race", False))
                              or str(race.get("format", "")).lower() == "sprint",
        })

    return jsonify(races_with_status)


@app.route("/api/drivers", methods=["GET"])
def get_drivers():
    return jsonify(DRIVERS_2026)


@app.route("/api/teams", methods=["GET"])
def get_teams():
    return jsonify(TEAMS_2026)


@app.route("/api/default-grid", methods=["GET"])
def get_default_grid():
    """Return a default grid keyed by driver code, ordered by championship pedigree."""
    sorted_drivers = sorted(
        DRIVERS_2026,
        key=lambda x: (x.get("championships", 0), x.get("experience", 0)),
        reverse=True,
    )
    grid_positions = {d["code"]: idx for idx, d in enumerate(sorted_drivers, 1)}
    return jsonify({"status": "success", "grid_positions": grid_positions})


@app.route("/api/predict", methods=["POST"])
def predict_race():
    if not model_loaded:
        return jsonify({"status": "error", "message": "Model not loaded"}), 500

    try:
        data = request.get_json(force=True) or {}

        race_name = data.get("race")
        race_info = race_by_name(race_name)
        if not race_info:
            return jsonify({"status": "error", "message": "Invalid race name"}), 400

        grid_positions = data.get("grid_positions") or {}
        if not isinstance(grid_positions, dict) or len(grid_positions) == 0:
            return jsonify({"status": "error", "message": "grid_positions missing"}), 400

        code_to_num  = build_code_to_number_map()
        num_to_driver = build_number_to_driver_map()

        grid_by_number = {}
        for code, pos in grid_positions.items():
            if code not in code_to_num:
                return jsonify({"status": "error",
                                "message": f"Unknown driver code: {code}"}), 400
            grid_by_number[str(code_to_num[code])] = int(pos)

        if len(grid_by_number) != 22:
            return jsonify({"status": "error",
                            "message": f"Expected 22 drivers, got {len(grid_by_number)}"}), 400

        positions = list(grid_by_number.values())
        if len(set(positions)) != len(positions):
            return jsonify({"status": "error", "message": "Duplicate grid positions"}), 400
        if not all(1 <= p <= 22 for p in positions):
            return jsonify({"status": "error",
                            "message": "Grid positions must be 1-22"}), 400

        raw_preds = predictor.predict(grid_by_number, race_info)

        predictions = []
        for p in raw_preds:
            driver_num = p["driver_number"]
            d = num_to_driver.get(driver_num, {})
            predictions.append({
                "position":        p["position"],
                "predictedPosition": p["predicted_position"],
                "driverNumber":    driver_num,
                "driverCode":      p.get("driver_code", d.get("code")),
                "driverName":      p["driver_name"],
                "team":            p["team"],
                "gridPosition":    p["grid_position"],
                "positionsGained": p["position_change"],
            })

        return jsonify({
            "status":     "success",
            "race":       race_info["name"],
            "race_info":  race_info,
            "predictions": predictions,
        })

    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status":       "healthy",
        "model_loaded": model_loaded,
        "timestamp":    datetime.now().isoformat(),
    })


if __name__ == "__main__":
    init_predictor()
    app.run(host=HOST, port=PORT, debug=False, use_reloader=False)
