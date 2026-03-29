"""
Enhanced Auto-Retrain Script for F1 2026 Race Predictor
========================================================

Features:
- Automatic detection of overfitting/underfitting
- Learning curve analysis
- Cross-validation monitoring
- Feature importance tracking
- Default grid from previous race results or qualifying
- Hyperparameter optimization
- Model performance regression detection

Usage:
    python auto_retrain_enhanced.py --round 1
    python auto_retrain_enhanced.py --all
"""

import os
import sys
import json
import pickle
import argparse
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import learning_curve, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Suppress warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from config import *
import fastf1
from feature_engineering.build_features import build_features
from model.train_model import (
    load_data, prepare_features,
    train_model as _train_model,
    evaluate_model, save_model, get_models, Config as TrainConfig,
)
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Thin wrappers so retrain_after_race() keeps its existing call signatures
# ---------------------------------------------------------------------------

def collect_latest_race_data(season: int, round_num: int) -> pd.DataFrame:
    """Fetch race results for a given season/round via FastF1 and append to historical CSV."""

    # Build lookup maps from config so team names match the training data
    driver_team_map   = {d['code']: d['team'] for d in DRIVERS_2026}
    driver_name_map   = {d['code']: d['name'] for d in DRIVERS_2026}
    driver_number_map = {d['code']: d['number'] for d in DRIVERS_2026}
    race_info = next((r for r in RACES_2026 if r['round'] == round_num), None)
    race_name = race_info['name'] if race_info else f'2026 Round {round_num}'

    cache_dir = project_root / 'data' / 'raw' / str(season) / 'cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))

    session = fastf1.get_session(season, round_num, 'R')
    session.load()

    results = session.results[['Abbreviation', 'Position', 'GridPosition', 'Points', 'Status']].copy()
    results.rename(columns={'Abbreviation': 'DriverCode'}, inplace=True)

    results['Year']         = season
    results['Round']        = round_num
    results['RaceName']     = race_name
    results['DriverName']   = results['DriverCode'].map(driver_name_map).fillna('Unknown')
    results['Team']         = results['DriverCode'].map(driver_team_map).fillna('Unknown')
    results['DriverNumber'] = results['DriverCode'].map(driver_number_map)

    new_rows = results[['Year', 'RaceName', 'Round', 'DriverNumber', 'DriverCode',
                         'DriverName', 'Team', 'GridPosition', 'Position', 'Points', 'Status']].copy()

    # Append to historical CSV (skip if already present to allow re-runs safely)
    raw_csv = project_root / 'data' / 'raw' / 'historical_race_results.csv'
    if raw_csv.exists():
        existing = pd.read_csv(raw_csv)
        already_exists = ((existing['Year'] == season) & (existing['Round'] == round_num)).any()
        if already_exists:
            print(f"  Data for {season} Round {round_num} already in historical CSV — skipping append")
        else:
            combined = pd.concat([existing, new_rows], ignore_index=True)
            combined.to_csv(raw_csv, index=False)
            print(f"  Appended {len(new_rows)} rows to {raw_csv.name}")
    else:
        new_rows.to_csv(raw_csv, index=False)
        print(f"  Created {raw_csv.name} with {len(new_rows)} rows")

    return new_rows


def build_features_for_season(season: int, include_historical: bool = True) -> pd.DataFrame:
    """Rebuild the feature dataset from the raw historical CSV and save it."""
    raw_csv = project_root / 'data' / 'raw' / 'historical_race_results.csv'
    if not raw_csv.exists():
        raise FileNotFoundError(f"Raw data not found: {raw_csv}")
    raw_df = pd.read_csv(raw_csv)
    features_df = build_features(raw_df)
    if features_df is not None:
        out_path = project_root / 'data' / 'processed' / 'f1_race_features.csv'
        out_path.parent.mkdir(parents=True, exist_ok=True)
        features_df.to_csv(out_path, index=False)
        print(f"  Saved {len(features_df)} feature rows to {out_path.name}")
    return features_df


def train_f1_model(features_df: pd.DataFrame, return_model: bool = False):
    """Train the ensemble on features_df and optionally return artefacts."""
    X, y, feature_cols = prepare_features(features_df)

    # Time-based split (chronological — avoids temporal leakage)
    if 'Date' in features_df.columns:
        sorted_idx = features_df['Date'].argsort().values
    else:
        sorted_idx = np.arange(len(features_df))
    split_point = int(len(sorted_idx) * (1 - TrainConfig.TEST_SIZE))
    train_idx, test_idx = sorted_idx[:split_point], sorted_idx[split_point:]

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index)
    X_test_s  = pd.DataFrame(scaler.transform(X_test),      columns=feature_cols, index=X_test.index)

    model, feat_imp = _train_model(X_train_s, y_train, X_test_s, y_test, feature_cols)
    metrics = evaluate_model(model, X_train_s, y_train, X_test_s, y_test, feat_imp)
    save_model(model, scaler, feature_cols, metrics, feat_imp)

    if not return_model:
        return metrics

    return {
        'model': model,
        'X_train': X_train_s.values,
        'y_train': y_train.values,
        'X_test':  X_test_s.values,
        'y_test':  y_test.values,
        'train_mae': metrics['train']['mae'],
        'test_mae':  metrics['test']['mae'],
        'train_r2':  metrics['train']['r2'],
        'test_r2':   metrics['test']['r2'],
        'train_accuracy_pm2': 0.0,   # filled by evaluate_model print output
        'test_accuracy_pm2':  metrics['accuracy_2pos'],
        'num_features': len(feature_cols),
    }


class ModelDiagnostics:
    """Advanced model diagnostics to detect overfitting/underfitting"""
    
    def __init__(self, output_dir='diagnostics'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.history = []
        
    def load_history(self):
        """Load training history from file"""
        history_file = self.output_dir / 'training_history.json'
        if history_file.exists():
            with open(history_file, 'r') as f:
                self.history = json.load(f)
        return self.history
    
    def save_history(self):
        """Save training history to file"""
        history_file = self.output_dir / 'training_history.json'
        with open(history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def analyze_learning_curves(self, model, X_train, y_train, cv=5):
        """
        Generate learning curves to detect overfitting/underfitting
        
        Returns:
            dict: Analysis results with overfitting score and recommendations
        """
        print("\n📊 Analyzing learning curves...")
        
        # Calculate learning curves
        train_sizes = np.linspace(0.1, 1.0, 10)
        train_sizes_abs, train_scores, val_scores = learning_curve(
            model, X_train, y_train,
            cv=cv,
            train_sizes=train_sizes,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            random_state=42
        )
        
        # Convert to positive MAE
        train_scores = -train_scores
        val_scores = -val_scores
        
        # Calculate means and stds
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        # Calculate overfitting score
        final_gap = val_mean[-1] - train_mean[-1]
        convergence_trend = np.mean(np.diff(val_mean[-3:]))  # Last 3 points
        
        # Determine model state
        if final_gap > 1.5:  # MAE difference > 1.5 positions
            if convergence_trend > 0:
                status = "SEVERE_OVERFIT"
                recommendation = "Reduce model complexity, add regularization, or get more data"
            else:
                status = "MILD_OVERFIT"
                recommendation = "Consider slight regularization increase"
        elif final_gap > 0.8:
            status = "ACCEPTABLE_FIT"
            recommendation = "Model is performing well, continue monitoring"
        else:
            if train_mean[-1] > 1.5:  # High training error
                status = "UNDERFITTING"
                recommendation = "Increase model complexity or add more features"
            else:
                status = "EXCELLENT_FIT"
                recommendation = "Model is well-balanced"
        
        # Plot learning curves
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes_abs, train_mean, 'o-', label='Training score', linewidth=2)
        plt.plot(train_sizes_abs, val_mean, 'o-', label='Validation score', linewidth=2)
        plt.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.1)
        plt.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, alpha=0.1)
        
        plt.xlabel('Training Examples', fontsize=12)
        plt.ylabel('Mean Absolute Error (positions)', fontsize=12)
        plt.title(f'Learning Curves - Status: {status}', fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Save plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_path = self.output_dir / f'learning_curves_{timestamp}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        analysis = {
            'status': status,
            'final_gap': float(final_gap),
            'train_error': float(train_mean[-1]),
            'val_error': float(val_mean[-1]),
            'convergence_trend': float(convergence_trend),
            'recommendation': recommendation,
            'plot_path': str(plot_path)
        }
        
        print(f"   Status: {status}")
        print(f"   Final Gap (Val - Train): {final_gap:.3f} positions")
        print(f"   Training Error: {train_mean[-1]:.3f} positions")
        print(f"   Validation Error: {val_mean[-1]:.3f} positions")
        print(f"   💡 Recommendation: {recommendation}")
        
        return analysis
    
    def cross_validate_performance(self, model, X, y, cv=5):
        """Perform cross-validation to assess model stability"""
        print("\n🔄 Running cross-validation...")
        
        # Calculate CV scores
        cv_scores_mae = cross_val_score(
            model, X, y, cv=cv,
            scoring='neg_mean_absolute_error',
            n_jobs=-1
        )
        cv_scores_r2 = cross_val_score(
            model, X, y, cv=cv,
            scoring='r2',
            n_jobs=-1
        )
        
        cv_mae = -cv_scores_mae
        cv_r2 = cv_scores_r2
        
        # Calculate stability metrics
        mae_std = np.std(cv_mae)
        r2_std = np.std(cv_r2)
        
        # Assess stability
        if mae_std < 0.1:
            stability = "EXCELLENT"
        elif mae_std < 0.2:
            stability = "GOOD"
        elif mae_std < 0.3:
            stability = "FAIR"
        else:
            stability = "POOR"
        
        print(f"   CV MAE: {np.mean(cv_mae):.3f} ± {mae_std:.3f}")
        print(f"   CV R²: {np.mean(cv_r2):.3f} ± {r2_std:.3f}")
        print(f"   Stability: {stability}")
        
        return {
            'cv_mae_mean': float(np.mean(cv_mae)),
            'cv_mae_std': float(mae_std),
            'cv_r2_mean': float(np.mean(cv_r2)),
            'cv_r2_std': float(r2_std),
            'stability': stability
        }
    
    def detect_performance_regression(self, current_metrics):
        """Detect if model performance has regressed compared to previous versions"""
        if len(self.history) == 0:
            return {'regression': False, 'message': 'First training run'}
        
        prev = self.history[-1]['metrics']
        
        # Compare test accuracy
        accuracy_drop = prev['test_accuracy_pm2'] - current_metrics['test_accuracy_pm2']
        mae_increase = current_metrics['test_mae'] - prev['test_mae']
        
        if accuracy_drop > 5.0:  # > 5% drop
            return {
                'regression': True,
                'severity': 'CRITICAL',
                'message': f"Accuracy dropped by {accuracy_drop:.1f}% (was {prev['test_accuracy_pm2']:.1f}%, now {current_metrics['test_accuracy_pm2']:.1f}%)"
            }
        elif accuracy_drop > 2.0:  # > 2% drop
            return {
                'regression': True,
                'severity': 'WARNING',
                'message': f"Accuracy dropped by {accuracy_drop:.1f}% (was {prev['test_accuracy_pm2']:.1f}%, now {current_metrics['test_accuracy_pm2']:.1f}%)"
            }
        elif mae_increase > 0.2:
            return {
                'regression': True,
                'severity': 'WARNING',
                'message': f"MAE increased by {mae_increase:.3f} (was {prev['test_mae']:.3f}, now {current_metrics['test_mae']:.3f})"
            }
        else:
            improvement = current_metrics['test_accuracy_pm2'] - prev['test_accuracy_pm2']
            return {
                'regression': False,
                'message': f"Performance improved by {improvement:.1f}%" if improvement > 0 else 'Performance stable'
            }
    
    def record_training_run(self, metrics, learning_analysis, cv_analysis):
        """Record training run in history"""
        run = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'learning_analysis': learning_analysis,
            'cv_analysis': cv_analysis
        }
        self.history.append(run)
        self.save_history()


class RaceResultsManager:
    """Manages race results and default grid positions"""
    
    def __init__(self, data_dir='data/2026_season'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True, parents=True)
        
    def save_race_result(self, round_num, results_df, session_type='R'):
        """
        Save race results for future use as default grid
        
        Args:
            round_num (int): Race round number (1-24)
            results_df (pd.DataFrame): Race results with columns [DriverCode, Position]
            session_type (str): 'R' for race, 'Q' for qualifying
        """
        filename = f'round_{round_num:02d}_{session_type}_results.json'
        filepath = self.data_dir / filename
        
        # Extract driver positions
        grid_positions = {}
        for _, row in results_df.iterrows():
            driver_code = row.get('DriverCode', row.get('Driver'))
            position = row.get('Position', row.get('Pos'))
            if driver_code and position:
                grid_positions[driver_code] = int(position)
        
        # Save to JSON
        data = {
            'round': round_num,
            'session_type': session_type,
            'timestamp': datetime.now().isoformat(),
            'grid_positions': grid_positions
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ Saved {session_type} results for Round {round_num} to {filepath}")
    
    def get_previous_race_result(self, current_round):
        """Get previous race result for default grid"""
        prev_round = current_round - 1
        if prev_round < 1:
            return None
        
        # Try race result first
        race_file = self.data_dir / f'round_{prev_round:02d}_R_results.json'
        if race_file.exists():
            with open(race_file, 'r') as f:
                return json.load(f)['grid_positions']
        
        return None
    
    def get_qualifying_result(self, current_round):
        """Get qualifying result for current race"""
        qual_file = self.data_dir / f'round_{current_round:02d}_Q_results.json'
        if qual_file.exists():
            with open(qual_file, 'r') as f:
                return json.load(f)['grid_positions']
        
        return None
    
    def get_default_grid(self, round_num, prefer_qualifying=True):
        """
        Get default grid for a race
        
        Priority:
        1. Qualifying results (if prefer_qualifying=True and available)
        2. Previous race results
        3. Championship order
        
        Args:
            round_num (int): Current race round
            prefer_qualifying (bool): Prefer qualifying over previous race
        
        Returns:
            dict: {driver_code: grid_position}
        """
        # Try qualifying first
        if prefer_qualifying:
            qual_grid = self.get_qualifying_result(round_num)
            if qual_grid:
                print(f"✓ Using qualifying results for Round {round_num}")
                return qual_grid
        
        # Try previous race result
        prev_race_grid = self.get_previous_race_result(round_num)
        if prev_race_grid:
            print(f"✓ Using previous race results (Round {round_num-1}) as default grid")
            return prev_race_grid
        
        # Fall back to championship order (from reference data)
        print(f"⚠ No previous results found, using championship order")
        return self.get_championship_order()
    
    def get_championship_order(self):
        """Get current championship order from reference data"""
        drivers_file = project_root / 'data' / 'reference' / 'drivers_2026.json'
        if drivers_file.exists():
            with open(drivers_file, 'r') as f:
                drivers = json.load(f)
            
            # Sort by points (assuming points field exists)
            drivers_sorted = sorted(
                drivers,
                key=lambda x: x.get('points', 0),
                reverse=True
            )
            
            grid = {}
            for pos, driver in enumerate(drivers_sorted, 1):
                grid[driver['code']] = pos
            
            return grid
        
        return {}


def retrain_after_race(round_num, diagnostics=None, results_manager=None):
    """
    Retrain model after a specific race
    
    Args:
        round_num (int): Race round number to collect and retrain on
        diagnostics (ModelDiagnostics): Diagnostics object for analysis
        results_manager (RaceResultsManager): Results manager for default grid
    
    Returns:
        dict: Training results and diagnostics
    """
    print(f"\n{'='*70}")
    print(f"  RETRAINING MODEL AFTER 2026 ROUND {round_num}")
    print(f"{'='*70}\n")
    
    # Initialize if not provided
    if diagnostics is None:
        diagnostics = ModelDiagnostics()
    if results_manager is None:
        results_manager = RaceResultsManager()
    
    # Step 1: Collect latest race data
    print("Step 1: Collecting race data...")
    try:
        race_data = collect_latest_race_data(2026, round_num)
        
        # Save race results for future default grid
        if race_data is not None and not race_data.empty:
            results_manager.save_race_result(round_num, race_data, 'R')
        
    except Exception as e:
        print(f"❌ Error collecting race data: {e}")
        return None
    
    # Step 2: Rebuild features
    print("\nStep 2: Rebuilding features with new data...")
    try:
        features_df = build_features_for_season(2026, include_historical=True)
        print(f"✓ Built {len(features_df)} feature rows")
    except Exception as e:
        print(f"❌ Error building features: {e}")
        return None
    
    # Step 3: Train model with diagnostics
    print("\nStep 3: Training enhanced model...")
    try:
        model_results = train_f1_model(features_df, return_model=True)
        
        if model_results is None:
            print("❌ Training failed")
            return None
        
        model = model_results['model']
        X_train = model_results['X_train']
        y_train = model_results['y_train']
        X_test = model_results['X_test']
        y_test = model_results['y_test']
        
        # Extract metrics
        metrics = {
            'train_mae': model_results['train_mae'],
            'test_mae': model_results['test_mae'],
            'train_r2': model_results['train_r2'],
            'test_r2': model_results['test_r2'],
            'train_accuracy_pm2': model_results['train_accuracy_pm2'],
            'test_accuracy_pm2': model_results['test_accuracy_pm2'],
            'features': model_results['num_features'],
            'samples': len(X_train) + len(X_test)
        }
        
    except Exception as e:
        print(f"❌ Error training model: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Step 4: Run diagnostics
    print("\nStep 4: Running advanced diagnostics...")
    
    # Learning curve analysis
    learning_analysis = diagnostics.analyze_learning_curves(model, X_train, y_train)
    
    # Cross-validation analysis
    cv_analysis = diagnostics.cross_validate_performance(
        model, 
        np.vstack([X_train, X_test]),
        np.concatenate([y_train, y_test])
    )
    
    # Performance regression detection
    diagnostics.load_history()
    regression_check = diagnostics.detect_performance_regression(metrics)
    
    # Record this training run
    diagnostics.record_training_run(metrics, learning_analysis, cv_analysis)
    
    # Step 5: Print summary report
    print("\n" + "="*70)
    print("  TRAINING SUMMARY")
    print("="*70)
    print(f"✓ Model trained successfully")
    print(f"✓ Accuracy: {metrics['test_accuracy_pm2']:.1f}% (±2 positions)")
    print(f"✓ Test MAE: {metrics['test_mae']:.3f} positions")
    print(f"✓ Test R²: {metrics['test_r2']:.3f}")
    print(f"✓ Samples: {metrics['samples']:,} total")
    print(f"✓ Features: {metrics['features']}")
    
    print(f"\n📊 Model Health:")
    print(f"   Learning Status: {learning_analysis['status']}")
    print(f"   CV Stability: {cv_analysis['stability']}")
    print(f"   Performance: {regression_check['message']}")
    
    if regression_check['regression']:
        print(f"\n⚠️  {regression_check['severity']}: {regression_check['message']}")
    
    print(f"\n💡 Recommendation: {learning_analysis['recommendation']}")
    print("="*70 + "\n")
    
    return {
        'metrics': metrics,
        'learning_analysis': learning_analysis,
        'cv_analysis': cv_analysis,
        'regression_check': regression_check
    }


def main():
    parser = argparse.ArgumentParser(description='Enhanced auto-retrain for F1 predictor')
    parser.add_argument('--round', type=int, help='Specific round number to retrain after (1-24)')
    parser.add_argument('--all', action='store_true', help='Retrain after all completed 2026 races in order')
    parser.add_argument('--auto', action='store_true', help='Auto-detect the latest completed race and retrain (for CI/cron)')
    parser.add_argument('--diagnostics-only', action='store_true', help='Run diagnostics on current model without retraining')

    args = parser.parse_args()

    # Initialize managers
    diagnostics = ModelDiagnostics()
    results_manager = RaceResultsManager()

    if args.diagnostics_only:
        print("Running diagnostics on current model...")
        print("Not yet implemented - use with --round to retrain")
        return

    if args.auto:
        # Find the latest race whose scheduled date has already passed
        today = datetime.now().date()
        completed = [r for r in RACES_2026 if datetime.strptime(r['date'], '%Y-%m-%d').date() < today]
        if not completed:
            print("No 2026 races have been completed yet based on today's date.")
            return
        latest = completed[-1]
        print(f"Auto-detected latest completed race: Round {latest['round']} — {latest['name']} ({latest['date']})")
        retrain_after_race(latest['round'], diagnostics, results_manager)

    elif args.all:
        # Retrain after every completed race in chronological order
        today = datetime.now().date()
        completed = [r for r in RACES_2026 if datetime.strptime(r['date'], '%Y-%m-%d').date() < today]
        if not completed:
            print("No 2026 races have been completed yet.")
            return
        print(f"Retraining after {len(completed)} completed races...")
        for race in completed:
            retrain_after_race(race['round'], diagnostics, results_manager)

    elif args.round:
        if 1 <= args.round <= 24:
            retrain_after_race(args.round, diagnostics, results_manager)
        else:
            print("Error: Round must be between 1 and 24")

    else:
        print("Error: Specify --round N, --auto, or --all")
        parser.print_help()


if __name__ == '__main__':
    main()