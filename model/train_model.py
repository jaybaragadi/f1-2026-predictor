"""
Enhanced F1 Model Training - Compatible with Existing Repository
================================================================

Enhancements:
- Overfitting/underfitting detection
- Hyperparameter tuning (optional, can disable for speed)
- Learning curves
- Feature selection
- Cross-validation
- Detailed metrics

Compatible with existing data files:
- data/processed/f1_training_dataset.csv OR
- data/processed/race_results_with_features.csv

Usage: python model/train_model.py
"""

import sys
import json
import pickle
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
class Config:
    # Paths (auto-detect from script location)
    SCRIPT_DIR = Path(__file__).parent
    PROJECT_ROOT = SCRIPT_DIR.parent
    DATA_DIR = PROJECT_ROOT / 'data'
    PROCESSED_DIR = DATA_DIR / 'processed'
    MODEL_DIR = SCRIPT_DIR / 'saved_models'
    RESULTS_DIR = SCRIPT_DIR / 'training_results'
    
    # Training data (try multiple names for compatibility)
    POSSIBLE_DATA_FILES = [
        PROCESSED_DIR / 'f1_training_dataset.csv',
        PROCESSED_DIR / 'race_results_with_features.csv',
        PROCESSED_DIR / 'f1_race_features.csv'
    ]
    
    # Model settings
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    CV_FOLDS = 5
    
    # Overfitting detection (set False to disable)
    CHECK_OVERFITTING = True
    MAX_TRAIN_TEST_GAP = 0.15  # 15% gap threshold
    
    # Hyperparameter tuning (set False for faster training)
    TUNE_HYPERPARAMETERS = False  # Set True for better accuracy
    
    # Feature selection (set True to reduce overfitting)
    USE_FEATURE_SELECTION = False


# ==================== DATA LOADING ====================
def find_training_data():
    """Find training data file"""
    for filepath in Config.POSSIBLE_DATA_FILES:
        if filepath.exists():
            return filepath
    return None


def load_data():
    """Load training data"""
    print("\n" + "="*70)
    print("🏎️  F1 2026 RACE PREDICTOR - ENHANCED TRAINING")
    print("="*70)
    
    data_file = find_training_data()
    
    if data_file is None:
        print("\n❌ Error: Training data not found!")
        print("\nSearched for:")
        for f in Config.POSSIBLE_DATA_FILES:
            print(f"  - {f}")
        print("\nRun feature engineering first:")
        print("  python feature_engineering/build_features.py")
        sys.exit(1)
    
    print(f"\nLoading data from: {data_file.name}")
    df = pd.read_csv(data_file)
    print(f"✓ Loaded {len(df)} training samples")
    
    return df


# ==================== FEATURE PREPARATION ====================
def prepare_features(df):
    """Prepare features for training"""
    print("\nPreparing features...")
    
    # Target
    target = 'Position'
    
    # Columns to exclude (auto-detect)
    # RacePaceVsQuali = Position / (GridPosition+1) uses the race result directly — leaky
    exclude_cols = [
        target, 'DriverCode', 'Team', 'TeamName', 'Circuit',
        'Date', 'RaceName', 'Year', 'Round', 'Season', 'Status',
        'Driver', 'DriverName', 'RacePaceVsQuali',
    ]
    
    # Get numeric feature columns
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    
    print(f"✓ Features: {len(numeric_cols)}")
    
    # Check for track-specific features
    track_features = [col for col in numeric_cols if 'track' in col.lower() or 'circuit' in col.lower()]
    if track_features:
        print(f"✓ Track-specific features detected: {len(track_features)}")
        print("  🏁 Model will be track-aware!")
    
    X = df[numeric_cols].copy()
    y = df[target].copy()
    
    # Handle NaN
    X.fillna(X.median(), inplace=True)
    
    return X, y, numeric_cols


# ==================== MODEL TRAINING ====================
def get_models(tune=False, X_train=None, y_train=None):
    """Get base models (with or without tuning)"""
    
    if tune and X_train is not None and y_train is not None:
        print("\n" + "="*70)
        print("HYPERPARAMETER TUNING (This may take 10-15 minutes)")
        print("="*70)
        
        # XGBoost tuning
        print("\nTuning XGBoost...")
        xgb_params = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [100, 200]
        }
        xgb_grid = GridSearchCV(
            XGBRegressor(random_state=Config.RANDOM_STATE),
            xgb_params, cv=3, scoring='r2', n_jobs=-1, verbose=0
        )
        xgb_grid.fit(X_train, y_train)
        xgb_model = xgb_grid.best_estimator_
        print(f"  ✓ Best R²: {xgb_grid.best_score_:.4f}")
        
        # RandomForest tuning
        print("\nTuning RandomForest...")
        rf_params = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5]
        }
        rf_grid = GridSearchCV(
            RandomForestRegressor(random_state=Config.RANDOM_STATE),
            rf_params, cv=3, scoring='r2', n_jobs=-1, verbose=0
        )
        rf_grid.fit(X_train, y_train)
        rf_model = rf_grid.best_estimator_
        print(f"  ✓ Best R²: {rf_grid.best_score_:.4f}")
        
        models = {
            'ridge': Ridge(alpha=10.0),
            'lasso': Lasso(alpha=1.0),
            'xgboost': xgb_model,
            'gb': GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=Config.RANDOM_STATE),
            'rf': rf_model
        }
    else:
        # Default models (fast training)
        models = {
            'ridge': Ridge(alpha=10.0),
            'lasso': Lasso(alpha=1.0),
            'xgboost': XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=Config.RANDOM_STATE),
            'gb': GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=Config.RANDOM_STATE),
            'rf': RandomForestRegressor(n_estimators=200, max_depth=20, random_state=Config.RANDOM_STATE)
        }
    
    return models


def train_model(X_train, y_train, X_test, y_test, feature_names):
    """Train stacking ensemble"""
    
    print("\n" + "="*70)
    print("MODEL TRAINING")
    print("="*70)
    
    # Get models
    if Config.TUNE_HYPERPARAMETERS:
        models = get_models(tune=True, X_train=X_train, y_train=y_train)
    else:
        print("\nUsing default hyperparameters (faster training)")
        models = get_models()
    
    # Train XGBoost separately to get feature importance
    print("\nTraining XGBoost for feature importance...")
    models['xgboost'].fit(X_train, y_train)
    feature_importance = dict(zip(feature_names, models['xgboost'].feature_importances_))
    
    # Create ensemble
    print("\nBuilding stacking ensemble...")
    estimators = [
        ('ridge', models['ridge']),
        ('lasso', models['lasso']),
        ('xgboost', models['xgboost']),
        ('gb', models['gb']),
        ('rf', models['rf'])
    ]
    
    ensemble = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=10.0),
        cv=Config.CV_FOLDS
    )
    
    print("  Base models: Ridge, Lasso, XGBoost, GradientBoosting, RandomForest")
    print("  Final estimator: Ridge")
    print("\nTraining ensemble...")
    ensemble.fit(X_train, y_train)
    print("✓ Training complete")
    
    return ensemble, feature_importance


# ==================== EVALUATION ====================
def evaluate_model(model, X_train, y_train, X_test, y_test, feature_importance):
    """Comprehensive evaluation"""
    
    print("\n" + "="*70)
    print("MODEL EVALUATION")
    print("="*70)
    
    # Predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    # Metrics
    train_mae = mean_absolute_error(y_train, train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    train_r2 = r2_score(y_train, train_pred)
    
    test_mae = mean_absolute_error(y_test, test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    test_r2 = r2_score(y_test, test_pred)
    
    print("\nTRAINING SET:")
    print(f"  MAE:  {train_mae:.3f} positions")
    print(f"  RMSE: {train_rmse:.3f} positions")
    print(f"  R²:   {train_r2:.3f}")
    
    print("\nTEST SET:")
    print(f"  MAE:  {test_mae:.3f} positions")
    print(f"  RMSE: {test_rmse:.3f} positions")
    print(f"  R²:   {test_r2:.3f}")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=Config.CV_FOLDS, scoring='r2', n_jobs=-1)
    print(f"\nCV R² (mean ± std): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    # Accuracy within N positions
    def accuracy_within_n(y_true, y_pred, n):
        return np.mean(np.abs(y_true - y_pred) <= n) * 100
    
    print("\nPREDICTION ACCURACY:")
    for n in [1, 2, 3]:
        train_acc = accuracy_within_n(y_train, train_pred, n)
        test_acc = accuracy_within_n(y_test, test_pred, n)
        print(f"  Within ±{n} position(s):")
        print(f"    Train: {train_acc:.1f}%")
        print(f"    Test:  {test_acc:.1f}%")
    
    # Overfitting check
    if Config.CHECK_OVERFITTING:
        gap = train_r2 - test_r2
        print("\n" + "="*70)
        print("OVERFITTING CHECK")
        print("="*70)
        print(f"Train R²: {train_r2:.3f}")
        print(f"Test R²:  {test_r2:.3f}")
        print(f"Gap:      {gap:.3f}")
        
        if gap > Config.MAX_TRAIN_TEST_GAP:
            print(f"⚠ WARNING: Potential overfitting (gap > {Config.MAX_TRAIN_TEST_GAP})")
            print("  Consider: Enable feature selection or add more data")
        else:
            print("✓ No significant overfitting detected")
    
    # Feature importance
    print("\n" + "="*70)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*70)
    
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:15]
    
    print("\nTop 15 Most Important Features:")
    print("-"*70)
    for feat, imp in top_features:
        bar_length = int(imp * 100)
        bar = "█" * bar_length
        print(f"{feat:35s} {bar} {imp:.4f}")
    
    # Track-specific features
    track_features = {k: v for k, v in feature_importance.items() if 'track' in k.lower() or 'circuit' in k.lower()}
    if track_features:
        print("\n🏁 Track-Specific Features:")
        print("-"*70)
        for feat, imp in sorted(track_features.items(), key=lambda x: x[1], reverse=True):
            print(f"  {feat:35s} Importance: {imp:.4f}")
        print(f"\n  Total track feature importance: {sum(track_features.values())*100:.2f}%")
        print("  💡 Higher = More circuit-aware predictions!")
    
    return {
        'train': {'mae': train_mae, 'rmse': train_rmse, 'r2': train_r2},
        'test': {'mae': test_mae, 'rmse': test_rmse, 'r2': test_r2},
        'cv': {'mean_r2': cv_scores.mean(), 'std_r2': cv_scores.std()},
        'accuracy_2pos': accuracy_within_n(y_test, test_pred, 2)
    }


# ==================== SAVE MODEL ====================
def save_model(model, scaler, feature_cols, metrics, feature_importance):
    """Save model and metadata"""
    
    print("\n" + "="*70)
    print("SAVING MODEL")
    print("="*70)
    
    Config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    Config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = Config.MODEL_DIR / 'f1_race_predictor_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✓ Model: {model_path}")
    
    # Save scaler
    scaler_path = Config.MODEL_DIR / 'scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✓ Scaler: {scaler_path}")
    
    # Save features
    features_path = Config.MODEL_DIR / 'feature_columns.pkl'
    with open(features_path, 'wb') as f:
        pickle.dump(feature_cols, f)
    print(f"✓ Features: {features_path}")
    
    # Save metadata
    metadata = {
        'training_date': datetime.now().isoformat(),
        'n_features': len(feature_cols),
        'test_accuracy': f"{metrics['accuracy_2pos']:.1f}%",
        'test_r2': metrics['test']['r2'],
        'test_mae': metrics['test']['mae'],
        'cv_r2_mean': metrics['cv']['mean_r2'],
        'cv_r2_std': metrics['cv']['std_r2'],
        'overfitting_gap': metrics['train']['r2'] - metrics['test']['r2'],
        'feature_importance': {k: float(v) for k, v in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:20]}
    }
    
    metadata_path = Config.MODEL_DIR / 'model_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata: {metadata_path}")
    
    print("\n✅ All artifacts saved successfully!")


# ==================== MAIN ====================
def main():
    """Main training pipeline"""
    
    # Load data
    df = load_data()
    
    # Prepare features
    X, y, feature_cols = prepare_features(df)
    
    print(f"\nDataset shape: {X.shape}")
    print(f"Train/test split: {int((1-Config.TEST_SIZE)*100)}% / {int(Config.TEST_SIZE*100)}% (TIME-BASED)")

    # Time-based split: sort by date so the test set is always the most recent races.
    # Random splitting causes temporal leakage — future races bleed into training.
    if 'Date' in df.columns:
        sorted_idx = df['Date'].argsort().values
    elif 'Year' in df.columns:
        sorted_idx = df['Year'].argsort().values
    else:
        sorted_idx = np.arange(len(df))

    split_point = int(len(sorted_idx) * (1 - Config.TEST_SIZE))
    train_idx = sorted_idx[:split_point]
    test_idx = sorted_idx[split_point:]

    X_train = X.iloc[train_idx]
    X_test  = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test  = y.iloc[test_idx]

    print(f"✓ Train set: {len(X_train)} samples (oldest {int((1-Config.TEST_SIZE)*100)}%)")
    print(f"✓ Test set: {len(X_test)} samples (most recent {int(Config.TEST_SIZE*100)}%)")
    
    # Scale
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols, index=X_test.index)
    print("✓ Features scaled")
    
    # Train
    model, feature_importance = train_model(X_train_scaled, y_train, X_test_scaled, y_test, feature_cols)
    
    # Evaluate
    metrics = evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test, feature_importance)
    
    # Save
    save_model(model, scaler, feature_cols, metrics, feature_importance)
    
    # Summary
    print("\n" + "="*70)
    print("✅ ENHANCED TRAINING COMPLETE!")
    print("="*70)
    print(f"✓ Model accuracy (±2): {metrics['accuracy_2pos']:.1f}%")
    print(f"✓ Test R²: {metrics['test']['r2']:.3f}")
    print(f"✓ Overfitting gap: {metrics['train']['r2'] - metrics['test']['r2']:.3f}")
    if Config.CHECK_OVERFITTING and (metrics['train']['r2'] - metrics['test']['r2']) <= Config.MAX_TRAIN_TEST_GAP:
        print("✓ Overfitting check: PASS")
    print("="*70)
    print("\n✅ Model ready for predictions!")
    print("\nNext steps:")
    print("  1. Run Flask app: python app/app.py")
    print("  2. Visit: http://127.0.0.1:5001")
    print("  3. Make predictions!\n")


if __name__ == "__main__":
    main()