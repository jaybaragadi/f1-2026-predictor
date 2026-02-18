"""
Sprint-Specific Model Training
Trains a model optimized for sprint race predictions
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import PROCESSED_DATA_DIR, MODEL_DIR, RANDOM_STATE, TEST_SIZE, CV_FOLDS


def train_sprint_model():
    """
    Train sprint-specific race prediction model
    """
    
    print("\n" + "="*70)
    print("🏁 F1 SPRINT RACE PREDICTOR - MODEL TRAINING")
    print("="*70 + "\n")
    
    # Load sprint training dataset
    print("Loading sprint training dataset...")
    data_file = PROCESSED_DATA_DIR / 'f1_sprint_training_dataset.csv'
    
    if not data_file.exists():
        print("❌ Sprint training dataset not found!")
        print("   Please run: python feature_engineering/build_sprint_features.py")
        return
    
    df = pd.read_csv(data_file)
    print(f"✓ Loaded {len(df)} training samples")
    
    # Separate features and target
    target_col = 'Position'
    exclude_cols = ['Position', 'Year', 'IsSprintRace']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols]
    y = df[target_col]
    
    print(f"✓ Features: {len(feature_cols)}")
    
    print("\n📊 SPRINT DATA ANALYSIS:")
    sprint_count = df['IsSprintRace'].sum() if 'IsSprintRace' in df.columns else 0
    main_count = len(df) - sprint_count
    print(f"  Sprint races: {sprint_count} samples")
    print(f"  Main races: {main_count} samples (for comparison)")
    print(f"  Total: {len(df)} samples")
    
    # Train/test split
    print("\nPreparing train/test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE,
        stratify=df['Year'] if 'Year' in df.columns else None
    )
    print(f"✓ Train set: {len(X_train)} samples")
    print(f"✓ Test set: {len(X_test)} samples")
    
    # Scale features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("✓ Features scaled")
    
    # Build SPRINT-OPTIMIZED stacking ensemble
    print("\nBuilding sprint-optimized stacking ensemble model...")
    
    # Base models - tuned for sprint characteristics
    base_models = [
        # Ridge - Good for high qualifying impact
        ('ridge', Ridge(
            alpha=1.0,
            random_state=RANDOM_STATE
        )),
        
        # Lasso - Feature selection (remove pit strategy features)
        ('lasso', Lasso(
            alpha=0.5,
            random_state=RANDOM_STATE,
            max_iter=2000
        )),
        
        # XGBoost - Captures overtaking patterns
        ('xgboost', XGBRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            verbosity=0
        )),
        
        # Gradient Boosting - Grid position importance
        ('gb', GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            random_state=RANDOM_STATE
        )),
        
        # Random Forest - Consistency in short races
        ('rf', RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ]
    
    # Meta-learner
    meta_learner = Ridge(alpha=1.0, random_state=RANDOM_STATE)
    
    # Create stacking ensemble
    sprint_model = StackingRegressor(
        estimators=base_models,
        final_estimator=meta_learner,
        cv=CV_FOLDS,
        n_jobs=-1
    )
    
    print("✓ Model architecture:")
    print("  Base models: Ridge, Lasso, XGBoost, GradientBoosting, RandomForest")
    print("  Final estimator: Ridge")
    print("  Optimization: Sprint-specific (qualifying weight, overtaking)")
    
    # Train model
    print("\nTraining sprint model...")
    print("This may take several minutes...")
    sprint_model.fit(X_train_scaled, y_train)
    print("✓ Model training complete")
    
    # Evaluate model
    print("\n" + "="*70)
    print("SPRINT MODEL EVALUATION")
    print("="*70 + "\n")
    
    # Predictions
    y_train_pred = sprint_model.predict(X_train_scaled)
    y_test_pred = sprint_model.predict(X_test_scaled)
    
    # Clip predictions to valid position range [1, 22]
    y_train_pred = np.clip(y_train_pred, 1, 22)
    y_test_pred = np.clip(y_test_pred, 1, 22)
    
    # Calculate metrics
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_r2 = r2_score(y_test, y_test_pred)
    
    print("TRAINING SET:")
    print(f"  MAE:  {train_mae:.3f} positions")
    print(f"  RMSE: {train_rmse:.3f} positions")
    print(f"  R²:   {train_r2:.3f}")
    
    print("\nTEST SET:")
    print(f"  MAE:  {test_mae:.3f} positions")
    print(f"  RMSE: {test_rmse:.3f} positions")
    print(f"  R²:   {test_r2:.3f}")
    
    # Accuracy within N positions
    def accuracy_within_n(y_true, y_pred, n):
        return np.mean(np.abs(y_true - y_pred) <= n) * 100
    
    print(f"\nAccuracy within ±1 position(s):")
    print(f"  Train: {accuracy_within_n(y_train, y_train_pred, 1):.1f}%")
    print(f"  Test:  {accuracy_within_n(y_test, y_test_pred, 1):.1f}%")
    
    print(f"\nAccuracy within ±2 position(s):")
    print(f"  Train: {accuracy_within_n(y_train, y_train_pred, 2):.1f}%")
    print(f"  Test:  {accuracy_within_n(y_test, y_test_pred, 2):.1f}%")
    
    print(f"\nAccuracy within ±3 position(s):")
    print(f"  Train: {accuracy_within_n(y_train, y_train_pred, 3):.1f}%")
    print(f"  Test:  {accuracy_within_n(y_test, y_test_pred, 3):.1f}%")
    
    print("="*70)
    
    # Save model artifacts
    print("\nSaving sprint model artifacts...")
    
    # Create sprint model directory
    sprint_model_dir = MODEL_DIR / 'sprint'
    sprint_model_dir.mkdir(exist_ok=True, parents=True)
    
    model_file = sprint_model_dir / 'f1_sprint_predictor_model.pkl'
    scaler_file = sprint_model_dir / 'sprint_scaler.pkl'
    features_file = sprint_model_dir / 'sprint_feature_columns.pkl'
    
    joblib.dump(sprint_model, model_file)
    joblib.dump(scaler, scaler_file)
    joblib.dump(feature_cols, features_file)
    
    print(f"✓ Model saved: {model_file}")
    print(f"✓ Scaler saved: {scaler_file}")
    print(f"✓ Feature columns saved: {features_file}")
    
    print("\n✅ All sprint model artifacts saved successfully!")
    
    # Compare with main race model (if exists)
    main_model_file = MODEL_DIR / 'f1_race_predictor_model.pkl'
    if main_model_file.exists():
        print("\n" + "="*70)
        print("📊 SPRINT vs MAIN RACE MODEL COMPARISON")
        print("="*70)
        print(f"\nSprint Model Test MAE: {test_mae:.3f} positions")
        print("Main Race Model Test MAE: ~1.27 positions (from training)")
        
        if test_mae < 1.27:
            print("\n✅ Sprint model is MORE accurate!")
            print("   Sprint-specific features improve predictions")
        else:
            print("\n⚠️  Sprint model needs more sprint-specific data")
            print("   Will improve as more 2026 sprint races are added")
        print("="*70)
    
    print("\n" + "="*70)
    print("✅ SPRINT MODEL TRAINING COMPLETE!")
    print("="*70)
    
    print("\nNext steps:")
    print("1. Use sprint model for sprint race predictions")
    print("2. Use main model for regular race predictions")
    print("3. After each 2026 sprint, run auto_retrain.py")
    print("4. Sprint model accuracy will improve throughout season")
    
    print("\n" + "="*70)
    print("✅ Sprint model ready for predictions!")
    print("="*70 + "\n")
    
    return sprint_model, scaler, feature_cols


if __name__ == "__main__":
    model, scaler, features = train_sprint_model()
    print("✅ Sprint predictor trained and saved!\n")