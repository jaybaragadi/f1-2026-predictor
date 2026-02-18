"""
Sprint-Specific Feature Engineering - FINAL WORKING VERSION
Compatible with actual data structure
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, REFERENCE_DATA_DIR, YEAR_WEIGHTS


def identify_sprint_races(df):
    """Identify sprint races in historical data"""
    sprint_keywords = ['Sprint', 'sprint']
    df['IsSprintRace'] = df['RaceName'].str.contains('|'.join(sprint_keywords), case=False, na=False).astype(int)
    return df


def calculate_sprint_specific_features(df):
    """Calculate features that matter more in sprint races"""
    
    # 1. QUALIFYING DOMINANCE
    df['QualifyingGap'] = df['GridPosition'] - df['Position']
    driver_quali_avg = df.groupby('DriverName')['GridPosition'].transform('mean')
    df['QualifyingPerformance'] = 11 - (df['GridPosition'] / (driver_quali_avg + 0.1))
    
    # 2. FIRST LAP PERFORMANCE
    df['FirstLapGain'] = df['GridPosition'] - df['Position']
    df['FirstLapGain_Rolling3'] = df.groupby('DriverName')['FirstLapGain'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    
    # 3. OVERTAKING ABILITY
    df['AvgPositionsGained'] = df.groupby('DriverName')['FirstLapGain'].transform('mean')
    df['OvertakingScore'] = df['AvgPositionsGained'] * 2
    
    # 4. CONSISTENCY
    driver_position_std = df.groupby('DriverName')['Position'].transform('std')
    df['ConsistencyScore'] = 1 / (driver_position_std + 1)
    
    # 5. RACE PACE vs QUALIFYING PACE
    df['QualifyingRaceDelta'] = df['GridPosition'] - df['Position']
    df['SprintRacePace'] = df.groupby('DriverName')['QualifyingRaceDelta'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
    )
    
    # 6. TRACK POSITION IMPORTANCE (1.5x more important in sprints)
    df['TrackPositionWeight'] = 1.5
    df['GridPosition_Weighted'] = df['GridPosition'] * df['TrackPositionWeight']
    
    # 7. AGGRESSIVE DRIVING STYLE
    df['AggressiveStyle'] = (df['FirstLapGain'] > 0).astype(int)
    df['AggressiveScore'] = df.groupby('DriverName')['AggressiveStyle'].transform('mean')
    
    return df


def build_sprint_training_dataset():
    """Build training dataset specifically for sprint races"""
    
    print("\n" + "="*70)
    print("🏁 SPRINT RACE FEATURE ENGINEERING")
    print("="*70 + "\n")
    
    # Load data
    print("Loading data files...")
    race_file = RAW_DATA_DIR / 'historical_race_results.csv'
    quali_file = RAW_DATA_DIR / 'historical_quali_results.csv'
    drivers_file = REFERENCE_DATA_DIR / '2026_drivers.csv'
    teams_file = REFERENCE_DATA_DIR / '2026_teams.csv'
    
    races = pd.read_csv(race_file)
    quali = pd.read_csv(quali_file)
    drivers_2026 = pd.read_csv(drivers_file)
    teams_2026 = pd.read_csv(teams_file)
    
    print(f"✓ Loaded {len(races)} race results")
    print(f"✓ Loaded {len(quali)} qualifying results")
    print(f"✓ Loaded {len(drivers_2026)} 2026 drivers")
    print(f"✓ Loaded {len(teams_2026)} 2026 teams")
    
    # Show available columns
    print(f"\n📊 Available columns:")
    print(f"  Race data: {list(races.columns)}")
    
    # Identify sprint races
    print("\nIdentifying sprint races...")
    races = identify_sprint_races(races)
    sprint_races = races[races['IsSprintRace'] == 1]
    print(f"✓ Found {len(sprint_races)} sprint race results")
    if len(sprint_races) == 0:
        print("  Note: No historical sprint races detected")
        print("  Model will train on all races with sprint-optimized features")
    
    # Merge race and qualifying data
    print("\nMerging race and qualifying data...")
    
    # Add DriverName to quali if needed
    if 'DriverName' not in quali.columns and 'DriverCode' in quali.columns:
        driver_code_map = races[['DriverName', 'DriverCode']].drop_duplicates()
        quali = quali.merge(driver_code_map, on='DriverCode', how='left')
        print("  ✓ Added DriverName to qualifying data")
    
    # Merge
    df = pd.merge(
        races,
        quali[['Year', 'RaceName', 'DriverName', 'QualifyingPosition']],
        on=['Year', 'RaceName', 'DriverName'],
        how='left'
    )
    print(f"✓ Merged data: {len(df)} records")
    
    # Fill missing qualifying positions with grid positions
    df['QualifyingPosition'] = df['QualifyingPosition'].fillna(df['GridPosition'])
    
    # Calculate sprint-specific features
    print("\nCalculating sprint-specific features...")
    df = calculate_sprint_specific_features(df)
    print("✓ Sprint features calculated")
    
    # Add standard race features
    print("\nAdding standard race features...")
    
    # Driver features
    df['Experience'] = df['Year'] - df.groupby('DriverName')['Year'].transform('min')
    df['CareerAvgPosition'] = df.groupby('DriverName')['Position'].transform('mean')
    df['Position_Rolling_3'] = df.groupby('DriverName')['Position'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    df['Position_Rolling_5'] = df.groupby('DriverName')['Position'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
    )
    
    # Team features
    df['TeamYearAvgPosition'] = df.groupby(['Team', 'Year'])['Position'].transform('mean')
    df['TeamYearPoints'] = df.groupby(['Team', 'Year'])['Points'].transform('sum')
    
    # Circuit features (use RaceName as proxy for circuit)
    print("  Creating circuit features using RaceName...")
    df['CircuitAvgPosition'] = df.groupby(['DriverName', 'RaceName'])['Position'].transform('mean')
    
    # Recency weighting
    df['RecencyWeight'] = df['Year'].map(YEAR_WEIGHTS).fillna(1.0)
    
    print("✓ Standard features added")
    
    # Clean dataset
    print("\nCleaning dataset...")
    initial_count = len(df)
    
    # Only drop rows with missing critical columns
    critical_cols = ['Position', 'GridPosition']
    df = df.dropna(subset=critical_cols)
    
    final_count = len(df)
    
    print(f"  Starting with {initial_count} records")
    print(f"  After cleaning: {final_count} records")
    
    # Select features for sprint model
    sprint_features = [
        # Target
        'Position',
        
        # Sprint-specific features (HIGH IMPORTANCE)
        'GridPosition_Weighted',
        'QualifyingPerformance',
        'FirstLapGain_Rolling3',
        'OvertakingScore',
        'ConsistencyScore',
        'SprintRacePace',
        'AggressiveScore',
        
        # Standard features (MODERATE IMPORTANCE)
        'GridPosition',
        'QualifyingPosition',
        'Experience',
        'CareerAvgPosition',
        'Position_Rolling_3',
        'Position_Rolling_5',
        'TeamYearAvgPosition',
        'TeamYearPoints',
        'CircuitAvgPosition',
        
        # Meta features
        'RecencyWeight',
        'IsSprintRace',
        'Year'
    ]
    
    # Ensure all features exist
    available_features = [f for f in sprint_features if f in df.columns]
    missing_features = [f for f in sprint_features if f not in df.columns]
    
    if missing_features:
        print(f"\n⚠️  Skipping missing features: {missing_features}")
    
    df_sprint = df[available_features].copy()
    
    # Fill any remaining NaN with 0
    df_sprint = df_sprint.fillna(0)
    
    print(f"\n✓ Selected {len(available_features)} features for sprint model")
    print(f"✓ Final dataset shape: {df_sprint.shape}")
    
    # Save sprint training dataset
    output_file = PROCESSED_DATA_DIR / 'f1_sprint_training_dataset.csv'
    df_sprint.to_csv(output_file, index=False)
    
    print("\n" + "="*70)
    print("✅ SPRINT FEATURE ENGINEERING COMPLETE!")
    print("="*70)
    print(f"\n✓ Saved: {output_file}")
    print(f"✓ Total samples: {len(df_sprint)}")
    print(f"✓ Sprint samples: {len(df_sprint[df_sprint['IsSprintRace'] == 1])}")
    print(f"✓ Main race samples: {len(df_sprint[df_sprint['IsSprintRace'] == 0])}")
    print(f"✓ Feature count: {len(available_features) - 3}")  # Exclude Position, Year, IsSprintRace
    print(f"✓ Years: {sorted(df_sprint['Year'].unique())}")
    
    print("\n💡 SPRINT MODEL NOTES:")
    print("   • Model trained with sprint-optimized features")
    print("   • Qualifying pace weighted 1.5x (vs 1.0x in main model)")
    print("   • Overtaking ability weighted 2.0x (vs 1.0x in main model)")
    print("   • First lap performance tracked separately")
    print("   • Will improve significantly after 2026 sprint races")
    
    print("\n" + "="*70)
    print("✅ Proceed to Sprint Model Training")
    print("   Run: python model\\train_sprint_model.py")
    print("="*70 + "\n")
    
    return df_sprint


if __name__ == "__main__":
    df = build_sprint_training_dataset()
    print("✅ Sprint features ready for model training!\n")