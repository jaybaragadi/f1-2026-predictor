"""
Feature Engineering for F1 2026 Race Predictor - TRACK-SPECIFIC VERSION
Includes track-specific performance features for circuit-aware predictions
"""

import os
import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import (RAW_DATA_DIR, PROCESSED_DATA_DIR, REFERENCE_DATA_DIR,
                   YEAR_WEIGHTS, MIN_RACES_FOR_AVG)


def load_all_data():
    """Load all necessary data files"""
    
    print("Loading data files...")
    
    # Historical data
    races = pd.read_csv(RAW_DATA_DIR / 'historical_race_results.csv')
    quali = pd.read_csv(RAW_DATA_DIR / 'historical_quali_results.csv')
    
    # Reference data
    drivers_2026 = pd.read_csv(REFERENCE_DATA_DIR / '2026_drivers.csv')
    teams_2026 = pd.read_csv(REFERENCE_DATA_DIR / '2026_teams.csv')
    driver_stats = pd.read_csv(REFERENCE_DATA_DIR / 'driver_historical_stats.csv')
    
    # Winter testing (if available)
    testing_file = PROCESSED_DATA_DIR / 'winter_testing_features.csv'
    testing = pd.read_csv(testing_file) if testing_file.exists() else None
    
    print(f"✓ Loaded {len(races)} race results")
    print(f"✓ Loaded {len(quali)} qualifying results")
    print(f"✓ Loaded {len(drivers_2026)} 2026 drivers")
    print(f"✓ Loaded {len(teams_2026)} 2026 teams")
    
    # DEBUG: Check Position column
    print(f"\nDEBUG: Position column stats:")
    print(f"  Non-null positions: {races['Position'].notna().sum()}")
    print(f"  Null positions: {races['Position'].isna().sum()}")
    print(f"  Position data type: {races['Position'].dtype}")
    
    # FIX: Convert Position to numeric, coercing errors
    races['Position'] = pd.to_numeric(races['Position'], errors='coerce')
    
    # Filter out DNFs, DSQs (keep only valid finishing positions 1-20)
    races = races[races['Position'].notna()].copy()
    races = races[(races['Position'] >= 1) & (races['Position'] <= 20)].copy()
    
    print(f"  After filtering: {len(races)} valid race results")
    
    return races, quali, drivers_2026, teams_2026, driver_stats, testing


def merge_race_quali_data(races, quali):
    """Merge race and qualifying data"""
    
    print("\nMerging race and qualifying data...")
    
    # Merge on Year, Round, and DriverNumber
    merged = races.merge(
        quali[['Year', 'Round', 'DriverNumber', 'QualifyingPosition', 'Q1', 'Q2', 'Q3']],
        on=['Year', 'Round', 'DriverNumber'],
        how='left'
    )
    
    # Use GridPosition as fallback for missing QualifyingPosition
    merged['QualifyingPosition'] = merged['QualifyingPosition'].fillna(merged['GridPosition'])
    
    print(f"✓ Merged race and qualifying data: {len(merged)} records")
    
    return merged


def calculate_rolling_averages(df, group_cols, target_col, windows=[3, 5, 10]):
    """
    Calculate rolling averages for a target column
    """
    df = df.sort_values(['Year', 'Round'])
    
    for window in windows:
        col_name = f'{target_col}_Rolling_{window}'
        df[col_name] = df.groupby(group_cols)[target_col].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
    
    return df


def create_driver_features(merged_df, driver_stats):
    """
    Create driver-specific features
    """
    print("\nCreating driver features...")
    
    # Sort by date for rolling calculations
    merged_df = merged_df.sort_values(['Year', 'Round']).reset_index(drop=True)
    
    # Convert Position to float for calculations
    merged_df['Position'] = merged_df['Position'].astype(float)
    merged_df['GridPosition'] = pd.to_numeric(merged_df['GridPosition'], errors='coerce')
    merged_df['Points'] = pd.to_numeric(merged_df['Points'], errors='coerce').fillna(0)
    
    # Rolling averages for finish position
    merged_df = calculate_rolling_averages(
        merged_df, ['DriverName'], 'Position', windows=[3, 5, 10]
    )
    
    # Rolling averages for grid position
    merged_df = calculate_rolling_averages(
        merged_df, ['DriverName'], 'GridPosition', windows=[3, 5]
    )
    
    # Rolling averages for points
    merged_df = calculate_rolling_averages(
        merged_df, ['DriverName'], 'Points', windows=[3, 5]
    )
    
    # Positions gained/lost (GridPosition - Position)
    merged_df['PositionsGained'] = merged_df['GridPosition'] - merged_df['Position']
    merged_df = calculate_rolling_averages(
        merged_df, ['DriverName'], 'PositionsGained', windows=[3, 5]
    )
    
    # Recent form (last 3 races average position)
    merged_df['RecentForm'] = merged_df.groupby('DriverName')['Position'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    
    # Merge historical stats
    merged_df = merged_df.merge(
        driver_stats[['DriverName', 'AvgFinishPosition', 'AvgGridPosition', 
                     'TotalPoints', 'BestFinish']],
        on='DriverName',
        how='left'
    )
    
    print(f"✓ Created rolling averages and driver statistics")
    
    return merged_df


def create_team_features(merged_df, teams_2026):
    """
    Create team-specific features
    """
    print("\nCreating team features...")
    
    # Team performance by year
    team_year_perf = merged_df.groupby(['Year', 'Team']).agg({
        'Points': 'sum',
        'Position': 'mean'
    }).reset_index()
    team_year_perf.columns = ['Year', 'Team', 'TeamYearPoints', 'TeamYearAvgPosition']
    
    merged_df = merged_df.merge(team_year_perf, on=['Year', 'Team'], how='left')
    
    # Team average position in race
    merged_df['TeamRaceAvgPosition'] = merged_df.groupby(['Year', 'Round', 'Team'])['Position'].transform('mean')
    
    # Merge team info
    merged_df = merged_df.merge(
        teams_2026[['Team', 'ChampionshipsWon', 'YearsInF1']],
        on='Team',
        how='left'
    )
    
    # Fill missing values for teams not in 2026 grid
    merged_df['ChampionshipsWon'] = merged_df['ChampionshipsWon'].fillna(0)
    merged_df['YearsInF1'] = merged_df['YearsInF1'].fillna(merged_df['YearsInF1'].median())
    
    print(f"✓ Created team performance features")
    
    return merged_df


def create_race_features(merged_df):
    """
    Create race-specific features
    """
    print("\nCreating race features...")
    
    # Circuit-specific driver performance
    circuit_driver_perf = merged_df.groupby(['RaceName', 'DriverName']).agg({
        'Position': ['mean', 'min', 'count']
    }).reset_index()
    circuit_driver_perf.columns = ['RaceName', 'DriverName', 'CircuitAvgPosition', 
                                    'CircuitBestPosition', 'CircuitRacesCount']
    
    merged_df = merged_df.merge(circuit_driver_perf, on=['RaceName', 'DriverName'], how='left')
    
    # Fill NaN for first-time combinations
    merged_df['CircuitAvgPosition'] = merged_df['CircuitAvgPosition'].fillna(
        merged_df['AvgFinishPosition']
    )
    merged_df['CircuitBestPosition'] = merged_df['CircuitBestPosition'].fillna(20)
    merged_df['CircuitRacesCount'] = merged_df['CircuitRacesCount'].fillna(0)
    
    print(f"✓ Created circuit-specific features")
    
    return merged_df


def add_track_specific_features(df):
    """
    Calculate driver and team performance at each specific track
    THIS IS THE KEY FEATURE FOR TRACK-AWARE PREDICTIONS!
    """
    
    print("\n🏁 Calculating track-specific performance...")
    print("   This makes predictions different for each circuit!")
    
    # Initialize new columns
    df['DriverTrackAvg'] = 0.0
    df['DriverTrackBest'] = 20
    df['DriverTrackRaces'] = 0
    df['TeamTrackAvg'] = 0.0
    df['DriverTrackConsistency'] = 0.0
    
    # For each row, calculate track-specific stats
    total_rows = len(df)
    processed = 0
    
    for idx in df.index:
        # Progress indicator
        processed += 1
        if processed % 100 == 0 or processed == total_rows:
            print(f"   Processing: {processed}/{total_rows} ({processed/total_rows*100:.1f}%)", end='\r')
        
        driver_code = df.loc[idx, 'DriverCode']
        team = df.loc[idx, 'Team']
        track = df.loc[idx, 'RaceName']
        
        # Get driver's history at this track (only past races to avoid data leakage)
        driver_track_data = df[
            (df['DriverCode'] == driver_code) & 
            (df['RaceName'] == track) &
            (df.index < idx)  # Only use past races
        ]
        
        if len(driver_track_data) > 0:
            # Driver has raced here before
            df.loc[idx, 'DriverTrackAvg'] = driver_track_data['Position'].mean()
            df.loc[idx, 'DriverTrackBest'] = driver_track_data['Position'].min()
            df.loc[idx, 'DriverTrackRaces'] = len(driver_track_data)
            
            # Track consistency (lower std = more consistent)
            if len(driver_track_data) >= 3:
                df.loc[idx, 'DriverTrackConsistency'] = driver_track_data['Position'].std()
            else:
                df.loc[idx, 'DriverTrackConsistency'] = 5.0  # Default
        else:
            # No history at this track - use overall average
            if 'AvgFinishPosition' in df.columns:
                df.loc[idx, 'DriverTrackAvg'] = df.loc[idx, 'AvgFinishPosition']
            else:
                df.loc[idx, 'DriverTrackAvg'] = 10.0
            
            df.loc[idx, 'DriverTrackBest'] = 20
            df.loc[idx, 'DriverTrackRaces'] = 0
            df.loc[idx, 'DriverTrackConsistency'] = 5.0
        
        # Get team's history at this track
        team_track_data = df[
            (df['Team'] == team) & 
            (df['RaceName'] == track) &
            (df.index < idx)
        ]
        
        if len(team_track_data) > 0:
            df.loc[idx, 'TeamTrackAvg'] = team_track_data['Position'].mean()
        else:
            # Use overall team average
            if 'TeamYearAvgPosition' in df.columns:
                df.loc[idx, 'TeamTrackAvg'] = df.loc[idx, 'TeamYearAvgPosition']
            else:
                df.loc[idx, 'TeamTrackAvg'] = 10.0
    
    print()  # New line after progress
    print(f"✓ Added track-specific features:")
    print(f"  - DriverTrackAvg: Driver's average finish at this track")
    print(f"  - DriverTrackBest: Driver's best finish at this track")
    print(f"  - DriverTrackRaces: Number of races driver completed at this track")
    print(f"  - TeamTrackAvg: Team's average position at this track")
    print(f"  - DriverTrackConsistency: How consistent driver is at this track")
    
    # Show example
    sample_track = df['RaceName'].iloc[0]
    sample_count = (df['RaceName'] == sample_track).sum()
    print(f"\n  Example: {sample_track}")
    print(f"  Entries for this track: {sample_count}")
    
    return df


def apply_recency_weights(merged_df):
    """
    Apply year-based recency weighting
    More recent data gets higher weight
    """
    print("\nApplying recency weights...")
    
    # Add weight column based on year
    merged_df['RecencyWeight'] = merged_df['Year'].map(YEAR_WEIGHTS)
    merged_df['RecencyWeight'] = merged_df['RecencyWeight'].fillna(0.5)  # Default for older years
    
    print(f"✓ Applied recency weights")
    print(f"  2025 data weight: {YEAR_WEIGHTS.get(2025, 'N/A')}")
    print(f"  2024 data weight: {YEAR_WEIGHTS.get(2024, 'N/A')}")
    
    return merged_df


def clean_and_finalize(merged_df):
    """
    Clean data and prepare final dataset
    """
    print("\nCleaning and finalizing dataset...")
    
    # Check initial count
    initial_count = len(merged_df)
    print(f"  Starting with {initial_count} records")
    
    # Handle infinite values
    merged_df = merged_df.replace([np.inf, -np.inf], np.nan)
    
    # Fill remaining NaNs with appropriate values
    numeric_cols = merged_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if merged_df[col].isna().any():
            # Use median for most columns, 0 for Points
            fill_value = 0 if 'Points' in col else merged_df[col].median()
            merged_df[col] = merged_df[col].fillna(fill_value)
    
    # Verify we still have Position data
    merged_df = merged_df[merged_df['Position'].notna()].copy()
    merged_df['Position'] = merged_df['Position'].astype(int)
    
    print(f"  After cleaning: {len(merged_df)} records")
    
    if len(merged_df) == 0:
        print("\n❌ ERROR: No data remaining after cleaning!")
        print("This usually means the Position column is completely empty.")
        print("Please check the race data collection step.")
        return None, None
    
    # Create final feature set
    feature_columns = [
        # Basic info
        'GridPosition', 'QualifyingPosition',
        
        # Driver rolling stats
        'Position_Rolling_3', 'Position_Rolling_5', 'Position_Rolling_10',
        'GridPosition_Rolling_3', 'GridPosition_Rolling_5',
        'Points_Rolling_3', 'Points_Rolling_5',
        'PositionsGained_Rolling_3', 'PositionsGained_Rolling_5',
        'RecentForm',
        
        # Driver historical stats
        'AvgFinishPosition', 'AvgGridPosition', 'TotalPoints', 'BestFinish',
        
        # Team features
        'TeamYearPoints', 'TeamYearAvgPosition', 'TeamRaceAvgPosition',
        'ChampionshipsWon', 'YearsInF1',
        
        # Circuit features
        'CircuitAvgPosition', 'CircuitBestPosition', 'CircuitRacesCount',
        
        # NEW: Track-specific features
        'DriverTrackAvg', 'DriverTrackBest', 'DriverTrackRaces',
        'TeamTrackAvg', 'DriverTrackConsistency',
        
        # Recency weight
        'RecencyWeight'
    ]
    
    # Select only existing columns
    available_features = [col for col in feature_columns if col in merged_df.columns]
    
    final_df = merged_df[['Year', 'Round', 'RaceName', 'DriverName', 'DriverCode', 
                          'Team', 'Position'] + available_features].copy()
    
    print(f"✓ Final dataset shape: {final_df.shape}")
    print(f"✓ Features: {len(available_features)}")
    print(f"✓ Track-specific features added: 5")
    
    return final_df, available_features


def main():
    """
    Main feature engineering pipeline with track-specific features
    """
    print("\n" + "="*70)
    print("🏎️  FEATURE ENGINEERING - TRACK-SPECIFIC VERSION")
    print("="*70 + "\n")
    
    # Load data
    races, quali, drivers_2026, teams_2026, driver_stats, testing = load_all_data()
    
    if len(races) == 0:
        print("\n❌ ERROR: No race data found!")
        print("Please check the data collection step.")
        return None, None
    
    # Merge race and qualifying
    merged = merge_race_quali_data(races, quali)
    
    # Create features
    merged = create_driver_features(merged, driver_stats)
    merged = create_team_features(merged, teams_2026)
    merged = create_race_features(merged)
    merged = apply_recency_weights(merged)
    
    # ⭐ ADD TRACK-SPECIFIC FEATURES (NEW!)
    merged = add_track_specific_features(merged)
    
    # Clean and finalize
    final_df, feature_columns = clean_and_finalize(merged)
    
    if final_df is None or len(final_df) == 0:
        print("\n❌ FAILED: No training data created!")
        return None, None
    
    # Save final dataset
    output_file = PROCESSED_DATA_DIR / 'f1_training_dataset.csv'
    final_df.to_csv(output_file, index=False)
    
    # Save feature columns list
    features_file = PROCESSED_DATA_DIR / 'feature_columns.txt'
    with open(features_file, 'w') as f:
        f.write('\n'.join(feature_columns))
    
    print("\n" + "="*70)
    print("✅ FEATURE ENGINEERING COMPLETE!")
    print("="*70)
    print(f"✓ Saved training dataset: {output_file}")
    print(f"✓ Total samples: {len(final_df)}")
    print(f"✓ Feature count: {len(feature_columns)}")
    print(f"✓ Years: {sorted(final_df['Year'].unique())}")
    print(f"✓ Drivers: {final_df['DriverName'].nunique()}")
    print(f"✓ Tracks: {final_df['RaceName'].nunique()}")
    print("\n🏁 TRACK-SPECIFIC FEATURES ENABLED!")
    print("   Predictions will now vary by circuit!")
    print("="*70 + "\n")
    
    return final_df, feature_columns


if __name__ == "__main__":
    dataset, features = main()
    if dataset is not None and len(dataset) > 0:
        print("✅ Proceed to Model Training\n")
        print("Run: python model/train_model.py")
    else:
        print("❌ Cannot proceed - fix data collection first\n")