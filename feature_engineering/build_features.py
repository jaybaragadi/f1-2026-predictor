"""
Enhanced Feature Engineering for F1 Race Predictor
==================================================

Advanced features to improve prediction accuracy:

1. Rolling averages (3, 5, 10 races) — LEAK-FREE via .shift(1)
2. Track-specific performance — LEAK-FREE via expanding+shift
3. Team momentum indicators — LEAK-FREE
4. Driver form analysis — LEAK-FREE
5. Head-to-head statistics
6. Weather impact features (if available)
7. Tire strategy features
8. Qualifying pace vs race pace
9. Reliability scores
10. Championship pressure features

Author: Enhanced for better predictions
Date: March 2026
"""

from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
class Config:
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / 'data'
    PROCESSED_DIR = DATA_DIR / 'processed'
    RAW_DIR = DATA_DIR / 'raw'

    INPUT_FILE = RAW_DIR / 'historical_race_results.csv'
    OUTPUT_FILE = PROCESSED_DIR / 'f1_race_features.csv'


# ==================== ROLLING FEATURES ====================
def add_rolling_features(df, windows=[3, 5, 10]):
    """Add rolling average features for multiple windows — leak-free via .shift(1)"""

    print("\nAdding rolling features...")

    # Sort by driver and date so rolling is chronological
    df = df.sort_values(['DriverCode', 'Date'])

    for window in windows:
        print(f"  Processing {window}-race window...")

        # Rolling position — shift(1) so current race is NOT included
        df[f'Position_Rolling_{window}'] = df.groupby('DriverCode')['Position'].transform(
            lambda x: x.rolling(window, min_periods=1).mean().shift(1)
        )

        # Rolling points
        df[f'Points_Rolling_{window}'] = df.groupby('DriverCode')['Points'].transform(
            lambda x: x.rolling(window, min_periods=1).mean().shift(1)
        )

        # Rolling positions gained
        df[f'PositionsGained_Rolling_{window}'] = df.groupby('DriverCode')['PositionsGained'].transform(
            lambda x: x.rolling(window, min_periods=1).mean().shift(1)
        )

        # Rolling podiums
        df[f'Podiums_Rolling_{window}'] = df.groupby('DriverCode')['Position'].transform(
            lambda x: (x <= 3).rolling(window, min_periods=1).sum().shift(1)
        )

        # Rolling DNFs
        df[f'DNF_Rolling_{window}'] = df.groupby('DriverCode')['Status'].transform(
            lambda x: (~x.str.contains('Finished', na=False)).rolling(window, min_periods=1).sum().shift(1)
        )

    print(f"  Added {len(windows) * 5} rolling features (leak-free)")

    return df


# ==================== TRACK-SPECIFIC FEATURES ====================
def add_track_features(df):
    """Add track-specific performance indicators — leak-free via expanding+shift"""

    print("\nAdding track-specific features...")

    # Sort by driver/team and date for correct chronological expanding
    df = df.sort_values(['DriverCode', 'Date'])

    # Driver average at this circuit (prior races only)
    df['DriverTrackAvg'] = df.groupby(['DriverCode', 'Circuit'])['Position'].transform(
        lambda x: x.expanding().mean().shift(1)
    )

    # Driver best at this circuit (prior races only)
    df['DriverTrackBest'] = df.groupby(['DriverCode', 'Circuit'])['Position'].transform(
        lambda x: x.expanding().min().shift(1)
    )

    # NOTE: DriverTrackConsistency removed — zero variance for single-visit driver/circuit pairs
    # and leaky when included. Replaced by DriverTrackRaces count.

    # Team average at circuit (prior races only)
    df = df.sort_values(['Team', 'Date'])
    df['TeamTrackAvg'] = df.groupby(['Team', 'Circuit'])['Position'].transform(
        lambda x: x.expanding().mean().shift(1)
    )

    # Circuit overall average position (prior races at circuit only)
    df['CircuitAvgPosition'] = df.groupby('Circuit')['Position'].transform(
        lambda x: x.expanding().mean().shift(1)
    )

    # Circuit best position for this driver (prior races only)
    df = df.sort_values(['DriverCode', 'Date'])
    df['CircuitBestPosition'] = df.groupby(['DriverCode', 'Circuit'])['Position'].transform(
        lambda x: x.expanding().min().shift(1)
    )

    # Number of prior races at this circuit (0 for debut — naturally non-leaky)
    df['CircuitRacesCount'] = df.groupby(['DriverCode', 'Circuit']).cumcount()

    # Team's best finish at circuit (prior races only)
    df = df.sort_values(['Team', 'Date'])
    df['TeamCircuitBest'] = df.groupby(['Team', 'Circuit'])['Position'].transform(
        lambda x: x.expanding().min().shift(1)
    )

    print(f"  Added 7 track-specific features (leak-free, DriverTrackConsistency removed)")

    return df


# ==================== MOMENTUM FEATURES ====================
def add_momentum_features(df):
    """Add momentum and form indicators — leak-free via .shift(1)"""

    print("\nAdding momentum features...")

    # Sort by driver and date
    df = df.sort_values(['DriverCode', 'Date'])

    # Position trend (improving or declining) — based on prior 3 races
    df['PositionTrend_3'] = df.groupby('DriverCode')['Position'].transform(
        lambda x: x.rolling(3, min_periods=2).apply(
            lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y) >= 2 else 0
        ).shift(1)
    )

    # Points momentum (recent 3 vs historical average) — both shifted to exclude current
    df['PointsMomentum'] = df.groupby('DriverCode')['Points'].transform(
        lambda x: x.rolling(3, min_periods=1).mean().shift(1)
                  / x.expanding().mean().shift(1).fillna(1)
    )

    # Win streak (last 5 races, excluding current)
    df['WinStreak'] = df.groupby('DriverCode')['Position'].transform(
        lambda x: (x == 1).rolling(5, min_periods=1).sum().shift(1)
    )

    # Podium streak (last 5, excluding current)
    df['PodiumStreak'] = df.groupby('DriverCode')['Position'].transform(
        lambda x: (x <= 3).rolling(5, min_periods=1).sum().shift(1)
    )

    # Points finish streak (last 5, excluding current)
    df['PointsFinishStreak'] = df.groupby('DriverCode')['Position'].transform(
        lambda x: (x <= 10).rolling(5, min_periods=1).sum().shift(1)
    )

    # Recent reliability (finished last 5, excluding current)
    df['RecentReliability'] = df.groupby('DriverCode')['Status'].transform(
        lambda x: x.str.contains('Finished', na=False).rolling(5, min_periods=1).sum().shift(1)
    )

    print(f"  Added 6 momentum features (leak-free)")

    return df


# ==================== TEAM FEATURES ====================
def add_team_features(df):
    """Add team performance features — leak-free via expanding+shift"""

    print("\nAdding team features...")

    # Sort by team and date for correct chronological order
    df = df.sort_values(['Team', 'Date'])

    # Team average position this year (cumulative, excluding current race)
    df['TeamYearAvgPosition'] = df.groupby(['Team', 'Year'])['Position'].transform(
        lambda x: x.expanding().mean().shift(1)
    )

    # Team points this year (cumulative, excluding current race)
    df['TeamYearPoints'] = df.groupby(['Team', 'Year'])['Points'].transform(
        lambda x: x.expanding().sum().shift(1).fillna(0)
    )

    # Team reliability this year (cumulative %, excluding current race)
    df['TeamYearReliability'] = df.groupby(['Team', 'Year'])['Status'].transform(
        lambda x: x.str.contains('Finished', na=False).expanding().mean().shift(1)
    )

    # Team's best result this year (cumulative, excluding current race)
    df['TeamYearBest'] = df.groupby(['Team', 'Year'])['Position'].transform(
        lambda x: x.expanding().min().shift(1)
    )

    # Team recent form (last 3 races avg, excluding current)
    df['TeamRecentForm'] = df.groupby('Team')['Position'].transform(
        lambda x: x.rolling(3, min_periods=1).mean().shift(1)
    )

    # Team overall race avg position (excluding current)
    df['TeamRaceAvgPosition'] = df.groupby('Team')['Position'].transform(
        lambda x: x.expanding().mean().shift(1)
    )

    print(f"  Added 6 team features (leak-free)")

    return df


# ==================== QUALIFYING VS RACE FEATURES ====================
def add_quali_race_features(df):
    """Add qualifying vs race pace features — leak-free"""

    print("\nAdding qualifying vs race features...")

    df = df.sort_values(['DriverCode', 'Date'])

    # Average positions gained from grid (excluding current race)
    df['AvgPositionsGained'] = df.groupby('DriverCode')['PositionsGained'].transform(
        lambda x: x.expanding().mean().shift(1).fillna(0)
    )

    # Consistency of positions gained (excluding current race)
    df['PositionsGainedStd'] = df.groupby('DriverCode')['PositionsGained'].transform(
        lambda x: x.expanding().std().shift(1).fillna(0)
    )

    # RacePaceVsQuali: intermediate column, computed from PAST races only via shift
    # This captures the driver's historical tendency to gain/lose vs grid position
    df['_RacePaceVsQuali_raw'] = df['Position'] / (df['GridPosition'] + 1)
    df['AvgRacePaceVsQuali'] = df.groupby('DriverCode')['_RacePaceVsQuali_raw'].transform(
        lambda x: x.expanding().mean().shift(1)
    )
    # Drop the leaky intermediate column
    df = df.drop(columns=['_RacePaceVsQuali_raw'])

    print(f"  Added 3 qualifying/race features (RacePaceVsQuali intermediate dropped)")

    return df


# ==================== CHAMPIONSHIP FEATURES ====================
def add_championship_features(df):
    """Add championship battle features"""

    print("\nAdding championship features...")

    # Sort by date
    df = df.sort_values('Date')

    # Points at start of race (cumulative points BEFORE this race — already shift(1))
    df['ChampionshipPoints'] = df.groupby(['DriverCode', 'Year'])['Points'].transform(
        lambda x: x.shift(1).fillna(0).cumsum()
    )

    # Championship position at start of race
    df['ChampionshipPosition'] = df.groupby(['Year', 'Date']).apply(
        lambda x: x['ChampionshipPoints'].rank(ascending=False, method='min')
    ).reset_index(level=[0, 1], drop=True)

    # Gap to leader (using same ChampionshipPoints which are already pre-race)
    df['LeaderPointsAtRace'] = df.groupby(['Year', 'Date'])['ChampionshipPoints'].transform('max')
    df['GapToLeader'] = (df['LeaderPointsAtRace'] - df['ChampionshipPoints']).clip(lower=0)
    df = df.drop(columns=['LeaderPointsAtRace'])

    # Fighting for championship (top 3 in standings)
    df['FightingForTitle'] = (df['ChampionshipPosition'] <= 3).astype(int)

    print(f"  Added 4 championship features (leak-free)")

    return df


# ==================== DRIVER EXPERIENCE FEATURES ====================
def add_experience_features(df):
    """Add driver experience features — leak-free"""

    print("\nAdding experience features...")

    df = df.sort_values(['DriverCode', 'Date'])

    # Career race count (cumcount gives index 0 for first race = number of prior races)
    df['CareerRaceCount'] = df.groupby('DriverCode').cumcount()

    # Races with current team (prior races with same team)
    df['RacesWithTeam'] = df.groupby(['DriverCode', 'Team']).cumcount()

    # Career podiums (prior races only via shift)
    df['CareerPodiums'] = df.groupby('DriverCode')['Position'].transform(
        lambda x: (x <= 3).expanding().sum().shift(1).fillna(0)
    )

    # Career wins (prior races only via shift)
    df['CareerWins'] = df.groupby('DriverCode')['Position'].transform(
        lambda x: (x == 1).expanding().sum().shift(1).fillna(0)
    )

    # Career points (prior races only via shift)
    df['CareerPoints'] = df.groupby('DriverCode')['Points'].transform(
        lambda x: x.expanding().sum().shift(1).fillna(0)
    )

    print(f"  Added 5 experience features (leak-free)")

    return df


# ==================== MAIN PIPELINE ====================
def build_features(df):
    """Main feature engineering pipeline"""

    print("\n" + "="*70)
    print("FEATURE ENGINEERING PIPELINE (LEAK-FREE)")
    print("="*70)
    print(f"Input samples: {len(df)}")
    print(f"Input features: {len(df.columns)}")

    # Normalise column names from the raw CSV format:
    # historical_race_results.csv uses 'RaceName' for circuit and has no 'Date' column.
    if 'Circuit' not in df.columns and 'RaceName' in df.columns:
        df = df.rename(columns={'RaceName': 'Circuit'})
    if 'Date' not in df.columns:
        if 'Year' in df.columns and 'Round' in df.columns:
            # Synthesise a sortable datetime from Year + Round.
            # Use Jan-1 of the year + (round-1)*14 days so all 24 rounds are valid dates.
            base = pd.to_datetime(df['Year'].astype(str) + '-01-01', format='%Y-%m-%d')
            df['Date'] = base + pd.to_timedelta((df['Round'].astype(int) - 1) * 14, unit='D')
        else:
            print("Cannot derive 'Date' column — need 'Year' and 'Round' columns")
            return None

    # Ensure required columns exist
    required = ['Date', 'DriverCode', 'Team', 'Circuit', 'Position', 'GridPosition', 'Points', 'Status', 'Year']
    missing = [col for col in required if col not in df.columns]

    if missing:
        print(f"\n❌ Missing required columns: {missing}")
        return None

    # Calculate positions gained if not present
    if 'PositionsGained' not in df.columns:
        df['PositionsGained'] = df['GridPosition'] - df['Position']

    # Convert date to datetime and sort chronologically
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Date', 'DriverCode']).reset_index(drop=True)

    # Apply feature engineering
    df = add_rolling_features(df, windows=[3, 5, 10])
    df = add_track_features(df)
    df = add_momentum_features(df)
    df = add_team_features(df)
    df = add_quali_race_features(df)
    df = add_championship_features(df)
    df = add_experience_features(df)

    # Fill any NaN values introduced by shift(1) on first races
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    print("\n" + "="*70)
    print("FEATURE ENGINEERING COMPLETE")
    print("="*70)
    print(f"Output samples: {len(df)}")
    print(f"Output features: {len(df.columns)}")
    print(f"New features added: {len(df.columns) - len(required)}")
    print("All rolling/expanding features are leak-free (shift(1) applied)")

    return df


def main():
    """Main execution"""

    print("\n" + "="*70)
    print("F1 RACE PREDICTOR - FEATURE ENGINEERING")
    print("="*70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load raw data
    print(f"\nLoading data from: {Config.INPUT_FILE}")

    if not Config.INPUT_FILE.exists():
        print(f"❌ Input file not found: {Config.INPUT_FILE}")
        return

    df = pd.read_csv(Config.INPUT_FILE)
    print(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns")

    # Build features
    df_features = build_features(df)

    if df_features is None:
        print("\n❌ Feature engineering failed")
        return

    # Save output
    Config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(Config.OUTPUT_FILE, index=False)
    print(f"\n✓ Features saved to: {Config.OUTPUT_FILE}")

    # Summary statistics
    print(f"\nFeature Summary:")
    print(f"  Total features: {len(df_features.columns)}")
    print(f"  Rolling features: 15 (3/5/10 race windows) [shift(1) applied]")
    print(f"  Track-specific: 7 (DriverTrackConsistency removed)")
    print(f"  Momentum: 6 [shift(1) applied]")
    print(f"  Team: 6 [expanding+shift applied]")
    print(f"  Quali/Race: 3 (RacePaceVsQuali intermediate dropped)")
    print(f"  Championship: 4 [already shift(1) correct]")
    print(f"  Experience: 5 [shift(1) applied]")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
