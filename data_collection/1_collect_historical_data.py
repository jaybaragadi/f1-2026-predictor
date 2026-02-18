"""
Step 1: Collect Historical F1 Data (2022-2025)
This script collects race and qualifying results from FastF1
"""

import os
import sys
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.fastf1_helpers import collect_full_season, get_driver_standings
from config import YEARS_TO_COLLECT, RAW_DATA_DIR


def main():
    """
    Main function to collect all historical data
    """
    print("\n" + "="*70)
    print("🏎️  F1 2026 PREDICTOR - DATA COLLECTION")
    print("="*70)
    
    all_race_results = []
    all_quali_results = []
    all_standings = []
    
    # Collect data for each year
    for year in YEARS_TO_COLLECT:
        # Collect race and qualifying data
        race_df, quali_df = collect_full_season(year, delay=3)
        
        if race_df is not None and not race_df.empty:
            all_race_results.append(race_df)
        
        if quali_df is not None and not quali_df.empty:
            all_quali_results.append(quali_df)
        
        # Collect final standings
        standings = get_driver_standings(year)
        if standings is not None:
            all_standings.append(standings)
    
    # Combine all years
    print("\n" + "-"*70)
    print("Combining data from all years...")
    
    combined_races = pd.concat(all_race_results, ignore_index=True)
    combined_quali = pd.concat(all_quali_results, ignore_index=True)
    combined_standings = pd.concat(all_standings, ignore_index=True)
    
    # Save to CSV
    race_file = RAW_DATA_DIR / 'historical_race_results.csv'
    quali_file = RAW_DATA_DIR / 'historical_quali_results.csv'
    standings_file = RAW_DATA_DIR / 'historical_standings.csv'
    
    combined_races.to_csv(race_file, index=False)
    combined_quali.to_csv(quali_file, index=False)
    combined_standings.to_csv(standings_file, index=False)
    
    print(f"\n✓ Saved {len(combined_races)} race results to {race_file}")
    print(f"✓ Saved {len(combined_quali)} qualifying results to {quali_file}")
    print(f"✓ Saved {len(combined_standings)} standings records to {standings_file}")
    
    # Display summary
    print("\n" + "="*70)
    print("DATA COLLECTION SUMMARY")
    print("="*70)
    print(f"Total Races Collected: {combined_races['RaceName'].nunique()}")
    print(f"Years Covered: {', '.join(map(str, YEARS_TO_COLLECT))}")
    print(f"Unique Drivers: {combined_races['DriverName'].nunique()}")
    print(f"Unique Teams: {combined_races['Team'].nunique()}")
    print("="*70 + "\n")
    
    return combined_races, combined_quali, combined_standings


if __name__ == "__main__":
    race_data, quali_data, standings_data = main()
    print("\n✅ Data collection complete! Proceed to Step 2: Driver Information Collection\n")