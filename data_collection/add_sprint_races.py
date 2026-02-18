"""
ADD SPRINT RACES TO EXISTING HISTORICAL DATA
This script collects ONLY sprint races from 2022-2025 and adds them to your existing data
"""

import os
import sys
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import YEARS_TO_COLLECT, RAW_DATA_DIR

try:
    import fastf1
    fastf1.Cache.enable_cache(str(RAW_DATA_DIR / '../cache'))
except ImportError:
    print("⚠️  FastF1 not installed. Run: pip install fastf1 --break-system-packages")
    sys.exit(1)


# Known sprint weekends by year
SPRINT_WEEKENDS = {
    2022: [
        {'round': 4, 'name': 'Emilia Romagna Grand Prix'},  # Imola
        {'round': 11, 'name': 'Austrian Grand Prix'},  # Austria
        {'round': 21, 'name': 'São Paulo Grand Prix'},  # Brazil
    ],
    2023: [
        {'round': 4, 'name': 'Azerbaijan Grand Prix'},  # Baku
        {'round': 11, 'name': 'Austrian Grand Prix'},  # Austria
        {'round': 12, 'name': 'Belgian Grand Prix'},  # Spa
        {'round': 15, 'name': 'Qatar Grand Prix'},  # Qatar
        {'round': 19, 'name': 'United States Grand Prix'},  # Austin
        {'round': 21, 'name': 'São Paulo Grand Prix'},  # Brazil
    ],
    2024: [
        {'round': 4, 'name': 'Chinese Grand Prix'},  # Shanghai
        {'round': 6, 'name': 'Miami Grand Prix'},  # Miami
        {'round': 11, 'name': 'Austrian Grand Prix'},  # Austria
        {'round': 19, 'name': 'United States Grand Prix'},  # Austin
        {'round': 21, 'name': 'São Paulo Grand Prix'},  # Brazil
        {'round': 23, 'name': 'Qatar Grand Prix'},  # Qatar
    ],
    2025: [
        {'round': 4, 'name': 'Chinese Grand Prix'},  # Shanghai
        {'round': 6, 'name': 'Miami Grand Prix'},  # Miami
        {'round': 9, 'name': 'Belgian Grand Prix'},  # Spa
        {'round': 13, 'name': 'British Grand Prix'},  # Silverstone
        {'round': 19, 'name': 'United States Grand Prix'},  # Austin
        {'round': 22, 'name': 'São Paulo Grand Prix'},  # Brazil
    ]
}


def collect_sprint_race(year, round_num, race_name):
    """Collect a single sprint race"""
    import time
    
    print(f"\n🏁 Round {round_num}: {race_name} (Sprint Weekend)")
    
    try:
        # Load event
        event = fastf1.get_event(year, round_num)
        
        # Try different sprint session names
        sprint_session = None
        for sprint_name in ['Sprint', 'Sprint Race', 'Sprint Shootout']:
            try:
                sprint_session = event.get_session(sprint_name)
                sprint_session.load()
                break
            except:
                continue
        
        if sprint_session is None:
            print(f"   ⚠️  Sprint session not found")
            return None
        
        if sprint_session.results is None or sprint_session.results.empty:
            print(f"   ⚠️  Sprint results empty")
            return None
        
        # Extract sprint data
        sprint_data = sprint_session.results.copy()
        
        sprint_df = pd.DataFrame({
            'Year': year,
            'RaceName': f"{race_name} - Sprint",  # Add " - Sprint" to distinguish
            'Round': round_num,
            'DriverNumber': sprint_data['DriverNumber'],
            'DriverCode': sprint_data['Abbreviation'],
            'DriverName': sprint_data['FullName'],
            'Team': sprint_data['TeamName'],
            'GridPosition': sprint_data['GridPosition'],
            'Position': sprint_data['Position'],
            'Points': sprint_data['Points'],
            'Status': sprint_data['Status']
        })
        
        print(f"   ✓ Sprint Race: {len(sprint_df)} drivers collected")
        time.sleep(2)  # Be nice to API
        return sprint_df
        
    except Exception as e:
        print(f"   ❌ Error: {str(e)[:100]}")
        return None


def main():
    """
    Collect sprint races and add to existing historical data
    """
    print("\n" + "="*70)
    print("🏁 ADDING SPRINT RACES TO HISTORICAL DATA")
    print("="*70)
    
    # Load existing data
    race_file = RAW_DATA_DIR / 'historical_race_results.csv'
    
    if not race_file.exists():
        print("\n❌ Error: historical_race_results.csv not found!")
        print("   Please run Step 1 data collection first")
        return
    
    print(f"\nLoading existing data from {race_file}...")
    existing_races = pd.read_csv(race_file)
    print(f"✓ Loaded {len(existing_races)} existing race results")
    
    # Check if sprints already exist
    existing_sprints = existing_races[existing_races['RaceName'].str.contains('Sprint', case=False, na=False)]
    print(f"  Current sprint results: {len(existing_sprints)}")
    
    if len(existing_sprints) > 0:
        print("\n⚠️  Warning: Sprint races already exist in data")
        response = input("   Do you want to re-collect them? (y/n): ")
        if response.lower() != 'y':
            print("   Cancelled.")
            return
        # Remove existing sprints
        existing_races = existing_races[~existing_races['RaceName'].str.contains('Sprint', case=False, na=False)]
        print(f"   Removed {len(existing_sprints)} existing sprint results")
    
    # Collect sprint races
    all_sprint_results = []
    
    for year in YEARS_TO_COLLECT:
        if year not in SPRINT_WEEKENDS:
            print(f"\n📅 {year}: No sprint weekends configured")
            continue
        
        print(f"\n{'='*70}")
        print(f"📅 COLLECTING {year} SPRINT RACES")
        print(f"{'='*70}")
        
        sprint_weekends = SPRINT_WEEKENDS[year]
        print(f"   Sprint weekends: {len(sprint_weekends)}")
        
        for sprint_info in sprint_weekends:
            sprint_df = collect_sprint_race(
                year, 
                sprint_info['round'], 
                sprint_info['name']
            )
            
            if sprint_df is not None:
                all_sprint_results.append(sprint_df)
    
    # Combine with existing data
    if all_sprint_results:
        print("\n" + "-"*70)
        print("Combining sprint races with existing data...")
        
        combined_sprints = pd.concat(all_sprint_results, ignore_index=True)
        print(f"✓ Collected {len(combined_sprints)} sprint race results")
        
        # Add to existing races
        updated_races = pd.concat([existing_races, combined_sprints], ignore_index=True)
        updated_races = updated_races.sort_values(['Year', 'Round']).reset_index(drop=True)
        
        # Save
        print(f"\nSaving updated data to {race_file}...")
        updated_races.to_csv(race_file, index=False)
        
        print("\n" + "="*70)
        print("✅ SPRINT RACES ADDED SUCCESSFULLY!")
        print("="*70)
        print(f"\nBEFORE:")
        print(f"  Total results: {len(existing_races)}")
        print(f"\nAFTER:")
        print(f"  Total results: {len(updated_races)}")
        print(f"  Main races: {len(updated_races[~updated_races['RaceName'].str.contains('Sprint', na=False)])}")
        print(f"  Sprint races: {len(updated_races[updated_races['RaceName'].str.contains('Sprint', na=False)])}")
        
        # Show breakdown by year
        print(f"\n📊 SPRINT RACES BY YEAR:")
        for year in YEARS_TO_COLLECT:
            year_sprints = updated_races[
                (updated_races['Year'] == year) & 
                (updated_races['RaceName'].str.contains('Sprint', na=False))
            ]
            if len(year_sprints) > 0:
                unique_sprints = year_sprints['RaceName'].nunique()
                print(f"  {year}: {unique_sprints} sprint races ({len(year_sprints)} results)")
        
        print("\n" + "="*70)
        print("✅ NEXT STEPS:")
        print("="*70)
        print("1. Rebuild features: python feature_engineering\\build_features.py")
        print("2. Rebuild sprint features: python feature_engineering\\build_sprint_features.py")
        print("3. Retrain main model: python model\\train_model.py")
        print("4. Train sprint model: python model\\train_sprint_model.py")
        print("="*70 + "\n")
        
    else:
        print("\n❌ No sprint races collected")
        print("   Check your internet connection and try again")


if __name__ == "__main__":
    main()
