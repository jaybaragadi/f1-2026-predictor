"""
Automatic Model Retraining After Each 2026 Race
IMPROVED: Handles FastF1 data availability delays
"""

import os
import sys
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import (RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, 
                   RACES_2026, AUTO_RETRAIN_ENABLED, CURRENT_SEASON)


def get_completed_races():
    """
    Check which 2026 races have been completed AND data is likely available
    """
    today = datetime.now()
    completed_races = []
    
    for race in RACES_2026:
        race_date = datetime.strptime(race['date'], '%Y-%m-%d')
        
        # Race is complete AND data should be available
        # FastF1 data typically available 60-120 min after race
        # Add 3 hours buffer to be safe
        data_available_time = race_date + timedelta(hours=3)
        
        if today >= data_available_time:
            completed_races.append(race)
    
    return completed_races


def check_if_new_race_data_available():
    """
    Check if there are new 2026 races with available data
    """
    training_file = PROCESSED_DATA_DIR / 'f1_training_dataset.csv'
    
    if not training_file.exists():
        print("⚠️  No existing training data found")
        return []
    
    training_data = pd.read_csv(training_file)
    
    # Get max year and round in training data
    if CURRENT_SEASON not in training_data['Year'].values:
        completed_races = get_completed_races()
        return completed_races
    
    # Get latest 2026 round in training data
    season_2026_data = training_data[training_data['Year'] == CURRENT_SEASON]
    max_round_in_data = season_2026_data['Round'].max() if len(season_2026_data) > 0 else 0
    
    # Check which completed races are not in data
    completed_races = get_completed_races()
    new_races = [race for race in completed_races if race['round'] > max_round_in_data]
    
    return new_races


def add_2026_race_results(round_number, retry_attempts=3):
    """
    Add results from a specific 2026 race to the historical data
    
    IMPROVED: Handles FastF1 timing delays with retries
    
    Parameters:
    -----------
    round_number : int
        Race round number (1-24)
    retry_attempts : int
        Number of times to retry if data not available (default: 3)
    """
    try:
        import fastf1
        fastf1.Cache.enable_cache('cache')
        
        print(f"\n📥 Fetching 2026 Round {round_number} data from FastF1...")
        
        # Try to load the race with retries
        for attempt in range(retry_attempts):
            try:
                race = fastf1.get_session(CURRENT_SEASON, round_number, 'R')
                race.load()
                
                # Success!
                break
            except Exception as e:
                if attempt < retry_attempts - 1:
                    print(f"⚠️  Attempt {attempt + 1} failed: {str(e)}")
                    print(f"   Retrying in 60 seconds...")
                    time.sleep(60)
                else:
                    # All attempts failed
                    raise e
        
        results = race.results
        
        if results is None or len(results) == 0:
            print(f"❌ No results available for Round {round_number}")
            print("\n💡 Possible reasons:")
            print("   • Race hasn't finished yet")
            print("   • Data not yet processed by F1 (wait 60-120 min after race)")
            print("   • Session was cancelled/postponed")
            return None
        
        # Extract race data
        race_data = pd.DataFrame({
            'Year': CURRENT_SEASON,
            'RaceName': race.event['EventName'],
            'Round': race.event['RoundNumber'],
            'DriverNumber': results.index.astype(str),
            'DriverCode': results['Abbreviation'],
            'DriverName': results['FullName'],
            'Team': results['TeamName'],
            'GridPosition': results['GridPosition'],
            'Position': results['Position'],
            'Points': results['Points'],
            'Status': results['Status'],
            'Time': results['Time'].astype(str),
        })
        
        # Clean data
        race_data['Position'] = pd.to_numeric(race_data['Position'], errors='coerce')
        race_data = race_data[race_data['Position'].notna()].copy()
        
        # Load existing race results
        race_file = RAW_DATA_DIR / 'historical_race_results.csv'
        
        if race_file.exists():
            existing_races = pd.read_csv(race_file)
            
            # Check if this race already exists
            existing_race = existing_races[
                (existing_races['Year'] == CURRENT_SEASON) &
                (existing_races['Round'] == round_number)
            ]
            
            if len(existing_race) > 0:
                print(f"⚠️  Round {round_number} already in database")
                print("   Use --force flag to overwrite")
                return None
            
            # Append new race
            updated_races = pd.concat([existing_races, race_data], ignore_index=True)
        else:
            updated_races = race_data
        
        # Save
        updated_races.to_csv(race_file, index=False)
        
        print(f"✅ Added Round {round_number} - {race.event['EventName']} to historical data")
        print(f"   Drivers finished: {len(race_data)}")
        print(f"   Winner: {race_data[race_data['Position']==1]['DriverName'].iloc[0]}")
        
        return race_data
        
    except Exception as e:
        print(f"❌ Error fetching race data: {e}")
        print("\n💡 Troubleshooting:")
        print("   • Wait 60-120 minutes after race finishes")
        print("   • Check FastF1 server status")
        print("   • Verify round number is correct")
        print("   • Or manually add results to historical_race_results.csv")
        return None


def trigger_full_pipeline():
    """Run the complete pipeline: Feature Engineering → Model Training"""
    import subprocess
    
    print("\n" + "="*70)
    print("🔄 TRIGGERING AUTOMATIC RETRAINING PIPELINE")
    print("="*70 + "\n")
    
    # Step 1: Feature Engineering
    print("Step 1: Running feature engineering...")
    try:
        result = subprocess.run(
            ['python', 'feature_engineering/build_features.py'],
            capture_output=True,
            text=True,
            check=True
        )
        print("✅ Feature engineering complete")
    except subprocess.CalledProcessError as e:
        print(f"❌ Feature engineering failed: {e}")
        return False
    
    # Step 2: Model Training
    print("\nStep 2: Training updated model...")
    try:
        result = subprocess.run(
            ['python', 'model/train_model.py'],
            capture_output=True,
            text=True,
            check=True
        )
        print("✅ Model training complete")
    except subprocess.CalledProcessError as e:
        print(f"❌ Model training failed: {e}")
        return False
    
    print("\n" + "="*70)
    print("✅ RETRAINING COMPLETE - Model updated with latest race data!")
    print("="*70)
    print("\n💡 Restart the Flask app to use the updated model:")
    print("   cd app")
    print("   python app.py")
    print("="*70 + "\n")
    
    return True


def estimate_data_availability(race_date_str):
    """
    Estimate when FastF1 data will be available for a race
    
    Parameters:
    -----------
    race_date_str : str
        Race date in 'YYYY-MM-DD' format
    
    Returns:
    --------
    tuple : (earliest_time, latest_time) as datetime objects
    """
    race_date = datetime.strptime(race_date_str, '%Y-%m-%d')
    
    # Assume race at 2 PM local time (rough average)
    race_start = race_date.replace(hour=14, minute=0)
    
    # Race duration ~2 hours
    race_end = race_start + timedelta(hours=2)
    
    # FastF1 data available 60-120 min after race
    earliest = race_end + timedelta(hours=1)
    latest = race_end + timedelta(hours=2)
    
    return earliest, latest


def main():
    """Main automatic retraining function with timing awareness"""
    
    print("\n" + "="*70)
    print("🤖 F1 2026 AUTOMATIC RETRAINING SYSTEM")
    print("="*70 + "\n")
    
    if not AUTO_RETRAIN_ENABLED:
        print("⚠️  Automatic retraining is DISABLED in config.py")
        print("   Set AUTO_RETRAIN_ENABLED = True to enable")
        return
    
    # Check for new races
    new_races = check_if_new_race_data_available()
    
    if not new_races:
        print("✅ No new race data available yet")
        completed = get_completed_races()
        
        if completed:
            print(f"\n  Races already in model: {len(completed)}")
            last_race = completed[-1]
            print(f"  Last race: {last_race['name']} ({last_race['date']})")
        
        # Check if any races happened recently
        today = datetime.now()
        for race in RACES_2026:
            race_date = datetime.strptime(race['date'], '%Y-%m-%d')
            time_since_race = today - race_date
            
            # Race happened in last 3 hours?
            if timedelta(hours=0) < time_since_race < timedelta(hours=3):
                earliest, latest = estimate_data_availability(race['date'])
                print(f"\n⏳ Recent race detected: {race['name']}")
                print(f"   Data likely available: {earliest.strftime('%H:%M')} - {latest.strftime('%H:%M')}")
                print(f"   Current time: {today.strftime('%H:%M')}")
                print("   Try again in ~1 hour")
        
        return
    
    print(f"📊 Found {len(new_races)} new race(s) to add:")
    for race in new_races:
        print(f"   • Round {race['round']}: {race['name']} ({race['date']})")
    
    # Try to fetch and add each new race
    successfully_added = []
    
    for race in new_races:
        print(f"\n{'='*70}")
        added = add_2026_race_results(race['round'])
        
        if added is not None:
            successfully_added.append(race)
        else:
            print(f"\n⚠️  Skipping Round {race['round']} - data not available")
    
    # Trigger retraining if we added any races
    if successfully_added:
        print("\n" + "-"*70)
        response = input(f"🔄 Added {len(successfully_added)} race(s). Retrain model now? (y/n): ")
        
        if response.lower() == 'y':
            success = trigger_full_pipeline()
            if success:
                print("\n🎉 Model successfully updated with 2026 season data!")
                print(f"   New races included: {', '.join([r['name'] for r in successfully_added])}")
        else:
            print("\n⏸️  Retraining skipped. Run this script again when ready.")
    else:
        print("\n⚠️  No new races were successfully added.")
        print("   Retraining not triggered.")


def manual_retrain():
    """Manually trigger retraining (bypasses checks)"""
    print("\n🔄 MANUAL RETRAINING MODE")
    print("This will retrain the model with all current data.\n")
    trigger_full_pipeline()


def force_add_race(round_number):
    """Force add a specific race (overwrites if exists)"""
    print(f"\n🔄 FORCE ADDING ROUND {round_number}")
    
    race_file = RAW_DATA_DIR / 'historical_race_results.csv'
    
    if race_file.exists():
        existing_races = pd.read_csv(race_file)
        
        # Remove existing entry for this race
        existing_races = existing_races[
            ~((existing_races['Year'] == CURRENT_SEASON) &
              (existing_races['Round'] == round_number))
        ]
        
        existing_races.to_csv(race_file, index=False)
        print(f"✓ Removed existing Round {round_number} data")
    
    # Add the race
    add_2026_race_results(round_number, retry_attempts=5)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='F1 2026 Automatic Retraining System')
    parser.add_argument('--manual', action='store_true', help='Force manual retrain')
    parser.add_argument('--check', action='store_true', help='Only check for new races')
    parser.add_argument('--force', type=int, metavar='ROUND', help='Force add specific round')
    parser.add_argument('--estimate', type=str, metavar='DATE', help='Estimate data availability for race date (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    if args.manual:
        manual_retrain()
    elif args.check:
        new_races = check_if_new_race_data_available()
        completed = get_completed_races()
        print(f"\nCompleted races: {len(completed)}")
        print(f"New races to add: {len(new_races)}")
        if new_races:
            for race in new_races:
                print(f"  • Round {race['round']}: {race['name']}")
    elif args.force:
        force_add_race(args.force)
    elif args.estimate:
        earliest, latest = estimate_data_availability(args.estimate)
        print(f"\nEstimated data availability for {args.estimate}:")
        print(f"  Earliest: {earliest.strftime('%Y-%m-%d %H:%M')}")
        print(f"  Latest: {latest.strftime('%Y-%m-%d %H:%M')}")
    else:
        main()