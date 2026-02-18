"""
Step 3: Integrate Winter Testing Data (February 2026)
Run this script AFTER winter testing is complete
"""

import os
import sys
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.fastf1_helpers import get_winter_testing_data
from config import CURRENT_SEASON, RAW_DATA_DIR, PROCESSED_DATA_DIR


def collect_winter_testing():
    """
    Collect winter testing data for 2026
    
    NOTE: This will only work AFTER February 2026 winter testing
    """
    print("\n" + "="*70)
    print("🏎️  COLLECTING 2026 WINTER TESTING DATA")
    print("="*70 + "\n")
    
    # Try to fetch winter testing data
    testing_data = get_winter_testing_data(CURRENT_SEASON)
    
    if testing_data is None:
        print("\n⚠️  Winter testing data not yet available")
        print("This is normal if we're before February 2026 testing")
        print("The model will work without this data (with reduced accuracy)")
        return None
    
    # Save raw testing data
    output_file = RAW_DATA_DIR / 'winter_testing_2026.csv'
    testing_data.to_csv(output_file, index=False)
    
    print(f"✓ Saved winter testing data to {output_file}")
    print(f"  Drivers tested: {len(testing_data)}")
    
    return testing_data


def process_winter_testing(testing_data):
    """
    Process winter testing data into features
    
    Parameters:
    -----------
    testing_data : pd.DataFrame
        Raw winter testing data
    
    Returns:
    --------
    pd.DataFrame : Processed testing features per driver
    """
    if testing_data is None:
        return None
    
    # Calculate pace ranking
    testing_data = testing_data.sort_values('BestLapTime').reset_index(drop=True)
    testing_data['TestingPaceRank'] = testing_data.index + 1
    
    # Normalize lap times (best = 1.0, worst = 0.0)
    best_time = testing_data['BestLapTime'].min()
    worst_time = testing_data['BestLapTime'].max()
    
    testing_data['NormalizedPace'] = 1 - (
        (testing_data['BestLapTime'] - best_time) / (worst_time - best_time)
    )
    
    # Reliability score (based on laps completed)
    max_laps = testing_data['LapsCompleted'].max()
    testing_data['ReliabilityScore'] = testing_data['LapsCompleted'] / max_laps
    
    # Create feature columns
    features = testing_data[[
        'Driver', 'Team', 'TestingPaceRank', 'NormalizedPace', 
        'ReliabilityScore', 'LapsCompleted'
    ]].copy()
    
    # Save processed features
    output_file = PROCESSED_DATA_DIR / 'winter_testing_features.csv'
    features.to_csv(output_file, index=False)
    
    print(f"✓ Processed winter testing features saved to {output_file}")
    
    return features


def create_placeholder_testing_data():
    """
    Create placeholder winter testing data for development
    Use this BEFORE actual testing data is available
    """
    print("\n" + "="*70)
    print("📝 CREATING PLACEHOLDER WINTER TESTING DATA")
    print("="*70 + "\n")
    
    # Load 2026 drivers
    drivers_file = RAW_DATA_DIR.parent / 'reference' / '2026_drivers.csv'
    
    if not drivers_file.exists():
        print("❌ 2026 drivers file not found. Run Step 2 first.")
        return None
    
    drivers = pd.read_csv(drivers_file)
    
    # Create placeholder data (neutral scores)
    placeholder = pd.DataFrame({
        'Driver': drivers['DriverCode'],
        'Team': drivers['Team'],
        'TestingPaceRank': range(1, len(drivers) + 1),  # Sequential ranking
        'NormalizedPace': 0.5,  # Neutral pace
        'ReliabilityScore': 1.0,  # Assume full reliability
        'LapsCompleted': 100  # Placeholder lap count
    })
    
    # Save placeholder
    output_file = PROCESSED_DATA_DIR / 'winter_testing_features.csv'
    placeholder.to_csv(output_file, index=False)
    
    print(f"✓ Created placeholder winter testing features: {output_file}")
    print("⚠️  Note: This is PLACEHOLDER data. Update after real testing!")
    
    return placeholder


def main():
    """
    Main function
    """
    # Try to collect real winter testing data
    testing_data = collect_winter_testing()
    
    if testing_data is not None:
        # Process real data
        features = process_winter_testing(testing_data)
    else:
        # Create placeholder for development
        features = create_placeholder_testing_data()
    
    print("\n" + "="*70)
    print("✅ Winter testing data integration complete!")
    print("="*70 + "\n")
    
    return features


if __name__ == "__main__":
    testing_features = main()
    print("✅ Proceed to Feature Engineering\n")