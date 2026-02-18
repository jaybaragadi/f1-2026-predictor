"""
Step 2: Create 2026 Driver and Team Reference Data
CORRECTED VERSION - Accurate 2026 lineup
"""

import os
import sys
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import REFERENCE_DATA_DIR, RAW_DATA_DIR, DRIVERS_2026, TEAMS_2026


def create_2026_drivers_reference():
    """
    Create 2026 driver reference file with CORRECTED lineup
    
    KEY CORRECTIONS:
    - Antonelli, Bortoleto, Hadjar, Colapinto, Bearman, Lawson = 2025 rookies (1 year exp in 2026)
    - Lindblad = ONLY 2026 rookie (0 years experience)
    - Hadjar promoted from RB to Red Bull Racing
    """
    
    # Convert config data to DataFrame
    df = pd.DataFrame(DRIVERS_2026)
    
    # Rename columns to match expected format
    df = df.rename(columns={
        'number': 'DriverNumber',
        'code': 'DriverCode',
        'name': 'DriverName',
        'team': 'Team',
        'experience': 'Experience',
        'championships': 'PreviousChampionships',
        'rookie_year': 'RookieYear'
    })
    
    # Add calculated fields
    df['IsRookie2026'] = (df['RookieYear'] == 2026).astype(int)  # Only Lindblad
    df['WasRookie2025'] = (df['RookieYear'] == 2025).astype(int)  # 6 drivers
    df['IsChampion'] = (df['PreviousChampionships'] > 0).astype(int)
    df['IsDefendingChampion'] = (df['DriverCode'] == 'NOR').astype(int)
    
    # Save
    output_file = REFERENCE_DATA_DIR / '2026_drivers.csv'
    df.to_csv(output_file, index=False)
    
    print(f"✓ Created 2026 driver reference: {output_file}")
    print(f"\n📊 DRIVER STATISTICS:")
    print(f"  Total Drivers: {len(df)} (11 teams)")
    print(f"  2026 Rookies: {df['IsRookie2026'].sum()} (Lindblad)")
    print(f"  2025 Rookies: {df['WasRookie2025'].sum()} (Antonelli, Bortoleto, Hadjar, Colapinto, Bearman, Lawson)")
    print(f"  Champions: {df['IsChampion'].sum()}")
    print(f"  Defending Champion: Lando Norris (#1)")
    
    print(f"\n🔄 KEY TEAM CHANGES:")
    print(f"  • Hamilton: Mercedes → Ferrari")
    print(f"  • Sainz: Ferrari → Williams")
    print(f"  • Hadjar: RB → Red Bull Racing (PROMOTED!)")
    print(f"  • Lindblad: NEW to RB (replaces Hadjar)")
    
    return df


def create_2026_teams_reference():
    """
    Create 2026 team reference file
    
    CORRECTED:
    - Audi = Renamed from Kick Sauber (NOT new team)
    - Cadillac = NEW 11th team
    """
    
    # Team championship history with CORRECTIONS
    team_data = [
        {'Team': 'McLaren', 'ChampionshipsWon': 8, 'YearsInF1': 60, 'TeamTier': 'Top', 'Notes': ''},
        {'Team': 'Red Bull Racing', 'ChampionshipsWon': 6, 'YearsInF1': 20, 'TeamTier': 'Top', 'Notes': ''},
        {'Team': 'Mercedes', 'ChampionshipsWon': 8, 'YearsInF1': 15, 'TeamTier': 'Top', 'Notes': ''},
        {'Team': 'Ferrari', 'ChampionshipsWon': 16, 'YearsInF1': 76, 'TeamTier': 'Top', 'Notes': ''},
        {'Team': 'Aston Martin', 'ChampionshipsWon': 0, 'YearsInF1': 5, 'TeamTier': 'Midfield', 'Notes': ''},
        {'Team': 'Audi', 'ChampionshipsWon': 0, 'YearsInF1': 30, 'TeamTier': 'Midfield', 'Notes': 'Renamed from Kick Sauber (same team)'},
        {'Team': 'Cadillac', 'ChampionshipsWon': 0, 'YearsInF1': 1, 'TeamTier': 'Midfield', 'Notes': 'NEW 11th team in 2026'},
        {'Team': 'Alpine', 'ChampionshipsWon': 2, 'YearsInF1': 7, 'TeamTier': 'Midfield', 'Notes': ''},
        {'Team': 'Williams', 'ChampionshipsWon': 9, 'YearsInF1': 48, 'TeamTier': 'Midfield', 'Notes': ''},
        {'Team': 'RB', 'ChampionshipsWon': 0, 'YearsInF1': 20, 'TeamTier': 'Back', 'Notes': ''},
        {'Team': 'Haas F1 Team', 'ChampionshipsWon': 0, 'YearsInF1': 10, 'TeamTier': 'Back', 'Notes': ''},
    ]
    
    df = pd.DataFrame(team_data)
    
    # Save
    output_file = REFERENCE_DATA_DIR / '2026_teams.csv'
    df.to_csv(output_file, index=False)
    
    print(f"\n✓ Created 2026 team reference: {output_file}")
    print(f"  Total Teams: {len(df)}")
    print(f"\n📝 TEAM NOTES:")
    print(f"  • Audi: Renamed from 'Kick Sauber' (2025)")
    print(f"    - Same team, same drivers (Hulkenberg + Bortoleto)")
    print(f"    - 30 years in F1 (Sauber legacy)")
    print(f"  • Cadillac: NEW 11th team (Perez + Bottas)")
    print(f"    - First year in F1")
    
    return df


def calculate_driver_stats_from_history():
    """
    Calculate driver performance statistics from historical data
    INCLUDES 2025 data for 2025 rookies
    """
    # Load historical data
    race_file = RAW_DATA_DIR / 'historical_race_results.csv'
    
    if not race_file.exists():
        print("⚠️  Historical race data not found. Run Step 1 first.")
        return None
    
    races = pd.read_csv(race_file)
    
    # Calculate per-driver statistics
    driver_stats = races.groupby('DriverName').agg({
        'Position': ['mean', 'median', 'min'],
        'GridPosition': ['mean'],
        'Points': ['sum', 'mean'],
        'RaceName': 'count'  # Number of races
    }).reset_index()
    
    driver_stats.columns = ['DriverName', 'AvgFinishPosition', 'MedianFinishPosition', 
                            'BestFinish', 'AvgGridPosition', 'TotalPoints', 
                            'AvgPointsPerRace', 'RacesCompleted']
    
    # Save
    output_file = REFERENCE_DATA_DIR / 'driver_historical_stats.csv'
    driver_stats.to_csv(output_file, index=False)
    
    print(f"\n✓ Created driver historical stats: {output_file}")
    print(f"  Drivers Analyzed: {len(driver_stats)}")
    print(f"\n📈 INCLUDES 2025 PERFORMANCE:")
    print(f"  • Antonelli, Bortoleto, Hadjar, Colapinto, Bearman, Lawson")
    print(f"  • These drivers have 2025 F1 statistics")
    print(f"  • Model will use their actual rookie season performance")
    
    return driver_stats


def main():
    """
    Main function to create all reference data
    """
    print("\n" + "="*70)
    print("🏎️  CREATING 2026 REFERENCE DATA (CORRECTED)")
    print("="*70)
    print("\n📋 2026 SEASON CORRECTIONS:")
    print("  ✅ Antonelli: 2025 rookie at Mercedes (1 year exp)")
    print("  ✅ Bortoleto: 2025 rookie at Sauber → now Audi (1 year exp)")
    print("  ✅ Hadjar: 2025 rookie at RB → PROMOTED to Red Bull (1 year exp)")
    print("  ✅ Colapinto: 2025 rookie at Alpine (1 year exp)")
    print("  ✅ Bearman: 2025 rookie at Haas (1 year exp)")
    print("  ✅ Lawson: 2025 rookie at RB (1 year exp)")
    print("  ✅ Lindblad: ONLY 2026 rookie at RB (0 years exp)")
    print("\n  ✅ Audi = Renamed from Kick Sauber (same team)")
    print("  ✅ Cadillac = NEW 11th team")
    print("="*70 + "\n")
    
    # Create reference files
    drivers_df = create_2026_drivers_reference()
    teams_df = create_2026_teams_reference()
    stats_df = calculate_driver_stats_from_history()
    
    print("\n" + "="*70)
    print("✅ Reference data creation complete!")
    print("="*70)
    print("\n💡 MODEL TRAINING NOTES:")
    print("  • 2025 rookies have 1 full season of data")
    print("  • Model will learn from their actual 2025 performance")
    print("  • Lindblad (2026 rookie) will use baseline features")
    print("  • Audi inherits Sauber's historical performance")
    print("="*70 + "\n")
    
    return drivers_df, teams_df, stats_df


if __name__ == "__main__":
    drivers, teams, stats = main()
    print("✅ Proceed to Step 3: Feature Engineering\n")