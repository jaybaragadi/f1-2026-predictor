"""
Default Grid Manager for F1 Race Predictor
==========================================

Automatically fetch and set default grids from:
1. Qualifying results (preferred for current race)
2. Previous race results (if qualifying not available)
3. Championship standings (fallback)

This allows quick predictions without manual grid entry.

Usage:
    python grid_manager.py --race-number 2 --from-quali
    python grid_manager.py --race-number 2 --from-prev-race
    python grid_manager.py --auto  # Auto-detect best source

Author: Enhanced for 2026 season
Date: March 2026
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
import fastf1
from fastf1.ergast import Ergast

# ==================== CONFIGURATION ====================
class Config:
    PROJECT_ROOT = Path(__file__).parent
    DATA_DIR = PROJECT_ROOT / 'data'
    RAW_DIR = DATA_DIR / 'raw' / '2026'
    REFERENCE_DIR = DATA_DIR / 'reference'
    
    DEFAULT_GRID_FILE = DATA_DIR / 'default_grid.json'
    
    SEASON = 2026
    
    # Driver numbers (2026 grid) — must match official FIA/FastF1 numbers
    # Cross-referenced against config.py DRIVERS_2026
    DRIVER_NUMBERS = {
        1:  'NOR',  # Lando Norris (2025 Champion)
        3:  'VER',  # Max Verstappen
        5:  'BOR',  # Gabriel Bortoleto (Audi)
        6:  'HAD',  # Isack Hadjar (Red Bull)
        10: 'GAS',  # Pierre Gasly (Alpine)
        11: 'PER',  # Sergio Perez (Cadillac)
        12: 'ANT',  # Andrea Kimi Antonelli (Mercedes)
        14: 'ALO',  # Fernando Alonso (Aston Martin)
        16: 'LEC',  # Charles Leclerc (Ferrari)
        18: 'STR',  # Lance Stroll (Aston Martin)
        23: 'ALB',  # Alexander Albon (Williams)
        27: 'HUL',  # Nico Hulkenberg (Audi)
        30: 'LAW',  # Liam Lawson (RB)
        31: 'OCO',  # Esteban Ocon (Haas)
        41: 'LIN',  # Arvid Lindblad (RB) — 2026 rookie
        43: 'COL',  # Franco Colapinto (Alpine)
        44: 'HAM',  # Lewis Hamilton (Ferrari)
        55: 'SAI',  # Carlos Sainz (Williams)
        63: 'RUS',  # George Russell (Mercedes)
        77: 'BOT',  # Valtteri Bottas (Cadillac)
        81: 'PIA',  # Oscar Piastri (McLaren)
        87: 'BEA',  # Oliver Bearman (Haas)
    }


# ==================== DATA FETCHING ====================
def fetch_qualifying_grid(race_number):
    """Fetch grid positions from qualifying session"""
    
    print(f"\n{'='*70}")
    print(f"FETCHING QUALIFYING GRID - RACE {race_number}")
    print(f"{'='*70}")
    
    try:
        # Enable cache
        cache_dir = Config.RAW_DIR / 'cache'
        cache_dir.mkdir(parents=True, exist_ok=True)
        fastf1.Cache.enable_cache(str(cache_dir))
        
        # Load qualifying session
        print(f"Loading qualifying session for Race {race_number}...")
        session = fastf1.get_session(Config.SEASON, race_number, 'Q')
        session.load()
        
        results = session.results
        
        grid = {
            'source': 'qualifying',
            'race_number': race_number,
            'race_name': session.event['EventName'],
            'circuit': session.event['Location'],
            'date': str(session.event['EventDate']),
            'positions': {}
        }
        
        print(f"✓ Race: {grid['race_name']}")
        print(f"✓ Circuit: {grid['circuit']}")
        print(f"\nGrid Positions from Qualifying:")
        print(f"{'-'*70}")
        
        for idx, row in results.iterrows():
            driver_code = row['Abbreviation']
            position = int(row['Position']) if pd.notna(row['Position']) else 99
            
            grid['positions'][driver_code] = position
            
            if position <= 10:
                print(f"  P{position:2d}  {driver_code:3s}  ({row['TeamName']})")
        
        print(f"{'-'*70}")
        print(f"✓ Grid positions extracted: {len(grid['positions'])} drivers")
        
        return grid
        
    except Exception as e:
        print(f"❌ Error fetching qualifying: {e}")
        return None


def fetch_previous_race_grid(race_number):
    """Fetch grid from previous race finishing positions"""
    
    if race_number <= 1:
        print("⚠ No previous race available for Race 1")
        return None
    
    prev_race = race_number - 1
    
    print(f"\n{'='*70}")
    print(f"FETCHING PREVIOUS RACE RESULTS - RACE {prev_race}")
    print(f"{'='*70}")
    
    try:
        cache_dir = Config.RAW_DIR / 'cache'
        cache_dir.mkdir(parents=True, exist_ok=True)
        fastf1.Cache.enable_cache(str(cache_dir))
        
        # Load previous race
        print(f"Loading Race {prev_race} results...")
        session = fastf1.get_session(Config.SEASON, prev_race, 'R')
        session.load()
        
        results = session.results
        
        grid = {
            'source': 'previous_race',
            'race_number': race_number,
            'source_race': prev_race,
            'race_name': session.event['EventName'],
            'positions': {}
        }
        
        print(f"✓ Using results from: {grid['race_name']}")
        print(f"\nGrid Positions from Previous Race:")
        print(f"{'-'*70}")
        
        for idx, row in results.iterrows():
            driver_code = row['Abbreviation']
            position = int(row['Position']) if pd.notna(row['Position']) else 99
            
            grid['positions'][driver_code] = position
            
            if position <= 10:
                print(f"  P{position:2d}  {driver_code:3s}  ({row['TeamName']})")
        
        print(f"{'-'*70}")
        print(f"✓ Grid positions extracted: {len(grid['positions'])} drivers")
        
        return grid
        
    except Exception as e:
        print(f"❌ Error fetching previous race: {e}")
        return None


def fetch_championship_grid():
    """Fetch grid from championship standings (fallback)"""
    
    print(f"\n{'='*70}")
    print("FETCHING CHAMPIONSHIP STANDINGS")
    print(f"{'='*70}")
    
    try:
        ergast = Ergast()
        standings = ergast.get_driver_standings(Config.SEASON)
        
        grid = {
            'source': 'championship',
            'season': Config.SEASON,
            'positions': {}
        }
        
        print(f"\nGrid Positions from Championship Order:")
        print(f"{'-'*70}")
        
        for idx, driver in standings.content[0].iterrows():
            driver_code = driver['Driver']['code']
            position = int(driver['position'])
            
            grid['positions'][driver_code] = position
            
            if position <= 10:
                print(f"  P{position:2d}  {driver_code:3s}  ({driver['points']} pts)")
        
        print(f"{'-'*70}")
        print(f"✓ Grid positions extracted: {len(grid['positions'])} drivers")
        
        return grid
        
    except Exception as e:
        print(f"❌ Error fetching championship: {e}")
        return None


def create_fallback_grid():
    """Create fallback grid based on 2025 season end standings"""
    
    print(f"\n{'='*70}")
    print("CREATING FALLBACK GRID (2025 STANDINGS)")
    print(f"{'='*70}")
    
    # 2025 season end approximate standings (adjust as needed)
    fallback_order = [
        'NOR', 'VER', 'LEC', 'PIA', 'HAM', 'RUS', 'SAI', 'PER',
        'ALO', 'GAS', 'HUL', 'STR', 'OCO', 'ALB', 'BOT', 'LAW',
        'HAD', 'BEA', 'COL', 'BOR', 'ANT', 'LIN'
    ]
    
    grid = {
        'source': 'fallback_2025',
        'positions': {}
    }
    
    for position, driver_code in enumerate(fallback_order, start=1):
        grid['positions'][driver_code] = position
    
    print(f"✓ Fallback grid created: {len(grid['positions'])} drivers")
    
    return grid


# ==================== GRID MANAGEMENT ====================
def save_default_grid(grid):
    """Save grid as default for predictions"""
    
    print(f"\n{'='*70}")
    print("SAVING DEFAULT GRID")
    print(f"{'='*70}")
    
    # Ensure data directory exists
    Config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Add metadata
    grid['updated_at'] = datetime.now().isoformat()
    
    # Save to file
    with open(Config.DEFAULT_GRID_FILE, 'w') as f:
        json.dump(grid, f, indent=2)
    
    print(f"✓ Default grid saved: {Config.DEFAULT_GRID_FILE}")
    print(f"✓ Source: {grid['source']}")
    print(f"✓ Drivers: {len(grid['positions'])}")
    
    # Also save to app data directory for Flask
    app_data_dir = Config.PROJECT_ROOT / 'app' / 'data'
    if app_data_dir.exists():
        app_grid_file = app_data_dir / 'default_grid.json'
        with open(app_grid_file, 'w') as f:
            json.dump(grid, f, indent=2)
        print(f"✓ Also saved to app: {app_grid_file}")


def load_default_grid():
    """Load saved default grid"""
    
    if not Config.DEFAULT_GRID_FILE.exists():
        print("⚠ No default grid file found")
        return None
    
    with open(Config.DEFAULT_GRID_FILE, 'r') as f:
        grid = json.load(f)
    
    print(f"\nLoaded Default Grid:")
    print(f"  Source: {grid['source']}")
    print(f"  Updated: {grid.get('updated_at', 'Unknown')}")
    print(f"  Drivers: {len(grid['positions'])}")
    
    return grid


def auto_fetch_grid(race_number):
    """Automatically fetch best available grid"""
    
    print(f"\n{'='*70}")
    print(f"AUTO-DETECTING BEST GRID SOURCE - RACE {race_number}")
    print(f"{'='*70}")
    
    # Try qualifying first (most accurate)
    print("\n1. Trying qualifying results...")
    grid = fetch_qualifying_grid(race_number)
    
    if grid:
        print("✓ Using qualifying grid")
        return grid
    
    # Try previous race
    print("\n2. Trying previous race results...")
    grid = fetch_previous_race_grid(race_number)
    
    if grid:
        print("✓ Using previous race grid")
        return grid
    
    # Try championship standings
    print("\n3. Trying championship standings...")
    grid = fetch_championship_grid()
    
    if grid:
        print("✓ Using championship grid")
        return grid
    
    # Last resort: fallback
    print("\n4. Using fallback grid...")
    grid = create_fallback_grid()
    
    return grid


# ==================== MAIN ====================
def main():
    parser = argparse.ArgumentParser(description='Manage default grid positions')
    parser.add_argument('--race-number', type=int, help='Race number for grid')
    parser.add_argument('--from-quali', action='store_true', help='Use qualifying results')
    parser.add_argument('--from-prev-race', action='store_true', help='Use previous race results')
    parser.add_argument('--from-championship', action='store_true', help='Use championship order')
    parser.add_argument('--auto', action='store_true', help='Auto-detect best source')
    parser.add_argument('--show', action='store_true', help='Show current default grid')
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print("F1 RACE PREDICTOR - GRID MANAGER")
    print(f"{'='*70}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Show current grid
    if args.show:
        load_default_grid()
        return
    
    # Require race number for fetching
    if not args.race_number and not args.from_championship:
        print("\n❌ Please specify --race-number or use --show")
        return
    
    # Fetch grid based on source
    grid = None
    
    if args.from_quali:
        grid = fetch_qualifying_grid(args.race_number)
    elif args.from_prev_race:
        grid = fetch_previous_race_grid(args.race_number)
    elif args.from_championship:
        grid = fetch_championship_grid()
    elif args.auto:
        grid = auto_fetch_grid(args.race_number)
    else:
        print("\n❌ Please specify grid source (--from-quali, --from-prev-race, --auto)")
        return
    
    if grid is None:
        print("\n❌ Failed to fetch grid")
        return
    
    # Save as default
    save_default_grid(grid)
    
    print(f"\n{'='*70}")
    print("GRID MANAGER COMPLETE")
    print(f"{'='*70}")
    print("✓ Default grid updated")
    print("✓ Ready for predictions")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
