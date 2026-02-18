"""
Configuration for F1 2026 Race Predictor
CORRECTED VERSION - Accurate 2025 final grid + 2026 changes
Based on actual mid-season changes and team transitions
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
REFERENCE_DATA_DIR = DATA_DIR / 'reference'
MODEL_DIR = PROJECT_ROOT / 'model' / 'saved_models'

# Model settings
CURRENT_SEASON = 2026
YEARS_TO_COLLECT = [2023, 2024, 2025]

# Year weighting
YEAR_WEIGHTS = {
    2025: 2.5,
    2024: 1.5,
    2023: 0.5,
}

# Quick fix - add this line to your config.py after line 27

# Add this after YEAR_WEIGHTS:
MIN_RACES_FOR_AVG = 3
RETRAIN_AFTER_RACES = 1
SECRET_KEY = 'f1-2026-predictor'
HOST = '0.0.0.0'
PORT = 8080
CORS_ORIGINS = ['*']

# Model training
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Auto-retraining
AUTO_RETRAIN_ENABLED = True

# 2026 CALENDAR (24 races)
RACES_2026 = [
    {'round': 1, 'name': 'Australian Grand Prix', 'location': 'Melbourne', 'date': '2026-03-08'},
    {'round': 2, 'name': 'Chinese Grand Prix', 'location': 'Shanghai', 'date': '2026-03-15', 'format': 'Sprint'},
    {'round': 3, 'name': 'Japanese Grand Prix', 'location': 'Suzuka', 'date': '2026-03-29'},
    {'round': 4, 'name': 'Bahrain Grand Prix', 'location': 'Sakhir', 'date': '2026-04-12'},
    {'round': 5, 'name': 'Saudi Arabian Grand Prix', 'location': 'Jeddah', 'date': '2026-04-19'},
    {'round': 6, 'name': 'Miami Grand Prix', 'location': 'Miami', 'date': '2026-05-03', 'format': 'Sprint'},
    {'round': 7, 'name': 'Canadian Grand Prix', 'location': 'Montreal', 'date': '2026-05-24', 'format': 'Sprint'},
    {'round': 8, 'name': 'Monaco Grand Prix', 'location': 'Monte Carlo', 'date': '2026-06-07'},
    {'round': 9, 'name': 'Spanish Grand Prix', 'location': 'Barcelona', 'date': '2026-06-14'},
    {'round': 10, 'name': 'Austrian Grand Prix', 'location': 'Spielberg', 'date': '2026-06-28'},
    {'round': 11, 'name': 'British Grand Prix', 'location': 'Silverstone', 'date': '2026-07-05', 'format': 'Sprint'},
    {'round': 12, 'name': 'Belgian Grand Prix', 'location': 'Spa-Francorchamps', 'date': '2026-07-19'},
    {'round': 13, 'name': 'Hungarian Grand Prix', 'location': 'Budapest', 'date': '2026-07-26'},
    {'round': 14, 'name': 'Dutch Grand Prix', 'location': 'Zandvoort', 'date': '2026-08-23', 'format': 'Sprint'},
    {'round': 15, 'name': 'Italian Grand Prix', 'location': 'Monza', 'date': '2026-09-06'},
    {'round': 16, 'name': 'Madrid Grand Prix', 'location': 'Madrid', 'date': '2026-09-13'},
    {'round': 17, 'name': 'Azerbaijan Grand Prix', 'location': 'Baku', 'date': '2026-09-26'},
    {'round': 18, 'name': 'Singapore Grand Prix', 'location': 'Marina Bay', 'date': '2026-10-11', 'format': 'Sprint'},
    {'round': 19, 'name': 'United States Grand Prix', 'location': 'Austin', 'date': '2026-10-25'},
    {'round': 20, 'name': 'Mexico City Grand Prix', 'location': 'Mexico City', 'date': '2026-11-01'},
    {'round': 21, 'name': 'São Paulo Grand Prix', 'location': 'Interlagos', 'date': '2026-11-08'},
    {'round': 22, 'name': 'Las Vegas Grand Prix', 'location': 'Las Vegas', 'date': '2026-11-21'},
    {'round': 23, 'name': 'Qatar Grand Prix', 'location': 'Lusail', 'date': '2026-11-29'},
    {'round': 24, 'name': 'Abu Dhabi Grand Prix', 'location': 'Yas Marina', 'date': '2026-12-06'},
]

# 2026 TEAMS (11 teams)
TEAMS_2026 = [
    {'name': 'McLaren', 'full_name': 'McLaren Mastercard F1 Team', 'power_unit': 'Mercedes'},
    {'name': 'Red Bull Racing', 'full_name': 'Oracle Red Bull Racing', 'power_unit': 'Red Bull Ford'},
    {'name': 'Mercedes', 'full_name': 'Mercedes-AMG PETRONAS F1 Team', 'power_unit': 'Mercedes'},
    {'name': 'Ferrari', 'full_name': 'Scuderia Ferrari HP', 'power_unit': 'Ferrari'},
    {'name': 'Aston Martin', 'full_name': 'Aston Martin Aramco F1 Team', 'power_unit': 'Honda'},  # CHANGED from Mercedes
    {'name': 'Audi', 'full_name': 'Audi Revolut F1 Team', 'power_unit': 'Audi'},
    {'name': 'Cadillac', 'full_name': 'Cadillac Formula 1 Team', 'power_unit': 'Ferrari'},  # NEW team
    {'name': 'Alpine', 'full_name': 'BWT Alpine F1 Team', 'power_unit': 'Mercedes'},  # CHANGED from Renault
    {'name': 'Williams', 'full_name': 'Atlassian Williams F1 Team', 'power_unit': 'Mercedes'},
    {'name': 'RB', 'full_name': 'Visa Cash App RB', 'power_unit': 'Red Bull Ford'},
    {'name': 'Haas F1 Team', 'full_name': 'TGR Haas F1 Team', 'power_unit': 'Ferrari'},
]

# 2026 DRIVERS (22 drivers) - CORRECTED
DRIVERS_2026 = [
    # McLaren
    {'number': 1, 'code': 'NOR', 'name': 'Lando Norris', 'team': 'McLaren', 'championships': 1},  # 2025 Champion!
    {'number': 81, 'code': 'PIA', 'name': 'Oscar Piastri', 'team': 'McLaren', 'championships': 0},
    
    # Red Bull Racing
    {'number': 3, 'code': 'VER', 'name': 'Max Verstappen', 'team': 'Red Bull Racing', 'championships': 3},  # #3 not #1!
    {'number': 6, 'code': 'HAD', 'name': 'Isack Hadjar', 'team': 'Red Bull Racing', 'championships': 0, 'rookie_year': 2025},
    
    # Mercedes
    {'number': 63, 'code': 'RUS', 'name': 'George Russell', 'team': 'Mercedes', 'championships': 0},
    {'number': 12, 'code': 'ANT', 'name': 'Andrea Kimi Antonelli', 'team': 'Mercedes', 'championships': 0, 'rookie_year': 2025},
    
    # Ferrari
    {'number': 16, 'code': 'LEC', 'name': 'Charles Leclerc', 'team': 'Ferrari', 'championships': 0},
    {'number': 44, 'code': 'HAM', 'name': 'Lewis Hamilton', 'team': 'Ferrari', 'championships': 7},
    
    # Aston Martin
    {'number': 14, 'code': 'ALO', 'name': 'Fernando Alonso', 'team': 'Aston Martin', 'championships': 2},
    {'number': 18, 'code': 'STR', 'name': 'Lance Stroll', 'team': 'Aston Martin', 'championships': 0},
    
    # Audi
    {'number': 27, 'code': 'HUL', 'name': 'Nico Hulkenberg', 'team': 'Audi', 'championships': 0},
    {'number': 5, 'code': 'BOR', 'name': 'Gabriel Bortoleto', 'team': 'Audi', 'championships': 0, 'rookie_year': 2025},
    
    # Cadillac
    {'number': 11, 'code': 'PER', 'name': 'Sergio Perez', 'team': 'Cadillac', 'championships': 0},
    {'number': 77, 'code': 'BOT', 'name': 'Valtteri Bottas', 'team': 'Cadillac', 'championships': 0},
    
    # Alpine
    {'number': 10, 'code': 'GAS', 'name': 'Pierre Gasly', 'team': 'Alpine', 'championships': 0},
    {'number': 43, 'code': 'COL', 'name': 'Franco Colapinto', 'team': 'Alpine', 'championships': 0, 'rookie_year': 2025},
    
    # Williams
    {'number': 23, 'code': 'ALB', 'name': 'Alexander Albon', 'team': 'Williams', 'championships': 0},
    {'number': 55, 'code': 'SAI', 'name': 'Carlos Sainz', 'team': 'Williams', 'championships': 0},
    
    # RB
    {'number': 30, 'code': 'LAW', 'name': 'Liam Lawson', 'team': 'RB', 'championships': 0, 'rookie_year': 2025},
    {'number': 41, 'code': 'LIN', 'name': 'Arvid Lindblad', 'team': 'RB', 'championships': 0, 'rookie_year': 2026},  # ONLY 2026 rookie
    
    # Haas
    {'number': 31, 'code': 'OCO', 'name': 'Esteban Ocon', 'team': 'Haas F1 Team', 'championships': 0},
    {'number': 87, 'code': 'BEA', 'name': 'Oliver Bearman', 'team': 'Haas F1 Team', 'championships': 0, 'rookie_year': 2025},
]