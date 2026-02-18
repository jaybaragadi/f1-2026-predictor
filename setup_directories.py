"""
Setup Script - Creates all necessary directories
Run this FIRST before any other scripts
"""

import os
from pathlib import Path

def create_directories():
    """Create all necessary project directories"""
    
    print("\n" + "="*70)
    print("🏎️  F1 2026 PREDICTOR - DIRECTORY SETUP")
    print("="*70 + "\n")
    
    # Get project root directory
    project_root = Path(__file__).parent
    
    # List of all directories to create
    directories = [
        'cache',                              # FastF1 cache
        'data/raw',                           # Raw data storage
        'data/processed',                     # Processed datasets
        'data/reference',                     # Reference data
        'model/saved_models',                 # Model artifacts
        'app/static/css',                     # CSS files
        'app/static/js',                      # JavaScript files
        'app/static/images/team_logos',       # Team logos
        'app/templates',                      # HTML templates
    ]
    
    created_count = 0
    existed_count = 0
    
    for directory in directories:
        dir_path = project_root / directory
        
        if dir_path.exists():
            print(f"✓ Already exists: {directory}")
            existed_count += 1
        else:
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created: {directory}")
            created_count += 1
    
    print("\n" + "="*70)
    print("SETUP SUMMARY")
    print("="*70)
    print(f"Directories created: {created_count}")
    print(f"Already existed: {existed_count}")
    print(f"Total directories: {created_count + existed_count}")
    print("="*70 + "\n")
    
    print("✅ Setup complete! You can now run the data collection scripts.")
    print("\nNext step:")
    print("  python data_collection/1_collect_historical_data.py")
    print("\n")


if __name__ == "__main__":
    create_directories()
