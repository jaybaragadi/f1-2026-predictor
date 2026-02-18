import pandas as pd
from pathlib import Path

print("\n🔍 DIAGNOSTIC CHECK\n")

# Check training data
training_file = Path("data/processed/f1_training_dataset.csv")

if training_file.exists():
    df = pd.read_csv(training_file)
    print(f"✓ Training data found: {len(df)} rows")
    print(f"\n📊 Columns ({len(df.columns)}):")
    for col in df.columns:
        print(f"  - {col}")
    
    # Check for required columns
    required = ["DriverCode", "Position", "Team", "RaceName"]
    print(f"\n✅ Required columns check:")
    for col in required:
        if col in df.columns:
            print(f"  ✓ {col}")
        else:
            print(f"  ❌ {col} MISSING!")
    
    # Check sample data
    if "DriverCode" in df.columns:
        print(f"\n📝 Sample DriverCodes:")
        print(f"  {df['DriverCode'].unique()[:10]}")
else:
    print("❌ Training data NOT FOUND!")
    print(f"   Expected: {training_file}")

# Check reference data
ref_file = Path("data/reference/2026_drivers.csv")
if ref_file.exists():
    drivers = pd.read_csv(ref_file)
    print(f"\n✓ 2026 drivers found: {len(drivers)} drivers")
    print(f"  Sample codes: {drivers['DriverCode'].head().tolist()}")
else:
    print("\n❌ 2026 drivers NOT FOUND!")