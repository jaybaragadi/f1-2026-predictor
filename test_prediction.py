import sys
sys.path.insert(0, 'app')
from app import F1RacePredictor

print("Testing predictor...")

predictor = F1RacePredictor()
success = predictor.load_model()

if success:
    print("✓ Model loaded")
    
    # Test prediction
    grid = {
        '1': 1,   # Norris
        '3': 2,   # Verstappen
        '44': 3,  # Hamilton
        '16': 4,  # Leclerc
        '63': 5,  # Russell
        '81': 6,  # Piastri
        '14': 7,  # Alonso
        '55': 8,  # Sainz
        '23': 9,  # Albon
        '18': 10, # Stroll
        '10': 11, # Gasly
        '27': 12, # Hulkenberg
        '31': 13, # Ocon
        '6': 14,  # Hadjar
        '87': 15, # Bearman
        '12': 16, # Antonelli
        '43': 17, # Colapinto
        '5': 18,  # Bortoleto
        '30': 19, # Lawson
        '11': 20, # Perez
        '77': 21, # Bottas
        '41': 22, # Lindblad
    }
    
    race_info = {'name': 'Australian Grand Prix'}
    
    try:
        predictions = predictor.predict(grid, race_info)
        print(f"✓ Predictions generated: {len(predictions)} drivers")
        print(f"\nTop 3:")
        for p in predictions[:3]:
            print(f"  {p['position']}. {p['driver_name']} (Grid P{p['grid_position']})")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
else:
    print("❌ Model failed to load")
