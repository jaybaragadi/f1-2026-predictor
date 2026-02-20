import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'app'))
from app import F1RacePredictor  # noqa: E402

def main():
    predictor = F1RacePredictor()
    if not predictor.load_model():
        print('SMOKE TEST FAILED: model failed to load')
        return 1
    
    drivers = predictor.get_driver_records()
    if len(drivers) != 22:
        print(f'SMOKE TEST FAILED: expected 22 drivers, got {len(drivers)}')
        return 1
    
    grid = {}
    for idx, d in enumerate(sorted(drivers, key=lambda x: int(x['number'])), start=1):
        grid[str(int(d['number']))] = idx
    
    race_info = {'name': 'Australian Grand Prix'}
    preds = predictor.predict(grid, race_info)
    if len(preds) != 22:
        print(f'SMOKE TEST FAILED: expected 22 predictions, got {len(preds)}')
        return 1
    
    positions = [p['position'] for p in preds]
    if positions != list(range(1, 23)):
        print('SMOKE TEST FAILED: output positions are not 1..22')
        return 1
    
    print('SMOKE TEST PASSED: model load + single prediction works')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())