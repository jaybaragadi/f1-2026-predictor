"""
Production Server for Windows Testing
Use waitress instead of gunicorn on Windows
FIXED: Handles paths correctly when running from app/ directory
"""

import os
import sys
from pathlib import Path

# Fix paths - go up one level to project root
current_dir = Path(__file__).parent
project_root = current_dir.parent

# Add project root to Python path
sys.path.insert(0, str(project_root))

# Change working directory to project root
os.chdir(project_root)

print("\n" + "="*70)
print("🏎️  F1 2026 Race Predictor - Production Mode (Windows)")
print("="*70)
print(f"\n📁 Working directory: {os.getcwd()}")
print(f"📁 Project root: {project_root}")

# Now import app (after paths are fixed)
from app.app import app

if __name__ == '__main__':
    from waitress import serve
    
    print("\n✓ Starting production server with Waitress...")
    print("✓ Server will run on: http://localhost:5000")
    print("✓ Press Ctrl+C to stop\n")
    print("="*70 + "\n")
    
    # Serve with waitress (production-ready for Windows)
    serve(app, host='0.0.0.0', port=5000, threads=4)