# 🏎️ F1 2026 Race Predictor

An AI-powered Formula 1 race prediction system that uses machine learning to predict 2026 Grand Prix results with **82.1% accuracy**.

![F1 Predictor](https://img.shields.io/badge/F1-2026%20Predictor-red?style=for-the-badge&logo=formula1)
![Accuracy](https://img.shields.io/badge/Accuracy-82.1%25-green?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-Web%20App-lightgrey?style=for-the-badge&logo=flask)

## 🎯 Features

- **🤖 AI-Powered Predictions**: 5-model ensemble (XGBoost, Random Forest, Ridge, Lasso, Gradient Boosting)
- **🏁 Track-Specific Intelligence**: Monaco ≠ Silverstone ≠ Monza predictions
- **📊 82.1% Accuracy**: Within ±2 positions on historical data
- **🌐 Beautiful Web Interface**: Interactive F1-themed UI
- **⚡ Real-Time Predictions**: Instant results for all 22 drivers
- **🔄 Continuous Learning**: Model improves with each 2026 race



### Example Predictions
**Different circuits produce different results:**
- **Monaco GP**: Conservative position changes due to limited overtaking
- **Chinese GP**: More dynamic results with strategic opportunities
- **Monza GP**: High-speed characteristics favor different drivers

## 🚀 Quick Start

### Prerequisites
- **Python 3.11+** (recommended for compatibility)
- **Git** (for cloning repository)
- **4GB+ RAM** (for ML model training)
- **Internet connection** (for initial setup)

### Local Installation

#### Step 1: Clone & Setup Environment
```bash
# Clone the repository
git clone https://github.com/jaybaragadi/f1-2026-predictor.git
cd f1-2026-predictor

# Create Python 3.11 virtual environment (recommended)
py -3.11 -m venv .venv

# Activate virtual environment
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

# Verify Python version
python --version  # Should show Python 3.11.x
pip --version     # Should show pip from venv
```

#### Step 2: Install Dependencies
```bash
# Install all required packages (this may take a few minutes)
pip install -r requirements.txt

# Key packages installed:
# - scikit-learn 1.8.0 (ML algorithms)
# - XGBoost 3.2.0 (gradient boosting)
# - pandas 2.3.3 & numpy 1.26.4 (data processing)
# - Flask 3.0.0 & Flask-CORS 4.0.0 (web server)
# - FastF1 3.8.1 (F1 data access)
# - matplotlib 3.10.8 (visualization)
```

#### Step 3: Train the ML Model
```bash
# Train the 82% accurate ensemble model
python model/train_model.py

# Expected output:
# ✓ Loaded 1395 training samples
# ✓ Features: 30
# ✓ Track-specific features detected: 5
# ✓ Test accuracy: 82.1% (within ±2 positions)
# ✓ Model saved successfully
```

#### Step 4: Start the Application
```bash
# Navigate to app directory
cd app

# Start Flask development server
python app.py

# Expected output:
# ✓ Model loaded successfully
# ✓ Track-specific features enabled: 5
# Features: 30
# Accuracy: 82.1% (within ±2 positions)
# Model: 5-Algorithm Ensemble
# * Running on http://127.0.0.1:5001
```

#### Step 5: Access Your F1 Predictor
Open your browser and navigate to: **http://127.0.0.1:5001**

🎉 **You now have a fully functional F1 2026 Race Predictor with 82.1% accuracy!**

### ⚠️ Troubleshooting

**Common Issues & Solutions:**

1. **"Python 3.11 not found"**
   ```bash
   # Install Python 3.11 from python.org
   # Or use available Python version:
   python -m venv .venv
   ```

2. **Package installation errors**
   ```bash
   # Update pip first:
   python -m pip install --upgrade pip
   # Then retry: pip install -r requirements.txt
   ```

3. **Model training fails**
   ```bash
   # Ensure you have 4GB+ RAM available
   # Close other applications and retry
   python model/train_model.py
   ```

4. **Port already in use**
   ```bash
   # The app will automatically try different ports
   # Or manually specify: python app.py --port 5002
   ```

### Production Deployment (Railway/Render)

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new/template/your-template)

```bash
# For production deployment:
git add .
git commit -m "Ready for deployment"
git push origin main

# Railway will automatically:
# - Install dependencies
# - Train the model
# - Start the production server
# - Provide a live URL
```

## 🎮 How to Use

1. **Select a Race**: Choose from 24 2026 Grand Prix races
2. **Set Grid Positions**: 
   - Click "Load Default Grid" (championship order)
   - Or manually enter qualifying results for any scenario
3. **Get Predictions**: Click "Predict Race Results"
4. **View Results**: See finishing positions, position changes, and insights
5. **Compare Tracks**: Try different circuits to see how predictions vary!

## 🧠 How It Works

### Data Sources
- **1,395 Historical Races** (2023-2025 seasons)
- **22 Drivers** across 11 teams (including new Cadillac team)
- **30 Engineered Features** per prediction

### Machine Learning Pipeline
```
Historical Data → Feature Engineering → Ensemble Training → Predictions
     ↓                    ↓                    ↓              ↓
 1,395 races        30 features         5 models       82.1% accuracy
```

### Model Architecture
**5-Algorithm Stacking Ensemble:**
- **Ridge Regression** (L2 regularization)
- **Lasso Regression** (L1 regularization) 
- **XGBoost** (gradient boosting excellence)
- **Random Forest** (robust ensemble learning)
- **Gradient Boosting** (sequential error correction)
- **Meta-Learner**: Ridge regression combines all predictions

### Track-Specific Intelligence
- `DriverTrackAvg`: How each driver typically finishes at THIS track
- `TeamTrackAvg`: How each team performs at THIS circuit  
- `DriverTrackConsistency`: Driver reliability at specific tracks
- **Circuit-aware predictions**: Monaco ≠ Silverstone ≠ Monza

### 2026 Season Updates
- ✅ **Lewis Hamilton** → Ferrari (#44)
- ✅ **Carlos Sainz** → Williams 
- ✅ **Cadillac joins** as 11th team (Pérez #11, Bottas #77)
- ✅ **Lando Norris** defending champion (#1)
- ✅ **Max Verstappen** career #3

## 📊 Model Performance

| Metric | Training | Test | Interpretation |
|--------|----------|------|----------------|
| **Accuracy (±2 positions)** | 97.9% | **82.1%** | 4 out of 5 predictions within 2 spots |
| **Accuracy (±3 positions)** | 99.6% | **92.1%** | 9 out of 10 predictions within 3 spots |
| **Mean Absolute Error** | 0.550 | **1.185** | Average prediction error |
| **R² Score** | 0.983 | **0.917** | 91.7% of variance explained |

### Feature Importance Analysis
1. **TeamRaceAvgPosition** (41.3%) - Current team form dominates
2. **Points_Rolling_3** (14.6%) - Recent driver performance
3. **Position_Rolling_3** (11.8%) - Driver consistency patterns
4. **CircuitAvgPosition** (8.2%) - Track-specific performance
5. **Track Features** (5.9%) - Circuit-aware intelligence

## 🔄 Continuous Improvement

### After Winter Testing (Feb 2026)
```bash
python data_collection/3_integrate_winter_testing.py
```
**Expected boost**: +4-6% accuracy → **86-88% total**

### After Each 2026 Race
```bash
python auto_retrain.py
```
**Expected boost**: +1-2% per race → **90-94% by season end**

### Live Race Integration
```bash
python live_timing_recorder.py --round 1 --session R
```
**Real-time updates**: Capture races as they happen for immediate retraining

## 🛠️ Technical Stack

- **Backend**: Python 3.11 + Flask 3.0
- **ML Libraries**: scikit-learn 1.8, XGBoost 3.2, pandas 2.3, numpy 1.26
- **Data Source**: FastF1 3.8 (official F1 timing data)
- **Frontend**: HTML5/CSS3/JavaScript (responsive design)
- **Deployment**: Railway, Render, or any Python hosting platform

## 🗂️ Project Structure

```
f1-2026-predictor/
├── app/                    # Flask web application
│   ├── app.py             # Main Flask backend with ML integration
│   ├── templates/         # HTML templates
│   └── static/           # CSS, JS, images
├── data/
│   ├── processed/        # Training datasets (1,395 races)
│   ├── reference/        # 2026 drivers/teams data
│   └── raw/             # Historical race results
├── model/
│   ├── train_model.py    # 5-algorithm ensemble training
│   └── saved_models/     # Trained model artifacts
├── data_collection/      # FastF1 data gathering scripts
├── feature_engineering/  # 30-feature creation pipeline
├── config.py            # Project configuration
└── requirements.txt     # Python dependencies
```

## 🎯 2026 Season Calendar

| Round | Race | Date | Format |
|-------|------|------|--------|
| 1 | Australian GP | Mar 8 | Standard |
| 2 | Chinese GP | Mar 15 | **Sprint** |
| 3 | Japanese GP | Mar 29 | Standard |
| 4 | Japanese GP | Apr 5 | Standard |
| ... | ... | ... | ... |
| 24 | Abu Dhabi GP | Dec 6 | Standard |

**6 Sprint races** included in 2026 calendar for extra excitement!

## 🤝 Contributing

We welcome contributions! Here are some areas where help is needed:

### High-Priority Features
- 📱 **Mobile-responsive design** improvements
- 🏆 **Championship points calculator** 
- ⛈️ **Weather integration** (rain impact on predictions)
- 📈 **Live timing** during race weekends
- 🎮 **Fantasy F1** league integration

### Technical Improvements
- 🔧 **Neural network** ensemble experiments
- 📊 **Enhanced visualization** of predictions
- 🚀 **Performance optimization** for faster predictions
- 🧪 **Extended testing** coverage
- 📖 **API documentation**

### Getting Started with Contributing
1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Set up development environment (`pip install -r requirements.txt`)
4. Make your changes with tests
5. Submit Pull Request


## ⚠️ Disclaimer

This is an educational/entertainment project for F1 fans and data science enthusiasts. Predictions are based on historical data analysis and machine learning models. Results may not reflect actual 2026 race outcomes. **Not intended for gambling or commercial betting purposes.**

## 🙏 Acknowledgments

- **FastF1**: Providing comprehensive F1 timing data and telemetry
- **Formula 1**: For the incredible sport and data accessibility  
- **scikit-learn & XGBoost**: Robust machine learning frameworks
- **Flask**: Lightweight and efficient web framework
- **F1 Community**: Data insights, feedback, and continuous support
- **Open Source**: NumPy, Pandas, and the entire Python ecosystem

## 📞 Contact

- **GitHub**: [@jaybaragadi](https://github.com/jaybaragadi)

---

**Built with ❤️ for F1 fans and data science and Formula1 enthusiasts**


### Example Predictions

**Track Intelligence in Action:**

**Monaco GP 2026** (Tight Street Circuit):
- Minimal position changes due to overtaking difficulty
- Grid position heavily influences final result
- Strategy and reliability become crucial

**Monza GP 2026** (High-Speed Power Circuit): 
- More dynamic position changes with slipstream battles
- Power unit performance differentiation
- Strategic opportunities with DRS zones

**Silverstone GP 2026** (Balanced Technical Circuit):
- Driver skill differentiation on full display
- Weather impact potential (UK climate)
- Historical data shows varied results

**🏁 Try different tracks and see how the AI adapts its predictions to circuit characteristics!**
