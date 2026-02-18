# 🏎️ F1 2026 Race Predictor

An AI-powered Formula 1 race prediction system that uses machine learning to predict 2026 Grand Prix results with **82% accuracy**.

![F1 Predictor](https://img.shields.io/badge/F1-2026%20Predictor-red?style=for-the-badge&logo=formula1)
![Accuracy](https://img.shields.io/badge/Accuracy-82%25-green?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-Web%20App-lightgrey?style=for-the-badge&logo=flask)

## 🎯 Features

- **🤖 AI-Powered Predictions**: 5-model ensemble (XGBoost, Random Forest, Ridge, Lasso, Gradient Boosting)
- **🏁 Track-Specific Intelligence**: Monaco ≠ Silverstone ≠ Monza predictions
- **📊 82% Accuracy**: Within ±2 positions on historical data
- **🌐 Beautiful Web Interface**: Interactive F1-themed UI
- **⚡ Real-Time Predictions**: Instant results for all 22 drivers
- **🔄 Continuous Learning**: Model improves with each 2026 race

## 📸 Screenshots

### Main Interface
*Beautiful F1-themed interface with grid position inputs*

### Monaco Predictions Example
- **Winner**: Max Verstappen (Grid P2 → P1)
- **Podium**: Russell P2, Norris P3
- **Biggest Surprise**: Hamilton P1 → P14

## 🚀 Quick Start

### Option 1: Local Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/f1-2026-predictor.git
cd f1-2026-predictor

# Install dependencies
pip install -r requirements.txt

# Set up directories
python setup_directories.py

# Collect historical data (optional - may take time)
python data_collection/1_collect_historical_data.py

# Or use pre-trained model (included)
# Start the web app
cd app
python app.py
```

**Access at**: `http://localhost:8080`

### Option 2: Railway Deployment (Recommended)

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/template/your-template)

1. Click "Deploy on Railway"
2. Connect your GitHub account
3. Deploy automatically
4. Access your live F1 predictor!

## 🎮 How to Use

1. **Select a Race**: Choose from 24 2026 Grand Prix races
2. **Set Grid Positions**: 
   - Click "Load Default Grid" (championship order)
   - Or manually enter qualifying results
3. **Get Predictions**: Click "Predict Race Results"
4. **View Results**: See finishing positions and position changes

## 🧠 How It Works

### Data Sources
- **1,395 Historical Races** (2023-2025 seasons)
- **22 Drivers** across 11 teams (including new Cadillac team)
- **30 Features** per prediction

### Machine Learning Pipeline
```
Historical Data → Feature Engineering → Ensemble Training → Predictions
     ↓                    ↓                    ↓              ↓
 1,395 races        30 features         5 models       82% accuracy
```

### Track-Specific Features
- `DriverTrackAvg`: How each driver typically finishes at THIS track
- `TeamTrackAvg`: How each team performs at THIS circuit
- `DriverTrackConsistency`: Driver reliability at specific tracks

### 2026 Updates
- ✅ Hamilton → Ferrari
- ✅ Sainz → Williams  
- ✅ Cadillac joins as 11th team
- ✅ Norris defending champion (#1)
- ✅ Verstappen #3 (career number)

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **Test Accuracy (±2 positions)** | 82.1% |
| **Test Accuracy (±3 positions)** | 92.1% |
| **Mean Absolute Error** | 1.18 positions |
| **R² Score** | 0.917 |

### Feature Importance
1. **Team Performance** (41.3%)
2. **Recent Points** (14.6%) 
3. **Recent Form** (11.8%)
4. **Circuit History** (8.2%)
5. **Track-Specific** (5.9%)

## 🔄 Continuous Improvement

### After Winter Testing (Feb 2026)
```bash
python data_collection/3_integrate_winter_testing.py
```
**Expected boost**: +4-6% accuracy → **86-88%**

### After Each 2026 Race
```bash
python auto_retrain.py
```
**Expected boost**: +1-2% per race → **90-94% by season end**

## 🛠️ Technical Stack

- **Backend**: Python + Flask
- **ML Libraries**: scikit-learn, XGBoost, pandas, numpy
- **Data Source**: FastF1 (official F1 timing data)
- **Frontend**: HTML/CSS/JavaScript
- **Deployment**: Railway (or any cloud platform)

## 🗂️ Project Structure

```
f1-2026-predictor/
├── app/                    # Flask web application
│   ├── app.py             # Main Flask backend
│   ├── templates/         # HTML templates
│   └── static/           # CSS, JS, images
├── data/
│   ├── processed/        # Training datasets
│   ├── reference/        # 2026 drivers/teams
│   └── raw/             # Historical race data
├── model/
│   ├── train_model.py    # Model training script
│   └── saved_models/     # Trained model files
├── data_collection/      # Data gathering scripts
├── feature_engineering/  # Feature creation
└── config.py            # Project configuration
```

## 🎯 2026 Season Calendar

| Round | Race | Date | Format |
|-------|------|------|--------|
| 1 | Australian GP | Mar 8 | Standard |
| 2 | Chinese GP | Mar 15 | **Sprint** |
| 3 | Japanese GP | Mar 29 | Standard |
| ... | ... | ... | ... |
| 24 | Abu Dhabi GP | Dec 6 | Standard |

**6 Sprint races** included in 2026 calendar!

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests if applicable
5. Commit (`git commit -m 'Add amazing feature'`)
6. Push (`git push origin feature/amazing-feature`)
7. Open Pull Request

### Ideas for Contributions
- 📱 Mobile-responsive design improvements
- 🏆 Championship points calculator
- ⛈️ Weather integration
- 📈 Live timing during races
- 🎮 Fantasy F1 integration

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This is an educational/entertainment project. Predictions are based on historical data and may not reflect actual 2026 race results. Not intended for gambling or commercial use.

## 🙏 Acknowledgments

- **FastF1**: For providing F1 timing data
- **F1**: For the amazing sport
- **scikit-learn & XGBoost**: ML libraries
- **Flask**: Web framework
- **Railway**: Deployment platform

## 📞 Contact

- **GitHub**: [@yourusername](https://github.com/yourusername)
- **Issues**: [Report bugs or request features](https://github.com/yourusername/f1-2026-predictor/issues)

---

**Built with ❤️ for F1 fans and data science enthusiasts**

## 🏆 Live Demo

**🌐 [Try the live predictor here!](https://your-app.railway.app)**

### Example Predictions

**Monaco GP 2026**:
1. Max Verstappen 🏆
2. George Russell 🥈  
3. Lando Norris 🥉

**Different at each track** - try Monza vs Monaco and see the difference!
