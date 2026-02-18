"""
Prediction utilities for F1 2026 Race Predictor
"""

import sys
sys.path.append('..')

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from config import MODEL_DIR, PROCESSED_DATA_DIR, REFERENCE_DATA_DIR


class F1RacePredictor:
    """
    F1 Race Predictor class for making predictions
    """
    
    def __init__(self):
        """Initialize predictor by loading model artifacts"""
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.drivers_2026 = None
        self.teams_2026 = None
        self.driver_stats = None
        
        self.load_model()
        self.load_reference_data()
    
    def load_model(self):
        """Load trained model, scaler, and feature columns"""
        
        model_file = MODEL_DIR / 'f1_race_predictor_model.pkl'
        scaler_file = MODEL_DIR / 'scaler.pkl'
        features_file = MODEL_DIR / 'feature_columns.pkl'
        
        if not all([model_file.exists(), scaler_file.exists(), features_file.exists()]):
            raise FileNotFoundError(
                "Model files not found. Please train the model first."
            )
        
        self.model = joblib.load(model_file)
        self.scaler = joblib.load(scaler_file)
        self.feature_columns = joblib.load(features_file)
        
        print(f"✓ Model loaded successfully")
        print(f"✓ Features: {len(self.feature_columns)}")
    
    def load_reference_data(self):
        """Load reference data for 2026 drivers and teams"""
        
        drivers_file = REFERENCE_DATA_DIR / '2026_drivers.csv'
        teams_file = REFERENCE_DATA_DIR / '2026_teams.csv'
        stats_file = REFERENCE_DATA_DIR / 'driver_historical_stats.csv'
        
        self.drivers_2026 = pd.read_csv(drivers_file)
        self.teams_2026 = pd.read_csv(teams_file)
        self.driver_stats = pd.read_csv(stats_file)
        
        print(f"✓ Loaded reference data for {len(self.drivers_2026)} drivers")
    
    def prepare_input_features(self, grid_positions):
        """
        Prepare input features from grid positions
        
        Parameters:
        -----------
        grid_positions : dict
            Dictionary mapping driver codes to grid positions
            Example: {'VER': 1, 'NOR': 2, 'LEC': 3, ...}
        
        Returns:
        --------
        pd.DataFrame : Feature matrix ready for prediction
        """
        
        # Create input dataframe
        input_data = []
        
        for driver_code, grid_pos in grid_positions.items():
            # Get driver info
            driver_info = self.drivers_2026[
                self.drivers_2026['DriverCode'] == driver_code
            ]
            
            if driver_info.empty:
                print(f"Warning: Driver {driver_code} not found in 2026 grid")
                continue
            
            driver_info = driver_info.iloc[0]
            
            # Get driver historical stats
            driver_hist = self.driver_stats[
                self.driver_stats['DriverName'] == driver_info['DriverName']
            ]
            
            # If driver is a rookie or not in history, use defaults
            if driver_hist.empty:
                avg_finish = 15.0
                avg_grid = 15.0
                total_points = 0.0
                best_finish = 20.0
            else:
                driver_hist = driver_hist.iloc[0]
                avg_finish = driver_hist['AvgFinishPosition']
                avg_grid = driver_hist['AvgGridPosition']
                total_points = driver_hist['TotalPoints']
                best_finish = driver_hist['BestFinish']
            
            # Get team info
            team_info = self.teams_2026[
                self.teams_2026['Team'] == driver_info['Team']
            ]
            
            if team_info.empty:
                championships_won = 0
                years_in_f1 = 5
            else:
                team_info = team_info.iloc[0]
                championships_won = team_info['ChampionshipsWon']
                years_in_f1 = team_info['YearsInF1']
            
            # Create feature dictionary with defaults
            features = {
                'GridPosition': grid_pos,
                'QualifyingPosition': grid_pos,  # Assume same as grid
                
                # Driver rolling stats (use historical averages as proxy)
                'Position_Rolling_3': avg_finish,
                'Position_Rolling_5': avg_finish,
                'Position_Rolling_10': avg_finish,
                'GridPosition_Rolling_3': avg_grid,
                'GridPosition_Rolling_5': avg_grid,
                'Points_Rolling_3': total_points / 20 if total_points > 0 else 0,
                'Points_Rolling_5': total_points / 20 if total_points > 0 else 0,
                'PositionsGained_Rolling_3': 0,
                'PositionsGained_Rolling_5': 0,
                'RecentForm': avg_finish,
                
                # Driver historical stats
                'AvgFinishPosition': avg_finish,
                'AvgGridPosition': avg_grid,
                'TotalPoints': total_points,
                'BestFinish': best_finish,
                
                # Team features (use defaults for 2026)
                'TeamYearPoints': 300,  # Neutral
                'TeamYearAvgPosition': 10,  # Neutral
                'TeamRaceAvgPosition': 10,  # Neutral
                'ChampionshipsWon': championships_won,
                'YearsInF1': years_in_f1,
                
                # Circuit features (neutral for new season)
                'CircuitAvgPosition': avg_finish,
                'CircuitBestPosition': best_finish,
                'CircuitRacesCount': 0,
                
                # Recency weight (maximum for 2026 prediction)
                'RecencyWeight': 2.0
            }
            
            # Add metadata
            features['DriverCode'] = driver_code
            features['DriverName'] = driver_info['DriverName']
            features['Team'] = driver_info['Team']
            
            input_data.append(features)
        
        # Convert to DataFrame
        input_df = pd.DataFrame(input_data)
        
        return input_df
    
    def predict_race(self, grid_positions):
        """
        Predict race finishing positions
        
        Parameters:
        -----------
        grid_positions : dict
            Dictionary mapping driver codes to grid positions
        
        Returns:
        --------
        pd.DataFrame : Predictions with driver info and predicted positions
        """
        
        # Prepare features
        input_df = self.prepare_input_features(grid_positions)
        
        # Extract feature columns for prediction
        X = input_df[self.feature_columns].copy()
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        
        # Clip to valid range [1, 20]
        predictions = np.clip(predictions, 1, 22)  # FIXED for 2026
        
        # Add predictions to dataframe
        input_df['PredictedPosition'] = predictions
        
        # Sort by predicted position
        input_df = input_df.sort_values('PredictedPosition').reset_index(drop=True)
        
        # Assign final positions (handle ties)
        input_df['FinalPosition'] = range(1, len(input_df) + 1)
        
        # Select output columns
        output = input_df[[
            'FinalPosition', 'DriverCode', 'DriverName', 'Team', 
            'GridPosition', 'PredictedPosition'
        ]].copy()
        
        return output
    
    def predict_with_confidence(self, grid_positions, n_iterations=100):
        """
        Predict with confidence intervals using bootstrap
        
        Parameters:
        -----------
        grid_positions : dict
            Dictionary mapping driver codes to grid positions
        n_iterations : int
            Number of bootstrap iterations
        
        Returns:
        --------
        pd.DataFrame : Predictions with confidence intervals
        """
        
        # For now, return standard prediction
        # TODO: Implement proper confidence intervals with bootstrap
        
        output = self.predict_race(grid_positions)
        
        # Add placeholder confidence intervals
        output['ConfidenceLower'] = np.maximum(1, output['PredictedPosition'] - 2)
        output['ConfidenceUpper'] = np.minimum(22, output['PredictedPosition'] + 2)
        
        return output


def example_prediction():
    """
    Example usage of the predictor
    """
    print("\n" + "="*70)
    print("🏎️  EXAMPLE PREDICTION")
    print("="*70 + "\n")
    
    # Initialize predictor
    predictor = F1RacePredictor()
    
    # Example grid positions (hypothetical qualifying result)
    grid_positions = {
    'VER': 1, 'NOR': 2, 'LEC': 3, 'PIA': 4, 'RUS': 5,
    'HAM': 6, 'SAI': 7, 'PER': 8, 'ALO': 9, 'ALB': 10,
    'GAS': 11, 'TSU': 12, 'OCO': 13, 'HUL': 14, 'STR': 15,
    'ANT': 16, 'BOT': 17, 'BEA': 18, 'HAD': 19, 'BOR': 20,
    'LAW': 21, 'COL': 22  # ADDED: Liam Lawson, Franco Colapinto
    }
    
    # Make prediction
    predictions = predictor.predict_race(grid_positions)
    
    # Display results
    print("PREDICTED RACE RESULTS:")
    print(predictions.to_string(index=False))
    print("\n" + "="*70 + "\n")
    
    return predictions


if __name__ == "__main__":
    example_prediction()