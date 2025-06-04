"""
SolarSense Model Training Module

This module implements the training pipeline for the SolarSense AI model,
including data preprocessing, model training, and evaluation with a focus
on ethical considerations and bias mitigation.
"""

import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Tuple, Dict, Any

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SolarSenseModel:
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the SolarSense model with configuration parameters.
        
        Args:
            config: Dictionary containing model configuration parameters
        """
        self.config = config or {
            'test_size': 0.2,
            'random_state': 42,
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 5,
            'min_samples_split': 5,
            'min_samples_leaf': 2
        }
        
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        self.bias_metrics = {}
        
    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        Load and validate the input data.
        
        Args:
            data_path: Path to the input data file
            
        Returns:
            DataFrame containing the processed data
        """
        try:
            data = pd.read_csv(data_path)
            logger.info(f"Successfully loaded data from {data_path}")
            
            # Validate data
            self._validate_data(data)
            
            return data
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def _validate_data(self, data: pd.DataFrame) -> None:
        """
        Validate input data for biases and quality issues.
        
        Args:
            data: Input DataFrame to validate
        """
        # Check for missing values
        missing_values = data.isnull().sum()
        if missing_values.any():
            logger.warning(f"Missing values found: {missing_values[missing_values > 0]}")
        
        # Check for geographic representation
        if 'region' in data.columns:
            region_counts = data['region'].value_counts()
            if (region_counts / len(data) < 0.1).any():
                logger.warning("Some regions are underrepresented")
        
        # Check for seasonal balance
        if 'season' in data.columns:
            season_counts = data['season'].value_counts()
            if (season_counts / len(data) < 0.2).any():
                logger.warning("Some seasons are underrepresented")
    
    def preprocess_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the input data for model training.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Tuple of (X, y) for model training
        """
        # Separate features and target
        X = data.drop(['energy_production'], axis=1)
        y = data['energy_production']
        
        # Handle categorical variables
        X = pd.get_dummies(X, columns=['region', 'season'])
        
        # Scale numerical features
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        X[numerical_cols] = self.scaler.fit_transform(X[numerical_cols])
        
        return X, y
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train the model with bias mitigation strategies.
        
        Args:
            X: Feature matrix
            y: Target variable
        """
        # Split data with stratification if possible
        if 'region' in X.columns:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.config['test_size'],
                random_state=self.config['random_state'],
                stratify=X['region']
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.config['test_size'],
                random_state=self.config['random_state']
            )
        
        # Initialize and train model
        self.model = GradientBoostingRegressor(
            n_estimators=self.config['n_estimators'],
            learning_rate=self.config['learning_rate'],
            max_depth=self.config['max_depth'],
            min_samples_split=self.config['min_samples_split'],
            min_samples_leaf=self.config['min_samples_leaf'],
            random_state=self.config['random_state']
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Calculate feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Evaluate model
        self._evaluate_model(X_test, y_test)
        
        # Check for biases
        self._check_bias(X_test, y_test)
    
    def _evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> None:
        """
        Evaluate model performance and log metrics.
        
        Args:
            X_test: Test feature matrix
            y_test: Test target variable
        """
        y_pred = self.model.predict(X_test)
        
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred)
        }
        
        logger.info("Model Performance Metrics:")
        for metric, value in metrics.items():
            logger.info(f"{metric.upper()}: {value:.4f}")
    
    def _check_bias(self, X_test: pd.DataFrame, y_test: pd.Series) -> None:
        """
        Check for potential biases in model predictions.
        
        Args:
            X_test: Test feature matrix
            y_test: Test target variable
        """
        y_pred = self.model.predict(X_test)
        
        # Check regional bias
        if 'region' in X_test.columns:
            for region in X_test['region'].unique():
                region_mask = X_test['region'] == region
                region_mae = mean_absolute_error(
                    y_test[region_mask],
                    y_pred[region_mask]
                )
                self.bias_metrics[f'region_{region}_mae'] = region_mae
        
        # Check seasonal bias
        if 'season' in X_test.columns:
            for season in X_test['season'].unique():
                season_mask = X_test['season'] == season
                season_mae = mean_absolute_error(
                    y_test[season_mask],
                    y_pred[season_mask]
                )
                self.bias_metrics[f'season_{season}_mae'] = season_mae
        
        logger.info("Bias Metrics:")
        for metric, value in self.bias_metrics.items():
            logger.info(f"{metric}: {value:.4f}")
    
    def save_model(self, model_path: str) -> None:
        """
        Save the trained model and associated metadata.
        
        Args:
            model_path: Path to save the model
        """
        if self.model is None:
            raise ValueError("No trained model to save")
        
        # Create model directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'feature_importance': self.feature_importance.to_dict(),
            'bias_metrics': self.bias_metrics
        }
        
        # Save model and metadata
        import joblib
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'metadata': metadata
        }, model_path)
        
        logger.info(f"Model saved to {model_path}")

def main():
    """Main training pipeline."""
    # Initialize model
    model = SolarSenseModel()
    
    # Load and preprocess data
    data = model.load_data('data/processed/solar_data.csv')
    X, y = model.preprocess_data(data)
    
    # Train model
    model.train(X, y)
    
    # Save model
    model.save_model('models/solarsense_model.joblib')

if __name__ == "__main__":
    main() 