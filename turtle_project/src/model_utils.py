"""
Turtle Trading Model Utilities
=============================

This module provides comprehensive utilities for model training, prediction, 
saving, loading, and analysis for the Turtle Trading Strategy project.

Author: Panwei Hu
Date: August 27, 2025
"""

import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# Project imports
from .config import get_key
from .feature_engineering import TurtleFeatureEngineer


class TurtleModelManager:
    """
    Comprehensive model management for Turtle Trading Strategy.
    
    Handles model training, prediction, saving, loading, and evaluation
    for both regression and classification tasks.
    """
    
    def __init__(self, model_dir: str = "model"):
        """
        Initialize the model manager.
        
        Args:
            model_dir: Directory to store model files
        """
        project_root = Path(__file__).resolve().parents[1]  
        self.model_dir = (project_root / model_dir).resolve()
        self.model_dir.mkdir(parents=True, exist_ok=True)
        # Model storage
        self.models = {}
        self.scalers = {}
        self.feature_engineer = None
        self.metadata = {}
        
        # Performance tracking
        self.performance_history = []
        
    def train_regression_models(self, X: pd.DataFrame, y: pd.Series, 
                               test_size: float = 0.2, random_state: int = 42) -> Dict:
        """
        Train multiple regression models for return prediction.
        
        Args:
            X: Feature matrix
            y: Target variable (returns)
            test_size: Proportion for test set
            random_state: Random seed
            
        Returns:
            Dictionary with model performance metrics
        """
        print("🔄 Training regression models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=False
        )
        
        # Initialize models
        models = {
            'Linear': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=random_state)
        }
        
        # Train and evaluate
        results = {}
        for name, model in models.items():
            print(f"  Training {name}...")
            
            # Train
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Evaluate
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = np.mean(np.abs(y_test - y_pred))
            
            # Directional accuracy
            dir_acc = np.mean((y_pred > 0) == (y_test > 0))
            
            results[name] = {
                'model': model,
                'R²': r2,
                'RMSE': rmse,
                'MAE': mae,
                'Dir_Acc': dir_acc,
                'y_pred': y_pred,
                'y_test': y_test
            }
            
            # Store model
            self.models[f'regression_{name}'] = model
            
            print(f"    R²: {r2:.4f}, RMSE: {rmse:.6f}, Dir_Acc: {dir_acc:.1%}")
        
        # Store performance
        self.performance_history.append({
            'timestamp': datetime.now().isoformat(),
            'task': 'regression',
            'results': {k: {m: v[m] for m in ['R²', 'RMSE', 'MAE', 'Dir_Acc']} 
                       for k, v in results.items()}
        })
        
        return results
    
    def train_classification_models(self, X: pd.DataFrame, y: pd.Series,
                                   task: str = 'binary', test_size: float = 0.2, 
                                   random_state: int = 42) -> Dict:
        """
        Train classification models for directional prediction.
        
        Args:
            X: Feature matrix
            y: Target variable (binary or multiclass)
            task: 'binary' or 'multiclass'
            test_size: Proportion for test set
            random_state: Random seed
            
        Returns:
            Dictionary with model performance metrics
        """
        print(f"🔄 Training {task} classification models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=False
        )
        
        # Initialize models
        if task == 'binary':
            models = {
                'Logistic': LogisticRegression(random_state=random_state),
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=random_state)
            }
        else:  # multiclass
            models = {
                'Logistic': LogisticRegression(random_state=random_state, max_iter=1000),
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=random_state)
            }
        
        # Train and evaluate
        results = {}
        for name, model in models.items():
            print(f"  Training {name}...")
            
            # Train
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            
            # Evaluate
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted' if task == 'multiclass' else 'binary')
            
            results[name] = {
                'model': model,
                'Accuracy': accuracy,
                'F1': f1,
                'y_pred': y_pred,
                'y_proba': y_proba,
                'y_test': y_test
            }
            
            # Store model
            self.models[f'classification_{task}_{name}'] = model
            
            print(f"    Accuracy: {accuracy:.1%}, F1: {f1:.3f}")
        
        # Store performance
        self.performance_history.append({
            'timestamp': datetime.now().isoformat(),
            'task': f'classification_{task}',
            'results': {k: {m: v[m] for m in ['Accuracy', 'F1']} 
                       for k, v in results.items()}
        })
        
        return results
    
    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "feature_names_") or self.feature_names_ is None:
            raise ValueError("Model feature schema not loaded. Train or load models first.")
        X = X.copy()
        # Add missing columns as 0, drop extras, and order columns
        for col in self.feature_names_:
            if col not in X.columns:
                X[col] = 0.0
        X = X[self.feature_names_]
        return X

    def predict(self, X: pd.DataFrame, model_name: str) -> np.ndarray:
        """
        Make predictions using a trained model.
        
        Args:
            X: Feature matrix
            model_name: Name of the model to use
            
        Returns:
            Predictions
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Available: {list(self.models.keys())}")
        Xa = self._align_features(X)
        return self.models[model_name].predict(Xa)

    def predict_proba(self, X: pd.DataFrame, model_name: str) -> np.ndarray:
        """
        Get prediction probabilities for classification models.
        
        Args:
            X: Feature matrix
            model_name: Name of the classification model
            
        Returns:
            Prediction probabilities
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        model = self.models[model_name]
        Xa = self._align_features(X)
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(Xa)
        raise ValueError(f"Model '{model_name}' does not support probability predictions")


    # def predict(self, X: pd.DataFrame, model_name: str) -> np.ndarray:
    #     """
    #     Make predictions using a trained model.
        
    #     Args:
    #         X: Feature matrix
    #         model_name: Name of the model to use
            
    #     Returns:
    #         Predictions
    #     """
    #     if model_name not in self.models:
    #         raise ValueError(f"Model '{model_name}' not found. Available: {list(self.models.keys())}")
        
    #     model = self.models[model_name]
    #     return model.predict(X)
    
    # def predict_proba(self, X: pd.DataFrame, model_name: str) -> np.ndarray:
        # """
        # Get prediction probabilities for classification models.
        
        # Args:
        #     X: Feature matrix
        #     model_name: Name of the classification model
            
        # Returns:
        #     Prediction probabilities
        # """
        # if model_name not in self.models:
        #     raise ValueError(f"Model '{model_name}' not found")
        
        # model = self.models[model_name]
        # if hasattr(model, 'predict_proba'):
        #     return model.predict_proba(X)
        # else:
        #     raise ValueError(f"Model '{model_name}' does not support probability predictions")
    
    def save_models(self, prefix: str = "turtle_models") -> str:
        """
        Save all trained models and metadata.
        
        Args:
            prefix: Prefix for saved files
            
        Returns:
            Path to saved models
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_path = self.model_dir / f"{prefix}_{timestamp}"
        # make sure directory exists
        base_path.mkdir(parents=True, exist_ok=True)
        # Save models
        for name, model in self.models.items():
            model_path = base_path / f"{name}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        
        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'models': list(self.models.keys()),
            'performance_history': self.performance_history,
            'feature_engineer': self.feature_engineer is not None,
            'feature_names': self.feature_names_,
        }
        
        metadata_path = base_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"✅ Models saved to: {base_path}")
        return str(base_path)
    
    def load_models(self, model_path: str) -> bool:
        """
        Load models from a saved directory.
        
        Args:
            model_path: Path to saved models directory
            
        Returns:
            True if successful
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model path not found: {model_path}")
        
        # Load metadata
        metadata_path = model_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            self.feature_names_ = self.metadata.get('feature_names', None)
        # Load models
        for pkl_file in model_path.glob("*.pkl"):
            if pkl_file.name == "metadata.json":
                continue
                
            model_name = pkl_file.stem
            with open(pkl_file, 'rb') as f:
                self.models[model_name] = pickle.load(f)
        
        print(f"✅ Loaded {len(self.models)} models from: {model_path}")
        return True
    
    def get_model_summary(self) -> Dict:
        """
        Get a summary of all trained models and their performance.
        
        Returns:
            Dictionary with model summary
        """
        summary = {
            'total_models': len(self.models),
            'model_types': list(self.models.keys()),
            'latest_performance': self.performance_history[-1] if self.performance_history else None,
            'metadata': self.metadata
        }
        
        return summary
    
    def run_full_analysis(self, data_path: str, output_dir: str = "reports") -> Dict:
        """
        Run complete analysis pipeline from data to predictions.
        
        Args:
            data_path: Path to processed data
            output_dir: Directory for outputs
            
        Returns:
            Analysis results
        """
        print("🚀 Running full Turtle Trading analysis...")
        
        # Load data
        data = pd.read_parquet(data_path)
        print(f"📊 Loaded data: {data.shape}")
        
        # Feature engineering
        if self.feature_engineer is None:
            self.feature_engineer = TurtleFeatureEngineer()
        
        features_df = self.feature_engineer.engineer_all_features(data)
        print(f"🔧 Created features: {features_df.shape}")
        
        # Prepare targets
        # y_reg = features_df['returns'].shift(-1).dropna()  # Next day returns
        # y_binary = (y_reg > 0).astype(int)  # Binary classification
        
        # # Align features with targets
        # X = features_df.drop(['returns', 'date', 'symbol'], axis=1, errors='ignore').iloc[:-1]
        # y_reg = y_reg.iloc[1:]  # Align with features
        
        # # Remove any remaining NaN
        # mask = ~(X.isna().any(axis=1) | y_reg.isna())
        # X = X[mask]
        # y_reg = y_reg[mask]
        # y_binary = y_binary[mask]
        
        
        # # Target: next-day return
        # y_next = features_df['returns'].shift(-1).rename('y')

        # # Features (do NOT slice with iloc here)
        # X_all = features_df.drop(['returns', 'date', 'symbol'], axis=1, errors='ignore')

        # # Align by index and drop any rows with NaNs in X or y
        # model_df = X_all.join(y_next).dropna()
        # X = model_df.drop(columns='y')
        # y_reg = model_df['y']
        # y_binary = (y_reg > 0).astype(int)

        # print("📈 Final dataset X/y shapes:", X.shape, y_reg.shape, y_binary.shape)  # same row count
        
        # --- Targets ---
        y_next = features_df['returns'].shift(-1).rename('y')

        # --- Features (drop obvious non-features) ---
        X_all = features_df.drop(columns=['returns', 'date', 'symbol'], errors='ignore').copy()

        # Identify non-numeric columns (object/category)
        cat_cols = X_all.select_dtypes(include=['object', 'category']).columns
        # Optional: see what's being encoded
        print("Categorical columns being encoded:", list(cat_cols))

        # One-hot encode categoricals
        if len(cat_cols) > 0:
            X_all = pd.get_dummies(X_all, columns=list(cat_cols), drop_first=True)

        # Ensure everything is numeric (coerce any leftovers)
        X_all = X_all.apply(pd.to_numeric, errors='coerce')

        # --- Align and drop NaNs ONCE ---
        model_df = X_all.join(y_next).dropna()
        X = model_df.drop(columns='y')
        y_reg = model_df['y']
        y_binary = (y_reg > 0).astype(int)

        print("📈 Final dataset X/y shapes:", X.shape, y_reg.shape, y_binary.shape)
        # after X, y_reg, y_binary are finalized
        self.feature_names_ = list(X.columns)

        # # Train models
        reg_results = self.train_regression_models(X, y_reg)
        bin_results = self.train_classification_models(X, y_binary, task='binary')
        
        # Save models
        model_path = self.save_models()
        
        # Generate outputs
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 1) Get the common test index from regression (Ridge) results
        test_idx = reg_results['Ridge']['y_test'].index

        # 2) Grab dates aligned to that index
        dates = features_df.loc[test_idx, 'date']

        # 3) Actuals (regression + classification), already aligned by index
        actual_returns = reg_results['Ridge']['y_test']                      # Series with test_idx
        direction_actual = bin_results['Random Forest']['y_test']           # Series with test_idx

        # 4) Predictions -> wrap arrays into Series with the SAME index
        predicted_returns = pd.Series(reg_results['Ridge']['y_pred'], index=test_idx)
        direction_predicted = pd.Series(bin_results['Random Forest']['y_pred'], index=test_idx)


        # # Save predictions
        # predictions = pd.DataFrame({
        #     'date': data['date'].iloc[1:][mask],
        #     'actual_returns': y_reg,
        #     'predicted_returns': reg_results['Ridge']['y_pred'],
        #     'direction_actual': y_binary,
        #     'direction_predicted': bin_results['Random Forest']['y_pred']
        # })
        
        # 5) Assemble export (all columns share the same index)
        predictions = pd.DataFrame({
            'date': dates,
            'actual_returns': actual_returns,
            'predicted_returns': predicted_returns,
            'direction_actual': direction_actual,
            'direction_predicted': direction_predicted,
        }).reset_index(drop=True)

        
        pred_path = output_path / f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        predictions.to_csv(pred_path, index=False)
        
        # Create summary report
        summary = {
            'analysis_date': datetime.now().isoformat(),
            'data_shape': data.shape,
            'features_shape': X.shape,
            'model_performance': {
                'regression': {k: {m: v[m] for m in ['R²', 'RMSE', 'MAE', 'Dir_Acc']} 
                              for k, v in reg_results.items()},
                'binary_classification': {k: {m: v[m] for m in ['Accuracy', 'F1']} 
                                        for k, v in bin_results.items()}
            },
            'outputs': {
                'models_saved': model_path,
                'predictions_saved': str(pred_path)
            }
        }
        
        # Save summary
        summary_path = output_path / f"analysis_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"✅ Analysis complete! Results saved to: {output_path}")
        return summary


def load_latest_model(model_dir: str = "model") -> Optional[TurtleModelManager]:
    """
    Load the most recent model from the model directory.
    
    Args:
        model_dir: Directory containing saved models
        
    Returns:
        TurtleModelManager instance or None
    """
    model_dir = Path(model_dir)
    
    if not model_dir.exists():
        return None
    
    # Find most recent model directory
    model_dirs = [d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith("turtle_models")]
    
    if not model_dirs:
        return None
    
    latest_dir = max(model_dirs, key=lambda x: x.name)
    
    # Load models
    manager = TurtleModelManager()
    manager.load_models(str(latest_dir))
    
    return manager


def create_sample_prediction_data(n_samples: int = 100) -> pd.DataFrame:
    """
    Create sample data for testing predictions.
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        Sample feature DataFrame
    """
    np.random.seed(42)
    
    # Generate realistic feature data
    features = {
        'sma_20': np.random.normal(100, 5, n_samples),
        'sma_50': np.random.normal(98, 5, n_samples),
        'donchian_high_20': np.random.normal(105, 8, n_samples),
        'donchian_low_20': np.random.normal(95, 8, n_samples),
        'atr_20': np.random.uniform(1, 5, n_samples),
        'volatility_20': np.random.uniform(0.01, 0.05, n_samples),
        'trend_strength': np.random.uniform(-1, 1, n_samples),
        'price_position': np.random.uniform(0, 1, n_samples)
    }
    
    return pd.DataFrame(features) 