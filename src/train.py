"""
Training Script for Battery SoH Prediction
Main script to train LSTM models on battery cycle data
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_preprocessing import BatteryDataProcessor
from lstm_model import BatterySoHPredictor
from visualization import BatteryVisualizer


class BatterySoHTrainer:
    """
    Main trainer class for battery SoH prediction
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize trainer with configuration
        
        Args:
            config_path: Path to configuration JSON file
        """
        self.config = self._load_config(config_path)
        self.processor = BatteryDataProcessor(
            sequence_length=self.config['sequence_length'],
            feature_columns=self.config['feature_columns']
        )
        self.predictor = BatterySoHPredictor(
            sequence_length=self.config['sequence_length'],
            n_features=len(self.config['feature_columns'])
        )
        self.visualizer = BatteryVisualizer()
        
        # Create output directories
        os.makedirs(self.config['model_save_dir'], exist_ok=True)
        os.makedirs(self.config['results_dir'], exist_ok=True)
        
    def _load_config(self, config_path: str) -> dict:
        """
        Load configuration from JSON file or use defaults
        
        Args:
            config_path: Path to config file
            
        Returns:
            Configuration dictionary
        """
        default_config = {
            "data_file": "data/battery_data.csv",
            "sequence_length": 50,
            "feature_columns": [
                "voltage", "current", "temperature", "capacity",
                "cycle_number", "discharge_time", "charge_time"
            ],
            "model_params": {
                "lstm_units": [128, 64, 32],
                "dropout_rate": 0.3,
                "learning_rate": 0.001
            },
            "training_params": {
                "epochs": 100,
                "batch_size": 32,
                "patience": 15,
                "test_size": 0.2,
                "val_size": 0.1
            },
            "model_save_dir": "models",
            "results_dir": "results",
            "use_attention": False
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                custom_config = json.load(f)
            # Merge with defaults
            default_config.update(custom_config)
        
        return default_config
    
    def prepare_data(self) -> tuple:
        """
        Load and prepare data for training
        
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        print("Loading and preparing data...")
        
        # Load data
        df = self.processor.load_data(self.config['data_file'])
        df = self.processor.clean_data(df)
        
        # Print data statistics
        stats = self.processor.get_data_statistics(df)
        print("\nData Statistics:")
        for feature, stat in stats.items():
            print(f"{feature}: Mean={stat['mean']:.3f}, Std={stat['std']:.3f}")
        
        # Create sequences
        X, y = self.processor.create_sequences(df)
        print(f"\nCreated sequences: X shape={X.shape}, y shape={y.shape}")
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.processor.split_data(
            X, y, 
            test_size=self.config['training_params']['test_size'],
            val_size=self.config['training_params']['val_size']
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def build_model(self):
        """
        Build the LSTM model based on configuration
        """
        print("Building LSTM model...")
        
        if self.config['use_attention']:
            self.predictor.build_attention_model(
                lstm_units=self.config['model_params']['lstm_units'][:2],
                dropout_rate=self.config['model_params']['dropout_rate'],
                learning_rate=self.config['model_params']['learning_rate']
            )
        else:
            self.predictor.build_model(
                lstm_units=self.config['model_params']['lstm_units'],
                dropout_rate=self.config['model_params']['dropout_rate'],
                learning_rate=self.config['model_params']['learning_rate']
            )
        
        print(self.predictor.get_model_summary())
    
    def train_model(self, X_train, y_train, X_val, y_val) -> dict:
        """
        Train the LSTM model
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Training history
        """
        print("Training model...")
        
        # Model save path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"battery_soh_model_{timestamp}.h5"
        model_save_path = os.path.join(self.config['model_save_dir'], model_name)
        
        # Train model
        history = self.predictor.train(
            X_train, y_train, X_val, y_val,
            epochs=self.config['training_params']['epochs'],
            batch_size=self.config['training_params']['batch_size'],
            patience=self.config['training_params']['patience'],
            model_save_path=model_save_path
        )
        
        # Plot training history
        history_plot_path = os.path.join(self.config['results_dir'], f"training_history_{timestamp}.png")
        self.predictor.plot_training_history(history_plot_path)
        
        return history
    
    def evaluate_model(self, X_test, y_test) -> dict:
        """
        Evaluate the trained model
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Evaluation metrics
        """
        print("Evaluating model...")
        
        # Get predictions
        y_pred = self.predictor.predict(X_test)
        
        # Inverse transform for original scale
        y_test_orig = self.processor.inverse_transform_predictions(y_test)
        y_pred_orig = self.processor.inverse_transform_predictions(y_pred)
        
        # Calculate metrics
        metrics = self.predictor.evaluate(X_test, y_test)
        
        print("\nEvaluation Metrics:")
        for metric, value in metrics.items():
            print(f"{metric.upper()}: {value:.4f}")
        
        # Plot predictions
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prediction_plot_path = os.path.join(self.config['results_dir'], f"predictions_{timestamp}.png")
        self.visualizer.plot_predictions(y_test_orig, y_pred_orig, prediction_plot_path)
        
        # Plot degradation curve
        degradation_plot_path = os.path.join(self.config['results_dir'], f"degradation_{timestamp}.png")
        self.visualizer.plot_degradation_curve(y_test_orig, y_pred_orig, degradation_plot_path)
        
        return metrics
    
    def predict_rul(self, X_test, threshold: float = 0.8) -> np.ndarray:
        """
        Predict Remaining Useful Life
        
        Args:
            X_test: Test features
            threshold: SoH threshold for EOL
            
        Returns:
            RUL predictions
        """
        print("Predicting RUL...")
        
        rul_predictions = self.predictor.predict_rul(X_test, threshold)
        
        print(f"RUL Statistics:")
        print(f"Mean RUL: {np.mean(rul_predictions):.1f} cycles")
        print(f"Min RUL: {np.min(rul_predictions):.1f} cycles")
        print(f"Max RUL: {np.max(rul_predictions):.1f} cycles")
        
        return rul_predictions
    
    def save_results(self, metrics: dict, rul_predictions: np.ndarray):
        """
        Save training results and metrics
        
        Args:
            metrics: Evaluation metrics
            rul_predictions: RUL predictions
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(self.config['results_dir'], f"results_{timestamp}.json")
        
        results = {
            "timestamp": timestamp,
            "config": self.config,
            "metrics": {k: float(v) for k, v in metrics.items()},
            "rul_stats": {
                "mean": float(np.mean(rul_predictions)),
                "std": float(np.std(rul_predictions)),
                "min": float(np.min(rul_predictions)),
                "max": float(np.max(rul_predictions))
            }
        }
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {results_path}")
    
    def run_complete_training(self):
        """
        Run the complete training pipeline
        """
        print("=" * 60)
        print("Battery SoH Prediction - Training Pipeline")
        print("=" * 60)
        
        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data()
        
        # Build model
        self.build_model()
        
        # Train model
        history = self.train_model(X_train, y_train, X_val, y_val)
        
        # Evaluate model
        metrics = self.evaluate_model(X_test, y_test)
        
        # Predict RUL
        rul_predictions = self.predict_rul(X_test)
        
        # Save results
        self.save_results(metrics, rul_predictions)
        
        print("\n" + "=" * 60)
        print("Training completed successfully!")
        print("=" * 60)


def main():
    """
    Main function to run training from command line
    """
    parser = argparse.ArgumentParser(description="Train Battery SoH Prediction Model")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--data", type=str, help="Path to data file")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--attention", action="store_true", help="Use attention model")
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = BatterySoHTrainer(args.config)
    
    # Override config with command line arguments
    if args.data:
        trainer.config['data_file'] = args.data
    if args.epochs:
        trainer.config['training_params']['epochs'] = args.epochs
    if args.batch_size:
        trainer.config['training_params']['batch_size'] = args.batch_size
    if args.attention:
        trainer.config['use_attention'] = True
    
    # Run training
    trainer.run_complete_training()


if __name__ == "__main__":
    main()
