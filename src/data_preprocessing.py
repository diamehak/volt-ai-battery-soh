"""
Data Preprocessing Module for Battery SoH Prediction
Handles loading, cleaning, and preparation of battery cycle data
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')


class BatteryDataProcessor:
    """
    Processes battery charge/discharge cycle data for LSTM training
    """
    
    def __init__(self, sequence_length: int = 50, feature_columns: List[str] = None):
        """
        Initialize the data processor
        
        Args:
            sequence_length: Number of time steps for LSTM sequences
            feature_columns: List of feature columns to use for training
        """
        self.sequence_length = sequence_length
        self.feature_columns = feature_columns or [
            'voltage', 'current', 'temperature', 'capacity', 
            'cycle_number', 'discharge_time', 'charge_time'
        ]
        self.scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load battery data from CSV file
        
        Args:
            file_path: Path to the CSV file containing battery data
            
        Returns:
            Loaded and cleaned DataFrame
        """
        try:
            df = pd.read_csv(file_path)
            print(f"Data loaded successfully. Shape: {df.shape}")
            return df
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return self._generate_sample_data()
    
    def _generate_sample_data(self, n_cycles: int = 1000) -> pd.DataFrame:
        """
        Generate sample battery degradation data for demonstration
        
        Args:
            n_cycles: Number of charge/discharge cycles to generate
            
        Returns:
            Synthetic battery data DataFrame
        """
        np.random.seed(42)
        
        # Base parameters
        nominal_capacity = 100.0  # Ah
        nominal_voltage = 3.7      # V
        nominal_temp = 25.0        # °C
        
        data = []
        for cycle in range(n_cycles):
            # Simulate capacity degradation (typical: 80% capacity at 1000 cycles)
            degradation_factor = 1.0 - (0.2 * cycle / n_cycles)
            capacity = nominal_capacity * degradation_factor
            
            # Add some noise
            noise = np.random.normal(0, 0.02)
            capacity *= (1 + noise)
            
            # Voltage variation during cycles
            voltage = nominal_voltage + np.random.normal(0, 0.1)
            
            # Current variation
            current = 1.0 + np.random.normal(0, 0.1)
            
            # Temperature variation
            temperature = nominal_temp + np.random.normal(0, 2)
            
            # Charge/discharge times (vary with degradation)
            discharge_time = 2.0 * (1 + (1 - degradation_factor) * 0.5) + np.random.normal(0, 0.1)
            charge_time = 1.8 * (1 + (1 - degradation_factor) * 0.3) + np.random.normal(0, 0.1)
            
            data.append({
                'cycle_number': cycle + 1,
                'voltage': voltage,
                'current': current,
                'temperature': temperature,
                'capacity': capacity,
                'discharge_time': discharge_time,
                'charge_time': charge_time,
                'soh': capacity / nominal_capacity  # State of Health
            })
        
        df = pd.DataFrame(data)
        print(f"Generated sample data with {n_cycles} cycles")
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess the raw battery data
        
        Args:
            df: Raw battery data DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle missing values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Remove outliers (using IQR method)
        for col in self.feature_columns:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df[col] = df[col].clip(lower_bound, upper_bound)
        
        # Calculate SoH if not present
        if 'soh' not in df.columns and 'capacity' in df.columns:
            nominal_capacity = df['capacity'].iloc[0]
            df['soh'] = df['capacity'] / nominal_capacity
        
        return df
    
    def create_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create LSTM sequences from time series data
        
        Args:
            df: Preprocessed battery data DataFrame
            
        Returns:
            Tuple of (X, y) where X is sequences of features and y is target SoH
        """
        # Select features
        features = df[self.feature_columns].values
        targets = df['soh'].values.reshape(-1, 1)
        
        # Normalize features
        features_scaled = self.scaler.fit_transform(features)
        targets_scaled = self.target_scaler.fit_transform(targets)
        
        # Create sequences
        X, y = [], []
        for i in range(len(features_scaled) - self.sequence_length):
            X.append(features_scaled[i:i + self.sequence_length])
            y.append(targets_scaled[i + self.sequence_length])
        
        return np.array(X), np.array(y)
    
    def split_data(self, X: np.ndarray, y: np.ndarray, 
                   test_size: float = 0.2, val_size: float = 0.1) -> Tuple:
        """
        Split data into train, validation, and test sets
        
        Args:
            X: Feature sequences
            y: Target values
            test_size: Proportion of data for testing
            val_size: Proportion of training data for validation
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=False
        )
        
        # Second split: separate train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size/(1-test_size), 
            random_state=42, shuffle=False
        )
        
        print(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def inverse_transform_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """
        Inverse transform scaled predictions back to original scale
        
        Args:
            predictions: Scaled predictions
            
        Returns:
            Predictions in original scale
        """
        return self.target_scaler.inverse_transform(predictions)
    
    def get_data_statistics(self, df: pd.DataFrame) -> dict:
        """
        Get statistical summary of the battery data
        
        Args:
            df: Battery data DataFrame
            
        Returns:
            Dictionary containing data statistics
        """
        stats = {}
        for col in self.feature_columns + ['soh']:
            if col in df.columns:
                stats[col] = {
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'median': df[col].median()
                }
        return stats


if __name__ == "__main__":
    # Example usage
    processor = BatteryDataProcessor(sequence_length=50)
    
    # Load or generate data
    df = processor.load_data("battery_data.csv")
    df = processor.clean_data(df)
    
    # Create sequences
    X, y = processor.create_sequences(df)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data(X, y)
    
    # Print statistics
    stats = processor.get_data_statistics(df)
    print("\nData Statistics:")
    for feature, stat in stats.items():
        print(f"{feature}: Mean={stat['mean']:.3f}, Std={stat['std']:.3f}")
