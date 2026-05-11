"""
Train LSTM Model on NASA Dataset
Battery SoH Prediction System
"""

import pandas as pd
import numpy as np
import sys
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import LSTM model
from lstm_model import BatterySoHPredictor

def load_nasa_dataset():
    """Load and preprocess NASA dataset"""
    print("Loading NASA dataset...")
    df = pd.read_csv('nasa_dataset.csv')
    print(f"Loaded {len(df)} records")
    
    return df

def preprocess_nasa_data(df, sequence_length=50):
    """Preprocess NASA dataset for LSTM training"""
    print("Preprocessing NASA dataset...")
    
    # Select features for training
    features = ['voltage', 'temperature', 'capacity', 'cycle']
    target = 'soh'
    
    # Convert soh from percentage to decimal (0-1 range)
    df[target] = df[target] / 100.0
    
    # Sort by cycle to ensure proper sequence
    df = df.sort_values('cycle').reset_index(drop=True)
    
    # Normalize features
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X = scaler_X.fit_transform(df[features].values)
    y = scaler_y.fit_transform(df[[target]].values)
    
    # Create sequences with proper overlap
    X_seq = []
    y_seq = []
    
    for i in range(len(df) - sequence_length):
        X_seq.append(X[i:i+sequence_length])
        y_seq.append(y[i+sequence_length])
    
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    print(f"Created {len(X_seq)} sequences of length {sequence_length}")
    print(f"SoH range: {df[target].min():.3f} - {df[target].max():.3f}")
    
    return X_seq, y_seq, scaler_X, scaler_y

def train_lstm_model(X, y):
    """Train LSTM model on NASA dataset"""
    print("Training LSTM model...")
    
    # Split data into train, validation, and test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Initialize model with correct sequence length and features
    model = BatterySoHPredictor(sequence_length=X.shape[1], n_features=X.shape[2])
    
    # Build model (without input_shape parameter)
    model.build_model(lstm_units=[128, 64, 32])
    
    # Train model with validation data
    history = model.train(X_train, y_train, X_val, y_val, epochs=50, batch_size=32)
    
    # Evaluate model
    metrics = model.evaluate(X_test, y_test)
    
    print(f"Model trained successfully!")
    print(f"Test MSE: {metrics['mse']:.6f}")
    print(f"Test MAE: {metrics['mae']:.6f}")
    if 'r2' in metrics:
        print(f"Test R2: {metrics['r2']:.6f}")
    
    return model, history, metrics

def save_model_and_scalers(model, scaler_X, scaler_y):
    """Save trained model and scalers"""
    print("Saving model and scalers...")
    
    # Save model
    model.save_model('nasa_lstm_model.h5')
    
    # Save scalers
    joblib.dump(scaler_X, 'scaler_X.pkl')
    joblib.dump(scaler_y, 'scaler_y.pkl')
    
    print("Model and scalers saved successfully!")

def main():
    """Main training function"""
    try:
        # Load NASA dataset
        df = load_nasa_dataset()
        
        # Preprocess data
        X, y, scaler_X, scaler_y = preprocess_nasa_data(df, sequence_length=50)
        
        # Train model
        model, history, metrics = train_lstm_model(X, y)
        
        # Save model and scalers
        save_model_and_scalers(model, scaler_X, scaler_y)
        
        print("\n✅ Training completed successfully!")
        print(f"Final Test MSE: {metrics['mse']:.6f}")
        print(f"Final Test MAE: {metrics['mae']:.6f}")
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
