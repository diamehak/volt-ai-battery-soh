"""
Train Dual LSTM Model on NASA Dataset
Battery SoH and RUL Prediction System
"""

import pandas as pd
import numpy as np
import sys
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def load_nasa_dataset():
    """Load and preprocess NASA dataset"""
    print("Loading NASA dataset...")
    df = pd.read_csv('nasa_dataset.csv')
    print(f"Loaded {len(df)} records")
    
    return df

def preprocess_dual_data(df, sequence_length=50):
    """Preprocess NASA dataset for dual LSTM training (SoH and RUL)"""
    print("Preprocessing NASA dataset for dual prediction...")
    
    # Select features for training
    features = ['voltage', 'temperature', 'capacity', 'cycle']
    targets = ['soh', 'rul']
    
    # Convert soh from percentage to decimal (0-1 range)
    df['soh'] = df['soh'] / 100.0
    
    # Sort by cycle to ensure proper sequence
    df = df.sort_values('cycle').reset_index(drop=True)
    
    # Normalize features
    scaler_X = MinMaxScaler()
    scaler_y_soh = MinMaxScaler()
    scaler_y_rul = MinMaxScaler()
    
    X = scaler_X.fit_transform(df[features].values)
    y_soh = scaler_y_soh.fit_transform(df[['soh']].values)
    y_rul = scaler_y_rul.fit_transform(df[['rul']].values)
    
    # Create sequences with proper overlap
    X_seq = []
    y_soh_seq = []
    y_rul_seq = []
    
    for i in range(len(df) - sequence_length):
        X_seq.append(X[i:i+sequence_length])
        y_soh_seq.append(y_soh[i+sequence_length])
        y_rul_seq.append(y_rul[i+sequence_length])
    
    X_seq = np.array(X_seq)
    y_soh_seq = np.array(y_soh_seq)
    y_rul_seq = np.array(y_rul_seq)
    
    print(f"Created {len(X_seq)} sequences of length {sequence_length}")
    print(f"SoH range: {df['soh'].min():.3f} - {df['soh'].max():.3f}")
    print(f"RUL range: {df['rul'].min()} - {df['rul'].max()}")
    
    return X_seq, y_soh_seq, y_rul_seq, scaler_X, scaler_y_soh, scaler_y_rul

def build_dual_lstm_model(sequence_length, n_features):
    """Build dual LSTM model for SoH and RUL prediction"""
    print("Building dual LSTM model...")
    
    # Input layer
    inputs = Input(shape=(sequence_length, n_features))
    
    # Shared LSTM layers
    x = LSTM(128, return_sequences=True)(inputs)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    
    x = LSTM(64, return_sequences=True)(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    
    x = LSTM(32, return_sequences=False)(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    
    # Shared dense layers
    shared = Dense(64, activation='relu')(x)
    shared = Dropout(0.3)(shared)
    shared = BatchNormalization()(shared)
    
    shared = Dense(32, activation='relu')(shared)
    shared = Dropout(0.3)(shared)
    shared = BatchNormalization()(shared)
    
    # Separate output branches
    # SoH output (0-1 range)
    soh_output = Dense(1, activation='sigmoid', name='soh_output')(shared)
    
    # RUL output (positive integers)
    rul_output = Dense(1, activation='relu', name='rul_output')(shared)
    
    # Create model
    model = Model(inputs=inputs, outputs=[soh_output, rul_output])
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss={
            'soh_output': 'mse',
            'rul_output': 'mse'
        },
        loss_weights={
            'soh_output': 1.0,
            'rul_output': 0.5
        },
        metrics={
            'soh_output': ['mae'],
            'rul_output': ['mae']
        }
    )
    
    return model

def train_dual_model():
    """Train dual LSTM model"""
    print("Training dual LSTM model...")
    
    # Load and preprocess data
    df = load_nasa_dataset()
    X_seq, y_soh_seq, y_rul_seq, scaler_X, scaler_y_soh, scaler_y_rul = preprocess_dual_data(df)
    
    # Split data
    X_train, X_temp, y_soh_train, y_soh_temp, y_rul_train, y_rul_temp = train_test_split(
        X_seq, y_soh_seq, y_rul_seq, test_size=0.3, random_state=42
    )
    
    X_val, X_test, y_soh_val, y_soh_test, y_rul_val, y_rul_test = train_test_split(
        X_temp, y_soh_temp, y_rul_temp, test_size=0.5, random_state=42
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Build model
    model = build_dual_lstm_model(sequence_length=50, n_features=4)
    model.summary()
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )
    
    # Train model
    history = model.fit(
        X_train, {'soh_output': y_soh_train, 'rul_output': y_rul_train},
        validation_data=(X_val, {'soh_output': y_soh_val, 'rul_output': y_rul_val}),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Evaluate model
    test_results = model.evaluate(
        X_test, {'soh_output': y_soh_test, 'rul_output': y_rul_test},
        verbose=1
    )
    
    print(f"Test Results: {dict(zip(model.metrics_names, test_results))}")
    
    # Save model and scalers
    model.save('nasa_dual_lstm_model.h5')
    scaler_X_filename = 'scaler_X_dual.pkl'
    scaler_y_soh_filename = 'scaler_y_soh.pkl'
    scaler_y_rul_filename = 'scaler_y_rul.pkl'
    
    joblib.dump(scaler_X, scaler_X_filename)
    joblib.dump(scaler_y_soh, scaler_y_soh_filename)
    joblib.dump(scaler_y_rul, scaler_y_rul_filename)
    
    print(f"✅ Model saved as 'nasa_dual_lstm_model.h5'")
    print(f"✅ Scalers saved as '{scaler_X_filename}', '{scaler_y_soh_filename}', '{scaler_y_rul_filename}'")
    
    return model, history

if __name__ == "__main__":
    model, history = train_dual_model()
