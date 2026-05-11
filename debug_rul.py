"""
Debug RUL Prediction Issue
"""

import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

def debug_rul():
    print("🔍 Debugging RUL Prediction...")
    
    # Load dataset
    df = pd.read_csv('nasa_dataset.csv')
    print(f"Dataset RUL range: {df['rul'].min()} - {df['rul'].max()}")
    print(f"Dataset SoH range: {df['soh'].min()}% - {df['soh'].max()}%")
    
    # Load model and scalers
    model = tf.keras.models.load_model('nasa_dual_lstm_model.h5')
    scaler_X = joblib.load('scaler_X_dual.pkl')
    scaler_y_soh = joblib.load('scaler_y_soh.pkl')
    scaler_y_rul = joblib.load('scaler_y_rul.pkl')
    
    # Test with actual data from dataset
    test_cycle = 100
    actual_data = df[df['cycle'] == test_cycle].iloc[0]
    
    print(f"\nTesting with actual cycle {test_cycle} data:")
    print(f"Actual SoH: {actual_data['soh']}%")
    print(f"Actual RUL: {actual_data['rul']}")
    
    # Prepare input using actual data
    recent_data = df.tail(50)[['voltage', 'temperature', 'capacity', 'cycle']].values
    recent_data[-1] = [actual_data['voltage'], actual_data['temperature'], actual_data['capacity'], test_cycle]
    
    # Scale and predict
    X_scaled = scaler_X.transform(recent_data)
    X_input = X_scaled.reshape(1, 50, 4)
    
    # Get raw predictions
    y_pred_soh_scaled, y_pred_rul_scaled = model.predict(X_input, verbose=0)
    
    print(f"\nRaw scaled predictions:")
    print(f"SoH scaled: {y_pred_soh_scaled[0][0]}")
    print(f"RUL scaled: {y_pred_rul_scaled[0][0]}")
    
    # Inverse transform
    soh_prediction = float(scaler_y_soh.inverse_transform(y_pred_soh_scaled)[0][0])
    rul_prediction = float(scaler_y_rul.inverse_transform(y_pred_rul_scaled)[0][0])
    
    print(f"\nInverse transformed predictions:")
    print(f"SoH: {soh_prediction} ({soh_prediction*100:.2f}%)")
    print(f"RUL: {rul_prediction}")
    
    # Check scaler ranges
    print(f"\nScaler ranges:")
    print(f"SoH scaler data range: [{scaler_y_soh.data_min_[0]:.3f}, {scaler_y_soh.data_max_[0]:.3f}]")
    print(f"RUL scaler data range: [{scaler_y_rul.data_min_[0]:.3f}, {scaler_y_rul.data_max_[0]:.3f}]")
    
    # Test with different cycles
    print(f"\nTesting different cycles:")
    for cycle in [10, 50, 100, 150]:
        cycle_data = df[df['cycle'] == cycle]
        if len(cycle_data) > 0:
            data = cycle_data.iloc[0]
            recent_data = df.tail(50)[['voltage', 'temperature', 'capacity', 'cycle']].values
            recent_data[-1] = [data['voltage'], data['temperature'], data['capacity'], cycle]
            
            X_scaled = scaler_X.transform(recent_data)
            X_input = X_scaled.reshape(1, 50, 4)
            
            y_pred_soh_scaled, y_pred_rul_scaled = model.predict(X_input, verbose=0)
            soh_pred = float(scaler_y_soh.inverse_transform(y_pred_soh_scaled)[0][0])
            rul_pred = float(scaler_y_rul.inverse_transform(y_pred_rul_scaled)[0][0])
            
            print(f"Cycle {cycle}: SoH={soh_pred*100:.1f}%, RUL={rul_pred:.1f} (Actual: SoH={data['soh']:.1f}%, RUL={data['rul']})")

if __name__ == "__main__":
    debug_rul()
