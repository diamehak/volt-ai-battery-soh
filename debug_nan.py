"""
Debug NaN Values in Predictions
"""

import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

def debug_nan():
    print("🔍 Debugging NaN Values...")
    
    # Load dataset
    try:
        df = pd.read_csv('nasa_dataset.csv')
        print(f"✅ Dataset loaded: {len(df)} records")
        print(f"Dataset columns: {df.columns.tolist()}")
        print(f"Dataset info:")
        print(df[['voltage', 'temperature', 'capacity', 'cycle', 'soh', 'rul']].describe())
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Check for NaN values in dataset
    print(f"\n🔍 Checking for NaN values in dataset:")
    nan_counts = df.isnull().sum()
    print(nan_counts)
    
    # Load model and scalers
    try:
        model = tf.keras.models.load_model('nasa_dual_lstm_model.h5')
        scaler_X = joblib.load('scaler_X_dual.pkl')
        scaler_y_soh = joblib.load('scaler_y_soh.pkl')
        scaler_y_rul = joblib.load('scaler_y_rul.pkl')
        print("✅ Model and scalers loaded")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Test with safe data
    print(f"\n🧪 Testing with safe data...")
    
    # Use actual data from dataset
    test_cycle = 100
    actual_data = df[df['cycle'] == test_cycle]
    
    if len(actual_data) > 0:
        data = actual_data.iloc[0]
        print(f"Using actual cycle {test_cycle} data:")
        print(f"Voltage: {data['voltage']}, Temp: {data['temperature']}, Capacity: {data['capacity']}")
        
        # Prepare input
        recent_data = df.tail(50)[['voltage', 'temperature', 'capacity', 'cycle']].copy()
        
        # Check for NaN in recent_data
        print(f"Recent data shape: {recent_data.shape}")
        print(f"Recent data NaN count: {recent_data.isnull().sum().sum()}")
        
        # Replace last row with test data
        recent_data.iloc[-1] = [data['voltage'], data['temperature'], data['capacity'], test_cycle]
        
        # Check again for NaN
        print(f"After replacement NaN count: {recent_data.isnull().sum().sum()}")
        
        # Convert to numpy
        X_input_data = recent_data.values
        print(f"Input data shape: {X_input_data.shape}")
        print(f"Input data NaN count: {np.isnan(X_input_data).sum()}")
        
        # Scale
        try:
            X_scaled = scaler_X.transform(X_input_data)
            print(f"Scaled data NaN count: {np.isnan(X_scaled).sum()}")
            
            # Reshape
            X_input = X_scaled.reshape(1, 50, 4)
            print(f"Final input shape: {X_input.shape}")
            
            # Predict
            y_pred_soh_scaled, y_pred_rul_scaled = model.predict(X_input, verbose=0)
            print(f"Raw predictions - SoH: {y_pred_soh_scaled[0][0]}, RUL: {y_pred_rul_scaled[0][0]}")
            
            # Check for NaN in predictions
            if np.isnan(y_pred_soh_scaled[0][0]) or np.isnan(y_pred_rul_scaled[0][0]):
                print("❌ NaN detected in model predictions!")
            else:
                # Inverse transform
                soh_prediction = float(scaler_y_soh.inverse_transform(y_pred_soh_scaled)[0][0])
                rul_prediction = float(scaler_y_rul.inverse_transform(y_pred_rul_scaled)[0][0])
                
                print(f"Final predictions - SoH: {soh_prediction}, RUL: {rul_prediction}")
                
                if np.isnan(soh_prediction) or np.isnan(rul_prediction):
                    print("❌ NaN detected after inverse transform!")
                else:
                    print("✅ No NaN values detected")
                    
        except Exception as e:
            print(f"❌ Error during prediction: {e}")
    else:
        print(f"❌ No data found for cycle {test_cycle}")

if __name__ == "__main__":
    debug_nan()
