"""
Test Script for SoH and RUL Predictions
"""

import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

def test_predictions():
    print("🧪 Testing Improved Prediction System...")
    
    # Load dataset
    try:
        df = pd.read_csv('nasa_dataset.csv')
        print(f"✅ Loaded dataset: {len(df)} records")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Load dual model
    try:
        model = tf.keras.models.load_model('nasa_dual_lstm_model.h5')
        scaler_X = joblib.load('scaler_X_dual.pkl')
        scaler_y_soh = joblib.load('scaler_y_soh.pkl')
        scaler_y_rul = joblib.load('scaler_y_rul.pkl')
        print("✅ Loaded dual LSTM model")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Test with different scenarios
    test_cases = [
        {"voltage": 3.53, "temperature": 32.5, "capacity": 1.86, "cycles": 10, "name": "Early Cycle"},
        {"voltage": 3.50, "temperature": 33.0, "capacity": 1.65, "cycles": 50, "name": "Mid Cycle"},
        {"voltage": 3.48, "temperature": 33.5, "capacity": 1.45, "cycles": 100, "name": "Late Cycle"},
        {"voltage": 3.47, "temperature": 34.0, "capacity": 1.35, "cycles": 150, "name": "Very Late Cycle"},
    ]
    
    print("\n📊 Prediction Results:")
    print("-" * 80)
    
    for test in test_cases:
        # Prepare input
        recent_data = df.tail(50)[['voltage', 'temperature', 'capacity', 'cycle']].values
        recent_data[-1] = [test["voltage"], test["temperature"], test["capacity"], test["cycles"]]
        
        # Scale and predict
        X_scaled = scaler_X.transform(recent_data)
        X_input = X_scaled.reshape(1, 50, 4)
        
        y_pred_soh_scaled, y_pred_rul_scaled = model.predict(X_input, verbose=0)
        
        soh_prediction = float(scaler_y_soh.inverse_transform(y_pred_soh_scaled)[0][0])
        rul_prediction = int(scaler_y_rul.inverse_transform(y_pred_rul_scaled)[0][0])
        
        # Get actual values from dataset for comparison
        actual_data = df[df['cycle'] == test["cycles"]]
        if len(actual_data) > 0:
            actual_soh = actual_data['soh'].values[0]
            actual_rul = actual_data['rul'].values[0]
            soh_error = abs(soh_prediction * 100 - actual_soh)
            rul_error = abs(rul_prediction - actual_rul)
        else:
            actual_soh = "N/A"
            actual_rul = "N/A"
            soh_error = "N/A"
            rul_error = "N/A"
        
        print(f"\n🔋 {test['name']} (Cycle {test['cycles']}):")
        print(f"   Predicted SoH: {soh_prediction*100:.2f}% | Actual: {actual_soh}% | Error: {soh_error}")
        print(f"   Predicted RUL: {rul_prediction} cycles | Actual: {actual_rul} | Error: {rul_error}")
        
        # Check if predictions are good
        if isinstance(soh_error, (int, float)) and soh_error < 2.0:
            print(f"   ✅ SoH prediction is accurate (error < 2%)")
        elif isinstance(soh_error, (int, float)):
            print(f"   ⚠️  SoH prediction has moderate error (error: {soh_error:.2f}%)")
        else:
            print(f"   ❓ Cannot verify SoH accuracy")
            
        if isinstance(rul_error, (int, float)) and rul_error < 10:
            print(f"   ✅ RUL prediction is accurate (error < 10 cycles)")
        elif isinstance(rul_error, (int, float)):
            print(f"   ⚠️  RUL prediction has moderate error (error: {rul_error} cycles)")
        else:
            print(f"   ❓ Cannot verify RUL accuracy")
    
    print("\n" + "=" * 80)
    print("🎯 Summary:")
    print("• If predictions show variable SoH values (not constant 79.3-79.4) → ✅ GOOD")
    print("• If RUL predictions are reasonable and not always 0 → ✅ GOOD") 
    print("• If error rates are low (<2% for SoH, <10 for RUL) → ✅ EXCELLENT")
    print("• If predictions vary based on input parameters → ✅ WORKING CORRECTLY")

if __name__ == "__main__":
    test_predictions()
