"""
Test the Fixed Server Predictions
"""

import requests
import json

def test_fixed_predictions():
    print("🧪 Testing Fixed Server Predictions...")
    
    # Test cases
    test_cases = [
        {"voltage": 3.53, "temperature": 32.5, "capacity": 1.86, "cycles": 10, "name": "Early Cycle"},
        {"voltage": 3.50, "temperature": 33.0, "capacity": 1.65, "cycles": 50, "name": "Mid Cycle"},
        {"voltage": 3.48, "temperature": 33.5, "capacity": 1.45, "cycles": 100, "name": "Late Cycle"},
        {"voltage": 3.47, "temperature": 34.0, "capacity": 1.35, "cycles": 150, "name": "Very Late Cycle"},
    ]
    
    try:
        # Test server health
        response = requests.get('http://localhost:5000/api/health')
        if response.status_code == 200:
            print("✅ Server is running")
        else:
            print("❌ Server is not responding")
            return
    except:
        print("❌ Cannot connect to server. Please start the server first.")
        print("Run: python simple_server_fixed.py")
        return
    
    print("\n📊 Prediction Results:")
    print("-" * 80)
    
    for test in test_cases:
        try:
            # Make prediction request
            response = requests.post(
                'http://localhost:5000/api/predict',
                json=test,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                result = response.json()
                if result['success']:
                    data = result['data']
                    print(f"\n🔋 {test['name']} (Cycle {test['cycles']}):")
                    print(f"   SoH: {data['soh']:.2f}%")
                    print(f"   RUL: {data['rul']} cycles")
                    print(f"   Category: {data['category']}")
                    print(f"   Confidence: {data['confidence']:.3f}")
                    
                    # Check if predictions are good
                    if data['soh'] > 70 and data['soh'] < 100:
                        print(f"   ✅ SoH prediction is reasonable")
                    else:
                        print(f"   ⚠️  SoH prediction may be unusual")
                    
                    if data['rul'] > 0 and data['rul'] <= 167:
                        print(f"   ✅ RUL prediction is reasonable")
                    else:
                        print(f"   ⚠️  RUL prediction may be unusual")
                        
                    # Check for variability
                    if test['cycles'] == 10 and data['soh'] > 90:
                        print(f"   ✅ Early cycle shows high SoH as expected")
                    elif test['cycles'] == 150 and data['soh'] < 80:
                        print(f"   ✅ Late cycle shows lower SoH as expected")
                        
                else:
                    print(f"❌ Prediction failed: {result.get('error', 'Unknown error')}")
            else:
                print(f"❌ Server error: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Request error: {e}")
    
    print("\n" + "=" * 80)
    print("🎯 Summary:")
    print("• If SoH values vary by cycle (higher for early cycles) → ✅ GOOD")
    print("• If RUL values are reasonable (not always 0) → ✅ GOOD")
    print("• If predictions show logical degradation patterns → ✅ EXCELLENT")
    print("• If both SoH and RUL are predicted successfully → ✅ WORKING CORRECTLY")

if __name__ == "__main__":
    test_fixed_predictions()
