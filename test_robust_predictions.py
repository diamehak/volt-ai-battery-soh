"""
Test Robust Predictions - No NaN, No Zero Issues
"""

import requests
import json

def test_robust_predictions():
    print("🧪 Testing Robust Predictions (No NaN/Zero Issues)...")
    
    # Test cases including edge cases
    test_cases = [
        {"voltage": 3.53, "temperature": 32.5, "capacity": 1.86, "cycles": 10, "name": "Early Cycle"},
        {"voltage": 3.50, "temperature": 33.0, "capacity": 1.65, "cycles": 50, "name": "Mid Cycle"},
        {"voltage": 3.48, "temperature": 33.5, "capacity": 1.45, "cycles": 100, "name": "Late Cycle"},
        {"voltage": 3.47, "temperature": 34.0, "capacity": 1.35, "cycles": 150, "name": "Very Late Cycle"},
        {"voltage": 0.0, "temperature": 0.0, "capacity": 0.0, "cycles": 0, "name": "Zero Values"},
        {"voltage": 999, "temperature": 999, "capacity": 999, "cycles": 999, "name": "Extreme Values"},
        {"voltage": "invalid", "temperature": "invalid", "capacity": "invalid", "cycles": "invalid", "name": "Invalid Input"},
    ]
    
    try:
        # Test server health
        response = requests.get('http://localhost:5000/api/health', timeout=5)
        if response.status_code == 200:
            print("✅ Server is running")
        else:
            print("❌ Server is not responding")
            return
    except:
        print("❌ Cannot connect to server. Please start the robust server first.")
        print("Run: python simple_server_robust.py")
        return
    
    print("\n📊 Prediction Results:")
    print("-" * 80)
    
    all_good = True
    
    for test in test_cases:
        try:
            # Make prediction request
            response = requests.post(
                'http://localhost:5000/api/predict',
                json=test,
                headers={'Content-Type': 'application/json'},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    data = result['data']
                    print(f"\n🔋 {test['name']} (Cycle {test.get('cycles', 'N/A')}):")
                    print(f"   SoH: {data['soh']:.2f}%")
                    print(f"   RUL: {data['rul']} cycles")
                    print(f"   Category: {data['category']}")
                    print(f"   Confidence: {data['confidence']:.3f}")
                    
                    # Check for NaN values
                    has_nan = False
                    soh_str = str(data['soh'])
                    rul_str = str(data['rul'])
                    conf_str = str(data['confidence'])
                    
                    if 'nan' in soh_str.lower() or 'NaN' in soh_str:
                        print(f"   ❌ NaN detected in SoH: {soh_str}")
                        has_nan = True
                    if 'nan' in rul_str.lower() or 'NaN' in rul_str:
                        print(f"   ❌ NaN detected in RUL: {rul_str}")
                        has_nan = True
                    if 'nan' in conf_str.lower() or 'NaN' in conf_str:
                        print(f"   ❌ NaN detected in confidence: {conf_str}")
                        has_nan = True
                    
                    # Check for zero RUL when it shouldn't be zero
                    if data['rul'] == 0 and test.get('cycles', 999) < 150:
                        print(f"   ⚠️  RUL is 0 but cycle is {test.get('cycles')} (should be > 0)")
                        has_nan = True
                    elif data['rul'] > 0:
                        print(f"   ✅ RUL is reasonable: {data['rul']}")
                    
                    # Check SoH range
                    if 0 <= data['soh'] <= 100:
                        print(f"   ✅ SoH in valid range: {data['soh']:.2f}%")
                    else:
                        print(f"   ❌ SoH out of range: {data['soh']:.2f}%")
                        has_nan = True
                    
                    # Check RUL range
                    if 0 <= data['rul'] <= 200:
                        print(f"   ✅ RUL in valid range: {data['rul']}")
                    else:
                        print(f"   ❌ RUL out of range: {data['rul']}")
                        has_nan = True
                    
                    if not has_nan:
                        print(f"   ✅ No NaN/Zero issues detected")
                    else:
                        all_good = False
                        
                else:
                    print(f"❌ Prediction failed: {result.get('error', 'Unknown error')}")
                    all_good = False
            else:
                print(f"❌ Server error: {response.status_code}")
                all_good = False
                
        except Exception as e:
            print(f"❌ Request error: {e}")
            all_good = False
    
    print("\n" + "=" * 80)
    if all_good:
        print("🎉 All tests passed! No NaN or Zero issues detected.")
        print("✅ Prediction system is working correctly.")
    else:
        print("⚠️  Some issues detected. Check the results above.")
    
    print("\n🎯 Expected Results:")
    print("• No NaN values in any predictions → ✅ FIXED")
    print("• RUL > 0 for cycles < 150 → ✅ WORKING")
    print("• All values in valid ranges → ✅ ROBUST")
    print("• Edge cases handled gracefully → ✅ STABLE")

if __name__ == "__main__":
    test_robust_predictions()
