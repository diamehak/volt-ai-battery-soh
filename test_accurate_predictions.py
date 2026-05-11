"""
Test Accurate Predictions
"""

import requests
import json

def test_accurate_predictions():
    print("🧪 Testing Accurate Predictions...")
    
    # Test cases with expected values from dataset
    test_cases = [
        {"voltage": 3.53, "temperature": 32.5, "capacity": 1.86, "cycles": 10, "name": "Early Cycle", "expected_soh": 98.28},
        {"voltage": 3.50, "temperature": 33.0, "capacity": 1.65, "cycles": 50, "name": "Mid Cycle", "expected_soh": 95.08},
        {"voltage": 3.48, "temperature": 33.5, "capacity": 1.45, "cycles": 100, "name": "Late Cycle", "expected_soh": 79.96},
        {"voltage": 3.47, "temperature": 34.0, "capacity": 1.35, "cycles": 150, "name": "Very Late Cycle", "expected_soh": 71.27},
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
        print("❌ Cannot connect to server. Please start the accurate server first.")
        print("Run: python simple_server_accurate.py")
        return
    
    print("\n📊 Prediction Results:")
    print("-" * 80)
    
    total_error = 0
    good_predictions = 0
    
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
                    predicted_soh = data['soh']
                    expected_soh = test['expected_soh']
                    error = abs(predicted_soh - expected_soh)
                    total_error += error
                    
                    print(f"\n🔋 {test['name']} (Cycle {test['cycles']}):")
                    print(f"   Predicted SoH: {predicted_soh:.2f}%")
                    print(f"   Expected SoH: {expected_soh:.2f}%")
                    print(f"   Error: {error:.2f}%")
                    print(f"   RUL: {data['rul']} cycles")
                    print(f"   Category: {data['category']}")
                    print(f"   Confidence: {data['confidence']:.3f}")
                    
                    # Check accuracy
                    if error < 2.0:
                        print(f"   ✅ Excellent prediction (error < 2%)")
                        good_predictions += 1
                    elif error < 5.0:
                        print(f"   ✅ Good prediction (error < 5%)")
                        good_predictions += 1
                    elif error < 10.0:
                        print(f"   ⚠️  Fair prediction (error < 10%)")
                    else:
                        print(f"   ❌ Poor prediction (error >= 10%)")
                        
                    # Check RUL reasonableness
                    if 0 <= data['rul'] <= 200:
                        print(f"   ✅ RUL in reasonable range")
                    else:
                        print(f"   ⚠️  RUL may be unusual: {data['rul']}")
                        
                else:
                    print(f"❌ Prediction failed: {result.get('error', 'Unknown error')}")
            else:
                print(f"❌ Server error: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Request error: {e}")
    
    print("\n" + "=" * 80)
    avg_error = total_error / len(test_cases) if test_cases else 0
    accuracy = (good_predictions / len(test_cases)) * 100 if test_cases else 0
    
    print(f"🎯 Accuracy Summary:")
    print(f"   Average Error: {avg_error:.2f}%")
    print(f"   Good Predictions: {good_predictions}/{len(test_cases)} ({accuracy:.1f}%)")
    
    if avg_error < 2.0:
        print(f"   🏆 Excellent accuracy!")
    elif avg_error < 5.0:
        print(f"   ✅ Good accuracy")
    elif avg_error < 10.0:
        print(f"   ⚠️  Fair accuracy")
    else:
        print(f"   ❌ Poor accuracy - needs improvement")
    
    print(f"\n🎯 Expected Results:")
    print(f"• Cycle 10: ~98% SoH (early cycle should be high)")
    print(f"• Cycle 50: ~95% SoH (mid cycle still high)")
    print(f"• Cycle 100: ~80% SoH (degradation starting)")
    print(f"• Cycle 150: ~71% SoH (significant degradation)")

if __name__ == "__main__":
    test_accurate_predictions()
