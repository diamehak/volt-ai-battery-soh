"""
Simple Battery SoH Prediction Demo
Works without TensorFlow for demonstration purposes
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class SimpleBatterySoHDemo:
    """
    Simple demo for Battery SoH prediction using scikit-learn
    """
    
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        
    def generate_battery_data(self, n_cycles=1000):
        """
        Generate synthetic battery degradation data
        """
        np.random.seed(42)
        
        nominal_capacity = 100.0
        nominal_voltage = 3.7
        nominal_temp = 25.0
        
        data = []
        for cycle in range(n_cycles):
            # Simulate capacity degradation
            degradation_factor = 1.0 - (0.2 * cycle / n_cycles)
            capacity = nominal_capacity * degradation_factor
            capacity *= (1 + np.random.normal(0, 0.02))
            
            # Other parameters
            voltage = nominal_voltage + np.random.normal(0, 0.1)
            current = 1.0 + np.random.normal(0, 0.1)
            temperature = nominal_temp + np.random.normal(0, 2)
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
                'soh': capacity / nominal_capacity
            })
        
        return pd.DataFrame(data)
    
    def prepare_data(self, df):
        """
        Prepare data for training
        """
        features = ['voltage', 'current', 'temperature', 'capacity', 
                   'cycle_number', 'discharge_time', 'charge_time']
        target = 'soh'
        
        X = df[features].values
        y = df[target].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train, y_train):
        """
        Train the model
        """
        print("Training Random Forest model...")
        self.model.fit(X_train, y_train)
        print("Training completed!")
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the model
        """
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        print("\nEvaluation Metrics:")
        print(f"MSE:  {mse:.4f}")
        print(f"MAE:  {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²:   {r2:.4f}")
        
        return y_pred, {'mse': mse, 'mae': mae, 'rmse': rmse, 'r2': r2}
    
    def plot_results(self, y_true, y_pred):
        """
        Plot prediction results
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Time series plot
        ax1.plot(y_true, label='True SoH', color='blue', linewidth=2, alpha=0.7)
        ax1.plot(y_pred, label='Predicted SoH', color='orange', linewidth=2, alpha=0.8)
        ax1.set_xlabel('Sample Index')
        ax1.set_ylabel('State of Health')
        ax1.set_title('SoH Prediction Results')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Scatter plot
        ax2.scatter(y_true, y_pred, alpha=0.6, color='green')
        ax2.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                'r--', lw=2, label='Perfect Prediction')
        ax2.set_xlabel('True SoH')
        ax2.set_ylabel('Predicted SoH')
        ax2.set_title('True vs Predicted SoH')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/simple_demo_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def predict_rul(self, soh_values, threshold=0.8):
        """
        Predict Remaining Useful Life
        """
        rul_predictions = []
        for soh in soh_values:
            if soh < threshold:
                rul = 0
            else:
                # Simple linear extrapolation
                degradation_rate = 0.0002  # Assumed degradation rate
                rul = int((soh - threshold) / degradation_rate)
            rul_predictions.append(max(0, rul))
        
        return np.array(rul_predictions)
    
    def run_demo(self):
        """
        Run the complete demo
        """
        print("=" * 60)
        print("Battery SoH Prediction Demo (Simplified Version)")
        print("=" * 60)
        
        # Generate data
        print("Generating battery data...")
        df = self.generate_battery_data(1000)
        print(f"Generated {len(df)} cycles of battery data")
        
        # Show sample data
        print("\nSample data:")
        print(df.head())
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(df)
        print(f"\nData split - Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Train model
        self.train_model(X_train, y_train)
        
        # Evaluate model
        y_pred, metrics = self.evaluate_model(X_test, y_test)
        
        # Plot results
        self.plot_results(y_test, y_pred)
        
        # RUL prediction
        rul_predictions = self.predict_rul(y_pred)
        print(f"\nRUL Predictions:")
        print(f"Mean RUL: {np.mean(rul_predictions):.1f} cycles")
        print(f"Min RUL: {np.min(rul_predictions):.1f} cycles")
        print(f"Max RUL: {np.max(rul_predictions):.1f} cycles")
        
        # Feature importance
        feature_names = ['voltage', 'current', 'temperature', 'capacity', 
                        'cycle_number', 'discharge_time', 'charge_time']
        importances = self.model.feature_importances_
        
        print("\nFeature Importance:")
        for name, importance in zip(feature_names, importances):
            print(f"{name}: {importance:.4f}")
        
        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("Results saved to: results/simple_demo_results.png")
        print("=" * 60)

if __name__ == "__main__":
    # Create results directory
    import os
    os.makedirs('results', exist_ok=True)
    
    # Run demo
    demo = SimpleBatterySoHDemo()
    demo.run_demo()
