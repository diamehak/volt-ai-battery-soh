# Battery SoH Prediction - Frontend User Guide

## 🚀 Getting Started with the Streamlit Frontend

### Prerequisites
- Python 3.8+ installed
- All dependencies installed: `pip install -r requirements.txt`

### Launching the Application
```bash
# Navigate to project directory
cd "battery-soh-prediction"

# Run Streamlit app
streamlit run app.py

# Or using Python 3.11 if installed
py -3.11 -m streamlit run app.py
```

The application will open in your web browser at `http://localhost:8501`

## 📱 Application Interface Overview

### Navigation Sidebar
The left sidebar provides:
- **🏠 Home**: System overview and quick actions
- **📊 Data Analysis**: Upload and analyze battery data
- **🧠 Model Configuration**: Train and configure prediction models
- **🔮 Prediction**: Make predictions on new data
- **📈 Results**: View detailed results and export data
- **📋 About**: Project information and documentation

### Quick Settings
Adjustable parameters in sidebar:
- **Sequence Length**: Time steps for LSTM (10-100)
- **Confidence Threshold**: Prediction confidence level (0.7-0.95)

## 📊 Data Analysis Page

### Data Sources
1. **📁 Upload CSV File**: Upload your battery cycle data
2. **🔧 Generate Sample Data**: Create synthetic data for testing
3. **📊 Use Existing Data**: Work with pre-loaded datasets

### Expected CSV Format
```csv
cycle_number,voltage,current,temperature,capacity,discharge_time,charge_time,soh
1,3.7,1.0,25.0,100.0,2.0,1.8,1.000
2,3.69,1.01,25.5,99.8,2.01,1.81,0.998
...
```

### Data Analysis Features
- **Statistical Summary**: Mean, std, min, max for all features
- **Data Quality Check**: Missing values and data types
- **Interactive Plots**: Battery parameters over cycles
- **Correlation Matrix**: Feature relationships

## 🧠 Model Configuration Page

### Model Types
1. **🔥 LSTM Neural Network**: Standard multi-layer LSTM
2. **⚡ LSTM with Attention**: Enhanced sequence understanding
3. **🌲 Random Forest**: Fallback for systems without TensorFlow

### LSTM Configuration
- **LSTM Units**: Number of neurons per layer (32-512)
- **Dropout Rate**: Regularization parameter (0.0-0.5)
- **Learning Rate**: Optimization step size (0.0001, 0.001, 0.01)
- **Batch Size**: Training batch size (16, 32, 64, 128)
- **Training Epochs**: Number of training iterations (10-500)
- **Early Stopping Patience**: Prevent overfitting (5-50)

### Training Configuration
- **Test Set Size**: Data for testing (10-40%)
- **Validation Set Size**: Data for validation (10-30%)
- **Sequence Length**: Input sequence length (10-100)
- **Feature Selection**: Choose which features to use

### Training Process
1. Load data from Data Analysis page
2. Configure model parameters
3. Click "🚀 Start Training"
4. Monitor progress with real-time updates
5. Review training metrics and results

## 🔮 Prediction Page

### Prediction Modes

#### 1. 📊 Batch Prediction
- **Purpose**: Analyze multiple cycles at once
- **Input**: Select cycle range from loaded data
- **Output**: Predictions for selected range with visualizations
- **Features**: SoH predictions, RUL estimates, performance metrics

#### 2. 🔧 Single Prediction
- **Purpose**: Predict SoH for specific battery parameters
- **Input**: Manual entry of battery parameters
- **Parameters**:
  - Voltage (V): 2.0-5.0
  - Current (A): 0.0-10.0
  - Temperature (°C): -20 to 60
  - Capacity (Ah): 10-200
  - Cycle Number: 1-10000
  - Discharge Time (h): 0.1-10.0
  - Charge Time (h): 0.1-10.0
- **Output**: SoH prediction, RUL estimate, health status

#### 3. 📈 Real-time Simulation
- **Purpose**: Simulate battery degradation over time
- **Parameters**:
  - Simulation Cycles: Number of cycles to simulate (50-1000)
  - Update Interval: Animation speed (100-2000ms)
  - Noise Level: Add realistic noise (0.0-0.1)
- **Output**: Live animated plots showing SoH and RUL evolution

### Health Status Indicators
- **🟢 Excellent**: SoH ≥ 0.9
- **🟡 Good**: 0.8 ≤ SoH < 0.9
- **🟠 Fair**: 0.7 ≤ SoH < 0.8
- **🔴 Poor**: SoH < 0.7

## 📈 Results Page

### Results Summary
- **Total Predictions**: Number of predictions made
- **Mean SoH**: Average predicted State of Health
- **Standard Deviation**: Variability in predictions
- **Min/Max SoH**: Range of predicted values

### Detailed Analysis
1. **Predictions vs Actual**: Scatter plot comparing predictions to true values
2. **Residuals**: Error analysis over time
3. **Error Distribution**: Histogram of prediction errors
4. **Prediction Confidence**: Confidence levels for predictions

### Export Options
- **📥 Download Results (CSV)**: Export predictions and actual values
- **📄 Download Report (TXT)**: Generate comprehensive text report

## 🎯 Best Practices

### Data Preparation
1. **Data Quality**: Ensure no missing values
2. **Feature Scaling**: Features should be in reasonable ranges
3. **Cycle Order**: Maintain chronological order
4. **Sufficient Data**: Minimum 100 cycles recommended

### Model Training
1. **Start Simple**: Begin with default parameters
2. **Monitor Training**: Watch for overfitting
3. **Validation**: Use validation set for tuning
4. **Save Models**: Enable model saving for reuse

### Prediction
1. **Data Consistency**: Use same feature format as training
2. **Confidence**: Consider confidence thresholds
3. **Validation**: Cross-check predictions with domain knowledge
4. **Documentation**: Keep track of prediction contexts

## 🔧 Troubleshooting

### Common Issues

#### Model Training Fails
- **Check Data**: Ensure data is loaded and properly formatted
- **Reduce Complexity**: Lower LSTM units or sequence length
- **Increase Patience**: Allow more training time
- **Check Memory**: Ensure sufficient RAM available

#### Poor Predictions
- **Data Quality**: Check for outliers or missing values
- **Feature Selection**: Try different feature combinations
- **Model Architecture**: Experiment with different configurations
- **Training Duration**: Increase training epochs

#### Frontend Issues
- **Dependencies**: Ensure all packages are installed
- **Browser Compatibility**: Use modern browser (Chrome, Firefox, Safari)
- **Memory**: Close other applications if system is slow
- **Updates**: Keep packages updated

### Performance Optimization
1. **Batch Processing**: Use batch predictions for large datasets
2. **Sequence Length**: Optimize for your specific use case
3. **Model Complexity**: Balance accuracy and speed
4. **Hardware**: Use GPU for LSTM training if available

## 📞 Support

### Getting Help
1. **Documentation**: Check this guide and README.md
2. **Error Messages**: Read error details carefully
3. **Logs**: Check Streamlit console for detailed errors
4. **Community**: Refer to project GitHub issues

### Feature Requests
- **UI Improvements**: Suggest interface enhancements
- **New Models**: Request additional model types
- **Data Formats**: Support for additional data formats
- **Export Options**: Additional export formats

---

## 🎉 Quick Start Tutorial

### 1. Launch Application
```bash
streamlit run app.py
```

### 2. Generate Sample Data
- Navigate to **📊 Data Analysis**
- Select **🔧 Generate Sample Data**
- Use default settings (1000 cycles)
- Click **🔧 Generate Data**

### 3. Train Model
- Go to **🧠 Model Configuration**
- Select **🔥 LSTM Neural Network**
- Use default parameters
- Click **🚀 Start Training**

### 4. Make Predictions
- Navigate to **🔮 Prediction**
- Try **🔧 Single Prediction**
- Enter sample values:
  - Voltage: 3.7
  - Current: 1.0
  - Temperature: 25.0
  - Capacity: 95.0
  - Cycle: 500
- Click **🔮 Predict SoH**

### 5. View Results
- Go to **📈 Results**
- Review performance metrics
- Download results if needed

Congratulations! You've successfully used the Battery SoH Prediction System! 🎉
