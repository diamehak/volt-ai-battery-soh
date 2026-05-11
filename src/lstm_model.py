"""
LSTM Neural Network for Battery State of Health Prediction
Implements deep learning architecture for SoH and RUL forecasting
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
import numpy as np
from typing import Tuple, Optional, List
import matplotlib.pyplot as plt
import os


class BatterySoHPredictor:
    """
    LSTM-based neural network for predicting battery State of Health
    and Remaining Useful Life
    """
    
    def __init__(self, sequence_length: int = 50, n_features: int = 7):
        """
        Initialize the LSTM model
        
        Args:
            sequence_length: Number of time steps in input sequences
            n_features: Number of input features
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = None
        self.history = None
        
    def build_model(self, lstm_units: List[int] = [128, 64, 32], 
                   dropout_rate: float = 0.3, 
                   learning_rate: float = 0.001) -> Model:
        """
        Build the LSTM architecture
        
        Args:
            lstm_units: List of units for each LSTM layer
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
            
        Returns:
            Compiled Keras model
        """
        # Input layer
        inputs = Input(shape=(self.sequence_length, self.n_features))
        
        # First LSTM layer with BatchNormalization
        x = LSTM(lstm_units[0], return_sequences=True, 
                kernel_regularizer=l2(0.01))(inputs)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        
        # Second LSTM layer
        x = LSTM(lstm_units[1], return_sequences=True, 
                kernel_regularizer=l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        
        # Third LSTM layer
        x = LSTM(lstm_units[2], return_sequences=False, 
                kernel_regularizer=l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        
        # Dense layers
        x = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        
        x = Dense(32, activation='relu', kernel_regularizer=l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        
        # Output layer for SoH prediction
        outputs = Dense(1, activation='linear')(x)
        
        # Create model
        self.model = Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        return self.model
    
    def build_attention_model(self, lstm_units: List[int] = [128, 64], 
                            dropout_rate: float = 0.3,
                            learning_rate: float = 0.001) -> Model:
        """
        Build LSTM model with attention mechanism
        
        Args:
            lstm_units: List of units for LSTM layers
            dropout_rate: Dropout rate
            learning_rate: Learning rate
            
        Returns:
            Compiled model with attention
        """
        inputs = Input(shape=(self.sequence_length, self.n_features))
        
        # LSTM layers
        lstm_out = LSTM(lstm_units[0], return_sequences=True, 
                       kernel_regularizer=l2(0.01))(inputs)
        lstm_out = BatchNormalization()(lstm_out)
        lstm_out = Dropout(dropout_rate)(lstm_out)
        
        lstm_out = LSTM(lstm_units[1], return_sequences=True, 
                       kernel_regularizer=l2(0.01))(lstm_out)
        lstm_out = BatchNormalization()(lstm_out)
        
        # Attention mechanism
        attention = Dense(1, activation='tanh')(lstm_out)
        attention = tf.keras.layers.Flatten()(attention)
        attention = tf.keras.layers.Activation('softmax')(attention)
        attention = tf.keras.layers.RepeatVector(lstm_units[1])(attention)
        attention = tf.keras.layers.Permute([2, 1])(attention)
        
        # Apply attention
        sent_representation = tf.keras.layers.Multiply()([lstm_out, attention])
        sent_representation = tf.keras.layers.Lambda(
            lambda x: tf.keras.backend.sum(x, axis=1)
        )(sent_representation)
        
        # Dense layers
        x = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(sent_representation)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        
        outputs = Dense(1, activation='linear')(x)
        
        # Create and compile model
        self.model = Model(inputs=inputs, outputs=outputs)
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mape'])
        
        return self.model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              epochs: int = 100, batch_size: int = 32,
              patience: int = 15, model_save_path: str = None) -> dict:
        """
        Train the LSTM model
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Number of training epochs
            batch_size: Batch size
            patience: Early stopping patience
            model_save_path: Path to save best model
            
        Returns:
            Training history
        """
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=patience, 
                         restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, 
                            patience=patience//2, min_lr=1e-7, verbose=1)
        ]
        
        if model_save_path:
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            callbacks.append(
                ModelCheckpoint(model_save_path, monitor='val_loss', 
                             save_best_only=True, verbose=1)
            )
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model
        
        Args:
            X: Input features
            
        Returns:
            Predicted SoH values
        """
        return self.model.predict(X)
    
    def predict_rul(self, X: np.ndarray, threshold: float = 0.8) -> np.ndarray:
        """
        Predict Remaining Useful Life based on SoH predictions
        
        Args:
            X: Input features
            threshold: SoH threshold for end-of-life (default 0.8)
            
        Returns:
            Predicted RUL in cycles
        """
        soh_predictions = self.predict(X)
        rul_predictions = []
        
        for soh_seq in soh_predictions:
            # Find when SoH drops below threshold
            if soh_seq[0] < threshold:
                rul = 0
            else:
                # Estimate RUL based on degradation rate
                if len(soh_seq) > 1:
                    degradation_rate = (soh_seq[0] - soh_seq[-1]) / len(soh_seq)
                    if degradation_rate > 0:
                        rul = int((soh_seq[0] - threshold) / degradation_rate)
                    else:
                        rul = 1000  # Default large value
                else:
                    rul = 1000
            
            rul_predictions.append(max(0, rul))
        
        return np.array(rul_predictions)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary of evaluation metrics
        """
        predictions = self.predict(X_test)
        
        # Calculate metrics
        mse = tf.keras.losses.mean_squared_error(y_test, predictions).numpy().mean()
        mae = tf.keras.losses.mean_absolute_error(y_test, predictions).numpy().mean()
        mape = tf.keras.losses.mean_absolute_percentage_error(y_test, predictions).numpy().mean()
        
        # R² score
        ss_res = np.sum((y_test.flatten() - predictions.flatten()) ** 2)
        ss_tot = np.sum((y_test.flatten() - np.mean(y_test)) ** 2)
        r2_score = 1 - (ss_res / ss_tot)
        
        # RMSE
        rmse = np.sqrt(mse)
        
        return {
            'mse': mse,
            'mae': mae,
            'mape': mape,
            'r2_score': r2_score,
            'rmse': rmse
        }
    
    def plot_training_history(self, save_path: str = None):
        """
        Plot training history
        
        Args:
            save_path: Path to save the plot
        """
        if not self.history:
            print("No training history available")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.history.history['loss'], label='Training Loss')
        ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # MAE plot
        ax2.plot(self.history.history['mae'], label='Training MAE')
        ax2.plot(self.history.history['val_mae'], label='Validation MAE')
        ax2.set_title('Model MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, filepath: str):
        """
        Save the trained model
        
        Args:
            filepath: Path to save the model
        """
        if self.model:
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
        else:
            print("No model to save")
    
    def load_model(self, filepath: str):
        """
        Load a trained model
        
        Args:
            filepath: Path to the saved model
        """
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
    
    def get_model_summary(self) -> str:
        """
        Get model architecture summary
        
        Returns:
            String representation of model summary
        """
        if self.model:
            import io
            import sys
            
            # Capture model summary
            old_stdout = sys.stdout
            sys.stdout = buffer = io.StringIO()
            self.model.summary()
            sys.stdout = old_stdout
            
            return buffer.getvalue()
        else:
            return "No model built yet"


if __name__ == "__main__":
    # Example usage
    predictor = BatterySoHPredictor(sequence_length=50, n_features=7)
    
    # Build model
    model = predictor.build_model()
    print(predictor.get_model_summary())
    
    # Generate dummy data for testing
    X_dummy = np.random.random((1000, 50, 7))
    y_dummy = np.random.random((1000, 1))
    
    # Test prediction
    predictions = predictor.predict(X_dummy[:10])
    print(f"Test predictions shape: {predictions.shape}")
