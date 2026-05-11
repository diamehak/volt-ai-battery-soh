"""
Visualization Module for Battery SoH Analysis
Creates comprehensive plots and visualizations for model results
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple
import os


class BatteryVisualizer:
    """
    Visualization tools for battery SoH prediction results
    """
    
    def __init__(self, style: str = 'seaborn-v0_8'):
        """
        Initialize visualizer with plotting style
        
        Args:
            style: Matplotlib style to use
        """
        plt.style.use(style)
        sns.set_palette("husl")
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    def plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                       save_path: Optional[str] = None, title: str = "SoH Predictions"):
        """
        Plot true vs predicted SoH values
        
        Args:
            y_true: True SoH values
            y_pred: Predicted SoH values
            save_path: Path to save the plot
            title: Plot title
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Time series plot
        ax1.plot(y_true, label='True SoH', color=self.colors[0], linewidth=2)
        ax1.plot(y_pred, label='Predicted SoH', color=self.colors[1], linewidth=2, alpha=0.8)
        ax1.set_xlabel('Cycle Number')
        ax1.set_ylabel('State of Health')
        ax1.set_title('SoH Prediction Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Scatter plot
        ax2.scatter(y_true, y_pred, alpha=0.6, color=self.colors[2])
        ax2.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                'r--', lw=2, label='Perfect Prediction')
        ax2.set_xlabel('True SoH')
        ax2.set_ylabel('Predicted SoH')
        ax2.set_title('True vs Predicted SoH')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add R² score
        r2 = 1 - np.sum((y_true.flatten() - y_pred.flatten())**2) / np.sum((y_true.flatten() - np.mean(y_true))**2)
        ax2.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax2.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_degradation_curve(self, y_true: np.ndarray, y_pred: np.ndarray,
                              save_path: Optional[str] = None):
        """
        Plot battery degradation curves
        
        Args:
            y_true: True SoH values
            y_pred: Predicted SoH values
            save_path: Path to save the plot
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        cycles = np.arange(len(y_true))
        
        # Plot degradation curves
        ax.plot(cycles, y_true, label='Actual Degradation', 
                color=self.colors[0], linewidth=2.5)
        ax.plot(cycles, y_pred, label='Predicted Degradation', 
                color=self.colors[1], linewidth=2.5, alpha=0.8)
        
        # Add threshold line (80% SoH)
        threshold = 0.8
        ax.axhline(y=threshold, color='red', linestyle='--', 
                  label=f'EOL Threshold ({threshold:.0%})', alpha=0.7)
        
        # Find EOL points
        actual_eol_idx = np.where(y_true < threshold)[0]
        predicted_eol_idx = np.where(y_pred < threshold)[0]
        
        if len(actual_eol_idx) > 0:
            actual_eol = actual_eol_idx[0]
            ax.axvline(x=actual_eol, color=self.colors[0], linestyle=':', 
                      alpha=0.7, label=f'Actual EOL: Cycle {actual_eol}')
        
        if len(predicted_eol_idx) > 0:
            predicted_eol = predicted_eol_idx[0]
            ax.axvline(x=predicted_eol, color=self.colors[1], linestyle=':', 
                      alpha=0.7, label=f'Predicted EOL: Cycle {predicted_eol}')
        
        ax.set_xlabel('Cycle Number')
        ax.set_ylabel('State of Health')
        ax.set_title('Battery Degradation Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_residuals(self, y_true: np.ndarray, y_pred: np.ndarray,
                      save_path: Optional[str] = None):
        """
        Plot prediction residuals
        
        Args:
            y_true: True values
            y_pred: Predicted values
            save_path: Path to save the plot
        """
        residuals = y_true.flatten() - y_pred.flatten()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Residuals vs predicted
        ax1.scatter(y_pred.flatten(), residuals, alpha=0.6, color=self.colors[3])
        ax1.axhline(y=0, color='red', linestyle='--')
        ax1.set_xlabel('Predicted Values')
        ax1.set_ylabel('Residuals')
        ax1.set_title('Residuals vs Predicted Values')
        ax1.grid(True, alpha=0.3)
        
        # Histogram of residuals
        ax2.hist(residuals, bins=30, color=self.colors[4], alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Residuals')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Residuals')
        ax2.grid(True, alpha=0.3)
        
        # Add statistics
        mean_res = np.mean(residuals)
        std_res = np.std(residuals)
        ax2.text(0.05, 0.95, f'Mean: {mean_res:.4f}\nStd: {std_res:.4f}', 
                transform=ax2.transAxes, bbox=dict(boxstyle="round,pad=0.3", 
                facecolor="white", alpha=0.8))
        
        plt.suptitle('Residual Analysis', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self, feature_names: List[str], 
                               importance_scores: np.ndarray,
                               save_path: Optional[str] = None):
        """
        Plot feature importance scores
        
        Args:
            feature_names: List of feature names
            importance_scores: Importance scores for each feature
            save_path: Path to save the plot
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Sort by importance
        sorted_idx = np.argsort(importance_scores)[::-1]
        sorted_features = [feature_names[i] for i in sorted_idx]
        sorted_scores = importance_scores[sorted_idx]
        
        bars = ax.barh(range(len(sorted_features)), sorted_scores, color=self.colors[2])
        ax.set_yticks(range(len(sorted_features)))
        ax.set_yticklabels(sorted_features)
        ax.set_xlabel('Importance Score')
        ax.set_title('Feature Importance')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{width:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_training_progress(self, history: dict, save_path: Optional[str] = None):
        """
        Plot training progress over epochs
        
        Args:
            history: Training history dictionary
            save_path: Path to save the plot
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        ax1.plot(history['loss'], label='Training Loss', color=self.colors[0])
        ax1.plot(history['val_loss'], label='Validation Loss', color=self.colors[1])
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # MAE
        ax2.plot(history['mae'], label='Training MAE', color=self.colors[0])
        ax2.plot(history['val_mae'], label='Validation MAE', color=self.colors[1])
        ax2.set_title('Mean Absolute Error')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # MAPE
        if 'mape' in history:
            ax3.plot(history['mape'], label='Training MAPE', color=self.colors[0])
            ax3.plot(history['val_mape'], label='Validation MAPE', color=self.colors[1])
            ax3.set_title('Mean Absolute Percentage Error')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('MAPE')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Learning rate (if available)
        if 'lr' in history:
            ax4.plot(history['lr'], color=self.colors[2])
            ax4.set_title('Learning Rate')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Learning Rate')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Learning Rate\nNot Available', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Learning Rate')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_interactive_dashboard(self, y_true: np.ndarray, y_pred: np.ndarray,
                                  cycles: Optional[np.ndarray] = None):
        """
        Create interactive dashboard using Plotly
        
        Args:
            y_true: True SoH values
            y_pred: Predicted SoH values
            cycles: Cycle numbers (if None, uses range)
        """
        if cycles is None:
            cycles = np.arange(len(y_true))
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('SoH Prediction Over Time', 'True vs Predicted', 
                          'Residuals', 'Error Distribution'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # SoH over time
        fig.add_trace(
            go.Scatter(x=cycles, y=y_true.flatten(), name='True SoH', 
                      line=dict(color='blue', width=2)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=cycles, y=y_pred.flatten(), name='Predicted SoH', 
                      line=dict(color='orange', width=2)),
            row=1, col=1
        )
        
        # True vs Predicted scatter
        fig.add_trace(
            go.Scatter(x=y_true.flatten(), y=y_pred.flatten(), 
                      mode='markers', name='Predictions',
                      marker=dict(color='green', size=6, opacity=0.6)),
            row=1, col=2
        )
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        fig.add_trace(
            go.Scatter(x=[min_val, max_val], y=[min_val, max_val], 
                      mode='lines', name='Perfect Prediction',
                      line=dict(color='red', dash='dash')),
            row=1, col=2
        )
        
        # Residuals
        residuals = y_true.flatten() - y_pred.flatten()
        fig.add_trace(
            go.Scatter(x=cycles, y=residuals, mode='markers', 
                      name='Residuals', marker=dict(color='purple', size=4)),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=cycles, y=[0]*len(cycles), mode='lines', 
                      name='Zero Line', line=dict(color='red', dash='dash')),
            row=2, col=1
        )
        
        # Error distribution
        fig.add_trace(
            go.Histogram(x=residuals, name='Error Distribution', 
                         marker=dict(color='orange', opacity=0.7)),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Battery SoH Prediction Dashboard",
            showlegend=True,
            height=800
        )
        
        fig.show()
    
    def plot_battery_health_map(self, soh_data: np.ndarray, 
                               cycles: np.ndarray, 
                               save_path: Optional[str] = None):
        """
        Create a heatmap visualization of battery health over cycles
        
        Args:
            soh_data: SoH data matrix (cycles x features)
            cycles: Cycle numbers
            save_path: Path to save the plot
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Create heatmap
        sns.heatmap(soh_data.T, cmap='RdYlBu_r', cbar_kws={'label': 'SoH'}, ax=ax)
        
        ax.set_xlabel('Cycle Number')
        ax.set_ylabel('Battery Features')
        ax.set_title('Battery Health Heatmap')
        
        # Set cycle labels
        if len(cycles) <= 20:  # Only show labels if not too many
            ax.set_xticks(np.arange(0, len(cycles), max(1, len(cycles)//10)))
            ax.set_xticklabels(cycles[::max(1, len(cycles)//10)])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    # Example usage
    visualizer = BatteryVisualizer()
    
    # Generate sample data
    np.random.seed(42)
    y_true = np.linspace(1.0, 0.7, 100) + np.random.normal(0, 0.02, 100)
    y_pred = y_true + np.random.normal(0, 0.01, 100)
    
    # Create visualizations
    visualizer.plot_predictions(y_true, y_pred)
    visualizer.plot_degradation_curve(y_true, y_pred)
    visualizer.plot_residuals(y_true, y_pred)
