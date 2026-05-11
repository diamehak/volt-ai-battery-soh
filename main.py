"""
Main entry point for Battery SoH Prediction Project
Final Year Engineering Internship Project
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.train import BatterySoHTrainer


def main():
    """
    Main function to run the Battery SoH prediction system
    """
    parser = argparse.ArgumentParser(
        description="Battery State of Health (SoH) Prediction using LSTM Networks"
    )
    
    # Configuration arguments
    parser.add_argument("--config", type=str, help="Path to configuration JSON file")
    parser.add_argument("--data", type=str, help="Path to battery data CSV file")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, help="Batch size for training")
    parser.add_argument("--sequence_length", type=int, help="LSTM sequence length")
    
    # Model arguments
    parser.add_argument("--attention", action="store_true", 
                       help="Use attention-based LSTM model")
    parser.add_argument("--lstm_units", type=int, nargs='+', 
                       help="LSTM units for each layer")
    parser.add_argument("--dropout", type=float, help="Dropout rate")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    
    # Output arguments
    parser.add_argument("--model_dir", type=str, help="Directory to save models")
    parser.add_argument("--results_dir", type=str, help="Directory to save results")
    
    # Demo mode
    parser.add_argument("--demo", action="store_true", 
                       help="Run demo with sample data")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Battery State of Health (SoH) Prediction System")
    print("Final Year Engineering Internship Project")
    print("=" * 60)
    
    if args.demo:
        print("Running demo mode with sample data...")
        # Create trainer with default config
        trainer = BatterySoHTrainer()
        trainer.run_complete_training()
    else:
        # Create trainer with custom config
        trainer = BatterySoHTrainer(args.config)
        
        # Override config with command line arguments
        if args.data:
            trainer.config['data_file'] = args.data
        if args.epochs:
            trainer.config['training_params']['epochs'] = args.epochs
        if args.batch_size:
            trainer.config['training_params']['batch_size'] = args.batch_size
        if args.sequence_length:
            trainer.config['sequence_length'] = args.sequence_length
        if args.attention:
            trainer.config['use_attention'] = True
        if args.lstm_units:
            trainer.config['model_params']['lstm_units'] = args.lstm_units
        if args.dropout:
            trainer.config['model_params']['dropout_rate'] = args.dropout
        if args.learning_rate:
            trainer.config['model_params']['learning_rate'] = args.learning_rate
        if args.model_dir:
            trainer.config['model_save_dir'] = args.model_dir
        if args.results_dir:
            trainer.config['results_dir'] = args.results_dir
        
        # Run training
        trainer.run_complete_training()
    
    print("\n" + "=" * 60)
    print("Project completed successfully!")
    print("Check the 'models' and 'results' directories for outputs.")
    print("=" * 60)


if __name__ == "__main__":
    main()
