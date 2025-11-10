#!/usr/bin/env python3
"""
Example script demonstrating the long-range dependencies modeling project.
This script shows how to use the project components to train and evaluate models.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data.data_generator import TimeSeriesDataGenerator
from src.models.neural_models import ModelFactory
from src.models.training import ModelTrainer, calculate_metrics
from src.utils.config import Config
from src.utils.logging_config import setup_logging
from src.visualization.plots import TimeSeriesVisualizer


def main():
    """Main example function."""
    print("Long-Range Dependencies Modeling - Example Script")
    print("=" * 50)
    
    # Setup logging
    logger = setup_logging(level="INFO", console=True)
    logger.info("Starting example script")
    
    # 1. Generate synthetic data
    print("\n1. Generating synthetic time series data...")
    generator = TimeSeriesDataGenerator(random_seed=42)
    x, y = generator.generate_long_dependency_data(
        n_samples=1500,
        dependency_lags=[30, 60],
        noise_scale=0.1
    )
    print(f"Generated {len(x)} samples")
    
    # 2. Create sequences for training
    print("\n2. Creating sequences for training...")
    seq_len = 80
    X, Y = generator.create_sequences(x, y, seq_len=seq_len)
    print(f"Created {len(X)} sequences of length {seq_len}")
    
    # 3. Split data
    print("\n3. Splitting data...")
    X_train, X_val, X_test, Y_train, Y_val, Y_test = generator.split_data(
        X, Y, test_size=0.2, val_size=0.1
    )
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # 4. Create data loaders
    batch_size = 32
    train_loader = DataLoader(
        TensorDataset(X_train, Y_train),
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val, Y_val),
        batch_size=batch_size,
        shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(X_test, Y_test),
        batch_size=batch_size,
        shuffle=False
    )
    
    # 5. Train TCN model
    print("\n4. Training TCN model...")
    tcn_config = {
        "input_channels": 1,
        "num_channels": [16, 32, 64],
        "kernel_size": 3,
        "dropout": 0.1
    }
    
    model = ModelFactory.create_model("tcn", tcn_config, input_size=1)
    
    # Model info
    model_info = ModelFactory.get_model_info(model)
    print(f"Model parameters: {model_info['total_parameters']:,}")
    
    # Training configuration
    config = Config()
    config.set('training.epochs', 30)
    config.set('training.learning_rate', 0.001)
    config.set('training.early_stopping_patience', 8)
    
    # Train model
    trainer = ModelTrainer(model, config)
    history = trainer.train(train_loader, val_loader)
    
    print(f"Training completed. Best validation loss: {trainer.best_val_loss:.6f}")
    
    # 6. Evaluate model
    print("\n5. Evaluating model...")
    y_pred = trainer.predict(test_loader)
    y_test_np = Y_test.numpy().flatten()
    
    metrics = calculate_metrics(y_test_np, y_pred)
    print("\nTest Results:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # 7. Create visualizations
    print("\n6. Creating visualizations...")
    visualizer = TimeSeriesVisualizer()
    
    # Create plots directory
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    
    # Training history
    visualizer.plot_training_history(
        history['train_losses'],
        history['val_losses'],
        title="TCN Training History",
        save_path=str(plots_dir / "tcn_training_history.png"),
        show=False
    )
    
    # Predictions vs actual
    visualizer.plot_predictions(
        y_test_np,
        y_pred,
        title="TCN Predictions vs Actual",
        save_path=str(plots_dir / "tcn_predictions.png"),
        show=False
    )
    
    # Residuals analysis
    visualizer.plot_residuals(
        y_test_np,
        y_pred,
        title="TCN Residuals Analysis",
        save_path=str(plots_dir / "tcn_residuals.png"),
        show=False
    )
    
    print(f"Plots saved to {plots_dir}")
    
    # 8. Summary
    print("\n" + "=" * 50)
    print("Example completed successfully!")
    print(f"Final RMSE: {metrics['RMSE']:.4f}")
    print(f"Final R²: {metrics['R2']:.4f}")
    print("\nNext steps:")
    print("- Run 'streamlit run src/streamlit_app.py' for interactive interface")
    print("- Run 'python train.py --model lstm' to train other models")
    print("- Check the plots/ directory for visualizations")


if __name__ == "__main__":
    main()
