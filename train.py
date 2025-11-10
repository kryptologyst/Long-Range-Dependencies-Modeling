"""Main training script for long-range dependencies modeling."""

import argparse
import sys
from pathlib import Path
import torch
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data.data_generator import TimeSeriesDataGenerator, DataPreprocessor, create_data_loader
from src.models.neural_models import ModelFactory
from src.models.training import ModelTrainer, calculate_metrics
from src.utils.config import Config
from src.utils.logging_config import setup_logging
from src.visualization.plots import TimeSeriesVisualizer


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train time series models for long-range dependencies")
    parser.add_argument("--model", type=str, default="tcn", choices=["tcn", "lstm", "transformer"],
                       help="Model type to train")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--epochs", type=int, default=None,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None,
                       help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=None,
                       help="Learning rate")
    parser.add_argument("--save-model", type=str, default=None,
                       help="Path to save the trained model")
    parser.add_argument("--save-plots", action="store_true",
                       help="Save training plots")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (cpu, cuda, auto)")
    
    args = parser.parse_args()
    
    # Setup logging
    config = Config(args.config)
    setup_logging(
        level=config.get("logging.level", "INFO"),
        log_file=config.get("logging.file", None),
        console=config.get("logging.console", True)
    )
    
    logger = setup_logging()
    logger.info(f"Starting training with {args.model} model")
    
    # Override config with command line arguments
    if args.epochs is not None:
        config.set("training.epochs", args.epochs)
    if args.batch_size is not None:
        config.set("training.batch_size", args.batch_size)
    if args.learning_rate is not None:
        config.set("training.learning_rate", args.learning_rate)
    
    # Generate data
    logger.info("Generating synthetic time series data")
    generator = TimeSeriesDataGenerator(
        random_seed=config.get("data.random_seed", 42)
    )
    
    x, y = generator.generate_long_dependency_data(
        n_samples=config.get("data.n_samples", 2000),
        dependency_lags=config.get("data.dependency_lags", [30, 60]),
        noise_scale=config.get("data.noise_scale", 0.1)
    )
    
    # Create sequences
    seq_len = config.get("sequence.seq_len", 100)
    X, Y = generator.create_sequences(x, y, seq_len=seq_len)
    
    # Split data
    X_train, X_val, X_test, Y_train, Y_val, Y_test = generator.split_data(
        X, Y,
        test_size=0.2,
        val_size=config.get("training.validation_split", 0.2)
    )
    
    # Create data loaders
    batch_size = config.get("training.batch_size", 32)
    train_loader = create_data_loader(X_train, Y_train, batch_size=batch_size, shuffle=True)
    val_loader = create_data_loader(X_val, Y_val, batch_size=batch_size, shuffle=False)
    test_loader = create_data_loader(X_test, Y_test, batch_size=batch_size, shuffle=False)
    
    # Create model
    logger.info(f"Creating {args.model} model")
    model_config = config.get(f"models.{args.model}", {})
    model = ModelFactory.create_model(args.model, model_config, input_size=1)
    
    # Print model info
    model_info = ModelFactory.get_model_info(model)
    logger.info(f"Model info: {model_info}")
    
    # Create trainer
    trainer = ModelTrainer(model, config, device=args.device)
    
    # Train model
    logger.info("Starting training")
    history = trainer.train(train_loader, val_loader, save_path=args.save_model)
    
    # Evaluate model
    logger.info("Evaluating model")
    y_pred = trainer.predict(test_loader)
    y_test_np = Y_test.numpy().flatten()
    
    # Calculate metrics
    metrics = calculate_metrics(y_test_np, y_pred)
    logger.info(f"Test metrics: {metrics}")
    
    # Create visualizations
    if args.save_plots:
        logger.info("Creating visualizations")
        visualizer = TimeSeriesVisualizer()
        
        # Create plots directory
        plots_dir = Path("plots")
        plots_dir.mkdir(exist_ok=True)
        
        # Training history
        visualizer.plot_training_history(
            history['train_losses'],
            history['val_losses'],
            title=f"{args.model.upper()} Training History",
            save_path=str(plots_dir / f"{args.model}_training_history.png"),
            show=False
        )
        
        # Predictions vs actual
        visualizer.plot_predictions(
            y_test_np,
            y_pred,
            title=f"{args.model.upper()} Predictions vs Actual",
            save_path=str(plots_dir / f"{args.model}_predictions.png"),
            show=False
        )
        
        # Residuals analysis
        visualizer.plot_residuals(
            y_test_np,
            y_pred,
            title=f"{args.model.upper()} Residuals Analysis",
            save_path=str(plots_dir / f"{args.model}_residuals.png"),
            show=False
        )
        
        logger.info(f"Plots saved to {plots_dir}")
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
