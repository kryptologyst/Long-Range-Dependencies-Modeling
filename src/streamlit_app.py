"""Streamlit interface for interactive time series analysis."""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch
from typing import Dict, Any
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.data.data_generator import TimeSeriesDataGenerator, DataPreprocessor, create_data_loader
from src.models.neural_models import ModelFactory
from src.models.training import ModelTrainer, calculate_metrics
from src.utils.config import Config
from src.utils.logging_config import setup_logging
from src.visualization.plots import TimeSeriesVisualizer, create_dashboard_data


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Long-Range Dependencies Modeling",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("📈 Long-Range Dependencies Modeling")
    st.markdown("Interactive time series analysis with neural networks")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Data generation parameters
    st.sidebar.subheader("Data Generation")
    n_samples = st.sidebar.slider("Number of samples", 500, 5000, 2000)
    noise_scale = st.sidebar.slider("Noise scale", 0.01, 0.5, 0.1)
    dependency_lags = st.sidebar.multiselect(
        "Dependency lags",
        [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        default=[30, 60]
    )
    
    # Model parameters
    st.sidebar.subheader("Model Configuration")
    model_type = st.sidebar.selectbox(
        "Model type",
        ["TCN", "LSTM", "Transformer"],
        index=0
    )
    
    seq_len = st.sidebar.slider("Sequence length", 50, 200, 100)
    batch_size = st.sidebar.slider("Batch size", 16, 128, 32)
    epochs = st.sidebar.slider("Epochs", 10, 100, 50)
    learning_rate = st.sidebar.slider("Learning rate", 0.0001, 0.01, 0.001)
    
    # Training parameters
    st.sidebar.subheader("Training")
    validation_split = st.sidebar.slider("Validation split", 0.1, 0.3, 0.2)
    early_stopping_patience = st.sidebar.slider("Early stopping patience", 5, 20, 10)
    
    # Initialize session state
    if 'data_generated' not in st.session_state:
        st.session_state.data_generated = False
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    
    # Generate data button
    if st.sidebar.button("Generate Data", type="primary"):
        with st.spinner("Generating synthetic time series data..."):
            # Generate data
            generator = TimeSeriesDataGenerator(random_seed=42)
            x, y = generator.generate_long_dependency_data(
                n_samples=n_samples,
                dependency_lags=dependency_lags,
                noise_scale=noise_scale
            )
            
            # Store in session state
            st.session_state.x = x
            st.session_state.y = y
            st.session_state.data_generated = True
            st.session_state.model_trained = False
        
        st.success("Data generated successfully!")
    
    # Main content area
    if st.session_state.data_generated:
        x = st.session_state.x
        y = st.session_state.y
        
        # Data visualization
        st.header("📊 Data Visualization")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Input Time Series")
            fig_input = go.Figure()
            fig_input.add_trace(go.Scatter(y=x, mode='lines', name='Input'))
            fig_input.update_layout(title="Input Time Series", xaxis_title="Time", yaxis_title="Value")
            st.plotly_chart(fig_input, use_container_width=True)
        
        with col2:
            st.subheader("Target Time Series")
            fig_target = go.Figure()
            fig_target.add_trace(go.Scatter(y=y, mode='lines', name='Target'))
            fig_target.update_layout(title="Target Time Series", xaxis_title="Time", yaxis_title="Value")
            st.plotly_chart(fig_target, use_container_width=True)
        
        # Data statistics
        st.subheader("📈 Data Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Samples", len(x))
        with col2:
            st.metric("Mean", f"{np.mean(x):.3f}")
        with col3:
            st.metric("Std", f"{np.std(x):.3f}")
        with col4:
            st.metric("Range", f"{np.max(x) - np.min(x):.3f}")
        
        # Training section
        st.header("🤖 Model Training")
        
        if st.button("Train Model", type="primary"):
            with st.spinner(f"Training {model_type} model..."):
                # Create sequences
                generator = TimeSeriesDataGenerator(random_seed=42)
                X, Y = generator.create_sequences(x, y, seq_len=seq_len)
                
                # Split data
                X_train, X_val, X_test, Y_train, Y_val, Y_test = generator.split_data(
                    X, Y, test_size=0.2, val_size=validation_split
                )
                
                # Create data loaders
                train_loader = create_data_loader(X_train, Y_train, batch_size=batch_size, shuffle=True)
                val_loader = create_data_loader(X_val, Y_val, batch_size=batch_size, shuffle=False)
                test_loader = create_data_loader(X_test, Y_test, batch_size=batch_size, shuffle=False)
                
                # Model configuration
                model_configs = {
                    "TCN": {
                        "input_channels": 1,
                        "num_channels": [16, 32, 64],
                        "kernel_size": 3,
                        "dropout": 0.1
                    },
                    "LSTM": {
                        "hidden_size": 64,
                        "num_layers": 2,
                        "dropout": 0.1,
                        "bidirectional": False
                    },
                    "Transformer": {
                        "d_model": 64,
                        "nhead": 8,
                        "num_layers": 3,
                        "dropout": 0.1,
                        "dim_feedforward": 256
                    }
                }
                
                # Create model
                model = ModelFactory.create_model(
                    model_type.lower(),
                    model_configs[model_type],
                    input_size=1
                )
                
                # Create config
                config = Config()
                config.set('training.learning_rate', learning_rate)
                config.set('training.epochs', epochs)
                config.set('training.early_stopping_patience', early_stopping_patience)
                
                # Train model
                trainer = ModelTrainer(model, config)
                history = trainer.train(train_loader, val_loader)
                
                # Make predictions
                y_pred = trainer.predict(test_loader)
                y_test_np = Y_test.numpy().flatten()
                
                # Calculate metrics
                metrics = calculate_metrics(y_test_np, y_pred)
                
                # Store results in session state
                st.session_state.model_trained = True
                st.session_state.trainer = trainer
                st.session_state.history = history
                st.session_state.y_test = y_test_np
                st.session_state.y_pred = y_pred
                st.session_state.metrics = metrics
                st.session_state.model_type = model_type
            
            st.success(f"{model_type} model trained successfully!")
        
        # Results section
        if st.session_state.model_trained:
            st.header("📊 Results")
            
            # Metrics
            st.subheader("Evaluation Metrics")
            col1, col2, col3, col4, col5 = st.columns(5)
            
            metrics = st.session_state.metrics
            with col1:
                st.metric("MAE", f"{metrics['MAE']:.4f}")
            with col2:
                st.metric("MSE", f"{metrics['MSE']:.4f}")
            with col3:
                st.metric("RMSE", f"{metrics['RMSE']:.4f}")
            with col4:
                st.metric("MAPE", f"{metrics['MAPE']:.2f}%")
            with col5:
                st.metric("R²", f"{metrics['R2']:.4f}")
            
            # Training history
            st.subheader("Training History")
            history = st.session_state.history
            
            fig_history = go.Figure()
            fig_history.add_trace(go.Scatter(
                y=history['train_losses'],
                mode='lines',
                name='Training Loss',
                line=dict(color='blue')
            ))
            fig_history.add_trace(go.Scatter(
                y=history['val_losses'],
                mode='lines',
                name='Validation Loss',
                line=dict(color='red')
            ))
            fig_history.update_layout(
                title="Training History",
                xaxis_title="Epoch",
                yaxis_title="Loss",
                template="plotly_white"
            )
            st.plotly_chart(fig_history, use_container_width=True)
            
            # Predictions vs Actual
            st.subheader("Predictions vs Actual")
            y_test = st.session_state.y_test
            y_pred = st.session_state.y_pred
            
            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(
                y=y_test,
                mode='lines',
                name='Actual',
                line=dict(color='blue')
            ))
            fig_pred.add_trace(go.Scatter(
                y=y_pred,
                mode='lines',
                name='Predicted',
                line=dict(color='red')
            ))
            fig_pred.update_layout(
                title="Predictions vs Actual Values",
                xaxis_title="Time",
                yaxis_title="Value",
                template="plotly_white"
            )
            st.plotly_chart(fig_pred, use_container_width=True)
            
            # Residuals analysis
            st.subheader("Residuals Analysis")
            residuals = y_test - y_pred
            
            fig_residuals = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Residuals over Time', 'Residuals Distribution', 
                              'Residuals vs Fitted', 'Q-Q Plot'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Residuals over time
            fig_residuals.add_trace(
                go.Scatter(y=residuals, mode='lines', name='Residuals'),
                row=1, col=1
            )
            
            # Histogram
            fig_residuals.add_trace(
                go.Histogram(x=residuals, name='Distribution'),
                row=1, col=2
            )
            
            # Residuals vs Fitted
            fig_residuals.add_trace(
                go.Scatter(x=y_pred, y=residuals, mode='markers', name='Residuals vs Fitted'),
                row=2, col=1
            )
            
            # Q-Q plot (simplified)
            from scipy import stats
            qq_data = stats.probplot(residuals, dist="norm")
            fig_residuals.add_trace(
                go.Scatter(x=qq_data[0][0], y=qq_data[0][1], mode='markers', name='Q-Q'),
                row=2, col=2
            )
            
            fig_residuals.update_layout(height=600, showlegend=False, template="plotly_white")
            st.plotly_chart(fig_residuals, use_container_width=True)
    
    else:
        st.info("👈 Please generate data first using the sidebar controls.")
    
    # Footer
    st.markdown("---")
    st.markdown("Built with Streamlit for interactive time series analysis")


if __name__ == "__main__":
    main()
