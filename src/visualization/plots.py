"""Comprehensive visualization utilities for time series analysis."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Dict, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class TimeSeriesVisualizer:
    """Comprehensive visualization class for time series analysis."""
    
    def __init__(self, style: str = "seaborn-v0_8", figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize visualizer.
        
        Args:
            style: Matplotlib style
            figsize: Default figure size
        """
        self.style = style
        self.figsize = figsize
        plt.style.use(style)
        
    def plot_time_series(
        self,
        data: np.ndarray,
        title: str = "Time Series",
        xlabel: str = "Time",
        ylabel: str = "Value",
        save_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """
        Plot a simple time series.
        
        Args:
            data: Time series data
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            save_path: Path to save the plot
            show: Whether to display the plot
        """
        plt.figure(figsize=self.figsize)
        plt.plot(data)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Predictions vs Actual",
        save_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """
        Plot predictions against actual values.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            title: Plot title
            save_path: Path to save the plot
            show: Whether to display the plot
        """
        plt.figure(figsize=self.figsize)
        
        # Plot actual and predicted values
        plt.plot(y_true, label="Actual", alpha=0.8)
        plt.plot(y_pred, label="Predicted", alpha=0.8)
        
        # Add scatter plot for better visibility
        plt.scatter(range(len(y_true)), y_true, alpha=0.3, s=10, label="Actual points")
        plt.scatter(range(len(y_pred)), y_pred, alpha=0.3, s=10, label="Predicted points")
        
        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_residuals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Residuals Analysis",
        save_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """
        Plot residual analysis.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            title: Plot title
            save_path: Path to save the plot
            show: Whether to display the plot
        """
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title)
        
        # Residuals over time
        axes[0, 0].plot(residuals)
        axes[0, 0].set_title("Residuals over Time")
        axes[0, 0].set_xlabel("Time")
        axes[0, 0].set_ylabel("Residuals")
        axes[0, 0].grid(True, alpha=0.3)
        
        # Histogram of residuals
        axes[0, 1].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title("Distribution of Residuals")
        axes[0, 1].set_xlabel("Residuals")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title("Q-Q Plot")
        axes[1, 0].grid(True, alpha=0.3)
        
        # Residuals vs Fitted
        axes[1, 1].scatter(y_pred, residuals, alpha=0.6)
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].set_title("Residuals vs Fitted")
        axes[1, 1].set_xlabel("Fitted Values")
        axes[1, 1].set_ylabel("Residuals")
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_training_history(
        self,
        train_losses: List[float],
        val_losses: List[float],
        title: str = "Training History",
        save_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """
        Plot training history.
        
        Args:
            train_losses: Training losses
            val_losses: Validation losses
            title: Plot title
            save_path: Path to save the plot
            show: Whether to display the plot
        """
        plt.figure(figsize=self.figsize)
        
        epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, label="Training Loss", marker='o')
        plt.plot(epochs, val_losses, label="Validation Loss", marker='s')
        
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_interactive_time_series(
        self,
        data: Dict[str, np.ndarray],
        title: str = "Interactive Time Series",
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create interactive time series plot with Plotly.
        
        Args:
            data: Dictionary of time series data
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        for name, series in data.items():
            fig.add_trace(go.Scatter(
                y=series,
                mode='lines',
                name=name,
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Value",
            hovermode='x unified',
            template="plotly_white"
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Interactive plot saved to {save_path}")
        
        return fig
    
    def plot_model_comparison(
        self,
        results: Dict[str, Dict[str, Any]],
        metric: str = "RMSE",
        title: str = "Model Comparison",
        save_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """
        Plot model comparison bar chart.
        
        Args:
            results: Dictionary with model results
            metric: Metric to compare
            title: Plot title
            save_path: Path to save the plot
            show: Whether to display the plot
        """
        models = list(results.keys())
        values = [results[model].get(metric, 0) for model in models]
        
        plt.figure(figsize=self.figsize)
        bars = plt.bar(models, values, alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.4f}', ha='center', va='bottom')
        
        plt.title(title)
        plt.xlabel("Model")
        plt.ylabel(metric)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_correlation_heatmap(
        self,
        data: np.ndarray,
        title: str = "Correlation Heatmap",
        save_path: Optional[str] = None,
        show: bool = True
    ) -> None:
        """
        Plot correlation heatmap for time series data.
        
        Args:
            data: Time series data
            title: Plot title
            save_path: Path to save the plot
            show: Whether to display the plot
        """
        # Create lagged features for correlation analysis
        max_lags = min(20, len(data) // 10)
        lagged_data = np.zeros((len(data) - max_lags, max_lags + 1))
        
        for i in range(max_lags + 1):
            lagged_data[:, i] = data[i:len(data) - max_lags + i]
        
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(lagged_data.T)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   xticklabels=[f'Lag {i}' for i in range(max_lags + 1)],
                   yticklabels=[f'Lag {i}' for i in range(max_lags + 1)])
        
        plt.title(title)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()


def create_dashboard_data(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    train_losses: List[float],
    val_losses: List[float],
    metrics: Dict[str, float]
) -> Dict[str, Any]:
    """
    Create data structure for dashboard visualization.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        train_losses: Training losses
        val_losses: Validation losses
        metrics: Evaluation metrics
        
    Returns:
        Dictionary with dashboard data
    """
    return {
        'predictions': {
            'actual': y_true.tolist(),
            'predicted': y_pred.tolist(),
            'residuals': (y_true - y_pred).tolist()
        },
        'training_history': {
            'epochs': list(range(1, len(train_losses) + 1)),
            'train_losses': train_losses,
            'val_losses': val_losses
        },
        'metrics': metrics
    }
