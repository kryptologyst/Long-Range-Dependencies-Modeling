"""Data generation and preprocessing utilities for time series modeling."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, List, Optional, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class TimeSeriesDataGenerator:
    """Generate synthetic time series data with long-range dependencies."""
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize data generator.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
    
    def generate_long_dependency_data(
        self,
        n_samples: int = 2000,
        dependency_lags: List[int] = [30, 60],
        noise_scale: float = 0.1,
        trend_strength: float = 0.0,
        seasonality_strength: float = 0.0,
        seasonality_period: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic time series with long-range dependencies.
        
        Args:
            n_samples: Number of samples to generate
            dependency_lags: List of lag indices for dependencies
            noise_scale: Standard deviation of noise
            trend_strength: Strength of linear trend
            seasonality_strength: Strength of seasonal component
            seasonality_period: Period of seasonal component
            
        Returns:
            Tuple of (input_series, target_series)
        """
        logger.info(f"Generating {n_samples} samples with dependencies at lags {dependency_lags}")
        
        # Generate base input series
        x = np.random.randn(n_samples)
        
        # Add trend if specified
        if trend_strength > 0:
            trend = np.linspace(0, trend_strength, n_samples)
            x += trend
        
        # Add seasonality if specified
        if seasonality_strength > 0:
            seasonal = seasonality_strength * np.sin(2 * np.pi * np.arange(n_samples) / seasonality_period)
            x += seasonal
        
        # Generate target series with long-range dependencies
        y = np.zeros_like(x)
        max_lag = max(dependency_lags)
        
        for t in range(max_lag, n_samples):
            # Sum contributions from all dependency lags
            dependency_sum = sum(x[t - lag] for lag in dependency_lags)
            y[t] = dependency_sum + np.random.normal(scale=noise_scale)
        
        logger.info(f"Generated data with {len(y[max_lag:])} valid samples")
        return x, y
    
    def create_sequences(
        self,
        x: np.ndarray,
        y: np.ndarray,
        seq_len: int = 100,
        prediction_horizon: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create sequences for time series modeling.
        
        Args:
            x: Input time series
            y: Target time series
            seq_len: Length of input sequences
            prediction_horizon: Number of steps to predict ahead
            
        Returns:
            Tuple of (input_sequences, target_sequences)
        """
        X, Y = [], []
        
        for i in range(seq_len, len(x) - prediction_horizon + 1):
            X.append(x[i-seq_len:i])
            Y.append(y[i:i+prediction_horizon])
        
        X_tensor = torch.FloatTensor(np.array(X)).unsqueeze(1)  # Add channel dimension
        Y_tensor = torch.FloatTensor(np.array(Y))
        
        logger.info(f"Created {len(X)} sequences of length {seq_len}")
        return X_tensor, Y_tensor
    
    def split_data(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            X: Input sequences
            Y: Target sequences
            test_size: Proportion of data for testing
            val_size: Proportion of remaining data for validation
            random_state: Random seed for splitting
            
        Returns:
            Tuple of (X_train, X_val, X_test, Y_train, Y_val, Y_test)
        """
        # First split: train+val vs test
        X_temp, X_test, Y_temp, Y_test = train_test_split(
            X, Y, test_size=test_size, random_state=random_state, shuffle=False
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, Y_train, Y_val = train_test_split(
            X_temp, Y_temp, test_size=val_size_adjusted, random_state=random_state, shuffle=False
        )
        
        logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        return X_train, X_val, X_test, Y_train, Y_val, Y_test


class DataPreprocessor:
    """Preprocessing utilities for time series data."""
    
    def __init__(self, scaler_type: str = "standard"):
        """
        Initialize preprocessor.
        
        Args:
            scaler_type: Type of scaler ('standard', 'minmax', or None)
        """
        self.scaler_type = scaler_type
        self.scaler = None
        self._fit_scaler = False
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Fit scaler and transform data.
        
        Args:
            data: Input data
            
        Returns:
            Transformed data
        """
        if self.scaler_type == "standard":
            self.scaler = StandardScaler()
        elif self.scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        else:
            return data
        
        self.scaler.fit(data.reshape(-1, 1))
        self._fit_scaler = True
        
        return self.scaler.transform(data.reshape(-1, 1)).flatten()
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted scaler.
        
        Args:
            data: Input data
            
        Returns:
            Transformed data
        """
        if not self._fit_scaler:
            raise ValueError("Scaler must be fitted before transforming")
        
        if self.scaler is None:
            return data
        
        return self.scaler.transform(data.reshape(-1, 1)).flatten()
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Inverse transform data.
        
        Args:
            data: Transformed data
            
        Returns:
            Original scale data
        """
        if not self._fit_scaler or self.scaler is None:
            return data
        
        return self.scaler.inverse_transform(data.reshape(-1, 1)).flatten()


def create_data_loader(
    X: torch.Tensor,
    Y: torch.Tensor,
    batch_size: int = 32,
    shuffle: bool = True
) -> DataLoader:
    """
    Create PyTorch DataLoader.
    
    Args:
        X: Input sequences
        Y: Target sequences
        batch_size: Batch size
        shuffle: Whether to shuffle data
        
    Returns:
        DataLoader instance
    """
    dataset = TensorDataset(X, Y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
