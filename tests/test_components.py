"""Unit tests for time series modeling components."""

import pytest
import numpy as np
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data.data_generator import TimeSeriesDataGenerator, DataPreprocessor, create_data_loader
from src.models.neural_models import TemporalConvNet, LSTMModel, TransformerModel, ModelFactory
from src.models.training import EarlyStopping, ModelTrainer, calculate_metrics
from src.utils.config import Config


class TestTimeSeriesDataGenerator:
    """Test cases for TimeSeriesDataGenerator."""
    
    def test_data_generation(self):
        """Test synthetic data generation."""
        generator = TimeSeriesDataGenerator(random_seed=42)
        x, y = generator.generate_long_dependency_data(
            n_samples=100,
            dependency_lags=[10, 20],
            noise_scale=0.1
        )
        
        assert len(x) == 100
        assert len(y) == 100
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
    
    def test_sequence_creation(self):
        """Test sequence creation for training."""
        generator = TimeSeriesDataGenerator(random_seed=42)
        x = np.random.randn(200)
        y = np.random.randn(200)
        
        X, Y = generator.create_sequences(x, y, seq_len=50)
        
        assert X.shape[0] == Y.shape[0]
        assert X.shape[1] == 1  # Channel dimension
        assert X.shape[2] == 50  # Sequence length
        assert isinstance(X, torch.Tensor)
        assert isinstance(Y, torch.Tensor)
    
    def test_data_splitting(self):
        """Test data splitting functionality."""
        generator = TimeSeriesDataGenerator(random_seed=42)
        X = torch.randn(100, 1, 50)
        Y = torch.randn(100, 1)
        
        X_train, X_val, X_test, Y_train, Y_val, Y_test = generator.split_data(
            X, Y, test_size=0.2, val_size=0.1
        )
        
        assert len(X_train) + len(X_val) + len(X_test) == len(X)
        assert len(Y_train) + len(Y_val) + len(Y_test) == len(Y)
        assert len(X_test) == 20  # 20% of 100
        assert len(X_val) == 8   # 10% of remaining 80


class TestDataPreprocessor:
    """Test cases for DataPreprocessor."""
    
    def test_standard_scaler(self):
        """Test standard scaling."""
        preprocessor = DataPreprocessor(scaler_type="standard")
        data = np.array([1, 2, 3, 4, 5])
        
        transformed = preprocessor.fit_transform(data)
        
        assert np.isclose(np.mean(transformed), 0, atol=1e-10)
        assert np.isclose(np.std(transformed), 1, atol=1e-10)
    
    def test_minmax_scaler(self):
        """Test min-max scaling."""
        preprocessor = DataPreprocessor(scaler_type="minmax")
        data = np.array([1, 2, 3, 4, 5])
        
        transformed = preprocessor.fit_transform(data)
        
        assert np.min(transformed) == 0
        assert np.max(transformed) == 1
    
    def test_inverse_transform(self):
        """Test inverse transformation."""
        preprocessor = DataPreprocessor(scaler_type="standard")
        original_data = np.array([1, 2, 3, 4, 5])
        
        transformed = preprocessor.fit_transform(original_data)
        inverse_transformed = preprocessor.inverse_transform(transformed)
        
        assert np.allclose(original_data, inverse_transformed, atol=1e-10)


class TestNeuralModels:
    """Test cases for neural network models."""
    
    def test_tcn_model(self):
        """Test Temporal Convolutional Network."""
        model = TemporalConvNet(
            input_channels=1,
            num_channels=[16, 32],
            kernel_size=3
        )
        
        x = torch.randn(32, 1, 100)  # batch_size, channels, seq_len
        output = model(x)
        
        assert output.shape == (32,)  # batch_size
        assert isinstance(output, torch.Tensor)
    
    def test_lstm_model(self):
        """Test LSTM model."""
        model = LSTMModel(
            input_size=1,
            hidden_size=32,
            num_layers=2
        )
        
        x = torch.randn(32, 100, 1)  # batch_size, seq_len, input_size
        output = model(x)
        
        assert output.shape == (32,)  # batch_size
        assert isinstance(output, torch.Tensor)
    
    def test_transformer_model(self):
        """Test Transformer model."""
        model = TransformerModel(
            input_size=1,
            d_model=32,
            nhead=4,
            num_layers=2
        )
        
        x = torch.randn(32, 100, 1)  # batch_size, seq_len, input_size
        output = model(x)
        
        assert output.shape == (32,)  # batch_size
        assert isinstance(output, torch.Tensor)
    
    def test_model_factory(self):
        """Test ModelFactory."""
        config = {"num_channels": [16, 32], "kernel_size": 3}
        
        tcn_model = ModelFactory.create_model("tcn", config, input_size=1)
        assert isinstance(tcn_model, TemporalConvNet)
        
        lstm_config = {"hidden_size": 32, "num_layers": 2}
        lstm_model = ModelFactory.create_model("lstm", lstm_config, input_size=1)
        assert isinstance(lstm_model, LSTMModel)
        
        transformer_config = {"d_model": 32, "nhead": 4, "num_layers": 2}
        transformer_model = ModelFactory.create_model("transformer", transformer_config, input_size=1)
        assert isinstance(transformer_model, TransformerModel)
    
    def test_model_info(self):
        """Test model information extraction."""
        model = TemporalConvNet(input_channels=1, num_channels=[16, 32])
        info = ModelFactory.get_model_info(model)
        
        assert "model_type" in info
        assert "total_parameters" in info
        assert "trainable_parameters" in info
        assert info["total_parameters"] > 0


class TestTraining:
    """Test cases for training components."""
    
    def test_early_stopping(self):
        """Test early stopping functionality."""
        early_stopping = EarlyStopping(patience=3)
        
        # Mock model
        model = torch.nn.Linear(1, 1)
        
        # Test improvement
        assert not early_stopping(0.5, model)
        assert not early_stopping(0.4, model)
        assert not early_stopping(0.3, model)
        
        # Test no improvement
        assert not early_stopping(0.3, model)
        assert not early_stopping(0.3, model)
        assert not early_stopping(0.3, model)
        assert early_stopping(0.3, model)  # Should trigger early stopping
    
    def test_calculate_metrics(self):
        """Test metrics calculation."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        
        metrics = calculate_metrics(y_true, y_pred)
        
        assert "MAE" in metrics
        assert "MSE" in metrics
        assert "RMSE" in metrics
        assert "MAPE" in metrics
        assert "R2" in metrics
        
        # All metrics should be positive (except R2 which can be negative)
        assert metrics["MAE"] > 0
        assert metrics["MSE"] > 0
        assert metrics["RMSE"] > 0
        assert metrics["MAPE"] > 0


class TestConfig:
    """Test cases for configuration management."""
    
    def test_config_creation(self):
        """Test configuration creation."""
        config = Config()
        
        # Test getting default values
        assert config.get("nonexistent.key", "default") == "default"
        
        # Test setting and getting values
        config.set("test.key", "test_value")
        assert config.get("test.key") == "test_value"
    
    def test_config_dict_conversion(self):
        """Test configuration to dictionary conversion."""
        config = Config()
        config.set("test.key1", "value1")
        config.set("test.key2", "value2")
        
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict["test"]["key1"] == "value1"
        assert config_dict["test"]["key2"] == "value2"


class TestDataLoader:
    """Test cases for data loader creation."""
    
    def test_data_loader_creation(self):
        """Test data loader creation."""
        X = torch.randn(100, 1, 50)
        Y = torch.randn(100, 1)
        
        loader = create_data_loader(X, Y, batch_size=32, shuffle=True)
        
        assert isinstance(loader, torch.utils.data.DataLoader)
        assert loader.batch_size == 32
        
        # Test iteration
        for batch_x, batch_y in loader:
            assert batch_x.shape[0] <= 32
            assert batch_x.shape[1] == 1
            assert batch_x.shape[2] == 50
            assert batch_y.shape[0] <= 32
            break  # Just test first batch


if __name__ == "__main__":
    pytest.main([__file__])
