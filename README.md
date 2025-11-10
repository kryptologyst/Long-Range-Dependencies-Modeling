# Long-Range Dependencies Modeling

A comprehensive time series analysis project focused on modeling long-range dependencies using state-of-the-art neural network architectures including Temporal Convolutional Networks (TCN), LSTM, and Transformer models.

## Overview

This project addresses the challenge of capturing long-range dependencies in time series data - patterns or correlations that span across long time horizons. Traditional time series models often struggle with these dependencies, making this an important area of research and application.

## Features

- **Multiple Model Architectures**: TCN, LSTM, and Transformer models for different types of long-range dependencies
- **Synthetic Data Generation**: Realistic time series data with configurable dependency patterns
- **Interactive Web Interface**: Streamlit-based dashboard for model exploration and visualization
- **Comprehensive Evaluation**: Multiple metrics including MAE, MSE, RMSE, MAPE, and R²
- **Modern Python Practices**: Type hints, comprehensive logging, configuration management
- **Extensible Design**: Easy to add new models and datasets

## Project Structure

```
├── src/
│   ├── data/
│   │   └── data_generator.py      # Data generation and preprocessing
│   ├── models/
│   │   ├── neural_models.py        # Neural network implementations
│   │   └── training.py             # Training utilities and metrics
│   ├── utils/
│   │   ├── config.py               # Configuration management
│   │   └── logging_config.py       # Logging setup
│   ├── visualization/
│   │   └── plots.py                # Visualization utilities
│   └── streamlit_app.py            # Interactive web interface
├── config/
│   └── config.yaml                 # Configuration file
├── tests/
│   └── test_components.py          # Unit tests
├── notebooks/                      # Jupyter notebooks for analysis
├── data/                           # Data storage
├── models/                         # Saved model checkpoints
├── logs/                           # Training logs
├── plots/                          # Generated plots
├── requirements.txt                # Python dependencies
├── train.py                        # Main training script
└── README.md                       # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Long-Range-Dependencies-Modeling.git
cd Long-Range-Dependencies-Modeling
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Interactive Web Interface

Launch the Streamlit dashboard for interactive exploration:

```bash
streamlit run src/streamlit_app.py
```

This will open a web interface where you can:
- Generate synthetic time series data with configurable parameters
- Train different model types (TCN, LSTM, Transformer)
- Visualize results and compare model performance
- Analyze residuals and training history

### 2. Command Line Training

Train models from the command line:

```bash
# Train TCN model
python train.py --model tcn --epochs 50 --batch-size 32

# Train LSTM model
python train.py --model lstm --epochs 50 --learning-rate 0.001

# Train Transformer model
python train.py --model transformer --epochs 50 --save-model models/transformer.pth
```

### 3. Configuration

Modify `config/config.yaml` to customize:
- Data generation parameters
- Model architectures
- Training hyperparameters
- Visualization settings

## Model Architectures

### Temporal Convolutional Network (TCN)
- Uses dilated convolutions to capture long-range dependencies
- Causal structure prevents information leakage
- Efficient training and inference

### LSTM
- Traditional recurrent architecture
- Bidirectional option available
- Good for sequential patterns

### Transformer
- Attention-based architecture
- Excellent for long-range dependencies
- Positional encoding for temporal information

## Usage Examples

### Basic Training

```python
from src.data.data_generator import TimeSeriesDataGenerator
from src.models.neural_models import ModelFactory
from src.models.training import ModelTrainer
from src.utils.config import Config

# Generate data
generator = TimeSeriesDataGenerator(random_seed=42)
x, y = generator.generate_long_dependency_data(
    n_samples=2000,
    dependency_lags=[30, 60],
    noise_scale=0.1
)

# Create sequences
X, Y = generator.create_sequences(x, y, seq_len=100)

# Create model
model = ModelFactory.create_model("tcn", {
    "num_channels": [16, 32, 64],
    "kernel_size": 3,
    "dropout": 0.1
})

# Train model
config = Config()
trainer = ModelTrainer(model, config)
history = trainer.train(train_loader, val_loader)
```

### Custom Data Integration

```python
import pandas as pd
import numpy as np

# Load your own data
df = pd.read_csv("your_data.csv")
x = df["input_column"].values
y = df["target_column"].values

# Use the same pipeline
generator = TimeSeriesDataGenerator()
X, Y = generator.create_sequences(x, y, seq_len=100)
# ... continue with training
```

## Evaluation Metrics

The project provides comprehensive evaluation metrics:

- **MAE**: Mean Absolute Error
- **MSE**: Mean Squared Error  
- **RMSE**: Root Mean Squared Error
- **MAPE**: Mean Absolute Percentage Error
- **R²**: Coefficient of Determination

## Visualization

The project includes extensive visualization capabilities:

- Time series plots with actual vs predicted values
- Training history (loss curves)
- Residual analysis (distribution, Q-Q plots, residuals vs fitted)
- Interactive plots with Plotly
- Model comparison charts

## Testing

Run the test suite:

```bash
pytest tests/
```

Or run specific test categories:

```bash
pytest tests/test_components.py::TestNeuralModels
pytest tests/test_components.py::TestDataGenerator
```

## Configuration

The project uses YAML configuration files for easy customization:

```yaml
# Data generation
data:
  n_samples: 2000
  noise_scale: 0.1
  dependency_lags: [30, 60]

# Model configuration
models:
  tcn:
    num_channels: [16, 32, 64]
    kernel_size: 3
    dropout: 0.1

# Training parameters
training:
  batch_size: 32
  learning_rate: 0.001
  epochs: 50
```

## Advanced Features

### Early Stopping
Prevents overfitting with configurable patience and minimum delta.

### Model Checkpointing
Save and load trained models for inference or continued training.

### Comprehensive Logging
Detailed logging of training progress, metrics, and model information.

### Extensible Architecture
Easy to add new models, datasets, or evaluation metrics.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-model`)
3. Commit your changes (`git commit -am 'Add new model'`)
4. Push to the branch (`git push origin feature/new-model`)
5. Create a Pull Request

## Requirements

- Python 3.10+
- PyTorch 2.0+
- Streamlit 1.25+
- Plotly 5.15+
- NumPy 1.24+
- Pandas 2.0+

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{long_range_dependencies,
  title={Long-Range Dependencies Modeling},
  author={Kryptologyst},
  year={2025},
  url={https://github.com/kryptologyst/Long-Range-Dependencies-Modeling}
}
```

## Acknowledgments

- PyTorch team for the excellent deep learning framework
- Streamlit team for the interactive web framework
- Plotly team for the visualization library
- The time series research community for foundational work
# Long-Range-Dependencies-Modeling
