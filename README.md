# LeverageGenerator

A deep learning project that predicts the **real leverage ratio** of the leveraged ETF **TQQQ** using market volatility signals and macroeconomic indicators.

## Overview

This project builds a deep learning model to predict the actual leverage ratio of TQQQ based on features derived from QQQ, VIX, interest rates, and other market conditions. The predicted leverage is then used to simulate TQQQ returns and assess performance accuracy.

## Prerequisites

- Python 3.8+
- CUDA 11.0+ (for GPU support)
- 8GB+ RAM recommended

## Installation

Clone this repository and install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Quick Start
```python
import yfinance as yf
from leverage_predictor import LeveragePredictor

# Load the model
predictor = LeveragePredictor('leverage_model.keras')

# Get predictions
leverage_ratio = predictor.predict(ticker='TQQQ')
```

## Example Notebook
See `examples/prediction_example.ipynb` for detailed usage examples.

## Model Description

### Architecture

- **Input shape**: (batch_size, 10, 60+)
- **Input features**: 60+ features including TQQQ price/volatility, VIX indicators, interest rates, macro signals.

### Network Structure

- **Initial feature processing**: Dense(384) → LayerNormalization

- **Attention Block 1**: MultiHeadAttention(16 heads, key_dim=64) + normalization & dropout

- **BiLSTM Block 1**: Bidirectional LSTM(384, return_sequences=True)

- **Attention Block 2**: MultiHeadAttention(12 heads, key_dim=48)

- **LSTM Blocks 2–3**: BiLSTM(256) → LSTM(192)

- **Dense Layers**:
  - Dense(256, swish) → BatchNorm + Dropout
  - Dense(128, swish) → BatchNorm
  - Dense(1, sigmoid) → OutputScaling([2.990, 3.010])

### Training Parameters

- **Batch size**: 20
- **Epochs**: 200 (10 warmup + 190 main)
- **Optimizer**: Adam(beta1=0.91, beta2=0.9995, eps=1e-8)
- **Learning rate**: 0.00008 with LR scheduler
- **Gradient clipping**: 1.0
- **Validation split**: 12%

### Custom Loss Function: tqqq_focused_loss

Combines MSE with custom penalties:
- Leverage range penalty (100.0)
- Return tracking penalty (60.0)
- Volatility penalty (40.0)
- Correlation penalty (50.0)
- Directional accuracy penalty (30.0)

### Callbacks

- **EarlyStopping**: patience=50, min_delta=1e-6, restore_best_weights=True
- **ReduceLROnPlateau**: patience=25, factor=0.5, min_lr=1e-5

### Feature Engineering

- **TQQQ-based**:
  - Price, volume, volatility
  - Moving average ratios (5, 20 days)
  - Momentum indicators (5, 10, 20 days)

- **VIX-based**:
  - VIX level and rate of change
  - Term structure signals
  - Moving averages and volatility

- **Interest rate features**:
  - 3-month and 10-year treasury rates
  - Yield curve slope and momentum

- **Market regime signals**:
  - Stress indices, trend strength, macro regimes

### Evaluation

- **Leverage Accuracy**:
  - Avg. actual leverage: 2.9986
  - Avg. predicted leverage: 3.0027
  - Correlation: -0.0388

- **Return Tracking**:
  - Correlation with actual: 0.9997
  - Annualized tracking error: 1.62%

- **Risk Metrics**:
  - Daily volatility (actual): 62.78%
  - Daily volatility (predicted): 63.52%
  - Max drawdown (actual): 58.04%
  - Max drawdown (predicted): 58.10%

## Performance

![Cumulative Returns](assets/cumulative_returns.png)
*Cumulative Returns Comparison (Initial Value: 100)*
- Blue: Actual TQQQ
- Orange: Predicted TQQQ
- Green (dashed): 3x QQQ

### Performance Metrics
- **Returns Tracking**
  - Returns Correlation: 0.9997
  - Annualized Tracking Error: 1.62%

- **Leverage Accuracy**
  - Average Actual Leverage: 2.9986
  - Average Predicted Leverage: 3.0027
  - Leverage Range: [2.990, 3.010]

- **Risk Metrics**
  - Daily Volatility (Actual/Predicted): 62.78% / 63.52%
  - Maximum Drawdown (Actual/Predicted): 58.04% / 58.10%
## Model Files

- **Trained model**: `leverage_model.keras`
- **Scaler**: `leverage_scaler.pkl`
- **Feature names**: `feature_names.json`

## License

This project is licensed under the MIT License.



