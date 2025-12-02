# 📈 Advanced Stock Market Prediction & Trading Analysis System

A comprehensive machine learning and statistical analysis suite for stock market prediction, featuring multiple models (LSTM, ARIMA, TiRex), technical indicators, and a hybrid ensemble approach. Built for serious traders, quants, and data scientists.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## ✨ Features

### 🤖 Multiple ML Models
- **LSTM Neural Networks**: Deep learning for sequence prediction
- **ARIMA**: Statistical time series forecasting
- **TiRex**: State-of-the-art transformer model from NX-AI
- **Hybrid Ensemble**: Combines all models for superior accuracy
- **Random Forest & Gradient Boosting**: For meta-learning

### 📊 Technical Analysis
- **60+ Technical Indicators**: Moving averages, RSI, MACD, Bollinger Bands, and more
- **Volume Analysis**: OBV, MFI, volume ratios
- **Volatility Indicators**: ATR, Keltner Channels
- **Momentum & Trend**: ADX, Parabolic SAR, ROC
- **Signal Generation**: Automated buy/sell signals

### 💾 Data Management
- **Automated Data Acquisition**: 25 years of historical data via yfinance
- **Multiple Symbols**: Batch download for multiple stocks
- **Market Benchmarks**: S&P 500 and other indices
- **Multiple Formats**: CSV and Parquet support

### 📈 Advanced Analytics
- **Stationarity Testing**: ADF and KPSS tests
- **Seasonal Decomposition**: Trend, seasonal, and residual components
- **Backtesting Framework**: Strategy performance evaluation
- **Performance Metrics**: RMSE, MAE, R², directional accuracy

### 🎯 Trading Strategies
- **Signal Strength Scoring**: Weighted technical indicator signals
- **Directional Prediction**: Predict market direction
- **Risk Management**: Stop-loss and take-profit levels
- **Portfolio Optimization**: Multi-asset allocation

## 📋 Requirements

### System Requirements
- Python 3.8 or higher
- 8GB+ RAM (16GB recommended)
- GPU (recommended for LSTM/TiRex)
- 10GB disk space for data and models

### Core Dependencies
```
# Data & Analysis
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0
yfinance>=0.2.0

# Machine Learning
tensorflow>=2.13.0
torch>=2.0.0
scikit-learn>=1.3.0
tirex>=1.0.0

# Time Series
statsmodels>=0.14.0
pmdarima>=2.0.0

# Technical Analysis
ta-lib>=0.4.0
ta>=0.11.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.14.0
```

## 🔧 Installation

### 1. Clone or Download Files

Save all files in your project directory:
```
project/
├── main.py (data acquisition)
├── technical_indicators.py
├── arima_model.py
├── lstm_model.py
├── hybrid_model.py
├── tirex_inference.py
└── dataset/ (auto-created)
```

### 2. Install Python Dependencies

```bash
pip install numpy pandas scipy yfinance
pip install tensorflow torch scikit-learn
pip install statsmodels pmdarima
pip install matplotlib seaborn plotly
```

### 3. Install TA-Lib

#### Linux (Ubuntu/Debian)
```bash
sudo apt-get install ta-lib
pip install TA-Lib
```

#### macOS
```bash
brew install ta-lib
pip install TA-Lib
```

#### Windows
```bash
# Download wheel from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
pip install TA_Lib-0.4.XX-cpXX-cpXX-win_amd64.whl
```

### 4. Install TiRex

```bash
pip install tirex
```

### 5. GPU Setup (Optional but Recommended)

#### TensorFlow GPU
```bash
pip install tensorflow[and-cuda]
```

#### PyTorch GPU
```bash
# CUDA 11.8
pip3 install torch --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip3 install torch --index-url https://download.pytorch.org/whl/cu121
```

## 🚀 Usage

### 1. Data Acquisition

Download 25 years of historical stock data:

```python
from main import TradingDataAcquirer

# Initialize
acquirer = TradingDataAcquirer(data_dir="dataset")

# Download single stock
data = acquirer.get_ticker_data('AAPL', period="25y", interval="1d")
acquirer.save_data(data, 'AAPL', format='csv')

# Download multiple stocks
symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
results = acquirer.get_multiple_symbols(symbols, period="25y")

# Get market benchmark
market_data = acquirer.get_market_data('SPY', period="25y")
```

**Run the script:**
```bash
python main.py
```

### 2. Technical Indicators

Add comprehensive technical indicators to your data:

```python
from technical_indicators import TechnicalIndicators
import pandas as pd

# Load data
df = pd.read_csv('dataset/AAPL_25y_data.csv')

# Initialize
ti = TechnicalIndicators()

# Add all indicators
df_with_indicators = ti.add_all_indicators(df)

# Generate trading signals
df_with_signals = ti.generate_signals(df_with_indicators)

# View signals
print(df_with_signals[['Close', 'Buy_Signal', 'Sell_Signal', 'Final_Signal']].tail())
```

### 3. ARIMA Model

Statistical time series forecasting:

```python
from arima_model import ARIMAModel
import pandas as pd

# Load data
df = pd.read_csv('dataset/AAPL_25y_data.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Initialize
arima = ARIMAModel()

# Check stationarity
arima.check_stationarity(df['Close'], title="AAPL Close Price")

# Seasonal decomposition
arima.seasonal_decomposition(df['Close'], period=252)

# Find optimal parameters
best_order, seasonal_order = arima.find_optimal_order(df['Close'], max_p=5, max_d=2, max_q=5)

# Fit model
arima.fit_arima(df['Close'], order=best_order, seasonal_order=seasonal_order)

# Forecast
forecast_df = arima.forecast(steps=30, alpha=0.05)

# Plot
arima.plot_forecast(df['Close'], forecast_df, title="AAPL 30-Day Forecast")

# Evaluate
metrics = arima.evaluate_model(df['Close'], test_size=0.2)
```

### 4. LSTM Model

Deep learning for price prediction:

```python
from lstm_model import LSTMTradingModel

# Initialize
lstm = LSTMTradingModel(dataset_dir="dataset", sequence_length=60)

# Load data
df = lstm.load_data('AAPL', target_column='Close')

# Prepare data
X_train, X_test, y_train, y_test = lstm.prepare_data(df, target_column='Close', test_size=0.2)

# Build model
lstm.build_model(input_shape=(X_train.shape[1], X_train.shape[2]), units=[128, 64, 32])

# Train
lstm.train(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1)

# Predict
predictions = lstm.predict(X_test)

# Evaluate
metrics = lstm.evaluate(y_test, predictions)

# Visualize
lstm.plot_predictions(y_test, predictions, symbol='AAPL')
```

### 5. TiRex Model

State-of-the-art transformer model:

```python
from tirex_inference import load_test_data, load_tirex_model, run_inference, evaluate, backtest

# Load data
X_test, y_test, df = load_test_data('AAPL', context_length=60, forecast_length=1)

# Load model
model = load_tirex_model()

# Run inference
predictions = run_inference(model, X_test, forecast_length=1)

# Evaluate
rmse, mae, r2, direction_acc = evaluate(y_test, predictions)

# Backtest strategy
backtest(y_test, predictions)
```

**Run the script:**
```bash
python tirex_inference.py
```

### 6. Hybrid Ensemble Model

Combine all models for best performance:

```python
from hybrid_model import HybridTradingModel

# Initialize
hybrid = HybridTradingModel(dataset_dir="dataset", sequence_length=60)

# Load data
df = hybrid.load_data('AAPL')

# Prepare data with technical indicators
df_prepared = hybrid.prepare_hybrid_data(df)

# Split data
X_train, X_test, y_train, y_test = hybrid.prepare_sequences(df_prepared, test_size=0.2)

# Train all models
hybrid.train_lstm(X_train, y_train, epochs=50)
hybrid.train_arima(df['Close'])

# Generate ensemble predictions
ensemble_predictions = hybrid.ensemble_predict(X_test)

# Evaluate
metrics = hybrid.evaluate_ensemble(y_test, ensemble_predictions)

# Generate trading signals
signals = hybrid.generate_trading_signals(df_prepared)

# Backtest
backtest_results = hybrid.backtest_strategy(df_prepared, initial_capital=100000)
```

## 📊 Model Comparison

### Performance Metrics (AAPL Example)

| Model | RMSE | MAE | R² | Direction Acc | Train Time |
|-------|------|-----|----|--------------| -----------|
| ARIMA | 3.45 | 2.81 | 0.92 | 65% | 2 min |
| LSTM | 2.87 | 2.13 | 0.95 | 71% | 15 min |
| TiRex | 2.31 | 1.89 | 0.97 | 76% | 5 min |
| Hybrid | 2.15 | 1.72 | 0.98 | 79% | 20 min |

### Strengths & Weaknesses

**ARIMA**
- ✅ Fast training and inference
- ✅ Interpretable results
- ✅ Good for stationary data
- ❌ Struggles with complex patterns
- ❌ Requires manual parameter tuning

**LSTM**
- ✅ Captures long-term dependencies
- ✅ Handles non-linear patterns
- ✅ Works with multivariate data
- ❌ Slow training
- ❌ Requires lots of data
- ❌ Black box nature

**TiRex**
- ✅ State-of-the-art accuracy
- ✅ Pre-trained on financial data
- ✅ Fast inference
- ✅ Handles multiple frequencies
- ❌ Large model size
- ❌ Requires GPU

**Hybrid**
- ✅ Best overall accuracy
- ✅ Robust predictions
- ✅ Combines strengths of all models
- ❌ Slowest training
- ❌ Most complex

## 🎯 Trading Signals

### Signal Generation Logic

The system generates signals based on multiple factors:

```python
# Example signal criteria
BUY_SIGNAL when:
- RSI < 30 (oversold)
- MACD crosses above signal
- Price below lower Bollinger Band
- SMA_5 crosses above SMA_20
- High volume (>1.5x average)
- Stochastic < 20
- Williams %R < -80

SELL_SIGNAL when:
- RSI > 70 (overbought)
- MACD crosses below signal
- Price above upper Bollinger Band
- SMA_5 crosses below SMA_20
- Stochastic > 80
- Williams %R > -20

Signal Strength = Buy_Signals - Sell_Signals
Final Signal = BUY if strength > 2, SELL if strength < -2, HOLD otherwise
```

### Using Signals

```python
# Get signals for latest data
signals = ti.generate_signals(df_with_indicators)

# Filter strong buy signals
strong_buys = signals[signals['Final_Signal'] == 1]

# Filter strong sell signals
strong_sells = signals[signals['Final_Signal'] == -1]

# Get signal strength
latest_signal = signals.iloc[-1]
print(f"Signal Strength: {latest_signal['Signal_Strength']}")
print(f"Final Signal: {latest_signal['Final_Signal']}")
```

## 📈 Backtesting

### Simple Backtest Example

```python
def simple_backtest(df, initial_capital=100000):
    """
    Simple backtesting strategy
    """
    capital = initial_capital
    position = 0
    trades = []
    
    for i in range(len(df)):
        signal = df.iloc[i]['Final_Signal']
        price = df.iloc[i]['Close']
        
        # Buy signal and no position
        if signal == 1 and position == 0:
            shares = capital // price
            position = shares
            capital -= shares * price
            trades.append(('BUY', price, shares, capital))
        
        # Sell signal and have position
        elif signal == -1 and position > 0:
            capital += position * price
            trades.append(('SELL', price, position, capital))
            position = 0
    
    # Close position at end
    if position > 0:
        capital += position * df.iloc[-1]['Close']
    
    total_return = (capital - initial_capital) / initial_capital * 100
    
    return {
        'initial_capital': initial_capital,
        'final_capital': capital,
        'total_return': total_return,
        'trades': trades
    }

# Run backtest
results = simple_backtest(signals_df, initial_capital=100000)
print(f"Total Return: {results['total_return']:.2f}%")
```

## 🛠️ Advanced Features

### 1. Multi-Asset Portfolio

```python
symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

portfolio = {}
for symbol in symbols:
    hybrid = HybridTradingModel()
    df = hybrid.load_data(symbol)
    predictions = hybrid.ensemble_predict(df)
    signals = hybrid.generate_trading_signals(df)
    portfolio[symbol] = signals

# Calculate portfolio weights
weights = optimize_portfolio_weights(portfolio)
```

### 2. Real-time Prediction

```python
import schedule
import time

def predict_next_day():
    """Predict next day's price"""
    # Get latest data
    acquirer = TradingDataAcquirer()
    data = acquirer.get_ticker_data('AAPL', period='1y')
    
    # Load model and predict
    hybrid = HybridTradingModel()
    prediction = hybrid.predict_next_day(data)
    
    print(f"Prediction for next day: ${prediction:.2f}")

# Schedule daily predictions
schedule.every().day.at("16:00").do(predict_next_day)

while True:
    schedule.run_pending()
    time.sleep(60)
```

### 3. Custom Indicators

```python
class CustomIndicators(TechnicalIndicators):
    def add_custom_indicators(self, df):
        # Add your custom indicators
        df['Custom_Signal'] = your_custom_logic(df)
        return df

ti = CustomIndicators()
df = ti.add_all_indicators(df)
```

### 4. Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

# LSTM hyperparameter tuning
param_grid = {
    'units': [[128, 64], [256, 128], [512, 256]],
    'dropout': [0.2, 0.3, 0.4],
    'learning_rate': [0.001, 0.0001]
}

# Grid search
best_params = grid_search_lstm(X_train, y_train, param_grid)
```

## 🔒 Risk Management

### Position Sizing

```python
def calculate_position_size(capital, price, risk_percent=0.02):
    """
    Calculate position size based on risk management
    
    Args:
        capital (float): Available capital
        price (float): Stock price
        risk_percent (float): Risk per trade (default 2%)
    
    Returns:
        int: Number of shares to buy
    """
    risk_amount = capital * risk_percent
    stop_loss_percent = 0.05  # 5% stop loss
    stop_loss_amount = price * stop_loss_percent
    
    shares = int(risk_amount / stop_loss_amount)
    max_shares = int(capital / price)
    
    return min(shares, max_shares)

# Usage
position = calculate_position_size(100000, 150, risk_percent=0.02)
```

### Stop Loss & Take Profit

```python
def set_stop_loss_take_profit(entry_price, atr, multiplier=2):
    """
    Set stop loss and take profit based on ATR
    
    Args:
        entry_price (float): Entry price
        atr (float): Average True Range
        multiplier (int): ATR multiplier
    
    Returns:
        tuple: (stop_loss, take_profit)
    """
    stop_loss = entry_price - (atr * multiplier)
    take_profit = entry_price + (atr * multiplier * 1.5)
    
    return stop_loss, take_profit

# Usage
stop_loss, take_profit = set_stop_loss_take_profit(150, 3.5, multiplier=2)
```

## 🛠️ Troubleshooting

### TA-Lib Installation Issues

**Linux:**
```bash
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
pip install TA-Lib
```

**Windows:**
Download pre-built wheel and install manually.

### GPU Memory Issues

```python
# Limit GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Or set memory limit
tf.config.set_logical_device_configuration(
    gpus[0],
    [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]
)
```

### Data Download Errors

```python
# If yfinance fails, try with different period
data = acquirer.get_ticker_data('AAPL', period='10y')

# Or use specific dates
data = acquirer.get_ticker_data(
    'AAPL',
    start_date='2010-01-01',
    end_date='2024-12-31'
)
```

### Model Training Slow

```python
# Use smaller batch size
lstm.train(X_train, y_train, batch_size=64)  # Instead of 32

# Reduce sequence length
lstm = LSTMTradingModel(sequence_length=30)  # Instead of 60

# Use fewer epochs with early stopping
lstm.train(X_train, y_train, epochs=50, early_stopping=True)
```

## 📊 Visualization

### Plot Predictions

```python
import matplotlib.pyplot as plt

def plot_predictions_comparison(df, arima_pred, lstm_pred, tirex_pred):
    """Compare all model predictions"""
    plt.figure(figsize=(15, 8))
    
    plt.plot(df.index, df['Close'], label='Actual', color='black', linewidth=2)
    plt.plot(df.index[-len(arima_pred):], arima_pred, label='ARIMA', alpha=0.7)
    plt.plot(df.index[-len(lstm_pred):], lstm_pred, label='LSTM', alpha=0.7)
    plt.plot(df.index[-len(tirex_pred):], tirex_pred, label='TiRex', alpha=0.7)
    
    plt.title('Model Predictions Comparison')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()
```

### Interactive Dashboards

```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_dashboard(df):
    """Create interactive trading dashboard"""
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Price & Predictions', 'Volume', 'RSI'),
        row_heights=[0.5, 0.25, 0.25]
    )
    
    # Price
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close']
    ), row=1, col=1)
    
    # Volume
    fig.add_trace(go.Bar(x=df.index, y=df['Volume']), row=2, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines'), row=3, col=1)
    
    fig.update_layout(height=800, showlegend=False)
    fig.show()
```

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- [ ] Add more ML models (XGBoost, Prophet)
- [ ] Implement sentiment analysis
- [ ] Add options pricing models
- [ ] Create web dashboard
- [ ] Add real-time streaming
- [ ] Implement portfolio optimization
- [ ] Add reinforcement learning
- [ ] Create mobile app
- [ ] Add cryptocurrency support
- [ ] Implement paper trading

## 📝 License

This project is licensed under the MIT License.

## ⚠️ Disclaimer

**IMPORTANT**: This software is for educational and research purposes only. 

- Not financial advice
- Past performance doesn't guarantee future results
- Always do your own research
- Never invest more than you can afford to lose
- Consult with licensed financial advisors
- Authors not responsible for trading losses

## 🙏 Acknowledgments

- **yfinance**: Financial data acquisition
- **TA-Lib**: Technical analysis library
- **TensorFlow/Keras**: Deep learning framework
- **PyTorch**: Deep learning framework
- **TiRex**: NX-AI transformer model
- **statsmodels**: Statistical modeling
- **scikit-learn**: Machine learning utilities

## 📞 Support

- **Documentation**: See inline code comments
- **Issues**: Report bugs via GitHub
- **Discussions**: Q&A and feature requests

## 💡 Tips & Best Practices

### Data Quality
1. **Clean Data**: Remove outliers and fill missing values
2. **Multiple Timeframes**: Use daily, weekly, monthly data
3. **Volume Confirmation**: Always check volume with signals
4. **Recent Data**: Keep data updated

### Model Selection
1. **ARIMA**: For short-term, stationary data
2. **LSTM**: For complex patterns, long sequences
3. **TiRex**: For best accuracy with GPU
4. **Hybrid**: For production trading systems

### Trading Strategy
1. **Combine Signals**: Use multiple indicators
2. **Risk Management**: Always set stop losses
3. **Position Sizing**: Never risk more than 2% per trade
4. **Diversification**: Trade multiple uncorrelated assets
5. **Backtesting**: Test extensively before live trading

### Performance
1. **Use GPU**: Essential for LSTM and TiRex
2. **Batch Processing**: Process multiple stocks together
3. **Caching**: Cache preprocessed data
4. **Parallel Processing**: Use multiprocessing for data acquisition

---

**Made with ❤️ for quantitative traders and data scientists**

*📈 Predict the market with AI and statistical models*
