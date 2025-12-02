import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LSTMTradingModel:
    def __init__(self, dataset_dir="dataset", sequence_length=60):
        """
        Initialize the LSTM Trading Model
        
        Args:
            dataset_dir (str): Directory containing the dataset files
            sequence_length (int): Number of time steps to look back
        """
        self.dataset_dir = dataset_dir
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.history = None
        
        # Set random seeds for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)
        
    def load_data(self, symbol, target_column='Close'):
        """
        Load data for a specific symbol
        
        Args:
            symbol (str): Stock symbol
            target_column (str): Column to predict (default: 'Close')
        
        Returns:
            pandas.DataFrame: Loaded data
        """
        try:
            # Try CSV first
            csv_path = os.path.join(self.dataset_dir, f"{symbol}_25y_data.csv")
            if os.path.exists(csv_path):
                data = pd.read_csv(csv_path)
                logger.info(f"Loaded {symbol} data from CSV: {len(data)} records")
            else:
                # Try Parquet
                parquet_path = os.path.join(self.dataset_dir, f"{symbol}_25y_data.parquet")
                if os.path.exists(parquet_path):
                    data = pd.read_parquet(parquet_path)
                    logger.info(f"Loaded {symbol} data from Parquet: {len(data)} records")
                else:
                    logger.error(f"No data file found for {symbol}")
                    return None
            
            # Convert Date column to datetime
            data['Date'] = pd.to_datetime(data['Date'])
            data = data.sort_values('Date').reset_index(drop=True)
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {str(e)}")
            return None
    
    def prepare_data(self, data, target_column='Close', train_years=10):
        """
        Prepare data for LSTM training
        
        Args:
            data (pandas.DataFrame): Raw data
            target_column (str): Column to predict
            train_years (int): Number of years to use for training
        
        Returns:
            tuple: (X_train, y_train, X_test, y_test, scaler)
        """
        if data is None or data.empty:
            logger.error("No data provided for preparation")
            return None, None, None, None, None
        
        # Select features for the model
        features = ['Open', 'High', 'Low', 'Close', 'Volume']
        if target_column not in features:
            features.append(target_column)
        
        # Create feature matrix
        feature_data = data[features].copy()
        
        # Handle missing values
        feature_data = feature_data.fillna(method='ffill')
        feature_data = feature_data.fillna(method='bfill')
        
        # Calculate additional features
        feature_data['Returns'] = feature_data['Close'].pct_change()
        feature_data['High_Low_Ratio'] = feature_data['High'] / feature_data['Low']
        feature_data['Volume_MA'] = feature_data['Volume'].rolling(window=20).mean()
        feature_data['Price_MA'] = feature_data['Close'].rolling(window=20).mean()
        feature_data['Price_Std'] = feature_data['Close'].rolling(window=20).std()
        
        # Remove NaN values
        feature_data = feature_data.dropna()
        
        # Scale the features
        scaled_data = self.scaler.fit_transform(feature_data)
        
        # Split data into train and test (10 years for training, rest for testing)
        total_days = len(scaled_data)
        train_days = train_years * 252  # Approximate trading days per year
        
        if total_days <= train_days:
            logger.warning(f"Not enough data for {train_years} years of training. Using 80% for training.")
            train_days = int(total_days * 0.8)
        
        train_data = scaled_data[:train_days]
        test_data = scaled_data[train_days:]
        
        logger.info(f"Training data: {len(train_data)} records ({train_years} years)")
        logger.info(f"Testing data: {len(test_data)} records ({len(test_data)/252:.1f} years)")
        
        # Create sequences for LSTM
        X_train, y_train = self.create_sequences(train_data, target_column)
        X_test, y_test = self.create_sequences(test_data, target_column)
        
        return X_train, y_train, X_test, y_test, self.scaler
    
    def create_sequences(self, data, target_column='Close'):
        """
        Create sequences for LSTM training
        
        Args:
            data (numpy.ndarray): Scaled data
            target_column (str): Target column name
        
        Returns:
            tuple: (X, y) sequences
        """
        X, y = [], []
        
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i])
            y.append(data[i, 3])  # Close price is at index 3
        
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape, units=50, dropout=0.2):
        """
        Build LSTM model
        
        Args:
            input_shape (tuple): Shape of input data
            units (int): Number of LSTM units
            dropout (float): Dropout rate
        
        Returns:
            tensorflow.keras.Model: Compiled model
        """
        model = Sequential([
            LSTM(units=units, return_sequences=True, input_shape=input_shape),
            BatchNormalization(),
            Dropout(dropout),
            
            LSTM(units=units, return_sequences=True),
            BatchNormalization(),
            Dropout(dropout),
            
            LSTM(units=units, return_sequences=False),
            BatchNormalization(),
            Dropout(dropout),
            
            Dense(units=25),
            Dense(units=1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train_model(self, X_train, y_train, X_test, y_test, epochs=100, batch_size=32):
        """
        Train the LSTM model
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Validation data
            epochs (int): Number of training epochs
            batch_size (int): Batch size
        
        Returns:
            tensorflow.keras.Model: Trained model
        """
        # Build model
        self.model = self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=0.0001,
            verbose=1
        )
        
        # Train model
        logger.info("Starting model training...")
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        logger.info("Model training completed!")
        return self.model
    
    def evaluate_model(self, X_test, y_test, scaler):
        """
        Evaluate the trained model
        
        Args:
            X_test, y_test: Test data
            scaler: Fitted scaler for inverse transformation
        
        Returns:
            dict: Evaluation metrics
        """
        if self.model is None:
            logger.error("No trained model available")
            return None
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Inverse transform predictions
        # Create dummy array for inverse transform
        dummy_array = np.zeros((len(y_pred), scaler.n_features_in_))
        dummy_array[:, 3] = y_pred.flatten()  # Close price is at index 3
        y_pred_original = scaler.inverse_transform(dummy_array)[:, 3]
        
        dummy_array = np.zeros((len(y_test), scaler.n_features_in_))
        dummy_array[:, 3] = y_test
        y_test_original = scaler.inverse_transform(dummy_array)[:, 3]
        
        # Calculate metrics
        mse = mean_squared_error(y_test_original, y_pred_original)
        mae = mean_absolute_error(y_test_original, y_pred_original)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_original, y_pred_original)
        
        # Calculate directional accuracy
        direction_accuracy = np.mean(
            np.sign(np.diff(y_test_original)) == np.sign(np.diff(y_pred_original))
        )
        
        metrics = {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'Directional_Accuracy': direction_accuracy
        }
        
        logger.info("Model Evaluation Results:")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        
        return metrics, y_test_original, y_pred_original
    
    def plot_results(self, y_test, y_pred, symbol):
        """
        Plot training results
        
        Args:
            y_test: Actual values
            y_pred: Predicted values
            symbol: Stock symbol
        """
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Training History
        plt.subplot(2, 2, 1)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title(f'{symbol} - Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot 2: Actual vs Predicted
        plt.subplot(2, 2, 2)
        plt.plot(y_test, label='Actual', alpha=0.7)
        plt.plot(y_pred, label='Predicted', alpha=0.7)
        plt.title(f'{symbol} - Actual vs Predicted Prices')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        
        # Plot 3: Scatter Plot
        plt.subplot(2, 2, 3)
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Price')
        plt.ylabel('Predicted Price')
        plt.title(f'{symbol} - Prediction Accuracy')
        plt.grid(True)
        
        # Plot 4: Residuals
        plt.subplot(2, 2, 4)
        residuals = y_test - y_pred
        plt.plot(residuals)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title(f'{symbol} - Residuals')
        plt.xlabel('Time')
        plt.ylabel('Residual')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{symbol}_lstm_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, symbol):
        """
        Save the trained model
        
        Args:
            symbol (str): Stock symbol
        """
        if self.model is None:
            logger.error("No trained model to save")
            return
        
        model_path = f'{symbol}_lstm_model.h5'
        self.model.save(model_path)
        logger.info(f"Model saved as: {model_path}")
    
    def predict_future(self, data, days_ahead=30):
        """
        Predict future prices
        
        Args:
            data (pandas.DataFrame): Recent data
            days_ahead (int): Number of days to predict
        
        Returns:
            numpy.ndarray: Predicted prices
        """
        if self.model is None:
            logger.error("No trained model available")
            return None
        
        # Prepare recent data
        features = ['Open', 'High', 'Low', 'Close', 'Volume']
        feature_data = data[features].copy()
        
        # Add calculated features
        feature_data['Returns'] = feature_data['Close'].pct_change()
        feature_data['High_Low_Ratio'] = feature_data['High'] / feature_data['Low']
        feature_data['Volume_MA'] = feature_data['Volume'].rolling(window=20).mean()
        feature_data['Price_MA'] = feature_data['Close'].rolling(window=20).mean()
        feature_data['Price_Std'] = feature_data['Close'].rolling(window=20).std()
        
        feature_data = feature_data.fillna(method='ffill').fillna(method='bfill')
        
        # Scale data
        scaled_data = self.scaler.transform(feature_data)
        
        # Get the last sequence
        last_sequence = scaled_data[-self.sequence_length:]
        
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(days_ahead):
            # Reshape for prediction
            current_sequence_reshaped = current_sequence.reshape(1, self.sequence_length, -1)
            
            # Make prediction
            next_pred = self.model.predict(current_sequence_reshaped, verbose=0)
            predictions.append(next_pred[0, 0])
            
            # Update sequence for next prediction
            new_row = current_sequence[-1].copy()
            new_row[3] = next_pred[0, 0]  # Update close price
            current_sequence = np.vstack([current_sequence[1:], new_row])
        
        # Inverse transform predictions
        dummy_array = np.zeros((len(predictions), self.scaler.n_features_in_))
        dummy_array[:, 3] = predictions
        predictions_original = self.scaler.inverse_transform(dummy_array)[:, 3]
        
        return predictions_original

def main():
    """Main function to demonstrate LSTM model usage"""
    
    # Initialize the model
    lstm_model = LSTMTradingModel(dataset_dir="dataset", sequence_length=60)
    
    # Example symbols to train on
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    for symbol in symbols:
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing {symbol}")
        logger.info(f"{'='*50}")
        
        # Load data
        data = lstm_model.load_data(symbol)
        if data is None:
            continue
        
        # Prepare data (10 years for training, rest for testing)
        X_train, y_train, X_test, y_test, scaler = lstm_model.prepare_data(
            data, target_column='Close', train_years=10
        )
        
        if X_train is None:
            continue
        
        # Train model
        model = lstm_model.train_model(X_train, y_train, X_test, y_test, epochs=100)
        
        # Evaluate model
        metrics, y_test_orig, y_pred_orig = lstm_model.evaluate_model(X_test, y_test, scaler)
        
        if metrics is not None:
            # Plot results
            lstm_model.plot_results(y_test_orig, y_pred_orig, symbol)
            
            # Save model
            lstm_model.save_model(symbol)
            
            # Predict future prices
            recent_data = data.tail(100)  # Last 100 days
            future_predictions = lstm_model.predict_future(recent_data, days_ahead=30)
            
            if future_predictions is not None:
                logger.info(f"Future 30-day predictions for {symbol}:")
                for i, pred in enumerate(future_predictions[:10]):  # Show first 10
                    logger.info(f"Day {i+1}: ${pred:.2f}")

if __name__ == "__main__":
    main() 