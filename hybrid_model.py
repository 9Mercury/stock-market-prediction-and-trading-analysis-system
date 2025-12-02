import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
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

from technical_indicators import TechnicalIndicators
from arima_model import ARIMAModel
from lstm_model import LSTMTradingModel

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HybridTradingModel:
    """
    Hybrid trading model combining LSTM, ARIMA, and technical indicators
    """
    
    def __init__(self, dataset_dir="dataset", sequence_length=60):
        """
        Initialize the Hybrid Trading Model
        
        Args:
            dataset_dir (str): Directory containing the dataset files
            sequence_length (int): Number of time steps to look back
        """
        self.dataset_dir = dataset_dir
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
        # Initialize individual models
        self.lstm_model = LSTMTradingModel(dataset_dir, sequence_length)
        self.arima_model = ARIMAModel()
        self.technical_indicators = TechnicalIndicators()
        
        # Ensemble models
        self.ensemble_model = None
        self.feature_importance = None
        
        # Set random seeds for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)
    
    def load_and_prepare_data(self, symbol, target_column='Close', train_years=10):
        """
        Load and prepare data with all features
        
        Args:
            symbol (str): Stock symbol
            target_column (str): Column to predict
            train_years (int): Number of years for training
        
        Returns:
            tuple: Prepared data for all models
        """
        logger.info(f"Loading and preparing data for {symbol}")
        
        # Load raw data
        data = self.lstm_model.load_data(symbol)
        if data is None:
            return None, None, None, None, None, None
        
        # Add technical indicators
        data_with_indicators = self.technical_indicators.add_all_indicators(data)
        data_with_signals = self.technical_indicators.generate_signals(data_with_indicators)
        
        # Prepare LSTM data
        X_train_lstm, y_train_lstm, X_test_lstm, y_test_lstm, scaler = self.lstm_model.prepare_data(
            data, target_column, train_years
        )
        
        if X_train_lstm is None:
            return None, None, None, None, None, None
        
        # Prepare ARIMA data
        train_size = int(len(data) * (train_years / 25))  # Assuming 25 years total
        train_series = data['Close'][:train_size]
        test_series = data['Close'][train_size:]
        
        # Prepare ensemble features
        ensemble_features = self.prepare_ensemble_features(data_with_signals, train_size)
        
        return (X_train_lstm, y_train_lstm, X_test_lstm, y_test_lstm, scaler), \
               (train_series, test_series), \
               ensemble_features
    
    def prepare_ensemble_features(self, data, train_size):
        """
        Prepare features for ensemble model
        
        Args:
            data (pandas.DataFrame): Data with technical indicators
            train_size (int): Training data size
        
        Returns:
            tuple: Training and testing features
        """
        # Select relevant features for ensemble
        feature_columns = [
            'RSI', 'MACD', 'BB_Position', 'Volume_Ratio', 'ATR_Ratio',
            'Stoch_K', 'Williams_R', 'CCI', 'MFI', 'ADX',
            'Price_ROC', 'HL_Ratio', 'Gap', 'Volatility_Ratio',
            'Price_vs_Support', 'Price_vs_Resistance',
            'SMA_5_20_Cross', 'EMA_5_20_Cross', 'MACD_Cross',
            'RSI_Overbought', 'RSI_Oversold', 'Stoch_Overbought', 'Stoch_Oversold',
            'Williams_R_Overbought', 'Williams_R_Oversold',
            'CCI_Overbought', 'CCI_Oversold', 'MFI_Overbought', 'MFI_Oversold',
            'ADX_Strong_Trend', 'SAR_Signal', 'Signal_Strength', 'Final_Signal'
        ]
        
        # Remove any missing columns
        available_features = [col for col in feature_columns if col in data.columns]
        
        # Prepare features
        X = data[available_features].fillna(0)
        y = data['Close']
        
        # Split data
        X_train = X[:train_size]
        X_test = X[train_size:]
        y_train = y[:train_size]
        y_test = y[train_size:]
        
        return (X_train, y_train, X_test, y_test, available_features)
    
    def train_individual_models(self, symbol, train_years=10):
        """
        Train individual models (LSTM, ARIMA)
        
        Args:
            symbol (str): Stock symbol
            train_years (int): Number of years for training
        
        Returns:
            dict: Trained models and results
        """
        logger.info(f"Training individual models for {symbol}")
        
        # Load and prepare data
        lstm_data, arima_data, ensemble_data = self.load_and_prepare_data(symbol, train_years=train_years)
        
        if lstm_data is None:
            return None
        
        results = {}
        
        # Train LSTM
        logger.info("Training LSTM model...")
        X_train_lstm, y_train_lstm, X_test_lstm, y_test_lstm, scaler = lstm_data
        lstm_model = self.lstm_model.train_model(X_train_lstm, y_train_lstm, X_test_lstm, y_test_lstm, epochs=50)
        lstm_metrics, y_test_orig, y_pred_orig = self.lstm_model.evaluate_model(X_test_lstm, y_test_lstm, scaler)
        
        results['lstm'] = {
            'model': lstm_model,
            'metrics': lstm_metrics,
            'predictions': y_pred_orig,
            'actual': y_test_orig,
            'scaler': scaler
        }
        
        # Train ARIMA
        logger.info("Training ARIMA model...")
        train_series, test_series = arima_data
        
        # Check stationarity and find optimal order
        self.arima_model.check_stationarity(train_series, f"{symbol} Training Data")
        optimal_order, seasonal_order = self.arima_model.find_optimal_order(train_series, seasonal=True)
        
        # Fit ARIMA model
        fitted_arima = self.arima_model.fit_arima(train_series, optimal_order, seasonal_order)
        
        if fitted_arima is not None:
            # Generate ARIMA forecasts
            arima_forecast = self.arima_model.forecast(steps=len(test_series))
            arima_metrics = self.arima_model.evaluate_model(train_series.append(test_series))
            
            results['arima'] = {
                'model': fitted_arima,
                'metrics': arima_metrics,
                'predictions': arima_forecast['forecast'].values if arima_forecast is not None else None,
                'actual': test_series.values,
                'order': optimal_order,
                'seasonal_order': seasonal_order
            }
        
        # Store ensemble data
        results['ensemble_data'] = ensemble_data
        
        return results
    
    def train_ensemble_model(self, ensemble_data, lstm_pred, arima_pred):
        """
        Train ensemble model combining all predictions
        
        Args:
            ensemble_data (tuple): Ensemble features
            lstm_pred (numpy.ndarray): LSTM predictions
            arima_pred (numpy.ndarray): ARIMA predictions
        
        Returns:
            dict: Ensemble model results
        """
        logger.info("Training ensemble model...")
        
        X_train, y_train, X_test, y_test, feature_names = ensemble_data
        
        # Create ensemble features
        ensemble_features = X_test.copy()
        
        # Add model predictions as features
        if lstm_pred is not None:
            ensemble_features['LSTM_Pred'] = lstm_pred[:len(ensemble_features)]
        
        if arima_pred is not None:
            ensemble_features['ARIMA_Pred'] = arima_pred[:len(ensemble_features)]
        
        # Fill any missing values
        ensemble_features = ensemble_features.fillna(0)
        
        # Train multiple ensemble models
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'LinearRegression': LinearRegression()
        }
        
        ensemble_results = {}
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            
            # Train model
            model.fit(ensemble_features, y_test)
            
            # Make predictions
            predictions = model.predict(ensemble_features)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, predictions)
            
            # Calculate directional accuracy
            actual_direction = np.sign(np.diff(y_test.values))
            predicted_direction = np.sign(np.diff(predictions))
            directional_accuracy = np.mean(actual_direction == predicted_direction)
            
            metrics = {
                'MSE': mse,
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2,
                'Directional_Accuracy': directional_accuracy
            }
            
            ensemble_results[name] = {
                'model': model,
                'metrics': metrics,
                'predictions': predictions,
                'feature_importance': getattr(model, 'feature_importances_', None)
            }
            
            logger.info(f"{name} Results:")
            for metric, value in metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
        
        return ensemble_results
    
    def generate_hybrid_predictions(self, symbol, train_years=10):
        """
        Generate hybrid predictions using all models
        
        Args:
            symbol (str): Stock symbol
            train_years (int): Number of years for training
        
        Returns:
            dict: All model results and predictions
        """
        logger.info(f"Generating hybrid predictions for {symbol}")
        
        # Train individual models
        individual_results = self.train_individual_models(symbol, train_years)
        
        if individual_results is None:
            return None
        
        # Train ensemble model
        lstm_pred = individual_results['lstm']['predictions']
        arima_pred = individual_results['arima']['predictions'] if 'arima' in individual_results else None
        
        ensemble_results = self.train_ensemble_model(
            individual_results['ensemble_data'], 
            lstm_pred, 
            arima_pred
        )
        
        # Combine all results
        all_results = {
            'individual_models': individual_results,
            'ensemble_models': ensemble_results,
            'symbol': symbol
        }
        
        return all_results
    
    def plot_hybrid_results(self, results, symbol):
        """
        Plot comprehensive results from all models
        
        Args:
            results (dict): Model results
            symbol (str): Stock symbol
        """
        if results is None:
            return
        
        # Create comprehensive plot
        fig, axes = plt.subplots(3, 2, figsize=(20, 15))
        
        # Get actual values
        actual = results['individual_models']['lstm']['actual']
        
        # Plot 1: LSTM Results
        axes[0, 0].plot(actual, label='Actual', alpha=0.7)
        axes[0, 0].plot(results['individual_models']['lstm']['predictions'], label='LSTM', alpha=0.7)
        axes[0, 0].set_title(f'{symbol} - LSTM Predictions')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot 2: ARIMA Results (if available)
        if 'arima' in results['individual_models']:
            axes[0, 1].plot(actual, label='Actual', alpha=0.7)
            axes[0, 1].plot(results['individual_models']['arima']['predictions'], label='ARIMA', alpha=0.7)
            axes[0, 1].set_title(f'{symbol} - ARIMA Predictions')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # Plot 3: Random Forest Results
        axes[1, 0].plot(actual, label='Actual', alpha=0.7)
        axes[1, 0].plot(results['ensemble_models']['RandomForest']['predictions'], label='Random Forest', alpha=0.7)
        axes[1, 0].set_title(f'{symbol} - Random Forest Predictions')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Plot 4: Gradient Boosting Results
        axes[1, 1].plot(actual, label='Actual', alpha=0.7)
        axes[1, 1].plot(results['ensemble_models']['GradientBoosting']['predictions'], label='Gradient Boosting', alpha=0.7)
        axes[1, 1].set_title(f'{symbol} - Gradient Boosting Predictions')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        # Plot 5: Model Comparison
        models = ['LSTM', 'ARIMA', 'RandomForest', 'GradientBoosting', 'LinearRegression']
        rmse_values = []
        
        rmse_values.append(results['individual_models']['lstm']['metrics']['RMSE'])
        if 'arima' in results['individual_models']:
            rmse_values.append(results['individual_models']['arima']['metrics']['RMSE'])
        else:
            rmse_values.append(np.nan)
        
        for model_name in ['RandomForest', 'GradientBoosting', 'LinearRegression']:
            rmse_values.append(results['ensemble_models'][model_name]['metrics']['RMSE'])
        
        axes[2, 0].bar(models, rmse_values)
        axes[2, 0].set_title('Model Comparison - RMSE')
        axes[2, 0].set_ylabel('RMSE')
        axes[2, 0].tick_params(axis='x', rotation=45)
        
        # Plot 6: Feature Importance (Random Forest)
        if results['ensemble_models']['RandomForest']['feature_importance'] is not None:
            feature_importance = results['ensemble_models']['RandomForest']['feature_importance']
            feature_names = results['individual_models']['ensemble_data'][4] + ['LSTM_Pred', 'ARIMA_Pred']
            
            # Get top 10 features
            top_indices = np.argsort(feature_importance)[-10:]
            top_features = [feature_names[i] for i in top_indices]
            top_importance = feature_importance[top_indices]
            
            axes[2, 1].barh(range(len(top_features)), top_importance)
            axes[2, 1].set_yticks(range(len(top_features)))
            axes[2, 1].set_yticklabels(top_features)
            axes[2, 1].set_title('Top 10 Feature Importance (Random Forest)')
            axes[2, 1].set_xlabel('Importance')
        
        plt.tight_layout()
        plt.savefig(f'{symbol}_hybrid_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_hybrid_model(self, results, symbol):
        """
        Save all trained models
        
        Args:
            results (dict): Model results
            symbol (str): Stock symbol
        """
        if results is None:
            return
        
        # Save LSTM model
        self.lstm_model.save_model(symbol)
        
        # Save ensemble models
        for name, model_data in results['ensemble_models'].items():
            import joblib
            model_path = f'{symbol}_{name.lower().replace(" ", "_")}_model.pkl'
            joblib.dump(model_data['model'], model_path)
            logger.info(f"Saved {name} model as: {model_path}")
    
    def predict_future_hybrid(self, symbol, days_ahead=30):
        """
        Generate future predictions using hybrid approach
        
        Args:
            symbol (str): Stock symbol
            days_ahead (int): Number of days to predict
        
        Returns:
            dict: Future predictions from all models
        """
        logger.info(f"Generating future predictions for {symbol}")
        
        # Load recent data
        data = self.lstm_model.load_data(symbol)
        if data is None:
            return None
        
        # Add technical indicators
        data_with_indicators = self.technical_indicators.add_all_indicators(data)
        data_with_signals = self.technical_indicators.generate_signals(data_with_indicators)
        
        # LSTM predictions
        recent_data = data.tail(100)
        lstm_predictions = self.lstm_model.predict_future(recent_data, days_ahead)
        
        # ARIMA predictions
        arima_predictions = None
        if self.arima_model.fitted_model is not None:
            arima_forecast = self.arima_model.forecast(steps=days_ahead)
            if arima_forecast is not None:
                arima_predictions = arima_forecast['forecast'].values
        
        # Ensemble predictions (simplified - using recent features)
        ensemble_predictions = {}
        recent_features = data_with_signals.tail(60)  # Last 60 days for features
        
        # Create future feature predictions (simplified approach)
        future_features = recent_features.tail(1).copy()
        for i in range(days_ahead):
            # Update features for next day (simplified)
            future_features = future_features.append(future_features.iloc[-1])
        
        # Use ensemble models for prediction
        for name in ['RandomForest', 'GradientBoosting', 'LinearRegression']:
            model_path = f'{symbol}_{name.lower().replace(" ", "_")}_model.pkl'
            try:
                import joblib
                model = joblib.load(model_path)
                
                # Prepare features for prediction
                pred_features = future_features[recent_features.columns].fillna(0)
                if 'LSTM_Pred' in pred_features.columns:
                    pred_features['LSTM_Pred'] = lstm_predictions[i] if lstm_predictions is not None else 0
                if 'ARIMA_Pred' in pred_features.columns:
                    pred_features['ARIMA_Pred'] = arima_predictions[i] if arima_predictions is not None else 0
                
                ensemble_predictions[name] = model.predict(pred_features)
                
            except FileNotFoundError:
                logger.warning(f"Model file not found: {model_path}")
        
        return {
            'LSTM': lstm_predictions,
            'ARIMA': arima_predictions,
            'Ensemble': ensemble_predictions
        }

def main():
    """Main function to demonstrate hybrid model usage"""
    
    # Initialize the hybrid model
    hybrid_model = HybridTradingModel(dataset_dir="dataset", sequence_length=60)
    
    # Example symbols to train on
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    
    for symbol in symbols:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {symbol} with Hybrid Model")
        logger.info(f"{'='*60}")
        
        # Generate hybrid predictions
        results = hybrid_model.generate_hybrid_predictions(symbol, train_years=10)
        
        if results is not None:
            # Plot results
            hybrid_model.plot_hybrid_results(results, symbol)
            
            # Save models
            hybrid_model.save_hybrid_model(results, symbol)
            
            # Generate future predictions
            future_predictions = hybrid_model.predict_future_hybrid(symbol, days_ahead=30)
            
            if future_predictions is not None:
                logger.info(f"Future 30-day predictions for {symbol}:")
                logger.info("LSTM predictions (first 5 days):")
                if future_predictions['LSTM'] is not None:
                    for i, pred in enumerate(future_predictions['LSTM'][:5]):
                        logger.info(f"  Day {i+1}: ${pred:.2f}")
                
                logger.info("ARIMA predictions (first 5 days):")
                if future_predictions['ARIMA'] is not None:
                    for i, pred in enumerate(future_predictions['ARIMA'][:5]):
                        logger.info(f"  Day {i+1}: ${pred:.2f}")
                
                logger.info("Ensemble predictions (first 5 days):")
                for model_name, predictions in future_predictions['Ensemble'].items():
                    logger.info(f"  {model_name}:")
                    for i, pred in enumerate(predictions[:5]):
                        logger.info(f"    Day {i+1}: ${pred:.2f}")

if __name__ == "__main__":
    main() 