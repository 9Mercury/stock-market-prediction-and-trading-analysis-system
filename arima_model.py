import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class ARIMAModel:
    """
    ARIMA model for time series forecasting
    """
    
    def __init__(self):
        self.model = None
        self.fitted_model = None
        self.order = None
        self.seasonal_order = None
        
    def check_stationarity(self, series, title="Time Series"):
        """
        Check if the time series is stationary
        
        Args:
            series (pandas.Series): Time series data
            title (str): Title for plots
        
        Returns:
            dict: Stationarity test results
        """
        # ADF Test
        adf_result = adfuller(series.dropna())
        adf_statistic = adf_result[0]
        adf_pvalue = adf_result[1]
        adf_critical_values = adf_result[4]
        
        # KPSS Test
        kpss_result = kpss(series.dropna())
        kpss_statistic = kpss_result[0]
        kpss_pvalue = kpss_result[1]
        kpss_critical_values = kpss_result[3]
        
        # Plot original series and differenced series
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Original series
        axes[0, 0].plot(series)
        axes[0, 0].set_title(f'{title} - Original Series')
        axes[0, 0].grid(True)
        
        # First difference
        diff1 = series.diff().dropna()
        axes[0, 1].plot(diff1)
        axes[0, 1].set_title(f'{title} - First Difference')
        axes[0, 1].grid(True)
        
        # Second difference
        diff2 = series.diff().diff().dropna()
        axes[1, 0].plot(diff2)
        axes[1, 0].set_title(f'{title} - Second Difference')
        axes[1, 0].grid(True)
        
        # ACF plot
        plot_acf(series.dropna(), ax=axes[1, 1], lags=40)
        axes[1, 1].set_title('Autocorrelation Function')
        
        plt.tight_layout()
        plt.show()
        
        # Print results
        print(f"ADF Test Results for {title}:")
        print(f"  ADF Statistic: {adf_statistic:.4f}")
        print(f"  p-value: {adf_pvalue:.4f}")
        print(f"  Critical values: {adf_critical_values}")
        
        print(f"\nKPSS Test Results for {title}:")
        print(f"  KPSS Statistic: {kpss_statistic:.4f}")
        print(f"  p-value: {kpss_pvalue:.4f}")
        print(f"  Critical values: {kpss_critical_values}")
        
        # Determine if stationary
        is_stationary_adf = adf_pvalue < 0.05
        is_stationary_kpss = kpss_pvalue > 0.05
        
        print(f"\nStationarity Assessment:")
        print(f"  ADF Test: {'Stationary' if is_stationary_adf else 'Non-stationary'}")
        print(f"  KPSS Test: {'Stationary' if is_stationary_kpss else 'Non-stationary'}")
        
        return {
            'adf_statistic': adf_statistic,
            'adf_pvalue': adf_pvalue,
            'kpss_statistic': kpss_statistic,
            'kpss_pvalue': kpss_pvalue,
            'is_stationary_adf': is_stationary_adf,
            'is_stationary_kpss': is_stationary_kpss
        }
    
    def seasonal_decomposition(self, series, period=252):
        """
        Perform seasonal decomposition
        
        Args:
            series (pandas.Series): Time series data
            period (int): Seasonal period (252 for daily data)
        
        Returns:
            dict: Decomposition results
        """
        try:
            decomposition = seasonal_decompose(series.dropna(), period=period, extrapolate_trend='freq')
            
            # Plot decomposition
            fig, axes = plt.subplots(4, 1, figsize=(15, 12))
            
            decomposition.observed.plot(ax=axes[0])
            axes[0].set_title('Original Time Series')
            axes[0].grid(True)
            
            decomposition.trend.plot(ax=axes[1])
            axes[1].set_title('Trend Component')
            axes[1].grid(True)
            
            decomposition.seasonal.plot(ax=axes[2])
            axes[2].set_title('Seasonal Component')
            axes[2].grid(True)
            
            decomposition.resid.plot(ax=axes[3])
            axes[3].set_title('Residual Component')
            axes[3].grid(True)
            
            plt.tight_layout()
            plt.show()
            
            return decomposition
            
        except Exception as e:
            print(f"Error in seasonal decomposition: {e}")
            return None
    
    def find_optimal_order(self, series, max_p=5, max_d=2, max_q=5, seasonal=False):
        """
        Find optimal ARIMA order using grid search
        
        Args:
            series (pandas.Series): Time series data
            max_p (int): Maximum AR order
            max_d (int): Maximum differencing order
            max_q (int): Maximum MA order
            seasonal (bool): Whether to include seasonal components
        
        Returns:
            tuple: Optimal (p, d, q) order
        """
        best_aic = np.inf
        best_order = None
        best_seasonal_order = None
        
        # Grid search for non-seasonal ARIMA
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    try:
                        if seasonal:
                            # Seasonal ARIMA
                            for P in range(2):
                                for D in range(2):
                                    for Q in range(2):
                                        try:
                                            model = ARIMA(series, order=(p, d, q), 
                                                         seasonal_order=(P, D, Q, 252))
                                            fitted_model = model.fit()
                                            
                                            if fitted_model.aic < best_aic:
                                                best_aic = fitted_model.aic
                                                best_order = (p, d, q)
                                                best_seasonal_order = (P, D, Q, 252)
                                        except:
                                            continue
                        else:
                            # Non-seasonal ARIMA
                            model = ARIMA(series, order=(p, d, q))
                            fitted_model = model.fit()
                            
                            if fitted_model.aic < best_aic:
                                best_aic = fitted_model.aic
                                best_order = (p, d, q)
                                
                    except:
                        continue
        
        print(f"Optimal ARIMA order: {best_order}")
        if seasonal and best_seasonal_order:
            print(f"Optimal seasonal order: {best_seasonal_order}")
        print(f"Best AIC: {best_aic:.2f}")
        
        self.order = best_order
        self.seasonal_order = best_seasonal_order
        
        return best_order, best_seasonal_order
    
    def fit_arima(self, series, order=None, seasonal_order=None):
        """
        Fit ARIMA model
        
        Args:
            series (pandas.Series): Time series data
            order (tuple): ARIMA order (p, d, q)
            seasonal_order (tuple): Seasonal order (P, D, Q, s)
        
        Returns:
            ARIMAResults: Fitted model
        """
        if order is None:
            order = self.order
        if seasonal_order is None:
            seasonal_order = self.seasonal_order
        
        try:
            if seasonal_order:
                model = ARIMA(series, order=order, seasonal_order=seasonal_order)
            else:
                model = ARIMA(series, order=order)
            
            self.fitted_model = model.fit()
            print(f"ARIMA Model Summary:")
            print(self.fitted_model.summary())
            
            return self.fitted_model
            
        except Exception as e:
            print(f"Error fitting ARIMA model: {e}")
            return None
    
    def forecast(self, steps=30, alpha=0.05):
        """
        Generate forecasts
        
        Args:
            steps (int): Number of steps to forecast
            alpha (float): Confidence level
        
        Returns:
            dict: Forecast results
        """
        if self.fitted_model is None:
            print("No fitted model available. Please fit the model first.")
            return None
        
        try:
            forecast_result = self.fitted_model.forecast(steps=steps, alpha=alpha)
            
            # Get confidence intervals
            conf_int = self.fitted_model.get_forecast(steps=steps).conf_int(alpha=alpha)
            
            forecast_df = pd.DataFrame({
                'forecast': forecast_result,
                'lower_ci': conf_int.iloc[:, 0],
                'upper_ci': conf_int.iloc[:, 1]
            })
            
            return forecast_df
            
        except Exception as e:
            print(f"Error generating forecast: {e}")
            return None
    
    def plot_forecast(self, series, forecast_df, title="ARIMA Forecast"):
        """
        Plot the forecast results
        
        Args:
            series (pandas.Series): Original time series
            forecast_df (pandas.DataFrame): Forecast results
            title (str): Plot title
        """
        plt.figure(figsize=(15, 8))
        
        # Plot original series
        plt.plot(series.index, series.values, label='Original Data', color='blue')
        
        # Plot forecast
        forecast_index = pd.date_range(start=series.index[-1] + pd.Timedelta(days=1), 
                                     periods=len(forecast_df), freq='D')
        plt.plot(forecast_index, forecast_df['forecast'], label='Forecast', color='red')
        
        # Plot confidence intervals
        plt.fill_between(forecast_index, forecast_df['lower_ci'], forecast_df['upper_ci'], 
                        alpha=0.3, color='red', label=f'Confidence Interval')
        
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def evaluate_model(self, series, test_size=0.2):
        """
        Evaluate ARIMA model performance
        
        Args:
            series (pandas.Series): Full time series
            test_size (float): Proportion of data for testing
        
        Returns:
            dict: Evaluation metrics
        """
        if self.fitted_model is None:
            print("No fitted model available.")
            return None
        
        # Split data
        split_idx = int(len(series) * (1 - test_size))
        train_series = series[:split_idx]
        test_series = series[split_idx:]
        
        # Fit model on training data
        if self.seasonal_order:
            model = ARIMA(train_series, order=self.order, seasonal_order=self.seasonal_order)
        else:
            model = ARIMA(train_series, order=self.order)
        
        fitted_model = model.fit()
        
        # Generate forecasts for test period
        forecast_steps = len(test_series)
        forecast_result = fitted_model.forecast(steps=forecast_steps)
        
        # Calculate metrics
        mse = np.mean((test_series.values - forecast_result.values) ** 2)
        mae = np.mean(np.abs(test_series.values - forecast_result.values))
        rmse = np.sqrt(mse)
        
        # Calculate directional accuracy
        actual_direction = np.sign(np.diff(test_series.values))
        predicted_direction = np.sign(np.diff(forecast_result.values))
        directional_accuracy = np.mean(actual_direction == predicted_direction)
        
        metrics = {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'Directional_Accuracy': directional_accuracy
        }
        
        print("ARIMA Model Evaluation:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # Plot results
        plt.figure(figsize=(15, 8))
        plt.plot(train_series.index, train_series.values, label='Training Data', color='blue')
        plt.plot(test_series.index, test_series.values, label='Actual Test Data', color='green')
        plt.plot(test_series.index, forecast_result.values, label='ARIMA Forecast', color='red')
        plt.title('ARIMA Model Performance')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        return metrics 