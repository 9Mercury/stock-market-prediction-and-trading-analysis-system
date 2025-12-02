import numpy as np
import pandas as pd
import talib
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class TechnicalIndicators:
    """
    Comprehensive technical indicators for trading analysis
    """
    
    def __init__(self):
        pass
    
    def add_all_indicators(self, df):
        """
        Add all technical indicators to the dataframe
        
        Args:
            df (pandas.DataFrame): DataFrame with OHLCV data
        
        Returns:
            pandas.DataFrame: DataFrame with all indicators added
        """
        df = df.copy()
        
        # Price-based indicators
        df = self.add_moving_averages(df)
        df = self.add_bollinger_bands(df)
        df = self.add_macd(df)
        df = self.add_rsi(df)
        df = self.add_stochastic(df)
        df = self.add_williams_r(df)
        df = self.add_cci(df)
        
        # Volume-based indicators
        df = self.add_volume_indicators(df)
        
        # Volatility indicators
        df = self.add_atr(df)
        df = self.add_keltner_channels(df)
        
        # Momentum indicators
        df = self.add_momentum_indicators(df)
        
        # Trend indicators
        df = self.add_trend_indicators(df)
        
        # Custom indicators
        df = self.add_custom_indicators(df)
        
        return df
    
    def add_moving_averages(self, df):
        """Add various moving averages"""
        # Simple Moving Averages
        df['SMA_5'] = talib.SMA(df['Close'], timeperiod=5)
        df['SMA_10'] = talib.SMA(df['Close'], timeperiod=10)
        df['SMA_20'] = talib.SMA(df['Close'], timeperiod=20)
        df['SMA_50'] = talib.SMA(df['Close'], timeperiod=50)
        df['SMA_200'] = talib.SMA(df['Close'], timeperiod=200)
        
        # Exponential Moving Averages
        df['EMA_5'] = talib.EMA(df['Close'], timeperiod=5)
        df['EMA_10'] = talib.EMA(df['Close'], timeperiod=10)
        df['EMA_20'] = talib.EMA(df['Close'], timeperiod=20)
        df['EMA_50'] = talib.EMA(df['Close'], timeperiod=50)
        df['EMA_200'] = talib.EMA(df['Close'], timeperiod=200)
        
        # Moving Average Crossovers
        df['SMA_5_20_Cross'] = np.where(df['SMA_5'] > df['SMA_20'], 1, 0)
        df['EMA_5_20_Cross'] = np.where(df['EMA_5'] > df['EMA_20'], 1, 0)
        
        return df
    
    def add_bollinger_bands(self, df):
        """Add Bollinger Bands"""
        upper, middle, lower = talib.BBANDS(df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2)
        df['BB_Upper'] = upper
        df['BB_Middle'] = middle
        df['BB_Lower'] = lower
        df['BB_Width'] = (upper - lower) / middle
        df['BB_Position'] = (df['Close'] - lower) / (upper - lower)
        
        return df
    
    def add_macd(self, df):
        """Add MACD indicator"""
        macd, macd_signal, macd_hist = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
        df['MACD'] = macd
        df['MACD_Signal'] = macd_signal
        df['MACD_Histogram'] = macd_hist
        df['MACD_Cross'] = np.where(macd > macd_signal, 1, 0)
        
        return df
    
    def add_rsi(self, df):
        """Add RSI indicator"""
        df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
        df['RSI_Overbought'] = np.where(df['RSI'] > 70, 1, 0)
        df['RSI_Oversold'] = np.where(df['RSI'] < 30, 1, 0)
        
        return df
    
    def add_stochastic(self, df):
        """Add Stochastic oscillator"""
        slowk, slowd = talib.STOCH(df['High'], df['Low'], df['Close'], 
                                  fastk_period=14, slowk_period=3, slowd_period=3)
        df['Stoch_K'] = slowk
        df['Stoch_D'] = slowd
        df['Stoch_Overbought'] = np.where(slowk > 80, 1, 0)
        df['Stoch_Oversold'] = np.where(slowk < 20, 1, 0)
        
        return df
    
    def add_williams_r(self, df):
        """Add Williams %R"""
        df['Williams_R'] = talib.WILLR(df['High'], df['Low'], df['Close'], timeperiod=14)
        df['Williams_R_Overbought'] = np.where(df['Williams_R'] > -20, 1, 0)
        df['Williams_R_Oversold'] = np.where(df['Williams_R'] < -80, 1, 0)
        
        return df
    
    def add_cci(self, df):
        """Add Commodity Channel Index"""
        df['CCI'] = talib.CCI(df['High'], df['Low'], df['Close'], timeperiod=14)
        df['CCI_Overbought'] = np.where(df['CCI'] > 100, 1, 0)
        df['CCI_Oversold'] = np.where(df['CCI'] < -100, 1, 0)
        
        return df
    
    def add_volume_indicators(self, df):
        """Add volume-based indicators"""
        # Volume SMA
        df['Volume_SMA_20'] = talib.SMA(df['Volume'], timeperiod=20)
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']
        
        # On Balance Volume
        df['OBV'] = talib.OBV(df['Close'], df['Volume'])
        
        # Money Flow Index
        df['MFI'] = talib.MFI(df['High'], df['Low'], df['Close'], df['Volume'], timeperiod=14)
        df['MFI_Overbought'] = np.where(df['MFI'] > 80, 1, 0)
        df['MFI_Oversold'] = np.where(df['MFI'] < 20, 1, 0)
        
        return df
    
    def add_atr(self, df):
        """Add Average True Range"""
        df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
        df['ATR_Ratio'] = df['ATR'] / df['Close']
        
        return df
    
    def add_keltner_channels(self, df):
        """Add Keltner Channels"""
        df['KC_Upper'] = df['EMA_20'] + (2 * df['ATR'])
        df['KC_Lower'] = df['EMA_20'] - (2 * df['ATR'])
        df['KC_Width'] = (df['KC_Upper'] - df['KC_Lower']) / df['EMA_20']
        
        return df
    
    def add_momentum_indicators(self, df):
        """Add momentum indicators"""
        # Rate of Change
        df['ROC'] = talib.ROC(df['Close'], timeperiod=10)
        
        # Momentum
        df['MOM'] = talib.MOM(df['Close'], timeperiod=10)
        
        # Relative Strength Index variations
        df['RSI_5'] = talib.RSI(df['Close'], timeperiod=5)
        df['RSI_21'] = talib.RSI(df['Close'], timeperiod=21)
        
        return df
    
    def add_trend_indicators(self, df):
        """Add trend indicators"""
        # ADX (Average Directional Index)
        df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)
        df['ADX_Strong_Trend'] = np.where(df['ADX'] > 25, 1, 0)
        
        # Parabolic SAR
        df['SAR'] = talib.SAR(df['High'], df['Low'], acceleration=0.02, maximum=0.2)
        df['SAR_Signal'] = np.where(df['Close'] > df['SAR'], 1, 0)
        
        return df
    
    def add_custom_indicators(self, df):
        """Add custom indicators"""
        # Price Rate of Change
        df['Price_ROC'] = df['Close'].pct_change(periods=10)
        
        # High-Low Ratio
        df['HL_Ratio'] = df['High'] / df['Low']
        
        # Gap Analysis
        df['Gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
        
        # Volatility Ratio
        df['Volatility_Ratio'] = df['Close'].rolling(window=20).std() / df['Close'].rolling(window=20).mean()
        
        # Support and Resistance levels (simplified)
        df['Support'] = df['Low'].rolling(window=20).min()
        df['Resistance'] = df['High'].rolling(window=20).max()
        df['Price_vs_Support'] = (df['Close'] - df['Support']) / df['Support']
        df['Price_vs_Resistance'] = (df['Resistance'] - df['Close']) / df['Close']
        
        return df
    
    def generate_signals(self, df):
        """
        Generate trading signals based on technical indicators
        
        Args:
            df (pandas.DataFrame): DataFrame with technical indicators
        
        Returns:
            pandas.DataFrame: DataFrame with trading signals
        """
        df = df.copy()
        
        # Initialize signal columns
        df['Buy_Signal'] = 0
        df['Sell_Signal'] = 0
        df['Signal_Strength'] = 0
        
        # MACD Signals
        df.loc[df['MACD'] > df['MACD_Signal'], 'Buy_Signal'] += 1
        df.loc[df['MACD'] < df['MACD_Signal'], 'Sell_Signal'] += 1
        
        # RSI Signals
        df.loc[df['RSI'] < 30, 'Buy_Signal'] += 1
        df.loc[df['RSI'] > 70, 'Sell_Signal'] += 1
        
        # Bollinger Bands Signals
        df.loc[df['Close'] < df['BB_Lower'], 'Buy_Signal'] += 1
        df.loc[df['Close'] > df['BB_Upper'], 'Sell_Signal'] += 1
        
        # Moving Average Signals
        df.loc[df['SMA_5'] > df['SMA_20'], 'Buy_Signal'] += 1
        df.loc[df['SMA_5'] < df['SMA_20'], 'Sell_Signal'] += 1
        
        # Volume Signals
        df.loc[df['Volume_Ratio'] > 1.5, 'Buy_Signal'] += 0.5
        df.loc[df['Volume_Ratio'] > 1.5, 'Sell_Signal'] += 0.5
        
        # Stochastic Signals
        df.loc[df['Stoch_K'] < 20, 'Buy_Signal'] += 1
        df.loc[df['Stoch_K'] > 80, 'Sell_Signal'] += 1
        
        # Williams %R Signals
        df.loc[df['Williams_R'] < -80, 'Buy_Signal'] += 1
        df.loc[df['Williams_R'] > -20, 'Sell_Signal'] += 1
        
        # Calculate signal strength
        df['Signal_Strength'] = df['Buy_Signal'] - df['Sell_Signal']
        
        # Generate final signals
        df['Final_Signal'] = np.where(df['Signal_Strength'] > 2, 1, 
                                     np.where(df['Signal_Strength'] < -2, -1, 0))
        
        return df 