import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TradingDataAcquirer:
    def __init__(self, data_dir="dataset"):
        """
        Initialize the TradingDataAcquirer
        
        Args:
            data_dir (str): Directory to store the downloaded data
        """
        self.data_dir = data_dir
        self.ensure_data_directory()
    
    def ensure_data_directory(self):
        """Create data directory if it doesn't exist"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            logger.info(f"Created data directory: {self.data_dir}")
    
    def get_ticker_data(self, symbol, period="25y", interval="1d", start_date=None, end_date=None):
        """
        Download trading data for a given symbol
        
        Args:
            symbol (str): Stock symbol (e.g., 'AAPL', 'MSFT', 'SPY')
            period (str): Data period ('25y' for 25 years)
            interval (str): Data interval ('1d' for daily, '1wk' for weekly, '1mo' for monthly)
            start_date (str): Start date in 'YYYY-MM-DD' format (optional)
            end_date (str): End date in 'YYYY-MM-DD' format (optional)
        
        Returns:
            pandas.DataFrame: Historical trading data
        """
        try:
            logger.info(f"Downloading data for {symbol}...")
            
            # Create ticker object
            ticker = yf.Ticker(symbol)
            
            # Get ticker info to validate symbol
            info = ticker.info
            if not info:
                logger.error(f"Invalid symbol: {symbol}")
                return None
            
            logger.info(f"Symbol: {symbol}")
            logger.info(f"Company: {info.get('longName', 'N/A')}")
            logger.info(f"Sector: {info.get('sector', 'N/A')}")
            
            # Download historical data
            if start_date and end_date:
                data = ticker.history(start=start_date, end=end_date, interval=interval)
            else:
                data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                logger.warning(f"No data found for {symbol}")
                return None
            
            # Add symbol column
            data['Symbol'] = symbol
            
            # Reset index to make Date a column
            data = data.reset_index()
            
            # Rename columns for clarity
            data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock_Splits', 'Symbol']
            
            logger.info(f"Successfully downloaded {len(data)} records for {symbol}")
            logger.info(f"Date range: {data['Date'].min().date()} to {data['Date'].max().date()}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error downloading data for {symbol}: {str(e)}")
            return None
    
    def save_data(self, data, symbol, format='csv'):
        """
        Save data to file
        
        Args:
            data (pandas.DataFrame): Data to save
            symbol (str): Stock symbol
            format (str): File format ('csv' or 'parquet')
        """
        if data is None or data.empty:
            logger.warning(f"No data to save for {symbol}")
            return
        
        try:
            if format.lower() == 'csv':
                filename = os.path.join(self.data_dir, f"{symbol}_25y_data.csv")
                data.to_csv(filename, index=False)
            elif format.lower() == 'parquet':
                filename = os.path.join(self.data_dir, f"{symbol}_25y_data.parquet")
                data.to_parquet(filename, index=False)
            else:
                logger.error(f"Unsupported format: {format}")
                return
            
            logger.info(f"Data saved to: {filename}")
            
        except Exception as e:
            logger.error(f"Error saving data for {symbol}: {str(e)}")
    
    def get_multiple_symbols(self, symbols, period="25y", interval="1d", save_format='csv'):
        """
        Download data for multiple symbols
        
        Args:
            symbols (list): List of stock symbols
            period (str): Data period
            interval (str): Data interval
            save_format (str): File format to save data
        
        Returns:
            dict: Dictionary with symbol as key and data as value
        """
        results = {}
        
        for symbol in symbols:
            logger.info(f"Processing {symbol} ({symbols.index(symbol) + 1}/{len(symbols)})")
            
            data = self.get_ticker_data(symbol, period, interval)
            if data is not None:
                results[symbol] = data
                self.save_data(data, symbol, save_format)
            else:
                logger.warning(f"Failed to get data for {symbol}")
        
        return results
    
    def get_market_data(self, market='SPY', period="25y", interval="1d"):
        """
        Get market benchmark data (default: S&P 500 ETF)
        
        Args:
            market (str): Market symbol (default: 'SPY' for S&P 500)
            period (str): Data period
            interval (str): Data interval
        
        Returns:
            pandas.DataFrame: Market data
        """
        logger.info(f"Downloading market benchmark data: {market}")
        return self.get_ticker_data(market, period, interval)

def main():
    """Main function to demonstrate usage"""
    
    # Initialize the acquirer
    acquirer = TradingDataAcquirer()
    
    # Example symbols (you can modify this list)
    symbols = [
        'AAPL',    # Apple
        'MSFT',    # Microsoft
        'GOOGL',   # Google
        'AMZN',    # Amazon
        'TSLA',    # Tesla
        'NVDA',    # NVIDIA
        'META',    # Meta (Facebook)
        'BRK-B',   # Berkshire Hathaway
        'JNJ',     # Johnson & Johnson
        'V',       # Visa
    ]
    
    # Download data for all symbols
    logger.info("Starting data acquisition for 25 years of trading data...")
    results = acquirer.get_multiple_symbols(symbols, period="25y", interval="1d")
    
    # Get market benchmark data
    market_data = acquirer.get_market_data('SPY', period="25y", interval="1d")
    if market_data is not None:
        acquirer.save_data(market_data, 'SPY', 'csv')
    
    # Summary
    logger.info(f"Data acquisition completed!")
    logger.info(f"Successfully downloaded data for {len(results)} symbols")
    logger.info(f"Data saved in: {acquirer.data_dir}")
    
    # Display summary for each symbol
    for symbol, data in results.items():
        if data is not None:
            logger.info(f"{symbol}: {len(data)} records, {data['Date'].min().date()} to {data['Date'].max().date()}")

if __name__ == "__main__":
    main()
