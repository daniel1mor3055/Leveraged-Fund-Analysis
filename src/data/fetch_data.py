"""
Fetch historical price data from Yahoo Finance.

This module downloads daily adjusted close prices for:
- QQQ: Invesco QQQ ETF (1999-present)
- TQQQ: ProShares UltraPro QQQ (2010-present)
- ^NDX: Nasdaq-100 Index (1985-present)
"""

import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataFetcher:
    """Fetch and save historical price data from Yahoo Finance."""
    
    TICKERS = {
        'QQQ': {'start': '1999-03-01', 'name': 'Invesco QQQ ETF'},
        'TQQQ': {'start': '2010-02-01', 'name': 'ProShares UltraPro QQQ'},
        '^NDX': {'start': '1985-01-01', 'name': 'Nasdaq-100 Index'}
    }
    
    def __init__(self, data_dir: str = 'data/raw'):
        """Initialize data fetcher with output directory."""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Data directory: {self.data_dir.absolute()}")
    
    def fetch_ticker(self, ticker: str) -> pd.DataFrame:
        """
        Fetch historical data for a single ticker.
        
        Args:
            ticker: Ticker symbol (QQQ, TQQQ, ^NDX)
            
        Returns:
            DataFrame with Date index and OHLCV columns
        """
        if ticker not in self.TICKERS:
            raise ValueError(f"Unknown ticker: {ticker}. Valid: {list(self.TICKERS.keys())}")
        
        info = self.TICKERS[ticker]
        start_date = info['start']
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        logger.info(f"Fetching {ticker} ({info['name']}) from {start_date} to {end_date}")
        
        try:
            df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=False  # We want both Close and Adj Close
            )
            
            if df.empty:
                raise ValueError(f"No data returned for {ticker}")
            
            # Flatten MultiIndex columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            logger.info(f"✓ Fetched {len(df)} rows for {ticker} (from {df.index[0].date()} to {df.index[-1].date()})")
            return df
            
        except Exception as e:
            logger.error(f"✗ Failed to fetch {ticker}: {e}")
            raise
    
    def fetch_all(self, force_refresh: bool = False) -> dict:
        """
        Fetch all tickers and save to CSV files.
        
        Args:
            force_refresh: If True, re-download even if files exist
            
        Returns:
            Dictionary mapping ticker to DataFrame
        """
        data = {}
        
        for ticker in self.TICKERS.keys():
            output_file = self.data_dir / f"{ticker.replace('^', '')}_raw.csv"
            
            if output_file.exists() and not force_refresh:
                logger.info(f"Loading cached data for {ticker} from {output_file}")
                df = pd.read_csv(output_file, index_col=0, parse_dates=True)
            else:
                df = self.fetch_ticker(ticker)
                df.to_csv(output_file)
                logger.info(f"Saved {ticker} data to {output_file}")
            
            data[ticker] = df
        
        return data
    
    def get_data_summary(self, data: dict) -> pd.DataFrame:
        """
        Generate summary statistics for fetched data.
        
        Args:
            data: Dictionary mapping ticker to DataFrame
            
        Returns:
            Summary DataFrame with ticker info
        """
        summary = []
        
        for ticker, df in data.items():
            summary.append({
                'Ticker': ticker,
                'Name': self.TICKERS[ticker]['name'],
                'Start Date': df.index[0].date(),
                'End Date': df.index[-1].date(),
                'Days': len(df),
                'Years': round((df.index[-1] - df.index[0]).days / 365.25, 1)
            })
        
        return pd.DataFrame(summary)


if __name__ == '__main__':
    # Example usage
    fetcher = DataFetcher()
    data = fetcher.fetch_all(force_refresh=False)
    summary = fetcher.get_data_summary(data)
    
    print("\n" + "="*60)
    print("DATA FETCH SUMMARY")
    print("="*60)
    print(summary.to_string(index=False))
    print("="*60)
