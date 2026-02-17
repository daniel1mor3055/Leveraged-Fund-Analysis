"""
Process raw price data into clean analysis-ready format.

Calculates:
- Daily returns
- Cumulative returns
- Realized variance
- Validates data quality
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataProcessor:
    """Process raw OHLCV data into analysis-ready format."""
    
    def __init__(self, raw_dir: str = 'data/raw', processed_dir: str = 'data/processed'):
        """Initialize processor with input/output directories."""
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Processing: {self.raw_dir} -> {self.processed_dir}")
    
    def process_ticker(self, ticker: str, use_adjusted: bool = True) -> pd.DataFrame:
        """
        Process a single ticker's raw data.
        
        Args:
            ticker: Ticker symbol (without ^)
            use_adjusted: If True, use Adj Close (includes dividends); if False, use Close
            
        Returns:
            Processed DataFrame with daily returns and metrics
        """
        input_file = self.raw_dir / f"{ticker}_raw.csv"
        
        if not input_file.exists():
            raise FileNotFoundError(f"Raw data not found: {input_file}")
        
        logger.info(f"Processing {ticker} from {input_file}")
        
        # Load raw data
        df = pd.read_csv(input_file, index_col=0, parse_dates=True)
        
        # Choose price column
        price_col = 'Adj Close' if use_adjusted and 'Adj Close' in df.columns else 'Close'
        logger.info(f"  Using {price_col} for {ticker}")
        
        # Create processed dataframe
        processed = pd.DataFrame(index=df.index)
        processed['Close'] = df[price_col]
        
        # Calculate daily returns
        processed['Daily_Return'] = processed['Close'].pct_change()
        processed['Daily_Return_Pct'] = processed['Daily_Return'] * 100
        
        # Cumulative metrics
        processed['Growth_1Dollar'] = (1 + processed['Daily_Return']).cumprod()
        processed['Cumulative_Return'] = processed['Growth_1Dollar'] - 1
        processed['Cumulative_Return_Pct'] = processed['Cumulative_Return'] * 100
        
        # Remove first row (NaN return)
        processed = processed.dropna()
        
        # Data quality checks
        self._validate_data(ticker, processed)
        
        logger.info(f"  ✓ Processed {len(processed)} days for {ticker}")
        return processed
    
    def _validate_data(self, ticker: str, df: pd.DataFrame) -> None:
        """Validate data quality and log warnings."""
        # Check for missing values
        missing = df.isnull().sum()
        if missing.any():
            logger.warning(f"  ⚠ Missing values in {ticker}: {missing[missing > 0].to_dict()}")
        
        # Check for suspicious returns (> +/- 50% in one day)
        extreme_returns = df[df['Daily_Return'].abs() > 0.5]
        if not extreme_returns.empty:
            logger.warning(f"  ⚠ {len(extreme_returns)} extreme daily returns (>50%) in {ticker}")
            for date, row in extreme_returns.head(3).iterrows():
                logger.warning(f"    {date.date()}: {row['Daily_Return_Pct']:.2f}%")
        
        # Check for zero/negative prices
        if (df['Close'] <= 0).any():
            logger.error(f"  ✗ Zero or negative prices found in {ticker}!")
    
    def calculate_rolling_variance(self, df: pd.DataFrame, window: int = 252) -> pd.DataFrame:
        """
        Calculate rolling realized variance.
        
        Args:
            df: Processed dataframe with Daily_Return column
            window: Rolling window in days (252 = 1 year)
            
        Returns:
            DataFrame with variance metrics added
        """
        df = df.copy()
        df[f'Rolling_Variance_{window}d'] = df['Daily_Return'].rolling(window=window).var() * 252
        df[f'Rolling_Volatility_{window}d'] = df['Daily_Return'].rolling(window=window).std() * np.sqrt(252)
        return df
    
    def process_all(self, tickers: list = None) -> dict:
        """
        Process all tickers.
        
        Args:
            tickers: List of ticker symbols (without ^), or None for all
            
        Returns:
            Dictionary mapping ticker to processed DataFrame
        """
        if tickers is None:
            # Find all raw CSV files
            tickers = [f.stem.replace('_raw', '') for f in self.raw_dir.glob('*_raw.csv')]
        
        processed_data = {}
        
        for ticker in tickers:
            try:
                df = self.process_ticker(ticker)
                
                # Add rolling variance
                df = self.calculate_rolling_variance(df, window=252)
                
                # Save processed data
                output_file = self.processed_dir / f"{ticker}_processed.csv"
                df.to_csv(output_file)
                logger.info(f"  Saved to {output_file}")
                
                processed_data[ticker] = df
                
            except Exception as e:
                logger.error(f"Failed to process {ticker}: {e}")
        
        return processed_data
    
    def get_summary_stats(self, data: dict) -> pd.DataFrame:
        """
        Generate summary statistics for processed data.
        
        Args:
            data: Dictionary mapping ticker to processed DataFrame
            
        Returns:
            Summary statistics DataFrame
        """
        summary = []
        
        for ticker, df in data.items():
            total_return = (df['Growth_1Dollar'].iloc[-1] - 1) * 100
            cagr = ((df['Growth_1Dollar'].iloc[-1] ** (252 / len(df))) - 1) * 100
            
            summary.append({
                'Ticker': ticker,
                'Start': df.index[0].date(),
                'End': df.index[-1].date(),
                'Days': len(df),
                'Total Return %': f"{total_return:.1f}",
                'CAGR %': f"{cagr:.2f}",
                'Avg Daily %': f"{df['Daily_Return_Pct'].mean():.3f}",
                'Daily Vol %': f"{df['Daily_Return_Pct'].std():.3f}",
                'Max Daily Gain %': f"{df['Daily_Return_Pct'].max():.2f}",
                'Max Daily Loss %': f"{df['Daily_Return_Pct'].min():.2f}"
            })
        
        return pd.DataFrame(summary)


if __name__ == '__main__':
    # Example usage
    processor = DataProcessor()
    data = processor.process_all()
    summary = processor.get_summary_stats(data)
    
    print("\n" + "="*80)
    print("DATA PROCESSING SUMMARY")
    print("="*80)
    print(summary.to_string(index=False))
    print("="*80)
