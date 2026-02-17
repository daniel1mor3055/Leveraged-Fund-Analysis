"""
Process raw price data into clean analysis-ready format.

Calculates:
- Daily returns (both price-return and total-return tracks)
- Cumulative returns
- Realized variance
- Validates data quality

Dual-Track Processing:
- Price Return: Based on unadjusted Close prices. Use for LETF modeling
  since leveraged ETFs target daily index PRICE performance (not total return).
- Total Return: Based on Adj Close (dividend-adjusted). Use for investor
  analysis showing actual returns including distributions.
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
    
    def process_ticker(self, ticker: str, default_to_price_return: bool = True) -> pd.DataFrame:
        """
        Process a single ticker's raw data with dual-track returns.
        
        Provides both price-return and total-return tracks:
        - Price Return: Use for LETF modeling (matches daily index objective)
        - Total Return: Use for investor analysis (includes dividends)
        
        Args:
            ticker: Ticker symbol (without ^)
            default_to_price_return: If True, default 'Daily_Return' uses price return
                                     (correct for LETF modeling). If False, uses total return.
            
        Returns:
            Processed DataFrame with dual-track daily returns and metrics
        """
        input_file = self.raw_dir / f"{ticker}_raw.csv"
        
        if not input_file.exists():
            raise FileNotFoundError(f"Raw data not found: {input_file}")
        
        logger.info(f"Processing {ticker} from {input_file}")
        
        # Load raw data
        df = pd.read_csv(input_file, index_col=0, parse_dates=True)
        
        # Create processed dataframe
        processed = pd.DataFrame(index=df.index)
        
        # ========================================================================
        # DUAL-TRACK PROCESSING
        # ========================================================================
        
        # Track 1: Price Return (for LETF modeling - matches daily index objective)
        processed['Close_Price'] = df['Close']
        processed['Daily_Return_Price'] = processed['Close_Price'].pct_change()
        processed['Daily_Return_Price_Pct'] = processed['Daily_Return_Price'] * 100
        
        # Track 2: Total Return (for investor analysis - includes dividends)
        if 'Adj Close' in df.columns:
            processed['Close_TotalReturn'] = df['Adj Close']
            processed['Daily_Return_TotalReturn'] = processed['Close_TotalReturn'].pct_change()
            processed['Daily_Return_TotalReturn_Pct'] = processed['Daily_Return_TotalReturn'] * 100
            logger.info(f"  Dual-track: Price Return (Close) + Total Return (Adj Close)")
        else:
            # If no Adj Close available, total return = price return
            processed['Close_TotalReturn'] = df['Close']
            processed['Daily_Return_TotalReturn'] = processed['Daily_Return_Price']
            processed['Daily_Return_TotalReturn_Pct'] = processed['Daily_Return_Price_Pct']
            logger.info(f"  Single-track: No Adj Close available, using Close for both")
        
        # ========================================================================
        # DEFAULT COLUMNS (for backward compatibility)
        # ========================================================================
        
        # Set default columns based on use case
        if default_to_price_return:
            # Default to price return (correct for LETF modeling)
            processed['Close'] = processed['Close_Price']
            processed['Daily_Return'] = processed['Daily_Return_Price']
            processed['Daily_Return_Pct'] = processed['Daily_Return_Price_Pct']
            logger.info(f"  Default track: Price Return (for LETF modeling)")
        else:
            # Default to total return (for investor analysis)
            processed['Close'] = processed['Close_TotalReturn']
            processed['Daily_Return'] = processed['Daily_Return_TotalReturn']
            processed['Daily_Return_Pct'] = processed['Daily_Return_TotalReturn_Pct']
            logger.info(f"  Default track: Total Return (for investor analysis)")
        
        # ========================================================================
        # CUMULATIVE METRICS (using default track)
        # ========================================================================
        
        processed['Growth_1Dollar'] = (1 + processed['Daily_Return']).cumprod()
        processed['Cumulative_Return'] = processed['Growth_1Dollar'] - 1
        processed['Cumulative_Return_Pct'] = processed['Cumulative_Return'] * 100
        
        # Also calculate cumulative for both tracks explicitly
        processed['Growth_1Dollar_Price'] = (1 + processed['Daily_Return_Price']).cumprod()
        processed['Growth_1Dollar_TotalReturn'] = (1 + processed['Daily_Return_TotalReturn']).cumprod()
        
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
