"""
Variance drag and volatility analysis.

Implements the Avellaneda-Zhang framework for analyzing the "variance drain"
effect in leveraged ETFs:

    L(t)/L(0) = (S(t)/S(0))^β * exp(...+ (β - β²)/2 * ∫σ²)

For β = 3, the term (β - β²)/2 = -3, meaning higher realized variance
reduces leveraged returns relative to naïve expectations.
"""

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VarianceDragAnalyzer:
    """Analyze volatility drag in leveraged ETFs."""
    
    def __init__(self, leverage: float = 3.0):
        """
        Initialize analyzer.
        
        Args:
            leverage: Leverage factor (default 3.0 for TQQQ)
        """
        self.leverage = leverage
        self.variance_coefficient = (leverage - leverage**2) / 2
        logger.info(f"Variance Drag Analyzer initialized ({leverage}x leverage)")
        logger.info(f"  Variance coefficient: {self.variance_coefficient:.2f}")
    
    def calculate_realized_variance(self, returns: pd.Series, window: int = None) -> float:
        """
        Calculate realized variance over a period.
        
        Args:
            returns: Daily returns (as decimal, not %)
            window: Rolling window in days, or None for entire period
            
        Returns:
            Annualized realized variance
        """
        if window is not None:
            variance = returns.rolling(window=window).var() * 252
        else:
            variance = returns.var() * 252
        
        return variance
    
    def calculate_variance_drag(self, returns: pd.Series) -> dict:
        """
        Calculate variance drag impact on leveraged returns.
        
        Args:
            returns: Daily returns of underlying asset
            
        Returns:
            Dictionary with variance drag metrics
        """
        # Realized variance (annualized)
        realized_var = returns.var() * 252
        realized_vol = returns.std() * np.sqrt(252)
        
        # Time period
        days = len(returns)
        years = days / 252
        
        # Cumulative variance drag term: (β - β²)/2 * ∫σ²
        # Approximate: (β - β²)/2 * σ² * T
        cumulative_drag = self.variance_coefficient * realized_var * years
        
        # Impact as percentage reduction
        drag_impact_pct = (np.exp(cumulative_drag) - 1) * 100
        
        logger.info(f"\nVariance Drag Analysis:")
        logger.info(f"  Period: {years:.2f} years ({days} days)")
        logger.info(f"  Realized volatility: {realized_vol*100:.2f}% annualized")
        logger.info(f"  Realized variance: {realized_var:.4f}")
        logger.info(f"  Cumulative drag term: {cumulative_drag:.4f}")
        logger.info(f"  Drag impact: {drag_impact_pct:.2f}%")
        
        return {
            'days': days,
            'years': years,
            'realized_volatility': realized_vol,
            'realized_variance': realized_var,
            'cumulative_drag_term': cumulative_drag,
            'drag_impact_pct': drag_impact_pct
        }
    
    def compare_naive_vs_actual(self, underlying_df: pd.DataFrame, 
                                leveraged_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compare naïve expectation (β × underlying) vs actual leveraged performance.
        
        Args:
            underlying_df: Processed dataframe of underlying asset (e.g., QQQ)
            leveraged_df: Processed dataframe of leveraged asset (e.g., TQQQ or synthetic)
            
        Returns:
            Comparison dataframe
        """
        # Align dates
        common_index = underlying_df.index.intersection(leveraged_df.index)
        
        if len(common_index) == 0:
            logger.warning("No overlapping dates!")
            return pd.DataFrame()
        
        comparison = pd.DataFrame(index=common_index)
        
        # Actual performance
        comparison['Underlying_Growth'] = underlying_df.loc[common_index, 'Growth_1Dollar']
        comparison['Leveraged_Growth'] = leveraged_df.loc[common_index, 'Growth_1Dollar']
        
        # Normalize to start at 1
        comparison['Underlying_Growth'] = comparison['Underlying_Growth'] / comparison['Underlying_Growth'].iloc[0]
        comparison['Leveraged_Growth'] = comparison['Leveraged_Growth'] / comparison['Leveraged_Growth'].iloc[0]
        
        # Naïve expectation: (underlying growth)^β
        comparison['Naive_Expectation'] = comparison['Underlying_Growth'] ** self.leverage
        
        # Performance gap
        comparison['Gap'] = comparison['Leveraged_Growth'] - comparison['Naive_Expectation']
        comparison['Gap_Pct'] = (comparison['Gap'] / comparison['Naive_Expectation']) * 100
        
        # Calculate variance drag
        underlying_returns = underlying_df.loc[common_index, 'Daily_Return']
        drag_metrics = self.calculate_variance_drag(underlying_returns)
        
        # Log comparison
        final_underlying = comparison['Underlying_Growth'].iloc[-1]
        final_leveraged = comparison['Leveraged_Growth'].iloc[-1]
        final_naive = comparison['Naive_Expectation'].iloc[-1]
        
        logger.info(f"\nPerformance Comparison:")
        logger.info(f"  Underlying: ${final_underlying:.2f} (+{(final_underlying-1)*100:.1f}%)")
        logger.info(f"  Naïve {self.leverage}x: ${final_naive:.2f} (+{(final_naive-1)*100:.1f}%)")
        logger.info(f"  Actual leveraged: ${final_leveraged:.2f} (+{(final_leveraged-1)*100:.1f}%)")
        logger.info(f"  Gap: {((final_leveraged/final_naive)-1)*100:+.2f}%")
        
        return comparison
    
    def rolling_variance_impact(self, underlying_df: pd.DataFrame, 
                               synthetic_df: pd.DataFrame, 
                               window: int = 252) -> pd.DataFrame:
        """
        Calculate rolling variance and its impact on performance gap.
        
        Args:
            underlying_df: Underlying asset dataframe
            synthetic_df: Synthetic leveraged ETF dataframe
            window: Rolling window in days
            
        Returns:
            DataFrame with rolling metrics
        """
        common_index = underlying_df.index.intersection(synthetic_df.index)
        
        result = pd.DataFrame(index=common_index)
        result['Rolling_Variance'] = underlying_df.loc[common_index, 'Daily_Return'].rolling(window).var() * 252
        result['Rolling_Volatility'] = underlying_df.loc[common_index, 'Daily_Return'].rolling(window).std() * np.sqrt(252)
        
        # Calculate expected drag
        result['Expected_Drag'] = self.variance_coefficient * result['Rolling_Variance']
        
        # Calculate actual performance
        underlying_growth = underlying_df.loc[common_index, 'Growth_1Dollar']
        leveraged_growth = synthetic_df.loc[common_index, 'Growth_1Dollar']
        
        # Normalize
        underlying_growth = underlying_growth / underlying_growth.iloc[0]
        leveraged_growth = leveraged_growth / leveraged_growth.iloc[0]
        
        result['Naive_Growth'] = underlying_growth ** self.leverage
        result['Actual_Growth'] = leveraged_growth
        result['Performance_Gap_Pct'] = ((result['Actual_Growth'] / result['Naive_Growth']) - 1) * 100
        
        return result


if __name__ == '__main__':
    # Example usage
    from pathlib import Path
    
    processed_dir = Path('data/processed')
    
    # Load data
    qqq = pd.read_csv(processed_dir / 'QQQ_processed.csv', index_col=0, parse_dates=True)
    
    # Create analyzer
    analyzer = VarianceDragAnalyzer(leverage=3.0)
    
    # Calculate variance drag for entire QQQ history
    drag_metrics = analyzer.calculate_variance_drag(qqq['Daily_Return'])
    
    # If synthetic TQQQ exists, compare
    synthetic_file = processed_dir / 'synthetic_tqqq_with_fees.csv'
    if synthetic_file.exists():
        synthetic = pd.read_csv(synthetic_file, index_col=0, parse_dates=True)
        
        # Compare naïve vs actual
        comparison = analyzer.compare_naive_vs_actual(qqq, synthetic)
        
        # Calculate rolling variance impact
        rolling = analyzer.rolling_variance_impact(qqq, synthetic, window=252)
        
        # Save results
        comparison.to_csv(processed_dir / 'variance_drag_comparison.csv')
        rolling.to_csv(processed_dir / 'variance_drag_rolling.csv')
        
        logger.info(f"\nSaved variance drag analysis to {processed_dir}")
