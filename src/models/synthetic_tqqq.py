"""
Synthetic leveraged ETF model using daily-reset 3x leverage.

Implements the Cheng-Madhavan framework:
    A(t+1) = A(t) * (1 + β * r(t))

where:
    - A(t) is the leveraged ETF NAV at time t
    - β is the leverage factor (3 for TQQQ)
    - r(t) is the underlying index return
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SyntheticLeveragedETF:
    """Model a synthetic 3x leveraged ETF with optional fees."""
    
    def __init__(self, leverage: float = 3.0, expense_ratio: float = 0.0095):
        """
        Initialize synthetic leveraged ETF.
        
        Args:
            leverage: Leverage factor (default 3.0 for 3x)
            expense_ratio: Annual expense ratio (default 0.95% for TQQQ)
        """
        self.leverage = leverage
        self.expense_ratio = expense_ratio
        self.daily_fee = expense_ratio / 252  # Convert annual to daily
        logger.info(f"Synthetic {leverage}x ETF initialized (expense ratio: {expense_ratio*100:.2f}%)")
    
    def simulate(self, underlying_returns: pd.Series, include_fees: bool = True, 
                 initial_value: float = 1.0) -> pd.DataFrame:
        """
        Simulate leveraged ETF performance.
        
        Args:
            underlying_returns: Daily returns of the underlying index/ETF (as decimal, not %)
            include_fees: If True, apply daily expense ratio drag
            initial_value: Starting NAV (default $1)
            
        Returns:
            DataFrame with synthetic leveraged ETF metrics
        """
        logger.info(f"Simulating {self.leverage}x ETF over {len(underlying_returns)} days")
        
        # Create result dataframe
        result = pd.DataFrame(index=underlying_returns.index)
        result['Underlying_Return'] = underlying_returns
        
        # Calculate leveraged returns (daily-reset formula)
        result['Leveraged_Return'] = self.leverage * underlying_returns
        
        # Apply fee drag if requested
        if include_fees:
            result['Fee_Adjusted_Return'] = result['Leveraged_Return'] - self.daily_fee
        else:
            result['Fee_Adjusted_Return'] = result['Leveraged_Return']
        
        # Compound returns to get NAV path
        result['NAV'] = initial_value * (1 + result['Fee_Adjusted_Return']).cumprod()
        result['Growth_1Dollar'] = result['NAV'] / initial_value
        result['Cumulative_Return'] = result['Growth_1Dollar'] - 1
        result['Cumulative_Return_Pct'] = result['Cumulative_Return'] * 100
        
        # Track daily % returns
        result['Daily_Return_Pct'] = result['Fee_Adjusted_Return'] * 100
        
        self._log_summary(result, include_fees)
        return result
    
    def _log_summary(self, result: pd.DataFrame, include_fees: bool) -> None:
        """Log simulation summary statistics."""
        final_value = result['NAV'].iloc[-1]
        total_return = (final_value - 1) * 100
        days = len(result)
        cagr = ((final_value ** (252 / days)) - 1) * 100
        
        fee_status = "with fees" if include_fees else "frictionless"
        logger.info(f"  ✓ Simulation complete ({fee_status})")
        logger.info(f"    Final value: ${final_value:.2f} (Total return: {total_return:.1f}%)")
        logger.info(f"    CAGR: {cagr:.2f}%")
        logger.info(f"    Avg daily return: {result['Daily_Return_Pct'].mean():.3f}%")
        logger.info(f"    Daily volatility: {result['Daily_Return_Pct'].std():.3f}%")
    
    def compare_to_actual(self, synthetic_df: pd.DataFrame, actual_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compare synthetic model to actual TQQQ performance.
        
        Args:
            synthetic_df: Synthetic leveraged ETF dataframe (from simulate())
            actual_df: Actual TQQQ processed dataframe
            
        Returns:
            Comparison dataframe with tracking error metrics
        """
        # Align dates (actual TQQQ only available from 2010-02-11)
        common_index = synthetic_df.index.intersection(actual_df.index)
        
        if len(common_index) == 0:
            logger.warning("No overlapping dates between synthetic and actual!")
            return pd.DataFrame()
        
        comparison = pd.DataFrame(index=common_index)
        comparison['Synthetic_NAV'] = synthetic_df.loc[common_index, 'NAV']
        comparison['Actual_NAV'] = actual_df.loc[common_index, 'Growth_1Dollar']
        
        # Normalize to same starting point
        comparison['Synthetic_NAV'] = comparison['Synthetic_NAV'] / comparison['Synthetic_NAV'].iloc[0]
        comparison['Actual_NAV'] = comparison['Actual_NAV'] / comparison['Actual_NAV'].iloc[0]
        
        # Calculate tracking difference
        comparison['Tracking_Diff'] = comparison['Actual_NAV'] - comparison['Synthetic_NAV']
        comparison['Tracking_Diff_Pct'] = (comparison['Tracking_Diff'] / comparison['Synthetic_NAV']) * 100
        
        self._log_tracking_error(comparison)
        return comparison
    
    def _log_tracking_error(self, comparison: pd.DataFrame) -> None:
        """Log tracking error statistics."""
        final_synthetic = comparison['Synthetic_NAV'].iloc[-1]
        final_actual = comparison['Actual_NAV'].iloc[-1]
        diff_pct = ((final_actual / final_synthetic) - 1) * 100
        
        logger.info(f"\nTracking Comparison:")
        logger.info(f"  Synthetic final value: ${final_synthetic:.2f}")
        logger.info(f"  Actual final value: ${final_actual:.2f}")
        logger.info(f"  Difference: {diff_pct:+.2f}%")
        logger.info(f"  Mean tracking error: {comparison['Tracking_Diff_Pct'].mean():.3f}%")
        logger.info(f"  Std tracking error: {comparison['Tracking_Diff_Pct'].std():.3f}%")


if __name__ == '__main__':
    # Example: simulate TQQQ using QQQ returns
    from pathlib import Path
    
    processed_dir = Path('data/processed')
    
    # Load QQQ data
    qqq = pd.read_csv(processed_dir / 'QQQ_processed.csv', index_col=0, parse_dates=True)
    
    # Create synthetic 3x leveraged ETF
    synthetic_tqqq = SyntheticLeveragedETF(leverage=3.0, expense_ratio=0.0095)
    
    # Simulate with fees
    result_with_fees = synthetic_tqqq.simulate(qqq['Daily_Return'], include_fees=True)
    
    # Simulate without fees (theoretical baseline)
    result_frictionless = synthetic_tqqq.simulate(qqq['Daily_Return'], include_fees=False)
    
    # Compare to actual TQQQ if available
    tqqq_file = processed_dir / 'TQQQ_processed.csv'
    if tqqq_file.exists():
        tqqq = pd.read_csv(tqqq_file, index_col=0, parse_dates=True)
        comparison = synthetic_tqqq.compare_to_actual(result_with_fees, tqqq)
        
        # Save comparison
        output_file = processed_dir / 'synthetic_vs_actual_comparison.csv'
        comparison.to_csv(output_file)
        logger.info(f"\nSaved comparison to {output_file}")
