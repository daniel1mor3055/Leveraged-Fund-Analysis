"""
Synthetic leveraged ETF model using daily-reset 3x leverage.

Implements the Cheng-Madhavan framework:
    A(t+1) = A(t) * (1 + β * r(t))

where:
    - A(t) is the leveraged ETF NAV at time t
    - β is the leverage factor (3 for TQQQ)
    - r(t) is the underlying index return

Enhanced with financing cost modeling:
    Daily cost = expense_ratio/252 + (leverage-1) * (risk_free + spread) / 252

For a 3x fund, this includes:
- Management fee: expense_ratio / 252 per day
- Financing cost: 2 * (risk_free_rate + spread) / 252 per day
  (The fund borrows 2x the NAV to achieve 3x exposure)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Historical average risk-free rates by decade (approximate Fed Funds rates)
# Used as fallback when actual rate data is not provided
HISTORICAL_RISK_FREE_RATES = {
    1980: 0.12,   # ~12% average in early 1980s
    1985: 0.08,   # ~8% late 1980s
    1990: 0.05,   # ~5% early 1990s
    1995: 0.05,   # ~5% late 1990s
    2000: 0.04,   # ~4% early 2000s
    2005: 0.03,   # ~3% mid 2000s
    2008: 0.01,   # Near zero during financial crisis
    2010: 0.0025, # ~0.25% ZIRP era
    2015: 0.005,  # ~0.5% beginning to normalize
    2018: 0.02,   # ~2% pre-pandemic
    2020: 0.0025, # Near zero COVID response
    2022: 0.03,   # ~3% rate hikes
    2023: 0.05,   # ~5% elevated rates
    2024: 0.05,   # ~5% 
    2025: 0.045,  # Assumed ~4.5%
    2026: 0.04,   # Assumed ~4%
}


class SyntheticLeveragedETF:
    """
    Model a synthetic leveraged ETF with financing costs.
    
    This model accounts for:
    - Daily leverage reset (daily compounding of leveraged returns)
    - Management expense ratio (annual fee applied daily)
    - Financing costs (cost of borrowing to achieve leverage)
    - Time-varying parameters for more realistic historical simulation
    """
    
    def __init__(self, leverage: float = 3.0, expense_ratio: float = 0.0095,
                 financing_spread: float = 0.004, default_risk_free: float = 0.03):
        """
        Initialize synthetic leveraged ETF.
        
        Args:
            leverage: Leverage factor (default 3.0 for 3x)
            expense_ratio: Annual expense ratio (default 0.95% for TQQQ)
            financing_spread: Annual spread over risk-free rate for borrowing
                             (default 0.4% = 40 basis points)
            default_risk_free: Default annual risk-free rate when not provided
                              (default 3% = 0.03)
        """
        self.leverage = leverage
        self.expense_ratio = expense_ratio
        self.financing_spread = financing_spread
        self.default_risk_free = default_risk_free
        self.daily_fee = expense_ratio / 252  # Convert annual to daily
        
        logger.info(f"Synthetic {leverage}x ETF initialized:")
        logger.info(f"  Expense ratio: {expense_ratio*100:.2f}%")
        logger.info(f"  Financing spread: {financing_spread*100:.2f}% over risk-free")
        logger.info(f"  Default risk-free rate: {default_risk_free*100:.2f}%")
    
    def _get_historical_risk_free_rate(self, year: int) -> float:
        """Get approximate historical risk-free rate for a given year."""
        # Find the closest year in our lookup table
        available_years = sorted(HISTORICAL_RISK_FREE_RATES.keys())
        closest_year = min(available_years, key=lambda y: abs(y - year))
        return HISTORICAL_RISK_FREE_RATES[closest_year]
    
    def _build_risk_free_series(self, index: pd.DatetimeIndex) -> pd.Series:
        """Build a series of historical risk-free rates for each date."""
        rates = pd.Series(index=index, dtype=float)
        for date in index:
            rates[date] = self._get_historical_risk_free_rate(date.year)
        return rates
    
    def simulate(self, underlying_returns: pd.Series, 
                 include_fees: bool = True,
                 include_financing_costs: bool = True,
                 expense_ratios: pd.Series = None,
                 risk_free_rates: pd.Series = None,
                 initial_value: float = 1.0) -> pd.DataFrame:
        """
        Simulate leveraged ETF performance with enhanced cost modeling.
        
        Args:
            underlying_returns: Daily returns of the underlying index/ETF (as decimal, not %)
            include_fees: If True, apply daily expense ratio drag
            include_financing_costs: If True, apply financing costs for leverage
            expense_ratios: Optional time-varying annual expense ratios (Series with same index)
                           If None, uses static self.expense_ratio
            risk_free_rates: Optional time-varying annual risk-free rates (Series with same index)
                            If None, uses historical estimates based on year
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
        
        # ========================================================================
        # COST CALCULATION
        # ========================================================================
        
        if include_fees or include_financing_costs:
            # Daily expense ratio
            if expense_ratios is not None:
                # Time-varying expense ratio
                daily_expense = expense_ratios.reindex(result.index, method='ffill') / 252
            else:
                # Static expense ratio
                daily_expense = self.daily_fee
            
            # Daily financing cost (only if leverage > 1)
            if include_financing_costs and self.leverage > 1:
                # Amount borrowed = (leverage - 1) * NAV
                # Financing cost = borrowed * (risk_free + spread)
                
                if risk_free_rates is not None:
                    # Use provided risk-free rates
                    rf = risk_free_rates.reindex(result.index, method='ffill')
                else:
                    # Use historical estimates
                    rf = self._build_risk_free_series(result.index)
                
                result['Risk_Free_Rate'] = rf
                daily_financing = (self.leverage - 1) * (rf + self.financing_spread) / 252
            else:
                daily_financing = 0
                result['Risk_Free_Rate'] = 0
            
            # Total daily drag
            result['Daily_Expense'] = daily_expense if isinstance(daily_expense, pd.Series) else daily_expense
            result['Daily_Financing'] = daily_financing if isinstance(daily_financing, pd.Series) else daily_financing
            result['Total_Daily_Drag'] = result['Daily_Expense'] + result['Daily_Financing']
            
            # Apply total drag
            if include_fees:
                result['Fee_Adjusted_Return'] = result['Leveraged_Return'] - result['Total_Daily_Drag']
            else:
                result['Fee_Adjusted_Return'] = result['Leveraged_Return']
            
            logger.info(f"  Cost modeling: fees={include_fees}, financing={include_financing_costs}")
            if include_financing_costs and self.leverage > 1:
                avg_financing = result['Daily_Financing'].mean() * 252 * 100
                logger.info(f"  Avg annual financing cost: {avg_financing:.2f}%")
        else:
            result['Fee_Adjusted_Return'] = result['Leveraged_Return']
            result['Daily_Expense'] = 0
            result['Daily_Financing'] = 0
            result['Total_Daily_Drag'] = 0
            result['Risk_Free_Rate'] = 0
        
        # ========================================================================
        # NAV CALCULATION
        # ========================================================================
        
        # Compound returns to get NAV path
        result['NAV'] = initial_value * (1 + result['Fee_Adjusted_Return']).cumprod()
        result['Growth_1Dollar'] = result['NAV'] / initial_value
        result['Cumulative_Return'] = result['Growth_1Dollar'] - 1
        result['Cumulative_Return_Pct'] = result['Cumulative_Return'] * 100
        
        # Track daily % returns
        result['Daily_Return_Pct'] = result['Fee_Adjusted_Return'] * 100
        result['Daily_Return'] = result['Fee_Adjusted_Return']  # For compatibility
        
        self._log_summary(result, include_fees, include_financing_costs)
        return result
    
    def _log_summary(self, result: pd.DataFrame, include_fees: bool, 
                      include_financing: bool = True) -> None:
        """Log simulation summary statistics."""
        final_value = result['NAV'].iloc[-1]
        total_return = (final_value - 1) * 100
        days = len(result)
        cagr = ((final_value ** (252 / days)) - 1) * 100
        
        # Build status string
        cost_components = []
        if include_fees:
            cost_components.append("expense ratio")
        if include_financing:
            cost_components.append("financing costs")
        fee_status = f"with {' + '.join(cost_components)}" if cost_components else "frictionless"
        
        logger.info(f"  ✓ Simulation complete ({fee_status})")
        logger.info(f"    Final value: ${final_value:.2f} (Total return: {total_return:.1f}%)")
        logger.info(f"    CAGR: {cagr:.2f}%")
        logger.info(f"    Avg daily return: {result['Daily_Return_Pct'].mean():.3f}%")
        logger.info(f"    Daily volatility: {result['Daily_Return_Pct'].std():.3f}%")
        
        # Log cost breakdown
        if 'Total_Daily_Drag' in result.columns:
            avg_annual_drag = result['Total_Daily_Drag'].mean() * 252 * 100
            logger.info(f"    Avg annual cost drag: {avg_annual_drag:.2f}%")
    
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
