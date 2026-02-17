"""
Dollar-Cost Averaging (DCA) simulation engine.

Simulates periodic investments into QQQ, TQQQ, or synthetic leveraged ETFs
across different start dates to test timing sensitivity.

Uses XIRR (Extended Internal Rate of Return) for accurate money-weighted
return calculation that properly accounts for the timing of each cash flow.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from pathlib import Path
import logging
from scipy.optimize import brentq

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DCASimulator:
    """Simulate dollar-cost averaging strategies."""
    
    def __init__(self, price_data: pd.DataFrame):
        """
        Initialize DCA simulator.
        
        Args:
            price_data: DataFrame with DatetimeIndex and 'Close' column (price per share)
        """
        self.price_data = price_data.copy()
        self.price_data = self.price_data.sort_index()
        logger.info(f"DCA Simulator initialized with {len(price_data)} days of data")
        logger.info(f"  Date range: {price_data.index[0].date()} to {price_data.index[-1].date()}")
    
    def simulate_dca(self, start_date: pd.Timestamp, end_date: pd.Timestamp, 
                     investment_amount: float = 1000.0, frequency: str = 'M') -> dict:
        """
        Simulate DCA strategy for a specific time period.
        
        Args:
            start_date: When to start investing
            end_date: When to stop (or evaluate portfolio)
            investment_amount: Amount to invest per period (default $1000)
            frequency: Investment frequency - 'M' (monthly), 'W' (weekly), 'Q' (quarterly)
            
        Returns:
            Dictionary with simulation results
        """
        # Get price data for the period
        mask = (self.price_data.index >= start_date) & (self.price_data.index <= end_date)
        period_data = self.price_data.loc[mask].copy()
        
        if len(period_data) == 0:
            logger.warning(f"No data available for {start_date.date()} to {end_date.date()}")
            return None
        
        # Generate investment dates
        investment_dates = self._generate_investment_dates(
            start_date, end_date, frequency, period_data.index
        )
        
        if len(investment_dates) == 0:
            logger.warning(f"No valid investment dates generated")
            return None
        
        # Simulate purchases
        total_invested = 0
        total_shares = 0
        purchases = []
        
        for inv_date in investment_dates:
            # Find closest trading day
            trading_date = self._get_closest_trading_day(inv_date, period_data.index)
            if trading_date is None:
                continue
            
            price = period_data.loc[trading_date, 'Close']
            shares = investment_amount / price
            total_shares += shares
            total_invested += investment_amount
            
            purchases.append({
                'date': trading_date,
                'price': price,
                'shares': shares,
                'invested': investment_amount,
                'total_invested': total_invested,
                'total_shares': total_shares
            })
        
        # Calculate final portfolio value
        final_price = period_data.iloc[-1]['Close']
        final_date = period_data.index[-1]
        final_value = total_shares * final_price
        
        # Calculate returns
        total_return = final_value - total_invested
        total_return_pct = (total_return / total_invested) * 100 if total_invested > 0 else 0
        
        # Calculate time period
        years = (end_date - start_date).days / 365.25
        
        # Calculate XIRR (correct money-weighted return for DCA)
        xirr = self._calculate_xirr(purchases, final_value, final_date)
        
        # Calculate simple CAGR (kept for reference - NOTE: this is INCORRECT for DCA
        # as it assumes all capital was invested at start_date)
        simple_cagr = ((final_value / total_invested) ** (1 / years) - 1) * 100 if years > 0 and total_invested > 0 else 0
        
        # Calculate average purchase price
        avg_price = total_invested / total_shares if total_shares > 0 else 0
        
        return {
            'start_date': start_date,
            'end_date': end_date,
            'num_investments': len(purchases),
            'total_invested': total_invested,
            'total_shares': total_shares,
            'avg_purchase_price': avg_price,
            'final_price': final_price,
            'final_value': final_value,
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'xirr': xirr,  # Correct money-weighted annualized return
            'simple_cagr': simple_cagr,  # Kept for backwards compatibility (incorrect for DCA)
            'cagr': xirr,  # Default to correct metric
            'years': years,
            'purchases': purchases
        }
    
    def _generate_investment_dates(self, start_date: pd.Timestamp, end_date: pd.Timestamp,
                                   frequency: str, available_dates: pd.DatetimeIndex) -> list:
        """Generate list of investment dates based on frequency."""
        dates = []
        current = start_date
        
        while current <= end_date:
            dates.append(current)
            
            if frequency == 'M':  # Monthly
                current += relativedelta(months=1)
            elif frequency == 'W':  # Weekly
                current += timedelta(weeks=1)
            elif frequency == 'Q':  # Quarterly
                current += relativedelta(months=3)
            else:
                raise ValueError(f"Unknown frequency: {frequency}")
        
        return dates
    
    def _get_closest_trading_day(self, target_date: pd.Timestamp, 
                                  available_dates: pd.DatetimeIndex) -> pd.Timestamp:
        """Find closest trading day to target date (forward-looking)."""
        future_dates = available_dates[available_dates >= target_date]
        if len(future_dates) > 0:
            return future_dates[0]
        return None
    
    def _calculate_xirr(self, purchases: list, final_value: float, 
                        final_date: pd.Timestamp) -> float:
        """
        Calculate XIRR (Extended Internal Rate of Return) for DCA cash flows.
        
        XIRR properly accounts for the timing of each investment (money-weighted return),
        unlike simple CAGR which incorrectly assumes all capital was invested at the start.
        
        Args:
            purchases: List of purchase dictionaries with 'date' and 'invested' keys
            final_value: Final portfolio value at end date
            final_date: End date for the investment period
            
        Returns:
            Annualized XIRR as a percentage (e.g., 10.5 for 10.5%)
        """
        if not purchases or final_value <= 0:
            return 0.0
        
        # Build cash flow series: negative for investments (outflows), positive for final value (inflow)
        dates = [p['date'] for p in purchases] + [final_date]
        amounts = [-p['invested'] for p in purchases] + [final_value]
        
        # Convert dates to year fractions from first date
        base_date = dates[0]
        year_fractions = [(d - base_date).days / 365.25 for d in dates]
        
        def npv(rate):
            """Calculate NPV for a given annual rate."""
            return sum(cf / ((1 + rate) ** t) for cf, t in zip(amounts, year_fractions))
        
        try:
            # Use Brent's method to find the rate where NPV = 0
            # Search between -99% (near total loss) and 1000% (10x annual return)
            xirr = brentq(npv, -0.99, 10.0, xtol=1e-8)
            return xirr * 100  # Convert to percentage
        except ValueError:
            # If no root found in range, try a wider/different range or return 0
            try:
                # Try more conservative range for edge cases
                xirr = brentq(npv, -0.9999, 100.0, xtol=1e-6)
                return xirr * 100
            except ValueError:
                logger.warning("XIRR calculation failed to converge")
                return 0.0
        except Exception as e:
            logger.warning(f"XIRR calculation error: {e}")
            return 0.0
    
    def rolling_start_dates_analysis(self, frequency: str = 'M', investment_amount: float = 1000.0,
                                     holding_period_years: int = 10, 
                                     start_date_frequency: str = 'Q') -> pd.DataFrame:
        """
        Run DCA simulations for multiple rolling start dates.
        
        This tests how timing (starting date) affects DCA outcomes.
        
        Args:
            frequency: DCA investment frequency ('M', 'W', 'Q')
            investment_amount: Amount per investment
            holding_period_years: How long to hold after starting DCA
            start_date_frequency: How often to test new start dates ('M', 'Q', 'Y')
            
        Returns:
            DataFrame with results for each start date
        """
        logger.info(f"\nRunning rolling DCA analysis:")
        logger.info(f"  Investment: ${investment_amount} every {frequency} for {holding_period_years} years")
        logger.info(f"  Testing start dates every {start_date_frequency}")
        
        results = []
        
        # Generate start dates to test
        first_date = self.price_data.index[0]
        last_possible_start = self.price_data.index[-1] - pd.DateOffset(years=holding_period_years)
        
        test_start_dates = self._generate_investment_dates(
            first_date, last_possible_start, start_date_frequency, self.price_data.index
        )
        
        logger.info(f"  Testing {len(test_start_dates)} different start dates")
        
        for i, start_date in enumerate(test_start_dates):
            end_date = start_date + pd.DateOffset(years=holding_period_years)
            
            # Make sure end date doesn't exceed available data
            if end_date > self.price_data.index[-1]:
                continue
            
            result = self.simulate_dca(start_date, end_date, investment_amount, frequency)
            
            if result is not None:
                results.append({
                    'start_date': result['start_date'],
                    'end_date': result['end_date'],
                    'years': result['years'],
                    'num_investments': result['num_investments'],
                    'total_invested': result['total_invested'],
                    'final_value': result['final_value'],
                    'total_return': result['total_return'],
                    'total_return_pct': result['total_return_pct'],
                    'xirr': result['xirr'],  # Correct money-weighted return
                    'simple_cagr': result['simple_cagr'],  # For reference only
                    'cagr': result['cagr'],  # Defaults to xirr (correct metric)
                    'avg_purchase_price': result['avg_purchase_price'],
                    'final_price': result['final_price']
                })
            
            if (i + 1) % 10 == 0:
                logger.info(f"  Processed {i + 1}/{len(test_start_dates)} start dates")
        
        df = pd.DataFrame(results)
        logger.info(f"\n✓ Completed {len(df)} simulations")
        
        return df


if __name__ == '__main__':
    # Example usage
    from pathlib import Path
    
    processed_dir = Path('data/processed')
    
    # Load QQQ data
    qqq = pd.read_csv(processed_dir / 'QQQ_processed.csv', index_col=0, parse_dates=True)
    
    # Create simulator
    simulator = DCASimulator(qqq[['Close']])
    
    # Run rolling analysis
    results = simulator.rolling_start_dates_analysis(
        frequency='M',
        investment_amount=1000,
        holding_period_years=10,
        start_date_frequency='Q'
    )
    
    # Save results
    output_file = processed_dir / 'dca_rolling_analysis_QQQ.csv'
    results.to_csv(output_file, index=False)
    logger.info(f"Saved results to {output_file}")
    
    # Summary statistics
    print("\n" + "="*60)
    print(f"DCA Analysis Summary (10-year holding periods)")
    print("="*60)
    print(f"Scenarios tested: {len(results)}")
    print(f"\nXIRR (Correct Money-Weighted Return):")
    print(f"  Median XIRR: {results['xirr'].median():.2f}%")
    print(f"  Mean XIRR: {results['xirr'].mean():.2f}%")
    print(f"  Best XIRR: {results['xirr'].max():.2f}%")
    print(f"  Worst XIRR: {results['xirr'].min():.2f}%")
    print(f"  Std Dev: {results['xirr'].std():.2f}%")
    print(f"\nSimple CAGR (For Reference - Incorrect for DCA):")
    print(f"  Median: {results['simple_cagr'].median():.2f}%")
    print(f"  Mean: {results['simple_cagr'].mean():.2f}%")
    print("="*60)
