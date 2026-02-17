"""
Comparative analysis of investment strategies.

Compares:
- QQQ buy-and-hold
- TQQQ DCA
- Synthetic TQQQ
across different start dates and market conditions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ComparativeAnalyzer:
    """Compare different investment strategies."""
    
    def __init__(self):
        """Initialize comparative analyzer."""
        logger.info("Comparative Analyzer initialized")
    
    def compare_strategies(self, qqq_dca_results: pd.DataFrame, 
                          tqqq_dca_results: pd.DataFrame,
                          synthetic_dca_results: pd.DataFrame = None) -> pd.DataFrame:
        """
        Compare DCA results across different assets.
        
        Args:
            qqq_dca_results: DCA results for QQQ
            tqqq_dca_results: DCA results for TQQQ
            synthetic_dca_results: Optional DCA results for synthetic TQQQ
            
        Returns:
            Merged comparison dataframe
        """
        # Merge on start_date
        comparison = qqq_dca_results.merge(
            tqqq_dca_results,
            on='start_date',
            suffixes=('_QQQ', '_TQQQ')
        )
        
        if synthetic_dca_results is not None:
            comparison = comparison.merge(
                synthetic_dca_results[['start_date', 'final_value', 'cagr', 'total_return_pct']],
                on='start_date',
                suffixes=('', '_Synthetic')
            )
            comparison.rename(columns={
                'final_value': 'final_value_Synthetic',
                'cagr': 'cagr_Synthetic',
                'total_return_pct': 'total_return_pct_Synthetic'
            }, inplace=True)
        
        # Calculate outperformance metrics
        comparison['TQQQ_vs_QQQ_value_ratio'] = comparison['final_value_TQQQ'] / comparison['final_value_QQQ']
        comparison['TQQQ_vs_QQQ_cagr_diff'] = comparison['cagr_TQQQ'] - comparison['cagr_QQQ']
        comparison['TQQQ_outperformed'] = comparison['final_value_TQQQ'] > comparison['final_value_QQQ']
        
        # Log summary
        win_rate = (comparison['TQQQ_outperformed'].sum() / len(comparison)) * 100
        avg_value_ratio = comparison['TQQQ_vs_QQQ_value_ratio'].mean()
        median_value_ratio = comparison['TQQQ_vs_QQQ_value_ratio'].median()
        
        logger.info(f"\nStrategy Comparison Summary:")
        logger.info(f"  Scenarios: {len(comparison)}")
        logger.info(f"  TQQQ win rate: {win_rate:.1f}%")
        logger.info(f"  Avg TQQQ/QQQ value ratio: {avg_value_ratio:.2f}x")
        logger.info(f"  Median TQQQ/QQQ value ratio: {median_value_ratio:.2f}x")
        logger.info(f"  Median CAGR diff: {comparison['TQQQ_vs_QQQ_cagr_diff'].median():+.2f}%")
        
        return comparison
    
    def identify_market_regimes(self, price_data: pd.DataFrame, 
                               window: int = 252) -> pd.DataFrame:
        """
        Identify market regimes (trending vs choppy).
        
        Args:
            price_data: Price dataframe with Daily_Return column
            window: Rolling window for regime classification
            
        Returns:
            DataFrame with regime indicators
        """
        df = price_data.copy()
        
        # Calculate rolling metrics
        df['Rolling_Return'] = df['Growth_1Dollar'].pct_change(window)
        df['Rolling_Volatility'] = df['Daily_Return'].rolling(window).std() * np.sqrt(252)
        
        # Simple regime classification
        # Trending: high absolute return, moderate volatility
        # Choppy: low absolute return, high volatility
        df['Abs_Rolling_Return'] = df['Rolling_Return'].abs()
        
        # Normalize metrics for comparison
        ret_median = df['Abs_Rolling_Return'].median()
        vol_median = df['Rolling_Volatility'].median()
        
        df['High_Return'] = df['Abs_Rolling_Return'] > ret_median
        df['High_Vol'] = df['Rolling_Volatility'] > vol_median
        
        # Classify regimes
        conditions = [
            (df['High_Return'] & ~df['High_Vol']),  # Trending
            (~df['High_Return'] & df['High_Vol']),  # Choppy
        ]
        choices = ['Trending', 'Choppy']
        df['Regime'] = np.select(conditions, choices, default='Mixed')
        
        return df
    
    def analyze_by_regime(self, comparison_df: pd.DataFrame, 
                         regime_df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze strategy performance by market regime.
        
        Args:
            comparison_df: Strategy comparison results
            regime_df: Market regime classification
            
        Returns:
            Summary by regime
        """
        # Map regimes to start dates
        comparison_df['Regime'] = comparison_df['start_date'].map(
            regime_df.set_index(regime_df.index)['Regime']
        )
        
        # Group by regime
        regime_summary = comparison_df.groupby('Regime').agg({
            'TQQQ_vs_QQQ_value_ratio': ['count', 'mean', 'median', 'min', 'max'],
            'TQQQ_vs_QQQ_cagr_diff': ['mean', 'median'],
            'TQQQ_outperformed': ['sum', 'mean']
        }).round(2)
        
        logger.info("\nPerformance by Market Regime:")
        print(regime_summary)
        
        return regime_summary
    
    def calculate_drawdowns(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate drawdown metrics.
        
        Args:
            price_data: DataFrame with Growth_1Dollar column
            
        Returns:
            DataFrame with drawdown metrics
        """
        df = price_data.copy()
        
        # Calculate running maximum
        df['Running_Max'] = df['Growth_1Dollar'].cummax()
        
        # Calculate drawdown
        df['Drawdown'] = (df['Growth_1Dollar'] / df['Running_Max']) - 1
        df['Drawdown_Pct'] = df['Drawdown'] * 100
        
        # Find drawdown periods
        df['In_Drawdown'] = df['Drawdown'] < 0
        
        return df
    
    def get_worst_scenarios(self, comparison_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
        """
        Identify worst-performing DCA start dates.
        
        Args:
            comparison_df: Strategy comparison results
            n: Number of worst scenarios to return
            
        Returns:
            DataFrame with worst scenarios
        """
        worst = comparison_df.nsmallest(n, 'cagr_TQQQ')[
            ['start_date', 'end_date_TQQQ', 'cagr_QQQ', 'cagr_TQQQ', 
             'TQQQ_vs_QQQ_cagr_diff', 'final_value_QQQ', 'final_value_TQQQ']
        ]
        
        logger.info(f"\nWorst {n} TQQQ DCA scenarios:")
        print(worst.to_string(index=False))
        
        return worst
    
    def get_best_scenarios(self, comparison_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
        """
        Identify best-performing DCA start dates.
        
        Args:
            comparison_df: Strategy comparison results
            n: Number of best scenarios to return
            
        Returns:
            DataFrame with best scenarios
        """
        best = comparison_df.nlargest(n, 'cagr_TQQQ')[
            ['start_date', 'end_date_TQQQ', 'cagr_QQQ', 'cagr_TQQQ', 
             'TQQQ_vs_QQQ_cagr_diff', 'final_value_QQQ', 'final_value_TQQQ']
        ]
        
        logger.info(f"\nBest {n} TQQQ DCA scenarios:")
        print(best.to_string(index=False))
        
        return best


if __name__ == '__main__':
    # Example usage
    processed_dir = Path('data/processed')
    
    # Load DCA results
    qqq_dca = pd.read_csv(processed_dir / 'dca_rolling_analysis_QQQ.csv', parse_dates=['start_date', 'end_date'])
    tqqq_dca = pd.read_csv(processed_dir / 'dca_rolling_analysis_TQQQ.csv', parse_dates=['start_date', 'end_date'])
    
    # Create analyzer
    analyzer = ComparativeAnalyzer()
    
    # Compare strategies
    comparison = analyzer.compare_strategies(qqq_dca, tqqq_dca)
    
    # Save comparison
    comparison.to_csv(processed_dir / 'strategy_comparison.csv', index=False)
    logger.info(f"Saved comparison to {processed_dir / 'strategy_comparison.csv'}")
    
    # Find best/worst scenarios
    worst = analyzer.get_worst_scenarios(comparison, n=10)
    best = analyzer.get_best_scenarios(comparison, n=10)
