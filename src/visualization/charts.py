"""
Visualization utilities for leveraged fund analysis.

Creates charts for:
- DCA outcome heatmaps
- Time-series comparisons
- Volatility drag impact
- Drawdown analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10


class Visualizer:
    """Create visualizations for analysis results."""
    
    def __init__(self, output_dir: str = 'output/figures'):
        """Initialize visualizer with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Visualizer initialized. Output: {self.output_dir}")
    
    def plot_dca_outcomes_by_start_date(self, comparison_df: pd.DataFrame, 
                                       metric: str = 'cagr') -> None:
        """
        Plot DCA outcomes (QQQ vs TQQQ) by start date.
        
        Args:
            comparison_df: Strategy comparison dataframe
            metric: Metric to plot ('cagr', 'total_return_pct', 'final_value')
        """
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # Plot both strategies
        ax.plot(comparison_df['start_date'], comparison_df[f'{metric}_QQQ'], 
                label='QQQ DCA', linewidth=2, alpha=0.8, marker='o', markersize=4)
        ax.plot(comparison_df['start_date'], comparison_df[f'{metric}_TQQQ'], 
                label='TQQQ DCA', linewidth=2, alpha=0.8, marker='s', markersize=4)
        
        # Add zero line if showing returns
        if metric in ['cagr', 'total_return_pct']:
            ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
        
        # Styling
        ax.set_xlabel('DCA Start Date', fontsize=12, fontweight='bold')
        
        if metric == 'cagr':
            ax.set_ylabel('CAGR (%)', fontsize=12, fontweight='bold')
            title = 'DCA Performance by Start Date: CAGR Comparison'
        elif metric == 'total_return_pct':
            ax.set_ylabel('Total Return (%)', fontsize=12, fontweight='bold')
            title = 'DCA Performance by Start Date: Total Return Comparison'
        else:
            ax.set_ylabel('Final Portfolio Value ($)', fontsize=12, fontweight='bold')
            title = 'DCA Performance by Start Date: Final Value Comparison'
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filename = f'dca_outcomes_by_start_date_{metric}.png'
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {filepath}")
        plt.close()
    
    def plot_dca_outperformance_heatmap(self, comparison_df: pd.DataFrame) -> None:
        """
        Create heatmap showing TQQQ/QQQ value ratio by start date.
        
        Args:
            comparison_df: Strategy comparison dataframe
        """
        if len(comparison_df) == 0:
            logger.warning("No data for heatmap - skipping")
            return
        
        fig, ax = plt.subplots(figsize=(16, 6))
        
        # Ensure start_date is datetime
        if not pd.api.types.is_datetime64_any_dtype(comparison_df['start_date']):
            comparison_df['start_date'] = pd.to_datetime(comparison_df['start_date'])
        
        # Extract year and quarter from start date
        comparison_df['Year'] = comparison_df['start_date'].dt.year
        comparison_df['Quarter'] = comparison_df['start_date'].dt.quarter
        
        # Pivot for heatmap
        pivot = comparison_df.pivot_table(
            values='TQQQ_vs_QQQ_value_ratio',
            index='Quarter',
            columns='Year',
            aggfunc='mean'
        )
        
        # Create heatmap
        sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn', center=1.0, 
                   cbar_kws={'label': 'TQQQ/QQQ Final Value Ratio'},
                   linewidths=0.5, ax=ax)
        
        ax.set_title('TQQQ vs QQQ Performance: DCA Value Ratio by Start Date\n(Green = TQQQ outperforms, Red = QQQ outperforms)', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Start Year', fontsize=12, fontweight='bold')
        ax.set_ylabel('Start Quarter', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        filepath = self.output_dir / 'dca_outperformance_heatmap.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {filepath}")
        plt.close()
    
    def plot_growth_comparison(self, qqq_df: pd.DataFrame, tqqq_df: pd.DataFrame,
                              synthetic_df: pd.DataFrame = None, 
                              start_date: str = None, end_date: str = None) -> None:
        """
        Plot growth of $1 invested for multiple assets.
        
        Args:
            qqq_df: QQQ processed dataframe
            tqqq_df: TQQQ processed dataframe
            synthetic_df: Optional synthetic TQQQ dataframe
            start_date: Optional start date for plot window
            end_date: Optional end date for plot window
        """
        fig, ax = plt.subplots(figsize=(16, 9))
        
        # Filter date range if specified
        if start_date:
            qqq_df = qqq_df[qqq_df.index >= start_date]
            tqqq_df = tqqq_df[tqqq_df.index >= start_date]
            if synthetic_df is not None:
                synthetic_df = synthetic_df[synthetic_df.index >= start_date]
        
        if end_date:
            qqq_df = qqq_df[qqq_df.index <= end_date]
            tqqq_df = tqqq_df[tqqq_df.index <= end_date]
            if synthetic_df is not None:
                synthetic_df = synthetic_df[synthetic_df.index <= end_date]
        
        # Normalize to start at $1
        qqq_growth = qqq_df['Growth_1Dollar'] / qqq_df['Growth_1Dollar'].iloc[0]
        tqqq_growth = tqqq_df['Growth_1Dollar'] / tqqq_df['Growth_1Dollar'].iloc[0]
        
        # Plot
        ax.plot(qqq_growth.index, qqq_growth, label='QQQ', linewidth=2.5, alpha=0.9)
        ax.plot(tqqq_growth.index, tqqq_growth, label='TQQQ (Actual)', linewidth=2.5, alpha=0.9)
        
        if synthetic_df is not None:
            synthetic_growth = synthetic_df['Growth_1Dollar'] / synthetic_df['Growth_1Dollar'].iloc[0]
            ax.plot(synthetic_growth.index, synthetic_growth, label='TQQQ (Synthetic)', 
                   linewidth=2, linestyle='--', alpha=0.7)
        
        # Logarithmic scale for better visibility
        ax.set_yscale('log')
        
        # Styling
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Growth of $1 (log scale)', fontsize=12, fontweight='bold')
        ax.set_title('Growth of $1 Investment: QQQ vs TQQQ', fontsize=14, fontweight='bold', pad=20)
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3, which='both')
        
        plt.tight_layout()
        
        filepath = self.output_dir / 'growth_comparison.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {filepath}")
        plt.close()
    
    def plot_drawdowns(self, drawdown_df: pd.DataFrame, title: str = 'Drawdown Analysis') -> None:
        """
        Plot drawdown over time.
        
        Args:
            drawdown_df: DataFrame with Drawdown_Pct column
            title: Chart title
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
        
        # Top: Growth
        ax1.plot(drawdown_df.index, drawdown_df['Growth_1Dollar'], linewidth=2, color='blue', alpha=0.8)
        ax1.set_ylabel('Growth of $1', fontsize=12, fontweight='bold')
        ax1.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Bottom: Drawdown
        ax2.fill_between(drawdown_df.index, 0, drawdown_df['Drawdown_Pct'], 
                        color='red', alpha=0.6, label='Drawdown')
        ax2.set_ylabel('Drawdown (%)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=11)
        
        plt.tight_layout()
        
        filename = title.lower().replace(' ', '_').replace(':', '') + '.png'
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {filepath}")
        plt.close()
    
    def plot_variance_drag_scatter(self, rolling_variance_df: pd.DataFrame) -> None:
        """
        Scatter plot: realized variance vs performance gap.
        
        Args:
            rolling_variance_df: DataFrame with variance and performance metrics
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Remove NaN values
        df = rolling_variance_df.dropna()
        
        # Scatter plot
        scatter = ax.scatter(df['Rolling_Volatility'] * 100, df['Performance_Gap_Pct'],
                           c=df.index.astype(np.int64), cmap='viridis', 
                           alpha=0.6, s=30)
        
        # Add trend line
        z = np.polyfit(df['Rolling_Volatility'] * 100, df['Performance_Gap_Pct'], 1)
        p = np.poly1d(z)
        ax.plot(df['Rolling_Volatility'] * 100, p(df['Rolling_Volatility'] * 100), 
               "r--", linewidth=2, label=f'Trend: y = {z[0]:.2f}x + {z[1]:.2f}')
        
        # Styling
        ax.set_xlabel('Rolling Volatility (% annualized)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Performance Gap: Actual vs Naïve 3x (%)', fontsize=12, fontweight='bold')
        ax.set_title('Variance Drag Effect: Volatility vs Leveraged Performance Gap', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Time progression', fontsize=10)
        
        plt.tight_layout()
        
        filepath = self.output_dir / 'variance_drag_scatter.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {filepath}")
        plt.close()
    
    def plot_distribution_of_outcomes(self, comparison_df: pd.DataFrame) -> None:
        """
        Plot distribution of DCA outcomes.
        
        Args:
            comparison_df: Strategy comparison dataframe
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # CAGR distribution
        axes[0, 0].hist(comparison_df['cagr_QQQ'], bins=30, alpha=0.6, label='QQQ', color='blue', edgecolor='black')
        axes[0, 0].hist(comparison_df['cagr_TQQQ'], bins=30, alpha=0.6, label='TQQQ', color='orange', edgecolor='black')
        axes[0, 0].axvline(comparison_df['cagr_QQQ'].median(), color='blue', linestyle='--', linewidth=2, label=f'QQQ Median: {comparison_df["cagr_QQQ"].median():.1f}%')
        axes[0, 0].axvline(comparison_df['cagr_TQQQ'].median(), color='orange', linestyle='--', linewidth=2, label=f'TQQQ Median: {comparison_df["cagr_TQQQ"].median():.1f}%')
        axes[0, 0].set_xlabel('CAGR (%)', fontsize=11, fontweight='bold')
        axes[0, 0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
        axes[0, 0].set_title('Distribution of CAGR', fontsize=12, fontweight='bold')
        axes[0, 0].legend(fontsize=9)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Final value distribution
        axes[0, 1].hist(comparison_df['final_value_QQQ'], bins=30, alpha=0.6, label='QQQ', color='blue', edgecolor='black')
        axes[0, 1].hist(comparison_df['final_value_TQQQ'], bins=30, alpha=0.6, label='TQQQ', color='orange', edgecolor='black')
        axes[0, 1].set_xlabel('Final Portfolio Value ($)', fontsize=11, fontweight='bold')
        axes[0, 1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
        axes[0, 1].set_title('Distribution of Final Values', fontsize=12, fontweight='bold')
        axes[0, 1].legend(fontsize=9)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Value ratio distribution
        axes[1, 0].hist(comparison_df['TQQQ_vs_QQQ_value_ratio'], bins=40, alpha=0.7, 
                       color='green', edgecolor='black')
        axes[1, 0].axvline(1.0, color='red', linestyle='--', linewidth=2, label='Break-even (1.0x)')
        axes[1, 0].axvline(comparison_df['TQQQ_vs_QQQ_value_ratio'].median(), 
                          color='darkgreen', linestyle='--', linewidth=2, 
                          label=f'Median: {comparison_df["TQQQ_vs_QQQ_value_ratio"].median():.2f}x')
        axes[1, 0].set_xlabel('TQQQ/QQQ Final Value Ratio', fontsize=11, fontweight='bold')
        axes[1, 0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
        axes[1, 0].set_title('Distribution of TQQQ vs QQQ Performance', fontsize=12, fontweight='bold')
        axes[1, 0].legend(fontsize=9)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Box plot comparison
        data_to_plot = [comparison_df['cagr_QQQ'], comparison_df['cagr_TQQQ']]
        bp = axes[1, 1].boxplot(data_to_plot, labels=['QQQ', 'TQQQ'], patch_artist=True,
                                showmeans=True, meanline=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightsalmon')
        axes[1, 1].set_ylabel('CAGR (%)', fontsize=11, fontweight='bold')
        axes[1, 1].set_title('CAGR Box Plot Comparison', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Distribution of DCA Outcomes Across Different Start Dates', 
                    fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        filepath = self.output_dir / 'outcome_distributions.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {filepath}")
        plt.close()


if __name__ == '__main__':
    # Example usage
    processed_dir = Path('data/processed')
    
    viz = Visualizer(output_dir='output/figures')
    
    # Load comparison data
    comparison = pd.read_csv(processed_dir / 'strategy_comparison.csv', 
                            parse_dates=['start_date', 'end_date_QQQ', 'end_date_TQQQ'])
    
    # Create visualizations
    viz.plot_dca_outcomes_by_start_date(comparison, metric='cagr')
    viz.plot_dca_outcomes_by_start_date(comparison, metric='final_value')
    viz.plot_dca_outperformance_heatmap(comparison)
    viz.plot_distribution_of_outcomes(comparison)
    
    logger.info("Visualization complete!")
