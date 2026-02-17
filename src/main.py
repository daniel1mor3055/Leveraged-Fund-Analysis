"""
Leveraged Fund Analysis - Main Orchestrator

Complete analysis pipeline:
1. Fetch historical data (QQQ, TQQQ, NDX)
2. Process data (calculate returns, variance)
3. Build synthetic TQQQ model
4. Run DCA simulations across rolling start dates
5. Calculate variance drag metrics
6. Compare strategies
7. Generate visualizations
8. Export results
"""

import logging
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from data.fetch_data import DataFetcher
from data.process_data import DataProcessor
from models.synthetic_tqqq import SyntheticLeveragedETF
from models.dca_simulator import DCASimulator
from analysis.variance_drag import VarianceDragAnalyzer
from analysis.comparative_metrics import ComparativeAnalyzer
from visualization.charts import Visualizer

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class LeveragedFundAnalysis:
    """Main orchestrator for complete analysis pipeline."""
    
    def __init__(self):
        """Initialize analysis pipeline."""
        logger.info("\n" + "="*80)
        logger.info("LEVERAGED FUND ANALYSIS PIPELINE")
        logger.info("="*80)
        
        # Create directories
        Path('data/raw').mkdir(parents=True, exist_ok=True)
        Path('data/processed').mkdir(parents=True, exist_ok=True)
        Path('output/figures').mkdir(parents=True, exist_ok=True)
        Path('output/results').mkdir(parents=True, exist_ok=True)
    
    def run_full_analysis(self, force_refresh: bool = False):
        """
        Run complete analysis pipeline.
        
        Args:
            force_refresh: If True, re-download all data
        """
        try:
            # Step 1: Fetch data
            logger.info("\n[STEP 1/8] Fetching historical data...")
            fetcher = DataFetcher()
            raw_data = fetcher.fetch_all(force_refresh=force_refresh)
            summary = fetcher.get_data_summary(raw_data)
            print("\n" + summary.to_string(index=False))
            
            # Step 2: Process data
            logger.info("\n[STEP 2/8] Processing raw data...")
            processor = DataProcessor()
            processed_data = processor.process_all()
            stats = processor.get_summary_stats(processed_data)
            print("\n" + stats.to_string(index=False))
            
            # Step 3: Build synthetic TQQQ
            logger.info("\n[STEP 3/8] Building synthetic TQQQ model...")
            qqq = processed_data['QQQ']
            synthetic_model = SyntheticLeveragedETF(leverage=3.0, expense_ratio=0.0095)
            
            # With fees
            synthetic_with_fees = synthetic_model.simulate(qqq['Daily_Return'], include_fees=True)
            synthetic_with_fees.to_csv('data/processed/synthetic_tqqq_with_fees.csv')
            
            # Frictionless (theoretical baseline)
            synthetic_frictionless = synthetic_model.simulate(qqq['Daily_Return'], include_fees=False)
            synthetic_frictionless.to_csv('data/processed/synthetic_tqqq_frictionless.csv')
            
            # Compare to actual TQQQ if available
            if 'TQQQ' in processed_data:
                comparison = synthetic_model.compare_to_actual(synthetic_with_fees, processed_data['TQQQ'])
                comparison.to_csv('data/processed/synthetic_vs_actual_comparison.csv')
            
            # Step 4: DCA simulations
            logger.info("\n[STEP 4/8] Running DCA simulations...")
            self._run_dca_simulations(processed_data, synthetic_with_fees)
            
            # Step 5: Variance drag analysis
            logger.info("\n[STEP 5/8] Calculating variance drag metrics...")
            self._analyze_variance_drag(qqq, synthetic_with_fees)
            
            # Step 6: Comparative analysis
            logger.info("\n[STEP 6/8] Running comparative analysis...")
            self._run_comparative_analysis()
            
            # Step 7: Visualizations
            logger.info("\n[STEP 7/8] Generating visualizations...")
            self._create_visualizations(processed_data)
            
            # Step 8: Export final results
            logger.info("\n[STEP 8/8] Exporting results...")
            self._export_summary_report()
            
            logger.info("\n" + "="*80)
            logger.info("✓ ANALYSIS COMPLETE!")
            logger.info("="*80)
            logger.info("Results saved to:")
            logger.info("  - Processed data: data/processed/")
            logger.info("  - Visualizations: output/figures/")
            logger.info("  - Summary report: output/results/summary_report.txt")
            logger.info("="*80 + "\n")
            
        except Exception as e:
            logger.error(f"\n✗ Analysis failed: {e}", exc_info=True)
            raise
    
    def _run_dca_simulations(self, processed_data: dict, synthetic_df: pd.DataFrame):
        """Run DCA simulations for QQQ, TQQQ, and synthetic."""
        
        # QQQ DCA
        logger.info("  Running QQQ DCA simulations...")
        qqq_simulator = DCASimulator(processed_data['QQQ'][['Close']])
        qqq_dca_results = qqq_simulator.rolling_start_dates_analysis(
            frequency='M',
            investment_amount=1000,
            holding_period_years=10,
            start_date_frequency='Q'
        )
        qqq_dca_results.to_csv('data/processed/dca_rolling_analysis_QQQ.csv', index=False)
        
        # TQQQ DCA (if available)
        if 'TQQQ' in processed_data:
            logger.info("  Running TQQQ DCA simulations...")
            tqqq_simulator = DCASimulator(processed_data['TQQQ'][['Close']])
            tqqq_dca_results = tqqq_simulator.rolling_start_dates_analysis(
                frequency='M',
                investment_amount=1000,
                holding_period_years=10,
                start_date_frequency='Q'
            )
            tqqq_dca_results.to_csv('data/processed/dca_rolling_analysis_TQQQ.csv', index=False)
        
        # Synthetic TQQQ DCA
        logger.info("  Running Synthetic TQQQ DCA simulations...")
        synthetic_price_df = pd.DataFrame({
            'Close': synthetic_df['NAV']
        }, index=synthetic_df.index)
        synthetic_simulator = DCASimulator(synthetic_price_df)
        synthetic_dca_results = synthetic_simulator.rolling_start_dates_analysis(
            frequency='M',
            investment_amount=1000,
            holding_period_years=10,
            start_date_frequency='Q'
        )
        synthetic_dca_results.to_csv('data/processed/dca_rolling_analysis_Synthetic.csv', index=False)
    
    def _analyze_variance_drag(self, qqq_df: pd.DataFrame, synthetic_df: pd.DataFrame):
        """Analyze variance drag effects."""
        analyzer = VarianceDragAnalyzer(leverage=3.0)
        
        # Overall variance drag
        drag_metrics = analyzer.calculate_variance_drag(qqq_df['Daily_Return'])
        
        # Compare naïve vs actual
        comparison = analyzer.compare_naive_vs_actual(qqq_df, synthetic_df)
        comparison.to_csv('data/processed/variance_drag_comparison.csv')
        
        # Rolling variance impact
        rolling = analyzer.rolling_variance_impact(qqq_df, synthetic_df, window=252)
        rolling.to_csv('data/processed/variance_drag_rolling.csv')
    
    def _run_comparative_analysis(self):
        """Run comparative analysis of strategies."""
        analyzer = ComparativeAnalyzer()
        
        # Load DCA results
        qqq_dca = pd.read_csv('data/processed/dca_rolling_analysis_QQQ.csv', 
                             parse_dates=['start_date', 'end_date'])
        
        # Check if TQQQ results exist
        tqqq_file = Path('data/processed/dca_rolling_analysis_TQQQ.csv')
        if tqqq_file.exists():
            tqqq_dca = pd.read_csv(tqqq_file, parse_dates=['start_date', 'end_date'])
            
            # Compare strategies (will only match overlapping start dates)
            comparison = analyzer.compare_strategies(qqq_dca, tqqq_dca)
            
            if len(comparison) > 0:
                comparison.to_csv('data/processed/strategy_comparison.csv', index=False)
                
                # Get best/worst scenarios
                worst = analyzer.get_worst_scenarios(comparison, n=min(10, len(comparison)))
                best = analyzer.get_best_scenarios(comparison, n=min(10, len(comparison)))
                
                # Save scenarios
                worst.to_csv('output/results/worst_scenarios.csv', index=False)
                best.to_csv('output/results/best_scenarios.csv', index=False)
            else:
                logger.warning("No overlapping DCA periods between QQQ and TQQQ")
                # Compare synthetic instead
                synthetic_dca = pd.read_csv('data/processed/dca_rolling_analysis_Synthetic.csv',
                                          parse_dates=['start_date', 'end_date'])
                comparison = analyzer.compare_strategies(qqq_dca, synthetic_dca)
                comparison.to_csv('data/processed/strategy_comparison.csv', index=False)
        else:
            # Use synthetic as fallback
            synthetic_dca = pd.read_csv('data/processed/dca_rolling_analysis_Synthetic.csv',
                                      parse_dates=['start_date', 'end_date'])
            comparison = analyzer.compare_strategies(qqq_dca, synthetic_dca)
            comparison.to_csv('data/processed/strategy_comparison.csv', index=False)
    
    def _create_visualizations(self, processed_data: dict):
        """Generate all visualizations."""
        viz = Visualizer(output_dir='output/figures')
        
        # Load comparison data
        comparison = pd.read_csv('data/processed/strategy_comparison.csv',
                                parse_dates=['start_date', 'end_date_QQQ', 'end_date_TQQQ'])
        
        if len(comparison) > 0:
            # DCA outcomes
            viz.plot_dca_outcomes_by_start_date(comparison, metric='cagr')
            viz.plot_dca_outcomes_by_start_date(comparison, metric='final_value')
            viz.plot_dca_outperformance_heatmap(comparison)
            viz.plot_distribution_of_outcomes(comparison)
        else:
            logger.warning("No comparison data available - skipping DCA visualizations")
        
        # Growth comparison
        qqq = processed_data['QQQ']
        synthetic = pd.read_csv('data/processed/synthetic_tqqq_with_fees.csv', 
                               index_col=0, parse_dates=True)
        
        if 'TQQQ' in processed_data:
            tqqq = processed_data['TQQQ']
            viz.plot_growth_comparison(qqq, tqqq, synthetic)
        else:
            # Use synthetic as TQQQ
            viz.plot_growth_comparison(qqq, synthetic)
        
        # Drawdown analysis
        from analysis.comparative_metrics import ComparativeAnalyzer
        analyzer = ComparativeAnalyzer()
        
        qqq_dd = analyzer.calculate_drawdowns(qqq)
        viz.plot_drawdowns(qqq_dd, title='QQQ Drawdown Analysis')
        
        if 'TQQQ' in processed_data:
            tqqq_dd = analyzer.calculate_drawdowns(processed_data['TQQQ'])
            viz.plot_drawdowns(tqqq_dd, title='TQQQ Drawdown Analysis')
        else:
            synthetic_dd = analyzer.calculate_drawdowns(synthetic)
            viz.plot_drawdowns(synthetic_dd, title='Synthetic TQQQ Drawdown Analysis')
        
        # Variance drag scatter
        rolling_variance = pd.read_csv('data/processed/variance_drag_rolling.csv',
                                      index_col=0, parse_dates=True)
        viz.plot_variance_drag_scatter(rolling_variance)
    
    def _export_summary_report(self):
        """Generate summary report."""
        comparison_file = Path('data/processed/strategy_comparison.csv')
        
        if not comparison_file.exists():
            logger.warning("No comparison data available - skipping summary report")
            return
        
        comparison = pd.read_csv(comparison_file)
        
        if len(comparison) == 0:
            logger.warning("Comparison data is empty - skipping summary report")
            return
        
        report = []
        report.append("="*80)
        report.append("LEVERAGED FUND ANALYSIS - SUMMARY REPORT")
        report.append("="*80)
        report.append("")
        
        report.append("ANALYSIS PARAMETERS:")
        report.append("  - Investment Strategy: Dollar-Cost Averaging (DCA)")
        report.append("  - Investment Amount: $1,000 per month")
        report.append("  - Holding Period: 10 years")
        report.append("  - Assets Analyzed: QQQ, TQQQ, Synthetic 3x TQQQ")
        report.append("")
        
        report.append("KEY FINDINGS:")
        report.append(f"  - Number of scenarios tested: {len(comparison)}")
        if 'TQQQ_outperformed' in comparison.columns:
            win_rate = (comparison['TQQQ_outperformed'].sum() / len(comparison)) * 100
            report.append(f"  - TQQQ win rate: {win_rate:.1f}%")
        report.append("")
        
        report.append("QQQ DCA PERFORMANCE:")
        report.append(f"  - Median CAGR: {comparison['cagr_QQQ'].median():.2f}%")
        report.append(f"  - Mean CAGR: {comparison['cagr_QQQ'].mean():.2f}%")
        report.append(f"  - Best CAGR: {comparison['cagr_QQQ'].max():.2f}%")
        report.append(f"  - Worst CAGR: {comparison['cagr_QQQ'].min():.2f}%")
        report.append(f"  - Std Dev: {comparison['cagr_QQQ'].std():.2f}%")
        report.append("")
        
        report.append("TQQQ DCA PERFORMANCE:")
        report.append(f"  - Median CAGR: {comparison['cagr_TQQQ'].median():.2f}%")
        report.append(f"  - Mean CAGR: {comparison['cagr_TQQQ'].mean():.2f}%")
        report.append(f"  - Best CAGR: {comparison['cagr_TQQQ'].max():.2f}%")
        report.append(f"  - Worst CAGR: {comparison['cagr_TQQQ'].min():.2f}%")
        report.append(f"  - Std Dev: {comparison['cagr_TQQQ'].std():.2f}%")
        report.append("")
        
        if 'TQQQ_vs_QQQ_value_ratio' in comparison.columns:
            report.append("RELATIVE PERFORMANCE:")
            report.append(f"  - Median TQQQ/QQQ value ratio: {comparison['TQQQ_vs_QQQ_value_ratio'].median():.2f}x")
            report.append(f"  - Mean TQQQ/QQQ value ratio: {comparison['TQQQ_vs_QQQ_value_ratio'].mean():.2f}x")
            report.append(f"  - Median CAGR difference: {comparison['TQQQ_vs_QQQ_cagr_diff'].median():+.2f}%")
            report.append("")
        
        report.append("="*80)
        
        # Write report
        report_text = "\n".join(report)
        with open('output/results/summary_report.txt', 'w') as f:
            f.write(report_text)
        
        print("\n" + report_text)


def main():
    """Main entry point."""
    analysis = LeveragedFundAnalysis()
    analysis.run_full_analysis(force_refresh=False)


if __name__ == "__main__":
    main()
