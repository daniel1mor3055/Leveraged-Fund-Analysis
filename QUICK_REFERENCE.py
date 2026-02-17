"""
Quick Reference Guide - Leveraged Fund Analysis

This document explains how to use and customize the analysis.
"""

# =============================================================================
# QUICK START
# =============================================================================

# 1. Run the complete analysis:
./run_analysis.sh

# 2. Or manually:
venv/bin/python3 src/main.py

# 3. View results:
open output/figures/
cat output/results/summary_report.txt


# =============================================================================
# CUSTOMIZING THE ANALYSIS
# =============================================================================

# You can modify parameters in src/main.py or run individual components:

# --- Fetch fresh data ---
from src.data.fetch_data import DataFetcher
fetcher = DataFetcher()
data = fetcher.fetch_all(force_refresh=True)  # Re-download data

# --- Process data ---
from src.data.process_data import DataProcessor
processor = DataProcessor()
processed = processor.process_all()

# --- Run DCA simulations with custom parameters ---
from src.models.dca_simulator import DCASimulator
import pandas as pd

qqq = pd.read_csv('data/processed/QQQ_processed.csv', index_col=0, parse_dates=True)
simulator = DCASimulator(qqq[['Close']])

# Custom simulation:
results = simulator.rolling_start_dates_analysis(
    frequency='M',              # Investment frequency: 'M' (monthly), 'W' (weekly), 'Q' (quarterly)
    investment_amount=2000,     # Amount to invest each period ($)
    holding_period_years=15,    # How long to hold (years)
    start_date_frequency='Q'    # How often to test new start dates
)

# --- Build custom synthetic leveraged ETF ---
from src.models.synthetic_tqqq import SyntheticLeveragedETF

# 2x leverage with custom expense ratio
model_2x = SyntheticLeveragedETF(leverage=2.0, expense_ratio=0.005)
result_2x = model_2x.simulate(qqq['Daily_Return'], include_fees=True)

# --- Analyze variance drag ---
from src.analysis.variance_drag import VarianceDragAnalyzer

analyzer = VarianceDragAnalyzer(leverage=3.0)
drag_metrics = analyzer.calculate_variance_drag(qqq['Daily_Return'])

# --- Create custom visualizations ---
from src.visualization.charts import Visualizer

viz = Visualizer(output_dir='custom_output')
comparison = pd.read_csv('data/processed/strategy_comparison.csv', parse_dates=['start_date'])
viz.plot_dca_outcomes_by_start_date(comparison, metric='cagr')


# =============================================================================
# KEY PARAMETERS TO ADJUST
# =============================================================================

# DCA Simulations (in src/main.py, _run_dca_simulations method):
#   - frequency: 'M' (monthly), 'W' (weekly), 'Q' (quarterly)
#   - investment_amount: dollars per period (default: 1000)
#   - holding_period_years: length of investment horizon (default: 10)
#   - start_date_frequency: how often to test new start dates (default: 'Q')

# Synthetic Leveraged ETF (in src/models/synthetic_tqqq.py):
#   - leverage: multiplier factor (default: 3.0 for TQQQ)
#   - expense_ratio: annual fee (default: 0.0095 = 0.95%)
#   - include_fees: True to apply expense drag, False for theoretical

# Variance Analysis (in src/analysis/variance_drag.py):
#   - window: rolling window for variance calculation (default: 252 = 1 year)


# =============================================================================
# OUTPUT FILES EXPLAINED
# =============================================================================

# DATA FILES (data/processed/):
#   *_processed.csv         - Daily returns and cumulative performance
#   dca_rolling_analysis_*.csv - DCA simulation results for each ticker
#   strategy_comparison.csv - Side-by-side QQQ vs TQQQ comparison
#   variance_drag_*.csv    - Variance drag metrics and rolling analysis

# VISUALIZATIONS (output/figures/):
#   dca_outcomes_by_start_date_*.png - Performance by DCA start date
#   dca_outperformance_heatmap.png   - Year/Quarter heatmap of TQQQ/QQQ ratio
#   outcome_distributions.png         - Histograms and box plots of results
#   growth_comparison.png             - Time series of $1 growth
#   *_drawdown_analysis.png           - Underwater charts showing drawdowns
#   variance_drag_scatter.png         - Volatility vs performance gap

# REPORTS (output/results/):
#   summary_report.txt    - Executive summary of key findings
#   best_scenarios.csv    - Top 10 DCA start dates
#   worst_scenarios.csv   - Bottom 10 DCA start dates


# =============================================================================
# ADVANCED: RUNNING SPECIFIC MODULES
# =============================================================================

# Example 1: Test 5-year DCA instead of 10-year
# Edit src/main.py line ~138 and change:
#     holding_period_years=10  →  holding_period_years=5

# Example 2: Compare different leverage ratios
from src.models.synthetic_tqqq import SyntheticLeveragedETF

leverage_ratios = [1.5, 2.0, 2.5, 3.0]
results = {}

for ratio in leverage_ratios:
    model = SyntheticLeveragedETF(leverage=ratio, expense_ratio=0.0095)
    results[f"{ratio}x"] = model.simulate(qqq['Daily_Return'], include_fees=True)

# Example 3: Analyze only specific market periods
# Filter data before running simulations:
qqq_2010_2020 = qqq[(qqq.index >= '2010-01-01') & (qqq.index <= '2020-12-31')]
simulator = DCASimulator(qqq_2010_2020[['Close']])


# =============================================================================
# TROUBLESHOOTING
# =============================================================================

# If data download fails:
#   - Check internet connection
#   - Yahoo Finance may have rate limits; wait and retry
#   - Use force_refresh=False to use cached data

# If analysis crashes mid-run:
#   - Check data/processed/ to see which steps completed
#   - Individual modules can be run separately
#   - Check analysis.log for detailed error messages

# If visualizations don't generate:
#   - Ensure matplotlib backend is configured correctly
#   - Check that output/figures/ directory exists
#   - Some charts require minimum data (e.g., heatmap needs multiple years)


# =============================================================================
# FURTHER READING
# =============================================================================

# Academic References:
#   - Cheng & Madhavan (2009): Mechanics of Leveraged Returns
#   - Avellaneda & Zhang (2010): Path-Dependence of Leveraged ETF Returns
#   - SEC Investor Bulletin: Updated Guidance on Leveraged/Inverse ETFs

# Key Formulas:
#   Daily-reset leverage: A(t+1) = A(t) × (1 + β × r(t))
#   Variance drag: (β - β²)/2 × ∫σ² 
#   CAGR: (Final/Initial)^(1/years) - 1
