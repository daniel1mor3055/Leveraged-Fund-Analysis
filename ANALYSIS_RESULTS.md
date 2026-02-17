# Leveraged Fund Analysis - Results

## Project Overview

Complete analysis of long-term TQQQ (3x leveraged QQQ) performance using Dollar-Cost Averaging (DCA) strategy across different market entry points.

## Key Findings

### Summary Statistics (10-year DCA, $1,000/month)

**QQQ Performance:**
- Median CAGR: **8.66%**
- Mean CAGR: 7.93%
- Best CAGR: 13.57%
- Worst CAGR: -3.47%
- Standard Deviation: 3.44%

**TQQQ (Synthetic) Performance:**
- Median CAGR: **21.74%**
- Mean CAGR: 19.20%
- Best CAGR: 39.72%
- Worst CAGR: -15.77%
- Standard Deviation: 12.38%

**Relative Performance:**
- TQQQ Win Rate: **83.8%** (TQQQ outperformed QQQ in 57 out of 68 scenarios)
- Median TQQQ/QQQ Final Value Ratio: **3.03x**
- Mean TQQQ/QQQ Final Value Ratio: 3.25x
- Median CAGR Difference: +12.78%

## Analysis Components

### 1. Data Collection ✓
- Downloaded daily historical data from Yahoo Finance
- QQQ: 6,777 days (1999-2026, ~27 years)
- TQQQ: 4,028 days (2010-2026, ~16 years)
- NDX Index: 10,173 days (1985-2026, ~40 years)

### 2. Synthetic TQQQ Model ✓
- Implemented Cheng-Madhavan daily-reset formula: A(t+1) = A(t) × (1 + 3×r(t))
- Applied 0.95% annual expense ratio
- Tracked divergence from actual TQQQ (tracking error: -12.78% mean)

### 3. DCA Simulations ✓
- Tested 68 different start dates (quarterly intervals)
- Each scenario: $1,000/month for 10 years
- Compared QQQ, TQQQ (actual), and Synthetic TQQQ

### 4. Variance Drag Analysis ✓
- Calculated realized volatility: 27.00% annualized for QQQ
- Variance drag impact: -99.72% over full 27-year period
- Demonstrated gap between naïve 3x expectation vs. actual compounded returns

### 5. Visualizations ✓
Generated 8 comprehensive charts:
- DCA outcomes by start date (CAGR and final value)
- Outperformance heatmap by year/quarter
- Distribution of outcomes (histograms and box plots)
- Growth comparison (QQQ vs TQQQ)
- Drawdown analysis (QQQ and TQQQ)
- Variance drag scatter plot

## Files Generated

### Processed Data
- `data/processed/QQQ_processed.csv` - QQQ daily returns and metrics
- `data/processed/TQQQ_processed.csv` - TQQQ daily returns and metrics
- `data/processed/NDX_processed.csv` - Nasdaq-100 Index data
- `data/processed/synthetic_tqqq_with_fees.csv` - Synthetic TQQQ model (with fees)
- `data/processed/synthetic_tqqq_frictionless.csv` - Theoretical 3x without fees
- `data/processed/dca_rolling_analysis_QQQ.csv` - QQQ DCA simulation results
- `data/processed/dca_rolling_analysis_TQQQ.csv` - TQQQ DCA simulation results
- `data/processed/strategy_comparison.csv` - Side-by-side comparison
- `data/processed/variance_drag_comparison.csv` - Variance drag metrics
- `data/processed/variance_drag_rolling.csv` - Rolling variance analysis

### Visualizations
All charts saved in `output/figures/`:
1. `dca_outcomes_by_start_date_cagr.png`
2. `dca_outcomes_by_start_date_final_value.png`
3. `dca_outperformance_heatmap.png`
4. `outcome_distributions.png`
5. `growth_comparison.png`
6. `qqq_drawdown_analysis.png`
7. `tqqq_drawdown_analysis.png`
8. `variance_drag_scatter.png`

### Reports
- `output/results/summary_report.txt` - Executive summary
- `output/results/best_scenarios.csv` - Top 10 performing start dates
- `output/results/worst_scenarios.csv` - Bottom 10 performing start dates

## How to Run

```bash
# Install dependencies
venv/bin/pip3 install -r requirements.txt

# Run full analysis
venv/bin/python3 src/main.py

# View results
open output/figures/
open output/results/summary_report.txt
```

## Key Insights

1. **DCA Mitigates Timing Risk**: Despite variance drag, TQQQ DCA outperformed QQQ in 83.8% of 10-year periods tested.

2. **Higher Returns, Higher Volatility**: TQQQ median CAGR (21.74%) is 2.5x higher than QQQ (8.66%), but with 3.6x higher standard deviation (12.38% vs 3.44%).

3. **Variance Drag is Real**: Over the full 27-year period, the naïve 3x expectation would be 2,670x growth, but synthetic TQQQ achieved only 5.6x growth due to compounding volatility.

4. **Start Date Matters**: Best scenario (2003 start) achieved 39.72% CAGR. Worst scenario (1999 start) suffered -15.77% CAGR.

5. **Long-Term Viability**: Despite warnings about leveraged ETFs being "short-term trading tools," systematic DCA into TQQQ has historically outperformed unleveraged QQQ DCA in most 10-year periods.

## Caveats

- Past performance does not guarantee future results
- Analysis uses synthetic TQQQ for pre-2010 periods
- Does not account for taxes or transaction costs
- Assumes continuous ability to invest monthly (no behavioral factors)
- Nasdaq-100 has been in a strong bull market over the analyzed period

## Technical Implementation

**Framework**: Python 3.9+
**Libraries**: pandas, numpy, matplotlib, seaborn, yfinance, scipy
**Methodology**: 
- Daily-reset leverage model (Cheng-Madhavan framework)
- Variance drag calculation (Avellaneda-Zhang framework)
- Monte Carlo-style rolling start date analysis

---

*Analysis completed: February 18, 2026*
