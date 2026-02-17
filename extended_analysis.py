"""
Extended Analysis: Test DCA from 1985 onwards using NDX Index
This tests if results are biased by only analyzing recent 30 years.

Uses NDX (Nasdaq-100 Index) as QQQ proxy back to 1985.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from models.synthetic_tqqq import SyntheticLeveragedETF
from models.dca_simulator import DCASimulator
import matplotlib.pyplot as plt
import seaborn as sns

print("="*80)
print("EXTENDED ANALYSIS: 40 YEARS OF DATA (1985-2026)")
print("="*80)
print("\nUsing NDX (Nasdaq-100 Index) as proxy for QQQ back to 1985")
print("This tests if our results are biased by only looking at recent decades")
print("="*80)

# Load NDX data (goes back to 1985)
ndx = pd.read_csv('data/processed/NDX_processed.csv', index_col=0, parse_dates=True)

print(f"\nNDX Data Range: {ndx.index[0].date()} to {ndx.index[-1].date()}")
print(f"Total days: {len(ndx):,}")
print(f"Total years: {(ndx.index[-1] - ndx.index[0]).days / 365.25:.1f}")

# Create synthetic 3x leveraged NDX
print("\nBuilding synthetic 3x leveraged NDX...")
synthetic_model = SyntheticLeveragedETF(leverage=3.0, expense_ratio=0.0095)
synthetic_3x = synthetic_model.simulate(ndx['Daily_Return'], include_fees=True)

# Prepare price data for DCA simulator
ndx_prices = ndx[['Close']].copy()
synthetic_prices = pd.DataFrame({'Close': synthetic_3x['NAV']}, index=synthetic_3x.index)

# Run DCA simulations with MONTHLY start dates (more granular)
print("\nRunning DCA simulations...")
print("  - Frequency: Monthly investment ($1,000/month)")
print("  - Holding period: 10 years")
print("  - Start dates: MONTHLY intervals (maximum coverage)")

# NDX DCA
print("\n  [1/2] Simulating NDX (1x) DCA...")
ndx_simulator = DCASimulator(ndx_prices)
ndx_dca_results = ndx_simulator.rolling_start_dates_analysis(
    frequency='M',
    investment_amount=1000,
    holding_period_years=10,
    start_date_frequency='M'  # Monthly start dates
)
print(f"  ✓ Completed {len(ndx_dca_results)} scenarios")

# Synthetic 3x NDX DCA
print("  [2/2] Simulating Synthetic 3x NDX DCA...")
synthetic_simulator = DCASimulator(synthetic_prices)
synthetic_dca_results = synthetic_simulator.rolling_start_dates_analysis(
    frequency='M',
    investment_amount=1000,
    holding_period_years=10,
    start_date_frequency='M'  # Monthly start dates
)
print(f"  ✓ Completed {len(synthetic_dca_results)} scenarios")

# Save results
ndx_dca_results.to_csv('data/processed/dca_extended_NDX.csv', index=False)
synthetic_dca_results.to_csv('data/processed/dca_extended_Synthetic3x.csv', index=False)

# Merge for comparison
comparison = ndx_dca_results.merge(
    synthetic_dca_results,
    on='start_date',
    suffixes=('_NDX', '_3x')
)

comparison['Outperformance_Ratio'] = comparison['final_value_3x'] / comparison['final_value_NDX']
comparison['CAGR_Diff'] = comparison['cagr_3x'] - comparison['cagr_NDX']
comparison['3x_Outperformed'] = comparison['final_value_3x'] > comparison['final_value_NDX']

comparison.to_csv('data/processed/extended_comparison.csv', index=False)

print("\n" + "="*80)
print("EXTENDED RESULTS SUMMARY")
print("="*80)
print(f"\nTotal scenarios tested: {len(comparison)}")
print(f"Date range: {comparison['start_date'].min().date()} to {comparison['start_date'].max().date()}")
print(f"Years covered: {(comparison['start_date'].max() - comparison['start_date'].min()).days / 365.25:.1f}")

# ============================================================================
# DISTRIBUTION ANALYSIS
# ============================================================================

def categorize_cagr(cagr):
    if cagr < 0:
        return "Terrible (Negative)"
    elif cagr < 5:
        return "Poor (0-5%)"
    elif cagr < 10:
        return "Fair (5-10%)"
    elif cagr < 15:
        return "Good (10-15%)"
    elif cagr < 25:
        return "Very Good (15-25%)"
    else:
        return "Excellent (25%+)"

category_order = [
    "Terrible (Negative)",
    "Poor (0-5%)",
    "Fair (5-10%)",
    "Good (10-15%)",
    "Very Good (15-25%)",
    "Excellent (25%+)"
]

comparison['NDX_Category'] = comparison['cagr_NDX'].apply(categorize_cagr)
comparison['3x_Category'] = comparison['cagr_3x'].apply(categorize_cagr)

print("\n" + "="*80)
print("3X LEVERAGED PERFORMANCE DISTRIBUTION (40 YEARS)")
print("="*80)

dist_3x = comparison['3x_Category'].value_counts().reindex(category_order, fill_value=0)
pct_3x = (dist_3x / len(comparison) * 100).round(1)

table_3x = pd.DataFrame({
    'Performance Level': dist_3x.index,
    'Count': dist_3x.values,
    'Percentage': [f"{p:.1f}%" for p in pct_3x.values],
    'Cumulative %': [f"{c:.1f}%" for c in pct_3x.cumsum().values]
})
print(table_3x.to_string(index=False))

print("\n" + "="*80)
print("KEY STATISTICS")
print("="*80)

print("\n📊 NDX (1x) DCA PERFORMANCE:")
print(f"  Median CAGR: {comparison['cagr_NDX'].median():.2f}%")
print(f"  Mean CAGR: {comparison['cagr_NDX'].mean():.2f}%")
print(f"  Best CAGR: {comparison['cagr_NDX'].max():.2f}%")
print(f"  Worst CAGR: {comparison['cagr_NDX'].min():.2f}%")
print(f"  Std Dev: {comparison['cagr_NDX'].std():.2f}%")

print("\n📊 SYNTHETIC 3x NDX DCA PERFORMANCE:")
print(f"  Median CAGR: {comparison['cagr_3x'].median():.2f}%")
print(f"  Mean CAGR: {comparison['cagr_3x'].mean():.2f}%")
print(f"  Best CAGR: {comparison['cagr_3x'].max():.2f}%")
print(f"  Worst CAGR: {comparison['cagr_3x'].min():.2f}%")
print(f"  Std Dev: {comparison['cagr_3x'].std():.2f}%")

loss_pct_3x = (comparison['cagr_3x'] < 0).sum() / len(comparison) * 100
win_rate_3x = comparison['3x_Outperformed'].sum() / len(comparison) * 100

print(f"\n⚠️  Probability of Loss (3x): {loss_pct_3x:.1f}%")
print(f"🎯 3x Win Rate vs 1x: {win_rate_3x:.1f}%")

# ============================================================================
# COMPARE RECENT vs HISTORICAL
# ============================================================================

print("\n" + "="*80)
print("BIAS TEST: RECENT (Last 27 Years) vs HISTORICAL (Full 40 Years)")
print("="*80)

# Split into periods
cutoff_date = pd.Timestamp('1999-01-01')
recent = comparison[comparison['start_date'] >= cutoff_date]
historical = comparison[comparison['start_date'] < cutoff_date]

print(f"\nRecent Period (1999+): {len(recent)} scenarios")
print(f"Historical Period (1985-1998): {len(historical)} scenarios")

print("\n📊 3x LEVERAGED PERFORMANCE BY ERA:")
print("-"*80)
print("RECENT PERIOD (1999-2016 starts):")
recent_dist = recent['3x_Category'].value_counts().reindex(category_order, fill_value=0)
recent_pct = (recent_dist / len(recent) * 100).round(1)
for cat, pct in zip(recent_dist.index, recent_pct.values):
    print(f"  {cat:25s}: {pct:5.1f}%")

print("\nHISTORICAL PERIOD (1985-1998 starts):")
hist_dist = historical['3x_Category'].value_counts().reindex(category_order, fill_value=0)
hist_pct = (hist_dist / len(historical) * 100).round(1)
for cat, pct in zip(hist_dist.index, hist_pct.values):
    print(f"  {cat:25s}: {pct:5.1f}%")

print("\nDIFFERENCE (Recent - Historical):")
for cat in category_order:
    diff = recent_pct.get(cat, 0) - hist_pct.get(cat, 0)
    print(f"  {cat:25s}: {diff:+5.1f} percentage points")

# Compare medians
print(f"\nMedian CAGR (3x):")
print(f"  Recent: {recent['cagr_3x'].median():.2f}%")
print(f"  Historical: {historical['cagr_3x'].median():.2f}%")
print(f"  Difference: {recent['cagr_3x'].median() - historical['cagr_3x'].median():+.2f} percentage points")

print(f"\nLoss Probability (3x):")
recent_loss = (recent['cagr_3x'] < 0).sum() / len(recent) * 100
hist_loss = (historical['cagr_3x'] < 0).sum() / len(historical) * 100
print(f"  Recent: {recent_loss:.1f}%")
print(f"  Historical: {hist_loss:.1f}%")
print(f"  Difference: {recent_loss - hist_loss:+.1f} percentage points")

# ============================================================================
# VISUALIZATION
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# 1. Full 40-year distribution
ax1 = axes[0, 0]
pct_3x.plot(kind='bar', ax=ax1, color='darkorange', alpha=0.8, edgecolor='black')
ax1.set_title('3x Leveraged DCA: Full 40-Year Distribution\n(1985-2026 start dates)', 
             fontsize=14, fontweight='bold')
ax1.set_xlabel('Performance Level', fontsize=11, fontweight='bold')
ax1.set_ylabel('Percentage of Scenarios (%)', fontsize=11, fontweight='bold')
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
ax1.grid(True, alpha=0.3, axis='y')
for i, v in enumerate(pct_3x.values):
    if v > 0:
        ax1.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=10, fontweight='bold')
ax1.text(0.98, 0.98, f'n={len(comparison)} scenarios', transform=ax1.transAxes,
        fontsize=10, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 2. Recent vs Historical comparison
ax2 = axes[0, 1]
x = np.arange(len(category_order))
width = 0.35
bars1 = ax2.bar(x - width/2, recent_pct.values, width, label='Recent (1999+)', 
               color='steelblue', alpha=0.8, edgecolor='black')
bars2 = ax2.bar(x + width/2, hist_pct.values, width, label='Historical (1985-1998)', 
               color='darkgreen', alpha=0.8, edgecolor='black')
ax2.set_title('3x Leveraged: Recent vs Historical Period\n(Testing for bias)', 
             fontsize=14, fontweight='bold')
ax2.set_xlabel('Performance Level', fontsize=11, fontweight='bold')
ax2.set_ylabel('Percentage of Scenarios (%)', fontsize=11, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(category_order, rotation=45, ha='right')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3, axis='y')

# 3. Time series of CAGR outcomes
ax3 = axes[1, 0]
ax3.scatter(comparison['start_date'], comparison['cagr_3x'], 
           c=comparison['cagr_3x'], cmap='RdYlGn', s=20, alpha=0.6, edgecolors='black', linewidth=0.5)
ax3.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Break-even')
ax3.axhline(y=comparison['cagr_3x'].median(), color='blue', linestyle='-', linewidth=2, 
           alpha=0.7, label=f'Median: {comparison["cagr_3x"].median():.1f}%')
ax3.axvline(x=cutoff_date, color='black', linestyle=':', linewidth=2, alpha=0.5, label='Recent Era Starts')
ax3.set_title('3x CAGR by Start Date (40 Years)\n(Color-coded: Green=Good, Red=Bad)', 
             fontsize=14, fontweight='bold')
ax3.set_xlabel('DCA Start Date', fontsize=11, fontweight='bold')
ax3.set_ylabel('CAGR (%)', fontsize=11, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

# 4. Summary statistics comparison
ax4 = axes[1, 1]
ax4.axis('tight')
ax4.axis('off')

summary_data = [
    ['Metric', 'Full 40Y', 'Recent', 'Historical'],
    ['', '', '', ''],
    ['Scenarios', f"{len(comparison)}", f"{len(recent)}", f"{len(historical)}"],
    ['', '', '', ''],
    ['Median CAGR', f"{comparison['cagr_3x'].median():.1f}%", 
     f"{recent['cagr_3x'].median():.1f}%", f"{historical['cagr_3x'].median():.1f}%"],
    ['Mean CAGR', f"{comparison['cagr_3x'].mean():.1f}%", 
     f"{recent['cagr_3x'].mean():.1f}%", f"{historical['cagr_3x'].mean():.1f}%"],
    ['', '', '', ''],
    ['Best CAGR', f"{comparison['cagr_3x'].max():.1f}%", 
     f"{recent['cagr_3x'].max():.1f}%", f"{historical['cagr_3x'].max():.1f}%"],
    ['Worst CAGR', f"{comparison['cagr_3x'].min():.1f}%", 
     f"{recent['cagr_3x'].min():.1f}%", f"{historical['cagr_3x'].min():.1f}%"],
    ['', '', '', ''],
    ['% Negative', f"{loss_pct_3x:.1f}%", f"{recent_loss:.1f}%", f"{hist_loss:.1f}%"],
    ['% Excellent (25%+)', f"{pct_3x.get('Excellent (25%+)', 0):.1f}%",
     f"{recent_pct.get('Excellent (25%+)', 0):.1f}%", f"{hist_pct.get('Excellent (25%+)', 0):.1f}%"],
    ['', '', '', ''],
    ['Std Dev', f"{comparison['cagr_3x'].std():.1f}%", 
     f"{recent['cagr_3x'].std():.1f}%", f"{historical['cagr_3x'].std():.1f}%"],
]

table = ax4.table(cellText=summary_data, cellLoc='left', loc='center',
                 colWidths=[0.35, 0.22, 0.22, 0.22])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.2)

# Style header
for i in range(4):
    table[(0, i)].set_facecolor('#40466e')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Style data rows
for i in range(1, len(summary_data)):
    if summary_data[i][0] == '':
        for j in range(4):
            table[(i, j)].set_facecolor('#e0e0e0')
    else:
        for j in range(4):
            table[(i, j)].set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')

ax4.set_title('Bias Check: Are Results Consistent?', fontsize=14, fontweight='bold', pad=20)

plt.suptitle('Extended Analysis: 40 Years of 3x Leveraged DCA\n(Testing if recent decades bias the results)',
            fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('output/figures/extended_40year_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved visualization: output/figures/extended_40year_analysis.png")

print("\n" + "="*80)
print("CONCLUSION: IS THERE BIAS?")
print("="*80)

diff_median = abs(recent['cagr_3x'].median() - historical['cagr_3x'].median())
diff_loss = abs(recent_loss - hist_loss)

if diff_median < 5 and diff_loss < 10:
    print("\n✅ NO SIGNIFICANT BIAS DETECTED")
    print(f"   - Median CAGR difference: {diff_median:.1f} percentage points (< 5% threshold)")
    print(f"   - Loss probability difference: {diff_loss:.1f} percentage points (< 10% threshold)")
    print("   - Distribution patterns are similar across eras")
    print("   - Results appear robust across different market conditions")
else:
    print("\n⚠️ POTENTIAL BIAS DETECTED")
    print(f"   - Median CAGR difference: {diff_median:.1f} percentage points")
    print(f"   - Loss probability difference: {diff_loss:.1f} percentage points")
    print("   - Distribution may be influenced by specific market conditions")

print("\n" + "="*80)
