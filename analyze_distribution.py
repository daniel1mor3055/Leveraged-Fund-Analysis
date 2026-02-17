"""
Analyze DCA outcome distribution by performance buckets.

Creates clear tables showing what % of start dates resulted in:
- Really bad outcomes
- Poor outcomes  
- Fair outcomes
- Good outcomes
- Excellent outcomes

NOTE: This script uses XIRR (money-weighted return) which correctly accounts
for the timing of each DCA investment. The 'cagr' column now contains XIRR
values by default from the updated DCA simulator.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load comparison data
comparison = pd.read_csv('data/processed/strategy_comparison.csv', 
                        parse_dates=['start_date', 'end_date_QQQ', 'end_date_TQQQ'])

print("="*80)
print("DCA OUTCOME DISTRIBUTION ANALYSIS")
print("="*80)
print(f"\nTotal scenarios analyzed: {len(comparison)}")
print(f"Investment: $1,000/month for 10 years")
print(f"Total invested per scenario: ${comparison['total_invested_QQQ'].iloc[0]:,.0f}")
print("")
print("NOTE: Returns shown are XIRR (money-weighted), which correctly")
print("accounts for the timing of each DCA investment.")
print("="*80)

# ============================================================================
# PART 1: ANNUALIZED RETURN (XIRR) DISTRIBUTION
# ============================================================================

def categorize_return(annual_return):
    """Categorize annualized return (XIRR) into performance buckets."""
    if annual_return < 0:
        return "Terrible (Negative)"
    elif annual_return < 5:
        return "Poor (0-5%)"
    elif annual_return < 10:
        return "Fair (5-10%)"
    elif annual_return < 15:
        return "Good (10-15%)"
    elif annual_return < 25:
        return "Very Good (15-25%)"
    else:
        return "Excellent (25%+)"

# Use xirr columns if available, otherwise fall back to cagr (which now = xirr)
qqq_return_col = 'xirr_QQQ' if 'xirr_QQQ' in comparison.columns else 'cagr_QQQ'
tqqq_return_col = 'xirr_TQQQ' if 'xirr_TQQQ' in comparison.columns else 'cagr_TQQQ'

# Categorize outcomes
comparison['QQQ_Category'] = comparison[qqq_return_col].apply(categorize_return)
comparison['TQQQ_Category'] = comparison[tqqq_return_col].apply(categorize_return)

# Define category order
category_order = [
    "Terrible (Negative)",
    "Poor (0-5%)",
    "Fair (5-10%)",
    "Good (10-15%)",
    "Very Good (15-25%)",
    "Excellent (25%+)"
]

print("\n" + "="*80)
print("ANNUALIZED RETURN (XIRR) DISTRIBUTION")
print("="*80)

# QQQ Distribution
print("\n📊 QQQ DCA OUTCOMES:")
print("-"*80)
qqq_dist = comparison['QQQ_Category'].value_counts().reindex(category_order, fill_value=0)
qqq_pct = (qqq_dist / len(comparison) * 100).round(1)

qqq_table = pd.DataFrame({
    'Performance Level': qqq_dist.index,
    'Count': qqq_dist.values,
    'Percentage': [f"{p:.1f}%" for p in qqq_pct.values],
    'Cumulative %': [f"{c:.1f}%" for c in qqq_pct.cumsum().values]
})
print(qqq_table.to_string(index=False))

# TQQQ Distribution
print("\n📊 TQQQ (Synthetic 3x) DCA OUTCOMES:")
print("-"*80)
tqqq_dist = comparison['TQQQ_Category'].value_counts().reindex(category_order, fill_value=0)
tqqq_pct = (tqqq_dist / len(comparison) * 100).round(1)

tqqq_table = pd.DataFrame({
    'Performance Level': tqqq_dist.index,
    'Count': tqqq_dist.values,
    'Percentage': [f"{p:.1f}%" for p in tqqq_pct.values],
    'Cumulative %': [f"{c:.1f}%" for c in tqqq_pct.cumsum().values]
})
print(tqqq_table.to_string(index=False))

# ============================================================================
# PART 2: FINAL VALUE DISTRIBUTION
# ============================================================================

print("\n" + "="*80)
print("FINAL PORTFOLIO VALUE DISTRIBUTION")
print("="*80)

def categorize_final_value(final_value, invested):
    """Categorize final value vs amount invested."""
    ratio = final_value / invested
    if ratio < 0.8:
        return "Terrible (<80% of invested)"
    elif ratio < 1.0:
        return "Poor (80-100% of invested)"
    elif ratio < 1.5:
        return "Fair (1.0-1.5x invested)"
    elif ratio < 2.0:
        return "Good (1.5-2.0x invested)"
    elif ratio < 3.0:
        return "Very Good (2.0-3.0x invested)"
    else:
        return "Excellent (3.0x+ invested)"

comparison['QQQ_Value_Category'] = comparison.apply(
    lambda row: categorize_final_value(row['final_value_QQQ'], row['total_invested_QQQ']), 
    axis=1
)
comparison['TQQQ_Value_Category'] = comparison.apply(
    lambda row: categorize_final_value(row['final_value_TQQQ'], row['total_invested_TQQQ']), 
    axis=1
)

value_category_order = [
    "Terrible (<80% of invested)",
    "Poor (80-100% of invested)",
    "Fair (1.0-1.5x invested)",
    "Good (1.5-2.0x invested)",
    "Very Good (2.0-3.0x invested)",
    "Excellent (3.0x+ invested)"
]

# QQQ Value Distribution
print("\n💰 QQQ FINAL VALUE OUTCOMES:")
print("-"*80)
qqq_val_dist = comparison['QQQ_Value_Category'].value_counts().reindex(value_category_order, fill_value=0)
qqq_val_pct = (qqq_val_dist / len(comparison) * 100).round(1)

qqq_val_table = pd.DataFrame({
    'Final Value Level': qqq_val_dist.index,
    'Count': qqq_val_dist.values,
    'Percentage': [f"{p:.1f}%" for p in qqq_val_pct.values],
    'Cumulative %': [f"{c:.1f}%" for c in qqq_val_pct.cumsum().values]
})
print(qqq_val_table.to_string(index=False))

# TQQQ Value Distribution
print("\n💰 TQQQ FINAL VALUE OUTCOMES:")
print("-"*80)
tqqq_val_dist = comparison['TQQQ_Value_Category'].value_counts().reindex(value_category_order, fill_value=0)
tqqq_val_pct = (tqqq_val_dist / len(comparison) * 100).round(1)

tqqq_val_table = pd.DataFrame({
    'Final Value Level': tqqq_val_dist.index,
    'Count': tqqq_val_dist.values,
    'Percentage': [f"{p:.1f}%" for p in tqqq_val_pct.values],
    'Cumulative %': [f"{c:.1f}%" for c in tqqq_val_pct.cumsum().values]
})
print(tqqq_val_table.to_string(index=False))

# ============================================================================
# PART 3: RISK METRICS
# ============================================================================

print("\n" + "="*80)
print("RISK ANALYSIS")
print("="*80)

# Calculate percentiles
qqq_percentiles = comparison[qqq_return_col].quantile([0.1, 0.25, 0.5, 0.75, 0.9])
tqqq_percentiles = comparison[tqqq_return_col].quantile([0.1, 0.25, 0.5, 0.75, 0.9])

print("\n📉 XIRR PERCENTILES (Money-Weighted Annualized Return):")
print("-"*80)
percentile_table = pd.DataFrame({
    'Percentile': ['10th (Bottom 10%)', '25th (Bottom 25%)', '50th (Median)', 
                   '75th (Top 25%)', '90th (Top 10%)'],
    'QQQ XIRR': [f"{p:.2f}%" for p in qqq_percentiles.values],
    'TQQQ XIRR': [f"{p:.2f}%" for p in tqqq_percentiles.values]
})
print(percentile_table.to_string(index=False))

print("\n⚠️  PROBABILITY OF LOSS (Negative Return):")
print("-"*80)
qqq_loss_pct = (comparison[qqq_return_col] < 0).sum() / len(comparison) * 100
tqqq_loss_pct = (comparison[tqqq_return_col] < 0).sum() / len(comparison) * 100
print(f"QQQ:  {qqq_loss_pct:.1f}% ({(comparison[qqq_return_col] < 0).sum()} out of {len(comparison)} scenarios)")
print(f"TQQQ: {tqqq_loss_pct:.1f}% ({(comparison[tqqq_return_col] < 0).sum()} out of {len(comparison)} scenarios)")

print("\n🎯 PROBABILITY OF BEATING QQQ:")
print("-"*80)
tqqq_win_rate = (comparison['TQQQ_outperformed']).sum() / len(comparison) * 100
print(f"TQQQ beat QQQ: {tqqq_win_rate:.1f}% ({comparison['TQQQ_outperformed'].sum()} out of {len(comparison)} scenarios)")

# ============================================================================
# PART 4: SPECIFIC SCENARIOS
# ============================================================================

print("\n" + "="*80)
print("WORST AND BEST SCENARIOS")
print("="*80)

print("\n❌ WORST 10 TQQQ SCENARIOS:")
print("-"*80)
worst = comparison.nsmallest(10, tqqq_return_col)[
    ['start_date', qqq_return_col, tqqq_return_col, 'final_value_TQQQ']
].copy()
worst['start_date'] = worst['start_date'].dt.strftime('%Y-%m-%d')
worst['final_value_TQQQ'] = worst['final_value_TQQQ'].apply(lambda x: f"${x:,.0f}")
worst[qqq_return_col] = worst[qqq_return_col].apply(lambda x: f"{x:.2f}%")
worst[tqqq_return_col] = worst[tqqq_return_col].apply(lambda x: f"{x:.2f}%")
worst.columns = ['Start Date', 'QQQ XIRR', 'TQQQ XIRR', 'TQQQ Final Value']
print(worst.to_string(index=False))

print("\n✅ BEST 10 TQQQ SCENARIOS:")
print("-"*80)
best = comparison.nlargest(10, tqqq_return_col)[
    ['start_date', qqq_return_col, tqqq_return_col, 'final_value_TQQQ']
].copy()
best['start_date'] = best['start_date'].dt.strftime('%Y-%m-%d')
best['final_value_TQQQ'] = best['final_value_TQQQ'].apply(lambda x: f"${x:,.0f}")
best[qqq_return_col] = best[qqq_return_col].apply(lambda x: f"{x:.2f}%")
best[tqqq_return_col] = best[tqqq_return_col].apply(lambda x: f"{x:.2f}%")
best.columns = ['Start Date', 'QQQ XIRR', 'TQQQ XIRR', 'TQQQ Final Value']
print(best.to_string(index=False))

# ============================================================================
# PART 5: VISUALIZATION
# ============================================================================

# Create distribution chart
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. XIRR Distribution - QQQ
ax1 = axes[0, 0]
qqq_pct.plot(kind='bar', ax=ax1, color='steelblue', alpha=0.8, edgecolor='black')
ax1.set_title('QQQ DCA: XIRR Distribution', fontsize=14, fontweight='bold')
ax1.set_xlabel('Performance Level', fontsize=11, fontweight='bold')
ax1.set_ylabel('Percentage of Scenarios (%)', fontsize=11, fontweight='bold')
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
ax1.grid(True, alpha=0.3, axis='y')
for i, v in enumerate(qqq_pct.values):
    if v > 0:
        ax1.text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=10, fontweight='bold')

# 2. XIRR Distribution - TQQQ
ax2 = axes[0, 1]
tqqq_pct.plot(kind='bar', ax=ax2, color='darkorange', alpha=0.8, edgecolor='black')
ax2.set_title('TQQQ DCA: XIRR Distribution', fontsize=14, fontweight='bold')
ax2.set_xlabel('Performance Level', fontsize=11, fontweight='bold')
ax2.set_ylabel('Percentage of Scenarios (%)', fontsize=11, fontweight='bold')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
ax2.grid(True, alpha=0.3, axis='y')
for i, v in enumerate(tqqq_pct.values):
    if v > 0:
        ax2.text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=10, fontweight='bold')

# 3. Side-by-side comparison
ax3 = axes[1, 0]
x = np.arange(len(category_order))
width = 0.35
bars1 = ax3.bar(x - width/2, qqq_pct.values, width, label='QQQ', 
               color='steelblue', alpha=0.8, edgecolor='black')
bars2 = ax3.bar(x + width/2, tqqq_pct.values, width, label='TQQQ', 
               color='darkorange', alpha=0.8, edgecolor='black')
ax3.set_title('QQQ vs TQQQ: XIRR Distribution Comparison', fontsize=14, fontweight='bold')
ax3.set_xlabel('Performance Level', fontsize=11, fontweight='bold')
ax3.set_ylabel('Percentage of Scenarios (%)', fontsize=11, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(category_order, rotation=45, ha='right')
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3, axis='y')

# 4. Summary statistics table
ax4 = axes[1, 1]
ax4.axis('tight')
ax4.axis('off')

summary_data = [
    ['Metric', 'QQQ', 'TQQQ'],
    ['', '', ''],
    ['Median XIRR', f"{comparison[qqq_return_col].median():.2f}%", f"{comparison[tqqq_return_col].median():.2f}%"],
    ['Mean XIRR', f"{comparison[qqq_return_col].mean():.2f}%", f"{comparison[tqqq_return_col].mean():.2f}%"],
    ['Best XIRR', f"{comparison[qqq_return_col].max():.2f}%", f"{comparison[tqqq_return_col].max():.2f}%"],
    ['Worst XIRR', f"{comparison[qqq_return_col].min():.2f}%", f"{comparison[tqqq_return_col].min():.2f}%"],
    ['Std Dev', f"{comparison[qqq_return_col].std():.2f}%", f"{comparison[tqqq_return_col].std():.2f}%"],
    ['', '', ''],
    ['% Negative Returns', f"{qqq_loss_pct:.1f}%", f"{tqqq_loss_pct:.1f}%"],
    ['% Returns > 10%', f"{(comparison[qqq_return_col] > 10).sum() / len(comparison) * 100:.1f}%", 
     f"{(comparison[tqqq_return_col] > 10).sum() / len(comparison) * 100:.1f}%"],
    ['% Returns > 20%', f"{(comparison[qqq_return_col] > 20).sum() / len(comparison) * 100:.1f}%",
     f"{(comparison[tqqq_return_col] > 20).sum() / len(comparison) * 100:.1f}%"],
    ['', '', ''],
    ['TQQQ Win Rate', '-', f"{tqqq_win_rate:.1f}%"],
]

table = ax4.table(cellText=summary_data, cellLoc='left', loc='center',
                 colWidths=[0.4, 0.3, 0.3])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# Style header row
for i in range(3):
    table[(0, i)].set_facecolor('#40466e')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Style data rows
for i in range(1, len(summary_data)):
    if summary_data[i][0] == '':  # Separator rows
        for j in range(3):
            table[(i, j)].set_facecolor('#e0e0e0')
    else:
        for j in range(3):
            table[(i, j)].set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')

ax4.set_title('Summary Statistics', fontsize=14, fontweight='bold', pad=20)

plt.suptitle('DCA Outcome Distribution Analysis\n$1,000/month for 10 years, 68 different start dates',
            fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('output/figures/outcome_distribution_detailed.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved visualization: output/figures/outcome_distribution_detailed.png")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
