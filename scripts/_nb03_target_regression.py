# === REGRESSION TARGET: max_discount_ever (matches notebook 04's regression target) ===
# We model the deepest discount each game has ever reached, on a 0-1 scale.
# This is the actual y in notebook 04's Step 4.4 regression.

target = (df['max_discount_ever'] / 100).dropna()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Original distribution
axes[0, 0].hist(target, bins=30, edgecolor='black', alpha=0.75, color=PALETTE[0])
axes[0, 0].axvline(target.median(), color='red', linestyle='--', linewidth=2.5,
                   label=f'Median: {target.median():.2f}')
axes[0, 0].axvline(target.mean(), color='orange', linestyle='--', linewidth=2.5,
                   label=f'Mean: {target.mean():.2f}')
axes[0, 0].set_xlabel('max_discount_ever / 100  (0 = never discounted, 1 = 100% off)',
                       fontweight='bold', fontsize=11)
axes[0, 0].set_ylabel('Number of games', fontweight='bold', fontsize=11)
axes[0, 0].set_title('Regression target distribution (raw)', fontweight='bold', fontsize=12)
axes[0, 0].legend(fontsize=9)
axes[0, 0].grid(True, alpha=0.3)

# CDF of the target -- shows the cumulative share of games that hit each depth
sorted_target = np.sort(target.values)
cdf_y = np.arange(1, len(sorted_target) + 1) / len(sorted_target)
axes[0, 1].plot(sorted_target, cdf_y, linewidth=2.5, color=PALETTE[1])
axes[0, 1].axhline(0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.6, label='Median')
axes[0, 1].axvline(target.median(), color='red', linestyle='--', linewidth=1.5, alpha=0.6)
axes[0, 1].set_xlabel('max_discount_ever / 100', fontweight='bold', fontsize=11)
axes[0, 1].set_ylabel('Cumulative fraction of games', fontweight='bold', fontsize=11)
axes[0, 1].set_title('CDF -- "What fraction of games discount at least X%?"',
                     fontweight='bold', fontsize=12)
axes[0, 1].legend(fontsize=9)
axes[0, 1].grid(True, alpha=0.3)

# Box plot -- outliers + spread
bp = axes[1, 0].boxplot(target, vert=True, patch_artist=True, widths=0.5,
                        boxprops=dict(facecolor=PALETTE[2], edgecolor='black', linewidth=1.5),
                        medianprops=dict(color='red', linewidth=2.5),
                        whiskerprops=dict(color='black', linewidth=1.5),
                        capprops=dict(color='black', linewidth=1.5))
axes[1, 0].set_ylabel('max_discount_ever / 100', fontweight='bold', fontsize=11)
axes[1, 0].set_title('Target -- box plot', fontweight='bold', fontsize=12)
axes[1, 0].set_xticklabels(['max_discount_ever'])
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Q-Q plot for normality assessment
stats.probplot(target, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q plot vs Normal', fontweight='bold', fontsize=12)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("="*80)
print("REGRESSION TARGET: max_discount_ever / 100  (notebook 04's regression target)")
print("="*80)
print(f"\nDescriptive Statistics:")
print(f"  Count:      {len(target):,}")
print(f"  Mean:       {target.mean():.3f}")
print(f"  Median:     {target.median():.3f}")
print(f"  Std Dev:    {target.std():.3f}")
print(f"  Min:        {target.min():.3f}")
print(f"  Max:        {target.max():.3f}")
print(f"  Q1 (25%):   {target.quantile(0.25):.3f}")
print(f"  Q3 (75%):   {target.quantile(0.75):.3f}")
print(f"\nDistribution shape:")
print(f"  Skewness:   {target.skew():.3f}")
print(f"  Kurtosis:   {target.kurtosis():.3f}")

# Concentration -- where does most of the mass sit?
hit_75 = (target >= 0.75).mean() * 100
hit_50 = (target >= 0.50).mean() * 100
hit_25 = (target >= 0.25).mean() * 100
print(f"\nThreshold concentration (share of games):")
print(f"  At least 25% off ever:  {hit_25:>5.1f}%")
print(f"  At least 50% off ever:  {hit_50:>5.1f}%")
print(f"  At least 75% off ever:  {hit_75:>5.1f}%")

print(f"\nINTERPRETATION:")
if target.skew() < -1:
    print(f"  Distribution is HEAVILY LEFT-SKEWED (skew = {target.skew():.2f}).")
    print(f"  Most games eventually reach a deep discount (Steam-sale floor at 75%);")
    print(f"  the rare cases are games that resist discounting.")
elif target.skew() < -0.3:
    print(f"  Distribution is moderately left-skewed (skew = {target.skew():.2f}).")
else:
    print(f"  Distribution is approximately symmetric (skew = {target.skew():.2f}).")
print(f"  RECOMMENDATION: model directly on the 0-1 scale; tree-based models handle")
print(f"  the bimodal/clustered nature without needing a transform.")
