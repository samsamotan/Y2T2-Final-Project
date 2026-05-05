# Feature correlations with the regression target (max_discount_ever / 100).
# Replaces the old version which correlated against current_price (PHP) and
# panelled box plots by the now-removed value_retention_tier.

# Align the target with X_full's index and append it for correlation calc
y_target = df.loc[X_full.index, 'max_discount_ever'] / 100
X_with_target = X_full.copy()
X_with_target['max_discount_ever'] = y_target

# Calculate correlations
target_corr = X_with_target.corr()['max_discount_ever'].drop('max_discount_ever')
target_corr_sorted = target_corr.abs().sort_values(ascending=False)

# For visualization, show top 20 features
n_features_show = min(20, len(target_corr_sorted))
target_corr_top = target_corr_sorted.head(n_features_show)

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Left panel: top 20 features by |correlation| with the target
y_pos = np.arange(len(target_corr_top))
colors = ['red' if target_corr[feat] < 0 else 'green'
          for feat in target_corr_top.index]
bars = axes[0].barh(y_pos, target_corr[target_corr_top.index],
                    color=colors, edgecolor='black', linewidth=1.5, alpha=0.85)
axes[0].set_yticks(y_pos)
display_names = [name[:35] + "..." if len(name) > 35 else name
                 for name in target_corr_top.index]
axes[0].set_yticklabels(display_names, fontsize=9)
axes[0].set_xlabel('Correlation coefficient', fontweight='bold', fontsize=11)
axes[0].set_title(f'Top {n_features_show} features: correlation with max_discount_ever\n'
                  f'(notebook 04 regression target)',
                  fontweight='bold', fontsize=12)
axes[0].axvline(x=0, color='black', linewidth=2)
axes[0].grid(True, alpha=0.3, axis='x')
for i, feature in enumerate(target_corr_top.index):
    corr = target_corr[feature]
    x_pos = corr + (0.03 if corr > 0 else -0.03)
    ha = 'left' if corr > 0 else 'right'
    axes[0].text(x_pos, i, f'{corr:.3f}', va='center', ha=ha,
                 fontweight='bold', fontsize=8)

# Right panel: best NUMERIC predictor split by quartile of the target
numeric_corrs = target_corr[numeric_features]
key_feature = numeric_corrs.abs().idxmax()

# Build quartile bins on the target -- equal-mass classes
quartile = pd.qcut(y_target, q=4, duplicates='drop',
                   labels=['Q1: shallowest', 'Q2: shallow-mid',
                           'Q3: mid-deep',   'Q4: deepest'])
qcolors = [PALETTE[2], PALETTE[1], PALETTE[3], PALETTE[0]]

data_to_plot = [df.loc[X_full.index][quartile == q][key_feature].dropna()
                for q in quartile.cat.categories]

bp = axes[1].boxplot(data_to_plot, labels=quartile.cat.categories,
                     patch_artist=True, widths=0.6,
                     showmeans=True, meanline=True)
for patch, color in zip(bp['boxes'], qcolors):
    patch.set_facecolor(color)
    patch.set_edgecolor('black')
    patch.set_linewidth(1.5)

axes[1].set_ylabel(key_feature.replace('_', ' ').title(),
                   fontweight='bold', fontsize=11)
axes[1].set_xlabel('Discount-depth quartile', fontweight='bold', fontsize=11)
axes[1].set_title(f'{key_feature.replace("_", " ").title()} by quartile of max_discount_ever',
                  fontweight='bold', fontsize=12)
axes[1].grid(True, alpha=0.3, axis='y')
axes[1].tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.show()

# --- summary text ---
print("\n" + "="*80)
print("FEATURE CORRELATION WITH max_discount_ever  (regression target)")
print("="*80)

print(f"\nTop 10 features overall (by |correlation|):")
for i, feature in enumerate(target_corr_sorted.head(10).index, 1):
    abs_corr = target_corr_sorted[feature]
    corr     = target_corr[feature]
    direction = "Positive" if corr > 0 else "Negative"
    feature_display = feature[:45] + "..." if len(feature) > 45 else feature
    print(f"{i:2d}. {feature_display:<48} |r|={abs_corr:.3f}  (r={corr:+.3f}, {direction})")

print(f"\nTop 5 NUMERIC features (excluding one-hot encoded):")
numeric_corr_sorted = numeric_corrs.abs().sort_values(ascending=False)
for i, feature in enumerate(numeric_corr_sorted.head(5).index, 1):
    abs_corr = numeric_corr_sorted[feature]
    corr     = target_corr[feature]
    direction = "Positive" if corr > 0 else "Negative"
    print(f"{i}. {feature:<25} |r|={abs_corr:.3f}  (r={corr:+.3f}, {direction})")

print(f"\nINTERPRETATION:")
top_feature = target_corr_sorted.index[0]
top_corr    = target_corr[top_feature]
print(f"  Strongest predictor overall: {top_feature}  (r = {top_corr:+.3f})")
top_numeric = numeric_corr_sorted.index[0]
print(f"  Strongest NUMERIC predictor: {top_numeric}  (r = {target_corr[top_numeric]:+.3f})")
if abs(top_corr) > 0.5:
    print(f"\n  STRONG correlation -- this feature will dominate linear models.")
elif abs(top_corr) > 0.2:
    print(f"\n  MODERATE correlation -- useful predictor but not dominant.")
else:
    print(f"\n  WEAK correlation -- no single feature strongly predicts discount depth;")
    print(f"  expect tree ensembles to outperform linear models.")
