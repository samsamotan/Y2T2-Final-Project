# === CLASSIFICATION-FRIENDLY VIEW: quartile bins on max_discount_ever ===
# Note: notebook 04's old `value_retention_tier` target was removed in the
# cleaning audit (the boundaries had no research justification, and the
# 'Premium Hold' tier conflicted with the modeling filter). For an EDA bar
# chart we derive 4 quartile-based tiers locally so each tier carries equal
# mass -- this is what we'd do if we wanted to bucket the regression target
# into classes for narrative purposes.

target = (df['max_discount_ever'] / 100).dropna()

# Quartile bins -- equal-mass classes by definition
qbins, qedges = pd.qcut(target, q=4, retbins=True, duplicates='drop',
                        labels=['Q1: shallowest', 'Q2: shallow-mid',
                                'Q3: mid-deep',   'Q4: deepest'])
counts = qbins.value_counts().reindex(qbins.cat.categories)
edge_labels = [f'[{qedges[i]:.2f}, {qedges[i+1]:.2f}{")" if i < len(qedges)-2 else "]"}'
               for i in range(len(qedges) - 1)]

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

quartile_colors = [PALETTE[2], PALETTE[1], PALETTE[3], PALETTE[0]]   # cool to hot
y_pos = np.arange(len(counts))
bars = axes[0].barh(y_pos, counts.values, color=quartile_colors,
                    edgecolor='black', linewidth=1.5, alpha=0.85)
axes[0].set_yticks(y_pos)
axes[0].set_yticklabels([f'{lbl}\n{rng}' for lbl, rng in zip(counts.index, edge_labels)])
axes[0].set_xlabel('Number of games', fontweight='bold', fontsize=11)
axes[0].set_title('max_discount_ever quartile bins\n(equal-mass classes for narrative)',
                  fontweight='bold', fontsize=12)
axes[0].grid(True, alpha=0.3, axis='x')
for i, (bar, count) in enumerate(zip(bars, counts.values)):
    pct = (count / counts.sum() * 100)
    axes[0].text(count + counts.max() * 0.01, i, f'{count:,} ({pct:.1f}%)',
                 va='center', fontweight='bold', fontsize=10)

# Pie chart
explode = [0.05] * len(counts)
axes[1].pie(counts.values, labels=counts.index, autopct='%1.1f%%',
            startangle=90, colors=quartile_colors, explode=explode,
            wedgeprops={'edgecolor': 'black', 'linewidth': 1.5},
            textprops={'fontsize': 10, 'fontweight': 'bold'})
axes[1].set_title('Quartile bin proportion', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.show()

print("\n" + "="*80)
print("CLASSIFICATION VIEW: quartile bins on max_discount_ever / 100")
print("="*80)
print(f"\nBy construction each tier holds ~25% of the dataset.")
print(f"Tier boundaries (data-derived, not hand-picked):")
for tier, rng in zip(counts.index, edge_labels):
    pct = counts[tier] / counts.sum() * 100
    print(f"  {tier:<22}  range {rng:<14}  n = {counts[tier]:>4,}  ({pct:.1f}%)")

print(f"\nNote: notebook 04 actually models on the continuous 0-1 target (Step 4.4)")
print(f"and on a 20-bin ordinal target (Step 4.5). These quartile bins are an EDA")
print(f"convenience -- they exist only in this cell, not in the cleaned tables.")
