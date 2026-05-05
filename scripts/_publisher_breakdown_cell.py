# Publisher influence breakdown -- the named-companies view of the publisher
# feature.
#
# Why this is a SECOND chart (not the same as the feature-importance bar):
#   Target encoding compresses 2,402 publishers into a single numeric column.
#   That column's importance answers "how much does *publisher* matter?" but
#   can't say which specific publishers move predictions most. To get the
#   named-publisher view, we read the fitted encoder directly: each publisher
#   maps to a smoothed mean discount, and that value is what the model uses.
#
# We show the deepest discounters (top encoded) and the strongest price-holders
# (lowest encoded), filtered to publishers with at least 10 games so the
# encoded value is stable.

# 1. Pull the fitted TargetEncoder out of the panel pipeline
pub_encoder = best_panel_model.named_steps['prep'].named_transformers_['pub']

# encodings_/categories_ are lists with one entry per input column; we have one
# publisher column so index 0.
encoded_values = pub_encoder.encodings_[0]      # array of smoothed means
publisher_names = pub_encoder.categories_[0]    # array of publisher strings

# Volume per publisher (from panel_clean) so we can filter sparse ones
pub_counts = panel_clean.groupby('publisher').agg(
    n_games=('appid', 'nunique'),
    n_obs=('appid', 'size'),
).reset_index()

pub_df = pd.DataFrame({
    'publisher':     publisher_names,
    'encoded_value': encoded_values,
}).merge(pub_counts, on='publisher', how='left')

MIN_GAMES = 10
pub_df = pub_df[pub_df['n_games'].fillna(0) >= MIN_GAMES].copy()
print(f'Publishers with >= {MIN_GAMES} games: {len(pub_df):,}  '
      f'(of {len(publisher_names):,} total)')

global_mean = pub_encoder.target_mean_
print(f'Global mean encoded value (max_discount): {global_mean:.1f}%')

# 2. Top 12 deepest discounters + top 12 price-holders
top_deep = pub_df.nlargest(12, 'encoded_value')
top_hold = pub_df.nsmallest(12, 'encoded_value')

fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

# Deepest discounters (red)
axes[0].barh(top_deep['publisher'][::-1], top_deep['encoded_value'][::-1],
             color=PALETTE['red'])
axes[0].axvline(global_mean, color=PALETTE['ink'], linestyle='--', linewidth=1,
                label=f'Global mean = {global_mean:.1f}%')
axes[0].set_xlabel('Encoded value (smoothed mean max-discount %)')
axes[0].set_title('Deepest Discounters\n(publishers the model expects to hit big discounts)')
axes[0].legend(loc='lower right')
for i, row in enumerate(top_deep[::-1].itertuples()):
    axes[0].text(row.encoded_value + 0.5, i,
                 f'  n={int(row.n_games)}', va='center', fontsize=8, color=PALETTE['ink'])
axes[0].set_xlim(0, 100)

# Price-holders (green)
axes[1].barh(top_hold['publisher'][::-1], top_hold['encoded_value'][::-1],
             color=PALETTE['green'])
axes[1].axvline(global_mean, color=PALETTE['ink'], linestyle='--', linewidth=1,
                label=f'Global mean = {global_mean:.1f}%')
axes[1].set_xlabel('Encoded value (smoothed mean max-discount %)')
axes[1].set_title('Strongest Price-Holders\n(publishers the model expects to discount least)')
axes[1].legend(loc='lower right')
for i, row in enumerate(top_hold[::-1].itertuples()):
    axes[1].text(row.encoded_value + 0.5, i,
                 f'  n={int(row.n_games)}', va='center', fontsize=8, color=PALETTE['ink'])
axes[1].set_xlim(0, 100)

plt.tight_layout()
plt.savefig(paths.outputs_dir / 'publisher_breakdown.png', dpi=150, bbox_inches='tight')
plt.show()

print('\nDeepest discounters (top 5):')
for _, r in top_deep.head(5).iterrows():
    print(f'  {r["publisher"]:<32s}  encoded={r["encoded_value"]:.1f}%  n_games={int(r["n_games"])}')
print('\nStrongest price-holders (top 5):')
for _, r in top_hold.head(5).iterrows():
    print(f'  {r["publisher"]:<32s}  encoded={r["encoded_value"]:.1f}%  n_games={int(r["n_games"])}')
