# 2D visualization of games in PC space, colored by max_discount_ever
# (notebook 04's regression target). The old version colored by value_retention_tier,
# which was removed during the cleaning audit.

y_disc = df.loc[X_pca.index, 'max_discount_ever'] / 100   # 0-1 scale
y_price = df.loc[X_pca.index, 'initial_price']            # USD MSRP

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Plot 1: colored by max_discount_ever (continuous viridis colormap)
sc1 = axes[0].scatter(X_pca_transformed[:, 0], X_pca_transformed[:, 1],
                      c=y_disc, cmap='viridis', alpha=0.6, s=40,
                      edgecolors='black', linewidths=0.5,
                      vmin=0, vmax=1)
cbar = plt.colorbar(sc1, ax=axes[0])
cbar.set_label('max_discount_ever (0-1)', fontweight='bold', fontsize=10)
axes[0].set_xlabel(f'PC1 ({explained_variance_ratio[0]*100:.1f}% variance)',
                   fontweight='bold', fontsize=11)
axes[0].set_ylabel(f'PC2 ({explained_variance_ratio[1]*100:.1f}% variance)',
                   fontweight='bold', fontsize=11)
axes[0].set_title('Games in 2D PC space\n(colored by deepest discount ever reached)',
                  fontweight='bold', fontsize=12)
axes[0].grid(True, alpha=0.3)

# Plot 2: colored by initial_price (USD)
sc2 = axes[1].scatter(X_pca_transformed[:, 0], X_pca_transformed[:, 1],
                      c=y_price, cmap='plasma', alpha=0.6, s=40,
                      edgecolors='black', linewidths=0.5)
cbar = plt.colorbar(sc2, ax=axes[1])
cbar.set_label('Initial Price (USD)', fontweight='bold', fontsize=10)
axes[1].set_xlabel(f'PC1 ({explained_variance_ratio[0]*100:.1f}% variance)',
                   fontweight='bold', fontsize=11)
axes[1].set_ylabel(f'PC2 ({explained_variance_ratio[1]*100:.1f}% variance)',
                   fontweight='bold', fontsize=11)
axes[1].set_title('Games in 2D PC space\n(colored by Initial Price USD)',
                  fontweight='bold', fontsize=12)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
