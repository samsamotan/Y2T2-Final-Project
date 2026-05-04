"""Rewrite cell `dddfb73b` in notebook 04 to:
  1. Drop the orange target-threshold horizontal line.
  2. Connect the actual-observed yearly maxes with a line (markers='X')
     instead of a bare scatter.
  3. Drop the green "Buy at year X" vertical line + buy-zone shading
     (the recommended year is still surfaced in the chart subtitle/message
     and the returned dict's 'recommended_age', just not drawn on the axes).
  4. Update the docstring to match.

We rewrite the entire cell source rather than doing surgical line edits,
which is what bit us last time. The new source is `compile()`-checked
before the notebook is written, so a syntax error here won't reach disk.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_PATH = ROOT / "notebooks" / "04_ml_part1_price_and_value_retention.ipynb"
CELL_ID = "dddfb73b"

NEW_SOURCE = '''def predict_buy_time(query, target_pct=50, max_wait=10,
                     df=None, model=None, panel_data=None):
    """Predict the optimal buy year for a specific game, with both the
    model's predicted trajectory and the game's actual observed yearly
    max-discounts (from the panel data) for comparison.

    Parameters
    ----------
    query : int or str
        appid (int) or title substring (case-insensitive) - the game to look up.
    target_pct : float
        The discount % the buyer wants to wait for (default 50%).
    max_wait : int
        Max years to consider waiting (default 10).
    df, model, panel_data
        Default to df_clean / best_panel_model / panel from the surrounding
        notebook scope. Override if you want to use a different source.
    """
    df = df_clean if df is None else df
    model = best_panel_model if model is None else model
    panel_data = panel if panel_data is None else panel_data

    # Find the game
    if isinstance(query, (int, np.integer)):
        matches = df[df['appid'] == int(query)]
        query_str = f'appid {query}'
    else:
        matches = df[df['title'].str.contains(query, case=False, na=False, regex=False)]
        query_str = f'title containing {query!r}'
    if len(matches) == 0:
        raise ValueError(f'No game in df_clean matching {query_str}. '
                         f'(Game may exist but be filtered out by max_discount_ever > 0.)')
    game = matches.iloc[0]

    # Build static feature row (everything except age_year)
    feature_cols = panel_num_features + panel_cat_features + panel_pub_feature
    static = {col: game[col] for col in feature_cols if col != 'age_year'}

    # Predict across ages
    ages = np.arange(0, max_wait + 1)
    grid = pd.DataFrame([{**static, 'age_year': age} for age in ages])
    predicted = model.predict(grid[feature_cols])

    # Pull the game's actual yearly max-discounts from the panel.
    # NB: panel was built with cut clipped to year-of-life, so this is the
    # ground truth the model was trained against (for this game, if it was
    # in the training set).
    actual = (
        panel_data[panel_data['appid'] == int(game['appid'])]
        [['age_year', 'max_discount']]
        .sort_values('age_year')
    )

    # First age that crosses the target
    crossings = np.where(predicted >= target_pct)[0]
    if len(crossings) > 0:
        rec_age = int(ages[crossings[0]])
        rec_disc = float(predicted[crossings[0]])
        msg = (f'{game["title"]}: buy at year {rec_age} '
               f'(predicted {rec_disc:.0f}% off, target was {target_pct}%)')
    else:
        rec_age = None
        rec_disc = float(predicted[-1])
        msg = (f'{game["title"]}: never reaches {target_pct}% within {max_wait} years. '
               f'Best predicted discount: {rec_disc:.0f}% at year {max_wait}.')

    return {
        'game':                game,
        'ages':                ages,
        'predicted_curve':     predicted,
        'actual_yearly_max':   actual,
        'recommended_age':     rec_age,
        'expected_discount':   rec_disc,
        'target_pct':          target_pct,
        'message':             msg,
    }


def plot_buy_recommendation(result, ax=None):
    """Visualize a single buy-time recommendation:
      cyan line     = model's predicted yearly peak discount
      dark X-line   = this game's actual observed yearly max (from panel)
    The recommended year is reported in the chart subtitle (and on the
    returned dict's 'recommended_age'), not drawn as a vertical guide.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    ages   = result['ages']
    pred   = result['predicted_curve']
    actual = result['actual_yearly_max']

    # Predicted trajectory
    ax.plot(ages, pred, marker='o', linewidth=2, color=PALETTE['cyan'],
            label='Predicted peak discount')

    # Actual observed yearly max-discounts (this game's real history) -
    # connected line with X markers, parallel to the predicted curve.
    if len(actual) > 0:
        ax.plot(actual['age_year'], actual['max_discount'],
                marker='X', linewidth=2, markersize=10,
                color=PALETTE['ink'], markeredgecolor='white', markeredgewidth=1.5,
                label=f'Actual observed (n={len(actual)})', zorder=5)

    title = result['game']['title']
    ax.set_title(f'{title}\\n{result["message"]}', fontsize=10)
    ax.set_xlabel('Game age (years)')
    ax.set_ylabel('Peak discount %')
    ax.set_ylim(0, 100)
    ax.set_xlim(0, max(ages))
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(alpha=0.3)
    return ax


print('predict_buy_time + plot_buy_recommendation defined (actual=line, no target line, no buy-year guide).')
'''

# Compile-check the new source so we can't write a syntax error to disk.
compile(NEW_SOURCE, "<dddfb73b>", "exec")

nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
target = next((c for c in nb["cells"] if c.get("id") == CELL_ID), None)
if target is None:
    raise SystemExit(f"Cell {CELL_ID} not found in {NB_PATH.name}")

# nbformat v4.5 stores source as a list of lines (each ending in \n except last).
lines = NEW_SOURCE.splitlines(keepends=True)
target["source"] = lines
# Clear any stale outputs / execution_count for this code cell.
if target.get("cell_type") == "code":
    target["outputs"] = []
    target["execution_count"] = None

NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Rewrote cell {CELL_ID} ({len(lines)} lines, {len(NEW_SOURCE)} chars). Syntax OK.")
