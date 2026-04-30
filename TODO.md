# SteamSale — TODO

Predicting Discount Depth and Value Retention of PC Games on Steam.

---

## Phase 0 — Setup

- [ ] Create project repo structure (`data/`, `notebooks/`, `src/`, `models/`, `reports/`)
- [ ] Set up Python environment (`requirements.txt` / `pyproject.toml`)
  - `requests`, `pandas`, `numpy`, `scikit-learn`, `xgboost`, `lightgbm`, `matplotlib`, `seaborn`, `jupyter`, `python-dotenv`, `tqdm`
- [ ] Register for IsThereAnyDeal API key (instant, free)
- [ ] Create `.env` file for API keys; add to `.gitignore`
- [ ] Write API rate-limit / retry / caching utility (Steam API throttles ~200 req / 5 min)

---

## Phase 1 — Data Collection (Part 1: Pricing & Value Retention)

### Steam Storefront API
- [ ] Pull list of ~5,000 appids (sample across genres, release years, popularity tiers)
- [ ] For each appid, fetch via `store.steampowered.com/api/appdetails?appids={id}&cc=ph`:
  - title, launch price (PHP), current price, discount percent
  - release date, genres, tags, categories
  - developer, publisher
  - Metacritic score
  - controller support, platform availability (win/mac/linux)
  - achievement count
  - multiplayer / co-op flags
- [ ] Cache raw JSON responses to `data/raw/steam/` to avoid re-pulls

### Steam Reviews
- [ ] For each appid, hit `store.steampowered.com/appreviews/{id}` for positive/negative review counts
- [ ] Store review timestamps for later velocity analysis (Part 2)

### SteamSpy
- [ ] Cross-reference each appid via `steamspy.com/api.php?request=appdetails&appid={id}`
  - ownership estimates, median playtime, average playtime

### Data Quality
- [ ] Drop F2P titles, demos, soundtracks, DLC, software (keep only base games)
- [ ] Drop entries missing launch price or release date
- [ ] Save consolidated cleaned dataset to `data/processed/games.parquet`

---

## Phase 2 — EDA & Feature Engineering (Part 1)

- [x] Create comprehensive EDA notebook (`notebooks/03_eda.ipynb`) with all required analyses
- [ ] Distribution plots: launch price, current price, discount %, age, review score
- [ ] Correlation heatmap of numeric features
- [ ] Genre × discount depth boxplots
- [ ] Engineer features:
  - `age_days` = today − release_date
  - `review_score` = positive / (positive + negative)
  - `review_volume_log` = log1p(total reviews)
  - `discount_pct` = (launch − current) / launch
  - One-hot / multi-hot encode genres and tags (top-N)
  - Target-encode developer / publisher (with smoothing)
- [ ] Define **value retention tier** label:
  - Premium Hold: discount_pct ≤ 10%
  - Standard Depreciation: 10–40%
  - Heavy Discount: 40–70%
  - Permanent Bargain: > 70%
- [ ] Train/val/test split (stratified by tier)

---

## Phase 3 — Modeling (Part 1)

### Regression: predict current price
- [ ] Baseline: linear regression with log-target
- [ ] Random Forest regressor
- [ ] Gradient Boosting (XGBoost / LightGBM)
- [ ] Metrics: RMSE, MAE, R², MAPE
- [ ] SHAP feature importance plot

### Classification: predict value retention tier
- [ ] Baseline: logistic regression (multinomial)
- [ ] Random Forest classifier
- [ ] XGBoost classifier
- [ ] Metrics: macro-F1, per-class precision/recall, confusion matrix
- [ ] Calibration check

### Save artifacts
- [ ] Pickle best models to `models/`
- [ ] Save evaluation report to `reports/part1_evaluation.md`

---

## Phase 4 — Data Collection (Part 2: Sale Effectiveness)

### IsThereAnyDeal
- [ ] Authenticate with API key
- [ ] For each appid in sample, pull historical price points + sale events
  - sale start date, end date, depth, sale type (if available)
- [ ] Store to `data/raw/itad/`

### Player counts — historical (SteamCharts)
- [ ] Run Stage 6 of `01_data_collection.ipynb` to scrape `steamcharts.com/app/{appid}/chart-data.json`
- [ ] Verify coverage: how many games have ≥ 100 history points? How far back does the median game go?
- [ ] Spot-check a few high-profile titles (CS2, Dota 2, TF2) against SteamCharts UI to confirm parser matches the chart
- [ ] Document granularity (~weekly per game) as a methodology limitation in the report

### Player counts — forward-looking (Steam Web API)
- [ ] Schedule Stage 7 to run daily via Windows Task Scheduler / cron
- [ ] After ~2 weeks of daily snapshots, layer them onto SteamCharts data for fresh post-Apr-2026 sales

### Review velocity
- [ ] Compute reviews-per-day series from review timestamps already collected

---

## Phase 5 — Longitudinal Analysis & Modeling (Part 2)

- [ ] **Derive sale events from `price_history`**: group consecutive rows where `cut > 0` per (appid, shop_id) into discrete (start, end, max_depth) intervals
- [ ] For each sale event, compute (using `steamcharts_history`):
  - 7-day post-sale player-count avg vs. 14-day pre-sale baseline
  - 7-day post-sale review velocity vs. baseline
  - `uplift_pct` = (post − pre) / pre
- [ ] **Within-game control**: each game's sale window vs. its own non-sale baseline (not cross-game)
- [ ] EDA: uplift distribution by discount depth, game age, sale type, prior discount frequency

### Regression: predict player-count uplift (%)
- [ ] Baseline + Gradient Boosting
- [ ] Features: discount depth, game age at sale, sale type, prior sale count, days since last sale, review score, genre
- [ ] Metrics: RMSE, R²

### Classification: sale effectiveness tier
- [ ] Define tiers: High Impact / Moderate Lift / Diminishing Returns / No Effect
- [ ] Train classifier; report macro-F1 + confusion matrix

### Caveats to document
- [ ] Acknowledge confounders: patches, streamer coverage, seasonality
- [ ] Note the within-game control mitigates but does not eliminate confounding
- [ ] Discuss survivorship bias (games that were delisted)

---

## Phase 6 — Deliverables

- [ ] Final report (`reports/final_report.md` or PDF)
  - Problem statement, data, methodology, results, business value, limitations
- [ ] Slide deck for presentation
- [ ] Notebook walkthroughs:
  - `01_data_collection.ipynb`
  - `02_eda.ipynb`
  - `03_part1_modeling.ipynb`
  - `04_part2_modeling.ipynb`
- [ ] README with setup instructions and reproducibility notes
- [ ] Stretch: prototype "fair price" Chrome extension overlay (mock up with screenshots if not built)

---

## Open Questions / Risks

- [ ] Confirm Steam API rate limits in practice — may need to spread collection over days
- [ ] Decide sampling strategy for the 5,000 games (random vs. stratified by year/genre/popularity)
- [ ] ITAD coverage may be patchy for very obscure indie titles — check before relying
- [ ] SteamCharts CCU is ~weekly per game (not daily as initially assumed) — fine for 7-day sale windows, but cannot resolve sub-week effects. Cite as a methodology limitation.
- [ ] SteamCharts scraping is not officially sanctioned — be polite (2s/req, descriptive UA) and cache; if blocked, fall back to live-only Stage 7 collection.
- [ ] PHP pricing assumes Steam regional pricing is stable — verify no mid-collection currency shifts
