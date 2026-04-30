# SteamSale — Presentation Script

A spoken-word script for presenting the project. Roughly 8–10 minutes at a natural pace. Section headers are stage directions; the prose is what you say.

---

## Opening (30 seconds)

Steam sells over a hundred thousand games. Every weekend, hundreds of them go on sale, with discounts ranging from 10% to 95%. As a buyer, you're constantly asking: is this discount real, or is it inflated against a fake "regular" price? Will the game drop further if I wait?

If you flip the question around, indie developers are asking the mirror version: am I leaving money on the table by joining the next big sale? Or is participation actually the only thing that moves my revenue?

Our project, **SteamSale**, builds a predictive model that answers both questions from the same dataset.

---

## The Two Research Questions (1 minute)

We've split the project into two parts.

**Part 1 is about pricing and value retention.** We want to predict the current Steam price of a game from its attributes — genre, developer, age since release, review score, multiplayer support, achievement count, and so on. From that, we classify each game into one of four value-retention tiers: *Premium Hold* games that rarely discount, *Standard Depreciation*, *Heavy Discount*, and *Permanent Bargain* — games that are essentially always on sale.

**Part 2 is about sale effectiveness.** When a game goes on sale, how much does player engagement actually increase compared to its non-sale baseline? We predict the player-count uplift from discount depth, game age, sale type, and prior discount history. Then we classify each sale event into four effectiveness tiers: *High Impact*, *Moderate Lift*, *Diminishing Returns*, and *No Effect*.

The dual framing is what makes this project useful. Part 1 tells buyers what to buy. Part 2 tells developers when to sell.

---

## Data Sources (1.5 minutes)

We're pulling from five sources, all free and open for academic use. Our sample size is around five thousand games, drawn from SteamSpy's top-owned list so the sample is stratified by popularity rather than purely random.

The **Steam Storefront API** is our primary source. It gives us the basic record per game: title, launch and current price in Philippine pesos directly via the country-code parameter, discount percent, release date, genres, categories, developer, publisher, Metacritic score, and platform support.

The **Steam Web API** gives us live concurrent player counts — one snapshot per call. We layer that on top of historical data from SteamCharts, which I'll come to in a moment.

The **Steam Reviews endpoint** gives us aggregate review counts and the famous review score description — *Overwhelmingly Positive*, *Mostly Mixed*, and so on. Plus the timestamp of every individual review, which we use to compute review velocity around sale events.

**SteamSpy** gives us ownership estimates as a bucket — for example, 10 to 20 million owners — along with median playtime and community tags.

**IsThereAnyDeal**, or ITAD, is the backbone of Part 2. They've been tracking historical price points across Steam for years. They give us the start date, end date, and depth of every discount per game.

And **SteamCharts** gives us the historical concurrent-player time series, going back to 2012, at roughly weekly granularity. Pair that with ITAD sale events and you can measure whether each sale actually moved the needle.

---

## The Data Warehouse (1 minute)

Everything lands in a single SQLite database with twelve tables. One worklist, one master table called `games`, and ten child tables — one per data domain.

The worklist is what makes the pipeline resumable. Every appid carries six progress flags: `has_details`, `has_reviews`, `has_steamspy`, `has_itad_id`, `has_price_history`, `has_steamcharts`. If we interrupt the pipeline overnight, the next run picks up exactly where we left off — only the unprocessed appids are retried.

Every child table has a foreign key back to `games`, so the schema is self-validating. If a game is delisted from Steam, every record about it cascades cleanly.

We also generated a schema diagram — `docs/db_schema.png` — which I'd point you to during a poster session.

---

## The ML Approach (1.5 minutes)

For Part 1, we're training two models on the same feature set. A **regressor** predicts the current price as a continuous value. A **classifier** predicts which of the four retention tiers the game falls into. We start with linear and logistic baselines, then move to Random Forest and gradient boosting — XGBoost or LightGBM — to capture interaction effects between genre, age, and reception.

Features include game age in days, log-scaled review volume, review score ratio, multi-hot encoded genres and categories, and target-encoded developer and publisher.

For Part 2, the unit of analysis shifts from "one game" to "one sale event." For every sale, we compute the seven-day post-sale player-count average versus a fourteen-day pre-sale baseline. The lift, normalized as a percentage, is our outcome variable.

The critical methodological choice is that **each game is its own control**. We compare each sale window to the same game's non-sale baseline, not to other games. This rules out a lot of confounding from popularity, genre mix, and seasonality.

---

## Expected Outputs (1 minute)

By the end of the project, we expect to deliver:

- Trained regressors and classifiers for both parts, with evaluation reports — RMSE, R-squared, macro F1, confusion matrices.
- SHAP feature-importance plots showing which game attributes most strongly drive price retention.
- A signature visualization: per-game CCU time series with sale events overlaid on the chart, so you can visually confirm that a Steam Summer Sale spike on a game's player count lines up with an ITAD discount window.
- A "fair price" scoring function that takes a game and its current Steam price and tells you whether this is a genuine deal or an inflated baseline.
- A "sale ROI" forecast for developers — given a proposed discount depth, what's the expected player-count lift?
- As a stretch goal, a Chrome extension overlay that shows model output inline on the Steam store page.

---

## Why It Matters (45 seconds)

Most buy-vs-wait advice on Steam is folklore. Wait for the Summer Sale. GOTY editions drop in November. Indie games never go below 50% off. We don't actually know which of these are true.

SteamSale grounds the question in five thousand games' worth of historical pricing and engagement data. Buyers get to distinguish genuine deals from inflated baselines and surface games that are basically always on sale. Indie developers get to benchmark their pricing against comparable titles and decide whether to participate in seasonal sales based on their game's actual profile, not gut feel. And publishers can refine their sale-cadence strategy — figure out which sales actually move the needle and which are just leaving money on the table.

---

## Limitations (1 minute)

We want to be upfront about what this project can't do.

First, sale uplift conflates the discount effect with confounders we can't observe — a major patch dropping the same week, a streamer covering the game, a seasonal event. The within-game baseline mitigates this but doesn't eliminate it.

Second, SteamCharts gives us roughly weekly granularity per game, not daily. That's fine for measuring seven-day sale windows, but we cannot resolve sub-week effects.

Third, SteamSpy ownership estimates are bucketed — 10 to 20 million, for example — because Valve restricted profile visibility back in 2018. Even paid SteamSpy data isn't fully precise. We treat ownership as a categorical feature, not a continuous one.

Fourth, ITAD coverage may miss obscure indie titles. Our dataset skews toward moderately popular games where all five sources have data.

Finally, there's survivorship bias. Delisted games drop out of our pool, so the model only sees games that survived.

These caveats go into the methodology section of the report — not as excuses, but as honest scope limits.

---

## Closing (15 seconds)

Part 1 tells you what to buy. Part 2 tells you when to sell — if you're a developer.

Same dataset, two audiences, one project.

Thank you. I'm happy to take questions.

---

## Q&A Cheat Sheet

If asked about something specific, here are quick answers:

- **Why PHP instead of USD?** Steam's regional pricing is meaningful — discount percent is a global signal, but absolute prices differ by region. We collect in PHP because that's our market and the API supports it directly via `cc=ph`.
- **Why SQLite instead of Postgres?** Single-developer, single-machine project, no concurrent writers. SQLite is faster to set up, the file is portable, and `.parquet` exports for modeling are trivial.
- **Why not just use SteamDB?** SteamDB is paid for bulk access and aggressive about anti-scraping. ITAD is the legitimate free path for the same data.
- **How long does the full pipeline take?** Storefront stage is the bottleneck at ~2–3 hours for 5,000 games due to throttle. ITAD adds about an hour. SteamCharts about three hours. We run overnight.
- **What if a game gets delisted mid-collection?** The Storefront API returns `success: false`, which we log and skip. The appid is marked with `last_error='not_found'`.
