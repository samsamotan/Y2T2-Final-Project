"""
Page 1 — Price Predictor
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from data_loader import TIER_COLORS


def _predict(df, genre, days, review_score, owners_log, price_tier):
    """Cohort-median estimate; formula fallback when cohort < 10."""
    mask = (
        (df["primary_genre"] == genre) &
        (df["days_since_release"].between(days * 0.55, days * 1.45)) &
        (df["price_tier"].astype(str) == price_tier)
    )
    cohort = df[mask]

    if len(cohort) >= 10:
        retention = cohort["price_retention"].median()
    else:
        age_fac      = np.exp(-days / 3500)
        review_boost = (review_score - 50) / 50 * 0.08
        pop_boost    = (owners_log - 3)   / 4  * 0.05
        retention    = float(np.clip(age_fac + review_boost + pop_boost, 0.05, 1.0))

    tier_mid = {"Budget": 150.0, "Mid": 850.0, "Premium": 2200.0}
    pred_price = tier_mid.get(price_tier, 500.0) * retention

    if retention > 0.85:   tier = "Premium Hold"
    elif retention > 0.50: tier = "Standard Depreciation"
    elif retention > 0.25: tier = "Heavy Discount"
    else:                  tier = "Permanent Bargain"

    return pred_price, retention, tier, len(cohort)


def render():
    if 'df' not in st.session_state:
        st.error("Data not loaded. Please return to the main page.")
        return

    df      = st.session_state['df']
    is_real = st.session_state['is_real']

    if not is_real:
        st.info("Demo mode — DB not found. Check CUSTOM_DB_PATH in data_loader.py.", icon="🗄️")

    st.markdown("<div class='section-header'>Price Predictor</div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='color:#666;font-size:0.82rem;margin-bottom:20px'>"
        "Estimate what a game should currently cost based on its genre, age, and reception.</div>",
        unsafe_allow_html=True
    )

    # ── Input controls ────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    genres = sorted(df["primary_genre"].dropna().unique().tolist())

    with c1:
        default_idx = genres.index("Indie") if "Indie" in genres else 0
        genre       = st.selectbox("Genre", genres, index=default_idx)
        price_tier  = st.selectbox("Launch Price Tier", ["Budget", "Mid", "Premium"], index=1)
    with c2:
        age_yr = st.slider("Game Age (years)", 0.5, 10.0, 2.0, 0.5)
        days   = int(age_yr * 365)
    with c3:
        review_score = st.slider("Review Score (%)", 20, 100, 80)
        owners_log   = st.slider("Ownership scale (log₁₀)", 2.0, 7.0, 4.5, 0.5,
                                  help="3 ≈ 1K | 4 ≈ 10K | 5 ≈ 100K | 6 ≈ 1M")

    # ── Compute prediction ────────────────────────────────────────────────────
    pred_price, retention, tier, n_cohort = _predict(
        df, genre, days, review_score, owners_log, price_tier
    )
    tc = TIER_COLORS.get(tier, "#00e5ff")

    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    cards = [
        (f"₱{pred_price:,.0f}", "#00e5ff", "Predicted Current Price"),
        (f"{retention*100:.0f}%", tc,       "Price Retention"),
        (tier,                    tc,       "Retention Tier"),
        (str(n_cohort),           "#888",   "Similar Games in Cohort"),
    ]
    for col, (val, color, label) in zip([m1, m2, m3, m4], cards):
        col.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-value' style='color:{color};font-size:1.5rem'>{val}</div>"
            f"<div class='metric-label'>{label}</div></div>",
            unsafe_allow_html=True
        )

    # ── Gauge + insight ───────────────────────────────────────────────────────
    g_col, i_col = st.columns([1, 1])
    with g_col:
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=retention * 100,
            number={"suffix": "%", "font": {"color": tc, "size": 34}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#333"},
                "bar":  {"color": tc, "thickness": 0.22},
                "bgcolor": "#141414",
                "steps": [
                    {"range": [0,  25],  "color": "#150a0a"},
                    {"range": [25, 50],  "color": "#15100a"},
                    {"range": [50, 85],  "color": "#0a150a"},
                    {"range": [85, 100], "color": "#0a1515"},
                ],
            },
            title={"text": "Price Retention", "font": {"color": "#666", "size": 12}}
        ))
        fig_g.update_layout(
            height=240, paper_bgcolor="#0a0a0a",
            font_color="#e0e0e0", margin=dict(l=20, r=20, t=40, b=10)
        )
        st.plotly_chart(fig_g, use_container_width=True)

    tier_desc = {
        "Premium Hold":          "> 85% retained — this game rarely goes on sale.",
        "Standard Depreciation": "50–85% retained — moderate discounts expected.",
        "Heavy Discount":        "25–50% retained — frequently discounted deeply.",
        "Permanent Bargain":     "< 25% retained — almost always on 75%+ sale.",
    }
    with i_col:
        st.markdown(
            f"<div class='insight-box' style='margin-top:30px'>"
            f"<b style='color:{tc}'>{tier}</b><br>{tier_desc.get(tier, '')}<br><br>"
            f"Expected discount depth: <b style='color:{tc}'>{(1-retention)*100:.0f}%</b><br>"
            f"Cohort: <b>{n_cohort}</b> comparable {genre} games in the <b>{price_tier}</b> tier."
            f"</div>",
            unsafe_allow_html=True
        )

    # ── Similar games table ───────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Similar Games in Dataset</div>", unsafe_allow_html=True)

    sim = df[
        (df["primary_genre"] == genre) &
        (df["days_since_release"].between(days * 0.4, days * 1.6)) &
        (df["review_score"].between(review_score - 20, review_score + 20))
    ].copy()

    if len(sim) < 5:
        sim = df[df["primary_genre"] == genre].copy()

    sim = sim.sort_values("price_retention", ascending=False).head(15)
    sim["Launch ₱"]  = sim["initial_price_php"].map(lambda x: f"₱{x:,.0f}")
    sim["Current ₱"] = sim["current_price_php"].map(lambda x: f"₱{x:,.0f}")
    sim["Retention"] = sim["price_retention"].map(lambda x: f"{x*100:.1f}%")
    sim["Score"]     = sim["review_score"].map(lambda x: f"{x:.0f}%")

    # Safe total_reviews access
    if "total_reviews" in sim.columns:
        sim["Reviews"] = sim["total_reviews"].fillna(0).astype(int).map(lambda x: f"{x:,}")
    else:
        sim["Reviews"] = "N/A"

    st.dataframe(
        sim[["title", "primary_genre", "value_retention_tier",
             "Launch ₱", "Current ₱", "Retention", "Score", "Reviews"]]
          .rename(columns={"title": "Game", "primary_genre": "Genre",
                            "value_retention_tier": "Tier"}),
        use_container_width=True, hide_index=True, height=340
    )

    # ── Scatter: age vs retention ─────────────────────────────────────────────
    st.markdown(
        f"<div class='section-header'>Age vs Price Retention — {genre}</div>",
        unsafe_allow_html=True
    )

    gdf = df[df["primary_genre"] == genre]
    fig = px.scatter(
        gdf, x="days_since_release", y="price_retention",
        color="value_retention_tier",
        color_discrete_map=TIER_COLORS,
        hover_data=["title", "current_price_php", "review_score"],
        labels={
            "days_since_release":    "Days Since Release",
            "price_retention":       "Price Retention",
            "value_retention_tier":  "Tier"
        },
        opacity=0.55,
    )
    fig.add_trace(go.Scatter(
        x=[days], y=[retention], mode="markers", name="Your Input",
        marker=dict(color=tc, size=18, symbol="star",
                    line=dict(color="#fff", width=2))
    ))
    fig.update_layout(
        paper_bgcolor="#0a0a0a", plot_bgcolor="#0d0d0d",
        font_color="#e0e0e0", height=400,
        margin=dict(l=10, r=10, t=30, b=10)
    )
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    render()