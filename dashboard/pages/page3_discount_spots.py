"""
Page 3 — Discount Sweet Spots
Identifying high-quality games with deep discounts.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from data_loader import TIER_COLORS

def render():
    # Pull data from global session state[cite: 1]
    if 'df' not in st.session_state:
        st.error("Data not loaded. Please return to the main page.")
        return
        
    df = st.session_state['df']
    is_real = st.session_state['is_real']

    st.markdown("<div class='section-header'>Discount Sweet Spots</div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='color:#666;font-size:0.82rem;margin-bottom:20px'>"
        "Finding the 'Hidden Gems': High review scores paired with heavy price depreciation.</div>",
        unsafe_allow_html=True
    )

    # ── Filters ───────────────────────────────────────────────────────────────
    f1, f2, f3 = st.columns([2, 1, 1])
    
    with f1:
        genres = ["All"] + sorted(df["primary_genre"].dropna().unique().tolist())
        sel_genre = st.multiselect("Filter Genres", genres, default=["All"])
    
    with f2:
        min_score = st.slider("Min Review Score (%)", 0, 100, 75)
        
    with f3:
        max_retention = st.slider("Max Price Retention (%)", 10, 100, 50)

    # ── Logic: Calculate Value Score ──────────────────────────────────────────
    # Value Score = (Review Score / 100) * (1 - Price Retention)
    # Higher score = Better "Deal"[cite: 1]
    fdf = df.copy()
    
    if "All" not in sel_genre:
        fdf = fdf[fdf["primary_genre"].isin(sel_genre)]
        
    fdf = fdf[fdf["review_score"] >= min_score]
    fdf = fdf[fdf["price_retention"] <= (max_retention / 100)]
    
    fdf["value_score"] = (fdf["review_score"] / 100) * (1 - fdf["price_retention"])
    fdf = fdf.sort_values("value_score", ascending=False).head(30)

    if fdf.empty:
        st.warning("No sweet spots found with these strict filters. Try lowering the Min Review Score!")
        return

    # ── Visualization: Value Matrix ───────────────────────────────────────────
    st.markdown("<div class='section-header'>The Value Matrix</div>", unsafe_allow_html=True)
    
    fig = px.scatter(
        fdf,
        x="price_retention",
        y="review_score",
        size="value_score",
        color="value_retention_tier",
        color_discrete_map=TIER_COLORS,
        hover_name="title",
        hover_data={
            "current_price_php": ":.2f",
            "price_retention": ":.2%",
            "value_score": ":.2f"
        },
        labels={
            "price_retention": "Price Retention (Lower is Cheaper)",
            "review_score": "Review Score (Higher is Better)",
            "value_retention_tier": "Tier"
        }
    )
    
    fig.update_layout(
        paper_bgcolor="#0a0a0a",
        plot_bgcolor="#0d0d0d",
        font_color="#e0e0e0",
        height=450,
        margin=dict(l=10, r=10, t=10, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Top Deals Table ───────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Top Curated Deals</div>", unsafe_allow_html=True)
    
    disp = fdf.copy()
    disp["Discount"] = ((1 - disp["price_retention"]) * 100).map(lambda x: f"{x:.0f}% OFF")
    disp["Price"] = disp["current_price_php"].map(lambda x: f"₱{x:,.0f}")
    disp["Rating"] = disp["review_score"].map(lambda x: f"{x:.0f}%")
    
    st.dataframe(
        disp[["title", "primary_genre", "Rating", "Discount", "Price", "value_retention_tier"]]
        .rename(columns={
            "title": "Game Title",
            "primary_genre": "Genre",
            "value_retention_tier": "Retention Tier"
        }),
        use_container_width=True,
        hide_index=True
    )

    # ── Dynamic Insight ───────────────────────────────────────────────────────
    best_deal = fdf.iloc[0]
    st.markdown(
        f"<div class='insight-box'>"
        f"🔥 <b>Top Pick:</b> <span style='color:#00e5ff'>{best_deal['title']}</span>. "
        f"It has a <b>{best_deal['review_score']:.0f}%</b> rating and is currently "
        f"<b>{(1-best_deal['price_retention'])*100:.0f}% off</b> its launch price."
        f"</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    render()