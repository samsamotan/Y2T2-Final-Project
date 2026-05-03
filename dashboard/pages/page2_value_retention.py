"""
Page 2 — Value Retention Leaderboard
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from data_loader import TIER_COLORS, TIER_ORDER


def render():
    if 'df' not in st.session_state:
        st.error("Data not loaded. Please return to the main page.")
        return

    df      = st.session_state['df']
    is_real = st.session_state['is_real']

    if not is_real:
        st.info("Demo mode — showing synthetic data.", icon="🗄️")

    st.markdown("<div class='section-header'>Value Retention Leaderboard</div>",
                unsafe_allow_html=True)
    st.markdown(
        "<div style='color:#666;font-size:0.82rem;margin-bottom:20px'>"
        "Games that hold their launch price best. "
        "<b style='color:#00e5ff'>Premium Hold</b> = rarely drops below 85% of launch price.</div>",
        unsafe_allow_html=True
    )

    # ── Summary tiles ─────────────────────────────────────────────────────────
    counts = df["value_retention_tier"].value_counts()
    cols   = st.columns(4)
    for col, tier in zip(cols, TIER_ORDER):
        n   = counts.get(tier, 0)
        pct = n / len(df) * 100
        c   = TIER_COLORS.get(tier, "#888")
        col.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-value' style='color:{c}'>{n}</div>"
            f"<div class='metric-label'>{tier}<br>"
            f"<span style='font-size:0.68rem;color:#444'>{pct:.1f}% of all games</span></div>"
            f"</div>",
            unsafe_allow_html=True
        )

    st.markdown("---")

    # ── Filters ───────────────────────────────────────────────────────────────
    f1, f2, f3, f4 = st.columns(4)

    genres = ["All"] + sorted(df["primary_genre"].dropna().unique().tolist())
    with f1:
        sel_genre = st.selectbox("Genre", genres)

    years     = sorted(df["release_date"].dt.year.dropna().unique().tolist(), reverse=True)
    year_opts = ["All"] + [str(int(y)) for y in years]
    with f2:
        sel_year = st.selectbox("Release Year", year_opts)

    with f3:
        sel_tiers = st.multiselect("Retention Tier", TIER_ORDER,
                                   default=["Premium Hold", "Standard Depreciation"])

    p_max = int(df["initial_price_php"].quantile(0.99)) if not df.empty else 5000
    with f4:
        p_range = st.slider("Launch Price (₱)", 0, p_max, (0, p_max))

    # ── Filter logic ──────────────────────────────────────────────────────────
    fdf = df.copy()
    if sel_genre != "All":
        fdf = fdf[fdf["primary_genre"] == sel_genre]
    if sel_year != "All":
        try:
            fdf = fdf[fdf["release_date"].dt.year == int(sel_year)]
        except Exception:
            pass
    if sel_tiers:
        fdf = fdf[fdf["value_retention_tier"].isin(sel_tiers)]

    fdf = fdf[fdf["initial_price_php"].between(*p_range)]
    fdf = fdf.sort_values("price_retention", ascending=False).head(50)

    if len(fdf) == 0:
        st.warning("No games match the current filters — try widening them.")
        return

    # ── Table ─────────────────────────────────────────────────────────────────
    st.markdown(f"<div class='section-header'>Top {len(fdf)} Games</div>",
                unsafe_allow_html=True)

    disp = fdf.copy()
    disp["#"]          = range(1, len(disp) + 1)
    disp["Launch ₱"]   = disp["initial_price_php"].map(lambda x: f"₱{x:,.0f}")
    disp["Current ₱"]  = disp["current_price_php"].map(lambda x: f"₱{x:,.0f}")
    disp["Retention"]  = disp["price_retention"].map(lambda x: f"{x*100:.1f}%")
    disp["Score"]      = disp["review_score"].map(lambda x: f"{x:.0f}%")
    disp["Age (days)"] = disp["days_since_release"].map(lambda x: f"{int(x):,}")

    # Safe total_reviews access
    if "total_reviews" in disp.columns:
        disp["Reviews"] = disp["total_reviews"].fillna(0).astype(int).map(lambda x: f"{x:,}")
    else:
        disp["Reviews"] = "N/A"

    st.dataframe(
        disp[["#", "title", "primary_genre", "value_retention_tier",
              "Launch ₱", "Current ₱", "Retention", "Score", "Reviews", "Age (days)"]]
          .rename(columns={"title": "Game", "primary_genre": "Genre",
                            "value_retention_tier": "Tier"}),
        use_container_width=True, hide_index=True, height=420
    )

    # ── Charts ────────────────────────────────────────────────────────────────
    ca, cb = st.columns(2)

    with ca:
        st.markdown("<div class='section-header'>Market Distribution</div>",
                    unsafe_allow_html=True)
        td  = df["value_retention_tier"].value_counts().reindex(TIER_ORDER).fillna(0)
        fig = go.Figure(go.Pie(
            labels=td.index, values=td.values,
            marker_colors=[TIER_COLORS.get(t, "#888") for t in td.index],
            hole=0.55, textinfo="percent+label", textfont_size=10
        ))
        fig.update_layout(
            paper_bgcolor="#0a0a0a", font_color="#e0e0e0",
            showlegend=False, height=300,
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)

    with cb:
        st.markdown("<div class='section-header'>Retention Leaders by Genre</div>",
                    unsafe_allow_html=True)
        gr = (
            df.groupby("primary_genre")["price_retention"]
            .median().sort_values(ascending=True).tail(12)
        )
        bar_colors = [
            TIER_COLORS["Premium Hold"]          if v > 0.85 else
            TIER_COLORS["Standard Depreciation"] if v > 0.50 else
            TIER_COLORS["Heavy Discount"]        if v > 0.25 else
            TIER_COLORS["Permanent Bargain"]
            for v in gr.values
        ]
        fig2 = go.Figure(go.Bar(
            x=gr.values * 100, y=gr.index, orientation="h",
            marker_color=bar_colors,
            text=[f"{v*100:.0f}%" for v in gr.values],
            textposition="outside", textfont_color="#aaa"
        ))
        fig2.update_layout(
            paper_bgcolor="#0a0a0a", plot_bgcolor="#0d0d0d",
            font_color="#e0e0e0", height=300,
            xaxis=dict(title="Median Retention (%)", gridcolor="#1a1a1a"),
            yaxis=dict(gridcolor="#1a1a1a"),
            margin=dict(l=10, r=40, t=10, b=30)
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Insight ───────────────────────────────────────────────────────────────
    ph_pct = counts.get("Premium Hold", 0) / len(df) * 100
    pb_pct = counts.get("Permanent Bargain", 0) / len(df) * 100
    st.markdown(
        f"<div class='insight-box'>"
        f"💡 Only <b style='color:#00e5ff'>{ph_pct:.1f}%</b> of games hold above 85% of launch price. "
        f"Meanwhile, <b style='color:#f87171'>{pb_pct:.1f}%</b> have dropped to Permanent Bargain levels."
        f"</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    render()