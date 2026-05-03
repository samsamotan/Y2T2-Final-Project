"""
Page 4 — Sale Effectiveness Dashboard
Predicts player uplift based on discount depth and game profile.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# Effectiveness tiers
EFF_COLORS = {
    "High Impact":  "#00e5ff",
    "Moderate":      "#a3e635",
    "Low":           "#fb923c",
    "Minimal":       "#f87171",
}

SALE_TYPES = ["Seasonal (Summer/Winter)", "Weekend Deal", "Publisher Sale", "Flash Sale"]

# Empirical uplift model
DISCOUNT_UPLIFT = {
    "<25%":   (5,  20),
    "25-50%": (15, 45),
    "50-75%": (30, 80),
    ">75%":   (40, 90),
}

SALE_TYPE_MULTIPLIER = {
    "Seasonal (Summer/Winter)": 1.35,
    "Weekend Deal":             1.20,
    "Publisher Sale":           0.90,
    "Flash Sale":               1.10,
}

def _disc_bucket(d):
    if d < 0.25:   return "<25%"
    if d < 0.50:   return "25-50%"
    if d < 0.75:   return "50-75%"
    return ">75%"

def _predict_uplift(discount, sale_type, days_since_release,
                    review_score, days_since_last_sale, sale_count):
    bucket = _disc_bucket(discount)
    lo, hi = DISCOUNT_UPLIFT[bucket]
    base   = (lo + hi) / 2

    # Modifiers based on research pipeline
    mult  = SALE_TYPE_MULTIPLIER.get(sale_type, 1.0)
    age_m = 1.0 + min(days_since_release / 3650, 0.4)    # older -> more room for new buyers
    rev_m = 0.85 + (review_score - 50) / 200             # better reviews -> higher conversion
    freq_m = max(0.6, 1.0 - (sale_count * 0.07))         # sale fatigue
    gap_m  = min(1.3, 1.0 + days_since_last_sale / 730)  # pent-up demand

    uplift = base * mult * age_m * rev_m * freq_m * gap_m
    uplift = float(np.clip(uplift, 0, 150))

    if uplift >= 50:   eff = "High Impact"
    elif uplift >= 20: eff = "Moderate"
    elif uplift >= 5:  eff = "Low"
    else:               eff = "Minimal"

    return uplift, eff

def render():
    # Use global session state data
    if 'df' not in st.session_state:
        st.error("Data not loaded. Please return to the main page.")
        return
        
    df = st.session_state['df']
    is_real = st.session_state['is_real']

    if not is_real:
        st.info("Demo mode — showing synthetic predictions based on market averages.", icon="🗄️")

    st.markdown("<div class='section-header'>Sale Effectiveness Dashboard</div>", unsafe_allow_html=True)
    
    st.markdown(
        "<div class='insight-box' style='border-left: 4px solid #fb923c;'>"
        "⚠️ <b>Note:</b> Estimates use a calibrated formula based on discount depth and game profile. "
        "Treat as directional guidance.</div>",
        unsafe_allow_html=True
    )

    # ── Input: Game profile ───────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Game Profile & Baseline</div>", unsafe_allow_html=True)

    g1, g2, g3 = st.columns(3)
    genres = sorted(df["primary_genre"].dropna().unique().tolist()) if not df.empty else ["Indie", "Action", "RPG"]
    with g1:
        genre = st.selectbox("Genre", genres, index=0)
        review_score = st.slider("Current Review Score (%)", 20, 100, 85)
    with g2:
        age_yr = st.slider("Game Age (years)", 0.5, 10.0, 2.0, 0.5)
        days_age = int(age_yr * 365)
    with g3:
        avg_players = st.number_input("Avg Daily Players (Current)", 100, 1_000_000, 1000, 100)

    # ── Input: Sale parameters ────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Proposed Sale Strategy</div>", unsafe_allow_html=True)

    s1, s2, s3 = st.columns(3)
    with s1:
        discount_pct = st.slider("Planned Discount (%)", 5, 90, 50)
        discount = discount_pct / 100
    with s2:
        sale_type = st.selectbox("Event Type", SALE_TYPES)
    with s3:
        days_last_sale = st.slider("Days Since Last Sale", 0, 730, 90)
        sale_count = st.slider("Previous Sales (12mo)", 0, 10, 2)

    # ── Prediction Logic ──────────────────────────────────────────────────────
    uplift_pct, eff_tier = _predict_uplift(discount, sale_type, days_age, review_score, days_last_sale, sale_count)
    ec = EFF_COLORS[eff_tier]
    peak_players = int(avg_players * (1 + uplift_pct / 100))

    st.markdown("---")
    r1, r2, r3, r4 = st.columns(4)
    result_tiles = [
        (f"+{uplift_pct:.0f}%", ec, "Predicted Uplift"),
        (eff_tier, ec, "Impact Level"),
        (f"{peak_players:,}", "#00e5ff", "Est. Peak CCU"),
        (_disc_bucket(discount), "#888", "Strategy Bucket"),
    ]
    for col, (val, color, label) in zip([r1, r2, r3, r4], result_tiles):
        col.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-value' style='color:{color};font-size:1.4rem'>{val}</div>"
            f"<div class='metric-label'>{label}</div></div>",
            unsafe_allow_html=True
        )

    # ── Charts: Sensitivity Analysis ──────────────────────────────────────────
    st.markdown("<div class='section-header'>Uplift Sensitivity Analysis</div>", unsafe_allow_html=True)
    ca, cb = st.columns([2, 1])

    with ca:
        # Discount Depth vs Uplift Line Chart
        disc_range = np.arange(0.10, 0.95, 0.05)
        uplifts = [_predict_uplift(d, sale_type, days_age, review_score, days_last_sale, sale_count)[0] for d in disc_range]
        
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(x=disc_range*100, y=uplifts, fill='tozeroy', line_color='#00e5ff', name="Projection"))
        fig_line.add_trace(go.Scatter(x=[discount_pct], y=[uplift_pct], marker=dict(size=15, color=ec, symbol='star'), name="Selected"))
        fig_line.update_layout(paper_bgcolor="#0a0a0a", plot_bgcolor="#0d0d0d", font_color="#e0e0e0", height=350, 
                              xaxis_title="Discount (%)", yaxis_title="Uplift (%)", margin=dict(t=20))
        st.plotly_chart(fig_line, use_container_width=True)

    with cb:
        # Genre Benchmark Bar Chart
        gdf = df[df["primary_genre"] == genre] if not df.empty else pd.DataFrame()
        if not gdf.empty:
            med_disc = gdf['discount_depth'].median() * 100
            st.markdown(f"**{genre} Market Norms**")
            st.metric("Median Discount", f"{med_disc:.0f}%")
            st.metric("50%+ Discount Adoption", f"{(gdf['discount_depth'] >= 0.5).mean()*100:.1f}%")
        else:
            st.info("Genre benchmarks unavailable in demo.")

    # ── Recommendation Text ───────────────────────────────────────────────────
    st.markdown(
        f"<div class='insight-box' style='background: #111; border: 1px solid #333;'>"
        f"💡 <b>Strategy Recommendation:</b> Based on current settings, a {discount_pct}% discount during a {sale_type} "
        f"is expected to perform with <b>{eff_tier}</b> efficiency. "
        f"{'Consider increasing discount to 50% for high-impact results.' if uplift_pct < 25 else 'This is a strong value proposition.'}"
        f"</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    render()