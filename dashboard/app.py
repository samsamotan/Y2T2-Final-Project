"""
app.py  —  SteamSale Analytics Dashboard
Run with:  streamlit run app.py
"""

import streamlit as st
from data_loader import load_games

st.set_page_config(
    page_title="SteamSale Analytics",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── DATA INITIALIZATION ────────────────────────────────────────────────────────
# Load the 3,816 games once and store them so the banner disappears
if 'df' not in st.session_state:
    df, is_real = load_games()
    st.session_state['df'] = df
    st.session_state['is_real'] = is_real

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Mono', monospace; }
h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

.stApp { background: #0a0a0a; color: #e0e0e0; }

[data-testid="stSidebar"] {
    background: #0d0d0d;
    border-right: 1px solid #1e1e1e;
}
[data-testid="stSidebar"] * { color: #e0e0e0 !important; }

.metric-card {
    background: #141414;
    border: 1px solid #2a2a2a;
    border-radius: 8px;
    padding: 20px;
    text-align: center;
    margin-bottom: 8px;
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #00e5ff;
}
.metric-label {
    font-size: 0.72rem;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 4px;
}
.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.2rem;
    font-weight: 700;
    color: #fff;
    border-left: 3px solid #00e5ff;
    padding-left: 12px;
    margin: 24px 0 14px 0;
}
.insight-box {
    background: #0d1117;
    border: 1px solid #00e5ff33;
    border-left: 3px solid #00e5ff;
    border-radius: 6px;
    padding: 14px 16px;
    margin: 10px 0;
    font-size: 0.83rem;
    color: #ccc;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar nav ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='font-family:Syne,sans-serif;font-size:1.4rem;"
        "font-weight:800;color:#00e5ff;margin-bottom:4px'>🎮 SteamSale</div>"
        "<div style='font-size:0.7rem;color:#444;margin-bottom:20px'>"
        "Steam Pricing & Sale Analytics · Y2T2 Final Project</div>",
        unsafe_allow_html=True
    )

    # Added a status indicator so you can see if the DB connected successfully
    status_color = "#00e5ff" if st.session_state['is_real'] else "#ff4b4b"
    status_text = "LIVE DATABASE" if st.session_state['is_real'] else "DEMO MODE"
    st.markdown(f"<div style='color:{status_color};font-size:0.8rem;font-weight:bold;'>● {status_text}</div>", unsafe_allow_html=True)
    
    page = st.radio(
        "Pages",
        ["Price Predictor",
         "Value Retention",
         "Discount Sweet Spots",
         "Sale Effectiveness"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.68rem;color:#333'>"
        "Data: Steam Store API · SteamSpy · ITAD<br>"
        "Models: Random Forest · XGBoost</div>",
        unsafe_allow_html=True
    )

# ── Route ──────────────────────────────────────────────────────────────────────
# Note: Ensure these pages use st.session_state['df'] instead of re-loading data[cite: 1]
if   "Price Predictor"     in page:
    from pages.page1_price_predictor   import render; render()
elif "Value Retention"     in page:
    from pages.page2_value_retention   import render; render()
elif "Discount Sweet Spots" in page:
    from pages.page3_discount_spots    import render; render()
elif "Sale Effectiveness"  in page:
    from pages.page4_sale_effectiveness import render; render()