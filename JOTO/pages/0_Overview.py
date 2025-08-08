import streamlit as st
from styles import inject_css
from utils import load_lst_prediction_data

st.set_page_config(page_title="Kilimani Heat Island", page_icon="🌍", layout="wide")
inject_css()

st.markdown('<h1 class="main-header">🌍 Kilimani Heat Island Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<h2 class="sub-header">AI-Powered Urban Climate Resilience Platform</h2>', unsafe_allow_html=True)

df = load_lst_prediction_data()

if df is not None:
    mean_temp = df['LST_Prediction'].mean()
    max_temp = df['LST_Prediction'].max()
    hii = df['LST_Prediction'].max() - df['LST_Prediction'].min()
    green_cover = (df['NDVI'] > 0.3).sum() / len(df) * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🌡️ Avg Temp", f"{mean_temp:.1f}°C")
    col2.metric("🔥 Max Temp", f"{max_temp:.1f}°C")
    col3.metric("🌡️ HII", f"{hii:.1f}°C")
    col4.metric("🌳 Green Cover", f"{green_cover:.0f}%")

    st.markdown("---")

    st.markdown("### 🧭 Navigate the Dashboard")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="glass-card">
            <h3>🤖 AI & Model Insights</h3>
            <p>AI analysis, model performance, predictions</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="glass-card">
            <h3>🏢 Urban Form & Vegetation</h3>
            <p>How buildings and green spaces affect heat</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="glass-card">
            <h3>🛡️ Mitigation Planning</h3>
            <p>Strategies, scenarios, and policy tools</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### 📌 Executive Summary")
    st.markdown(f"""
    <div class="insight-box">
        <p>The Kilimani area faces significant urban heat stress, with temperatures reaching <strong>{max_temp:.1f}°C</strong> 
        in densely built zones. Building density and low vegetation are the primary drivers.</p>
        
        <p><strong>Recommended Actions:</strong> Expand green infrastructure, implement cool building standards, 
        and use AI-powered planning tools to mitigate future development impacts.</p>
    </div>
    """, unsafe_allow_html=True)

else:
    st.error("❌ Unable to load Kilimani LST prediction data")
    st.info("Please ensure the resources/Kilimani_LST_Prediction.tif file exists")

st.markdown("---")
st.markdown("### 🧭 Navigation")
st.info("💡 Use the sidebar to explore all analysis pages")