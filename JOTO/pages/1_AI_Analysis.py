import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_lst_prediction_data, analyze_lst_prediction_with_ai, generate_ai_response

# Page configuration
st.set_page_config(
    page_title="AI Analysis - Kilimani Heat Island",
    page_icon="🤖",
    layout="wide"
)

# Load the same CSS as main app
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        color: #ffffff;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 20px rgba(0,0,0,0.3);
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4, #45b7d1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .sub-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #ffffff;
        text-align: center;
        margin-bottom: 2rem;
        opacity: 0.9;
    }
    
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #3498db;
    }
    
    .metric-container {
        background: rgba(255,255,255,0.95);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.3);
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .insight-box {
        background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
        backdrop-filter: blur(15px);
        padding: 2rem;
        border-radius: 20px;
        border: 1px solid rgba(255,255,255,0.2);
        margin: 2rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        color: #ffffff;
    }
    
    .insight-box h4 {
        color: #4ecdc4;
        font-weight: 600;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Page header
st.markdown('<h1 class="main-header">🤖 AI Analysis</h1>', unsafe_allow_html=True)
st.markdown('<h2 class="sub-header">Comprehensive AI-Powered LST Insights</h2>', unsafe_allow_html=True)

# Load data
df = load_lst_prediction_data()

if df is not None:
    # Generate AI analysis
    ai_analysis = analyze_lst_prediction_with_ai(df)
    
    if ai_analysis:
        # Overview Section
        st.markdown("### 📋 Analysis Overview")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-container">
                <h4>🌡️ Average Temperature</h4>
                <h2>{ai_analysis['overview']['mean_temperature']}°C</h2>
                <p>Across {ai_analysis['overview']['total_area_analyzed']:,} points</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-container">
                <h4>🔥 Heat Island Intensity</h4>
                <h2>{ai_analysis['overview']['heat_island_intensity']}°C</h2>
                <p>Max - Min Temperature</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-container">
                <h4>⚠️ Risk Level</h4>
                <h2>{ai_analysis['risk_assessment']['heat_stress_risk']}</h2>
                <p>Heat Stress Assessment</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Primary Insights
        st.markdown("### 🧠 AI Insights")
        
        st.markdown(f"""
        <div class="glass-card">
            <h4 style="color: #e74c3c;">🎯 Primary Concern</h4>
            <p style="font-size: 1.1em; margin-bottom: 20px;">{ai_analysis['ai_insights']['primary_concern']}</p>
            
            <h4 style="color: #3498db;">📊 Intervention Priority</h4>
            <p style="font-size: 1.1em;">{ai_analysis['ai_insights']['intervention_priority']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Spatial Analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🗺️ Spatial Patterns")
            spatial = ai_analysis['spatial_patterns']
            
            st.markdown(f"""
            <div class="glass-card">
                <h4>🔥 Hot Spots</h4>
                <p><strong>Count:</strong> {spatial['hot_spots_count']:,}</p>
                <p><strong>Percentage:</strong> {spatial['hot_spots_percentage']}%</p>
                <p><strong>Threshold:</strong> >{spatial['hot_threshold']}°C</p>
                
                <h4 style="margin-top: 20px;">❄️ Cool Spots</h4>
                <p><strong>Count:</strong> {spatial['cool_spots_count']:,}</p>
                <p><strong>Percentage:</strong> {spatial['cool_spots_percentage']}%</p>
                <p><strong>Threshold:</strong> <{spatial['cool_threshold']}°C</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### ⚠️ Risk Assessment")
            risk = ai_analysis['risk_assessment']
            
            st.markdown(f"""
            <div class="glass-card">
                <h4>🌡️ Heat Stress Risk</h4>
                <p><strong>Level:</strong> {risk['heat_stress_risk']}</p>
                
                <h4>🏝️ Heat Island Severity</h4>
                <p><strong>Level:</strong> {risk['urban_heat_island_severity']}</p>
                
                <h4>🚨 Areas of Concern</h4>
                <p><strong>High Risk Areas:</strong> {risk['areas_of_concern']:,}</p>
                <p><strong>Priority Zones:</strong> {risk['priority_intervention_zones']:,}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Spatial Recommendations
        st.markdown("### 💡 AI Spatial Recommendations")
        
        recommendations = ai_analysis['ai_insights']['spatial_recommendations']
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"""
            <div class="glass-card">
                <h4 style="color: #2ecc71;">Recommendation {i}</h4>
                <p>{rec}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Climate Adaptation Needs
        st.markdown("### 🌍 Climate Adaptation Needs")
        
        adaptation_needs = ai_analysis['ai_insights']['climate_adaptation_needs']
        
        cols = st.columns(2)
        for i, need in enumerate(adaptation_needs):
            with cols[i % 2]:
                st.markdown(f"""
                <div class="glass-card">
                    <p>• {need}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Interactive Q&A Section
        st.markdown("---")
        st.markdown("### 💬 Ask the AI About Your Predictions")
        
        user_question = st.text_input("Ask a question about the LST predictions:", 
                                    placeholder="e.g., Which areas need immediate intervention?")
        
        if user_question:
            # Generate AI response based on the analysis
            response = generate_ai_response(user_question, ai_analysis, df)
            st.markdown(f"""
            <div class="glass-card">
                <h4 style="color: #3498db;">🤖 AI Response</h4>
                <p>{response}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Temperature Distribution Analysis
        st.markdown("---")
        st.markdown("### 📊 Temperature Distribution Analysis")
        
        temp_distribution = ai_analysis['temperature_distribution']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🌡️ Temperature Zones")
            for zone, count in temp_distribution.items():
                percentage = (count / ai_analysis['overview']['total_area_analyzed']) * 100
                st.markdown(f"""
                <div class="metric-container">
                    <h4>{zone}</h4>
                    <h3>{count:,} areas</h3>
                    <p>{percentage:.1f}% of total</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### 🎯 Key Statistics")
            overview = ai_analysis['overview']
            st.markdown(f"""
            <div class="glass-card">
                <p><strong>Mean Temperature:</strong> {overview['mean_temperature']}°C</p>
                <p><strong>Maximum Temperature:</strong> {overview['max_temperature']}°C</p>
                <p><strong>Minimum Temperature:</strong> {overview['min_temperature']}°C</p>
                <p><strong>Temperature Variation:</strong> {overview['temperature_variation']}°C</p>
                <p><strong>Heat Island Intensity:</strong> {overview['heat_island_intensity']}°C</p>
            </div>
            """, unsafe_allow_html=True)

else:
    st.error("❌ Unable to load Kilimani LST prediction data")
    st.info("Please ensure the resources/Kilimani_LST_Prediction.tif file exists")

# Navigation back to homepage
st.markdown("---")
st.markdown("### 🧭 Navigation")
st.info("💡 Use the sidebar navigation to explore other analysis pages or return to the homepage")
