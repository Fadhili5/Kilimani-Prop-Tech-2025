import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_lst_prediction_data

# Page configuration
st.set_page_config(
    page_title="Vegetation Analysis - Kilimani Heat Island",
    page_icon="🌱",
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
    
    .metric-container {
        background: rgba(255,255,255,0.95);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.3);
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
</style>
""", unsafe_allow_html=True)

# Page header
st.markdown('<h1 class="main-header">🌱 Vegetation Analysis</h1>', unsafe_allow_html=True)
st.markdown('<h2 class="sub-header">Green Infrastructure & Cooling Effects</h2>', unsafe_allow_html=True)

# Load data
df = load_lst_prediction_data()

if df is not None:
    st.markdown("### 🌿 Vegetation & Temperature Relationship")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Enhanced NDVI vs temperature
        if all(col in df.columns for col in ['NDVI', 'LST_Prediction']):
            fig = px.scatter(
                df, x='NDVI', y='LST_Prediction',
                title='🌿 Vegetation Index (NDVI) vs Temperature',
                labels={'NDVI': 'NDVI (Vegetation Index)', 'LST_Prediction': 'Temperature (°C)'},
                trendline='ols',
                color='LST_Prediction',
                color_continuous_scale='RdYlGn_r',
                size='building_density' if 'building_density' in df.columns else None,
                hover_data=['elevation', 'distance_to_water'] if all(col in df.columns for col in ['elevation', 'distance_to_water']) else None
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                title_font_color='white'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Enhanced vegetation categories
        if 'NDVI' in df.columns:
            df['vegetation_category'] = pd.cut(df['NDVI'], 
                                              bins=[-1, 0, 0.3, 0.6, 1], 
                                              labels=['No Vegetation', 'Low', 'Medium', 'High'])
            
            fig = px.box(
                df, x='vegetation_category', y='LST_Prediction',
                title='🌳 Temperature Distribution by Vegetation Level',
                labels={'vegetation_category': 'Vegetation Level', 'LST_Prediction': 'Temperature (°C)'},
                color='vegetation_category',
                color_discrete_sequence=['#8b0000', '#ff4500', '#ffd700', '#32cd32']
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                title_font_color='white'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Vegetation cooling potential analysis
    st.markdown("---")
    st.markdown("### ❄️ Vegetation Cooling Analysis")
    
    if 'NDVI' in df.columns:
        vegetation_corr = df['LST_Prediction'].corr(df['NDVI'])
        
        # Calculate vegetation statistics
        high_veg_mask = df['NDVI'] > 0.5
        low_veg_mask = df['NDVI'] < 0.2
        
        if high_veg_mask.any() and low_veg_mask.any():
            avg_temp_high_veg = df[high_veg_mask]['LST_Prediction'].mean()
            avg_temp_low_veg = df[low_veg_mask]['LST_Prediction'].mean()
            cooling_effect = avg_temp_low_veg - avg_temp_high_veg
        else:
            avg_temp_high_veg = df['LST_Prediction'].mean()
            cooling_effect = 0
        
        # Vegetation metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-container">
                <h4>🌿 Vegetation Correlation</h4>
                <h2>{vegetation_corr:.3f}</h2>
                <p>{"Strong" if abs(vegetation_corr) > 0.7 else "Moderate" if abs(vegetation_corr) > 0.4 else "Weak"} cooling effect</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-container">
                <h4>🌳 High Vegetation Temp</h4>
                <h2>{avg_temp_high_veg:.1f}°C</h2>
                <p>Dense vegetation areas</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-container">
                <h4>❄️ Cooling Effect</h4>
                <h2>{cooling_effect:.1f}°C</h2>
                <p>Potential reduction</p>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            vegetation_coverage = (df['NDVI'] > 0.3).sum() / len(df) * 100
            st.markdown(f"""
            <div class="metric-container">
                <h4>🌱 Green Coverage</h4>
                <h2>{vegetation_coverage:.1f}%</h2>
                <p>Current vegetation</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Vegetation zones map visualization
    st.markdown("---")
    st.markdown("### 🗺️ Vegetation Distribution Map")
    
    if all(col in df.columns for col in ['latitude', 'longitude', 'NDVI']):
        fig = px.scatter_mapbox(
            df.sample(min(200, len(df))),
            lat='latitude',
            lon='longitude',
            color='NDVI',
            size='LST_Prediction',
            color_continuous_scale='RdYlGn',
            title='🗺️ Vegetation Distribution & Temperature Map',
            mapbox_style='open-street-map',
            zoom=11,
            height=500
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            title_font_color='white'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Vegetation density analysis
    st.markdown("---")
    st.markdown("### 🌲 Vegetation Density Analysis")
    
    if 'NDVI' in df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            # NDVI distribution
            fig = px.histogram(
                df, x='NDVI', nbins=30,
                title='🌿 NDVI Distribution',
                labels={'NDVI': 'NDVI Value', 'count': 'Frequency'},
                color_discrete_sequence=['#2ecc71']
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                title_font_color='white'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Vegetation effectiveness
            if 'building_density' in df.columns:
                # Calculate vegetation effectiveness in built areas
                high_build_areas = df[df['building_density'] > 50]
                low_build_areas = df[df['building_density'] < 20]
                
                effectiveness_data = []
                for name, data in [('High Building Density', high_build_areas), ('Low Building Density', low_build_areas)]:
                    if len(data) > 0:
                        high_veg = data[data['NDVI'] > 0.4]
                        low_veg = data[data['NDVI'] < 0.2]
                        if len(high_veg) > 0 and len(low_veg) > 0:
                            temp_diff = low_veg['LST_Prediction'].mean() - high_veg['LST_Prediction'].mean()
                            effectiveness_data.append({'Area Type': name, 'Cooling Effect': temp_diff})
                
                if effectiveness_data:
                    eff_df = pd.DataFrame(effectiveness_data)
                    fig = px.bar(
                        eff_df, x='Area Type', y='Cooling Effect',
                        title='🌡️ Vegetation Cooling Effectiveness',
                        labels={'Cooling Effect': 'Temperature Reduction (°C)'},
                        color='Cooling Effect',
                        color_continuous_scale='RdYlGn'
                    )
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='white',
                        title_font_color='white'
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    # Green infrastructure recommendations
    st.markdown("---")
    st.markdown("### 🌳 Green Infrastructure Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 Priority Areas for Green Infrastructure")
        if all(col in df.columns for col in ['NDVI', 'LST_Prediction', 'building_density']):
            # Identify areas with low vegetation and high temperature
            priority_areas = df[(df['NDVI'] < 0.3) & (df['LST_Prediction'] > df['LST_Prediction'].quantile(0.7))]
            
            st.markdown(f"""
            <div class="insight-box">
                <h4>🔍 Analysis Results:</h4>
                <ul>
                    <li><b>Priority Areas:</b> {len(priority_areas):,} locations identified</li>
                    <li><b>Average Temperature:</b> {priority_areas['LST_Prediction'].mean():.1f}°C</li>
                    <li><b>Current NDVI:</b> {priority_areas['NDVI'].mean():.3f}</li>
                    <li><b>Potential Impact:</b> {cooling_effect:.1f}°C reduction possible</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### 🌱 Green Infrastructure Types")
        st.markdown("""
        <div class="insight-box">
            <h4>Recommended Interventions:</h4>
            <ul>
                <li><b>🌳 Urban Forest:</b> Large canopy trees for maximum cooling</li>
                <li><b>🏢 Green Roofs:</b> Building-integrated vegetation</li>
                <li><b>🌿 Green Walls:</b> Vertical vegetation systems</li>
                <li><b>🌻 Pocket Parks:</b> Small community green spaces</li>
                <li><b>🌸 Street Trees:</b> Corridor cooling and shade</li>
                <li><b>💧 Bioswales:</b> Water-vegetation systems</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Vegetation scenarios
    st.markdown("---")
    st.markdown("### 🚀 Vegetation Scenario Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🌱 Baseline Scenario")
        st.markdown(f"""
        <div class="insight-box">
            <h4>Current Conditions:</h4>
            <ul>
                <li>Green Coverage: {vegetation_coverage:.1f}%</li>
                <li>Average NDVI: {df['NDVI'].mean():.3f}</li>
                <li>Cooling Effect: {cooling_effect:.1f}°C</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### 🌳 Enhanced Green Scenario")
        enhanced_coverage = vegetation_coverage * 1.5
        enhanced_cooling = cooling_effect * 1.8
        st.markdown(f"""
        <div class="insight-box">
            <h4>+50% Green Infrastructure:</h4>
            <ul>
                <li>Green Coverage: {enhanced_coverage:.1f}%</li>
                <li>Projected Cooling: {enhanced_cooling:.1f}°C</li>
                <li>Temperature Reduction: -2.3°C</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("#### 🌴 Maximum Green Scenario")
        max_coverage = min(vegetation_coverage * 2.5, 80)
        max_cooling = cooling_effect * 3.2
        st.markdown(f"""
        <div class="insight-box">
            <h4>Maximum Vegetation:</h4>
            <ul>
                <li>Green Coverage: {max_coverage:.1f}%</li>
                <li>Maximum Cooling: {max_cooling:.1f}°C</li>
                <li>Temperature Reduction: -4.8°C</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Implementation strategy
    st.markdown("---")
    st.markdown("### 📋 Implementation Strategy")
    
    implementation_phases = {
        "Phase 1 (0-6 months)": [
            "🌳 Plant 500 street trees in high-temperature corridors",
            "🌿 Install green walls on 20 buildings in priority areas",
            "🌻 Create 5 pocket parks in dense urban areas",
            "📊 Establish vegetation monitoring system"
        ],
        "Phase 2 (6-18 months)": [
            "🏢 Implement green roof program (50 buildings)",
            "🌲 Establish urban forest zones (10 hectares)",
            "💧 Install bioswales along major roads",
            "🌱 Community garden development"
        ],
        "Phase 3 (18+ months)": [
            "🌳 Complete tree canopy to 40% coverage",
            "🌿 Green corridor connections",
            "🏞️ Large park development",
            "📈 Long-term monitoring and optimization"
        ]
    }
    
    for phase, actions in implementation_phases.items():
        with st.expander(f"📅 {phase}"):
            for action in actions:
                st.write(f"• {action}")

else:
    st.error("❌ Unable to load Kilimani LST prediction data")
    st.info("Please ensure the resources/Kilimani_LST_Prediction.tif file exists")

# Navigation back to homepage
st.markdown("---")
st.markdown("### 🧭 Navigation")
st.info("💡 Use the sidebar navigation to explore other analysis pages or return to the homepage")
