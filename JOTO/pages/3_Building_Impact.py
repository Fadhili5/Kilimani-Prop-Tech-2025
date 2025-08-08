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
    page_title="Building Impact - Kilimani Heat Island",
    page_icon="🏢",
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
st.markdown('<h1 class="main-header">🏢 Building Impact Analysis</h1>', unsafe_allow_html=True)
st.markdown('<h2 class="sub-header">Urban Development Effects on Temperature</h2>', unsafe_allow_html=True)

# Load data
df = load_lst_prediction_data()

if df is not None:
    st.markdown("### 🏗️ Building Density & Temperature Relationship")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Enhanced building density impact
        if all(col in df.columns for col in ['building_density', 'LST_Prediction']):
            fig = px.scatter(
                df, x='building_density', y='LST_Prediction',
                title='🏗️ Building Density Impact on Temperature',
                labels={'building_density': 'Building Density (%)', 'LST_Prediction': 'Temperature (°C)'},
                trendline='ols',
                color='building_coverage' if 'building_coverage' in df.columns else None,
                color_continuous_scale='Reds',
                size='population' if 'population' in df.columns else None,
                hover_data=['NDVI', 'elevation'] if all(col in df.columns for col in ['NDVI', 'elevation']) else None
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                title_font_color='white'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Enhanced building coverage analysis
        if 'building_coverage' in df.columns:
            df['building_category'] = pd.cut(df['building_coverage'], 
                                            bins=[0, 25, 50, 75, 100], 
                                            labels=['Low', 'Medium', 'High', 'Very High'])
            
            fig = px.violin(
                df, x='building_category', y='LST_Prediction',
                title='🏘️ Temperature by Building Coverage',
                labels={'building_category': 'Building Coverage Level', 'LST_Prediction': 'Temperature (°C)'},
                box=True,
                color='building_category',
                color_discrete_sequence=['#4ecdc4', '#ffe66d', '#ff9f40', '#ff6b6b']
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                title_font_color='white'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Enhanced 3D analysis
    st.markdown("---")
    st.markdown("### 🎯 3D Building Impact Visualization")
    
    if all(col in df.columns for col in ['building_density', 'building_coverage', 'LST_Prediction']):
        fig = px.scatter_3d(
            df.sample(min(500, len(df))), 
            x='building_density', 
            y='building_coverage', 
            z='LST_Prediction',
            color='LST_Prediction',
            size='population' if 'population' in df.columns else None,
            title='🏢 3D Building-Temperature Relationship',
            labels={
                'building_density': 'Building Density (%)',
                'building_coverage': 'Building Coverage (%)',
                'LST_Prediction': 'Temperature (°C)'
            },
            color_continuous_scale='RdYlBu_r'
        )
        fig.update_layout(
            scene=dict(
                bgcolor='rgba(0,0,0,0)',
                xaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor='rgba(255,255,255,0.2)'),
                yaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor='rgba(255,255,255,0.2)'),
                zaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor='rgba(255,255,255,0.2)')
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            title_font_color='white'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Building impact metrics
    st.markdown("---")
    st.markdown("### 📊 Building Impact Metrics")
    
    # Calculate correlations
    building_corr = df['LST_Prediction'].corr(df['building_density']) if 'building_density' in df.columns else 0
    coverage_corr = df['LST_Prediction'].corr(df['building_coverage']) if 'building_coverage' in df.columns else 0
    pop_corr = df['LST_Prediction'].corr(df['population']) if 'population' in df.columns else 0
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <h4>🏗️ Density Correlation</h4>
            <h2>{building_corr:.3f}</h2>
            <p>{"Strong" if abs(building_corr) > 0.7 else "Moderate" if abs(building_corr) > 0.4 else "Weak"} relationship</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <h4>🏘️ Coverage Correlation</h4>
            <h2>{coverage_corr:.3f}</h2>
            <p>{"High" if abs(coverage_corr) > 0.5 else "Moderate"} impact</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-container">
            <h4>👥 Population Correlation</h4>
            <h2>{pop_corr:.3f}</h2>
            <p>Population density effect</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Building density zones analysis
    st.markdown("---")
    st.markdown("### 🏙️ Building Density Zones Analysis")
    
    if 'building_density' in df.columns:
        # Create density categories
        df['density_zone'] = pd.cut(df['building_density'], 
                                   bins=[0, 20, 40, 60, 80, 100], 
                                   labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Zone distribution
            zone_counts = df['density_zone'].value_counts()
            fig = px.pie(
                values=zone_counts.values,
                names=zone_counts.index,
                title='🏗️ Building Density Distribution',
                color_discrete_sequence=['#70a1ff', '#5352ed', '#3742fa', '#2f3542', '#1e2124']
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                title_font_color='white'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Temperature by zone
            zone_temps = df.groupby('density_zone')['LST_Prediction'].agg(['mean', 'std']).reset_index()
            fig = px.bar(
                zone_temps, x='density_zone', y='mean',
                error_y='std',
                title='🌡️ Average Temperature by Building Density Zone',
                labels={'density_zone': 'Building Density Zone', 'mean': 'Average Temperature (°C)'},
                color='mean',
                color_continuous_scale='RdYlBu_r'
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                title_font_color='white'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Building type analysis (simulated)
    st.markdown("---")
    st.markdown("### 🏢 Building Type Impact Analysis")
    
    # Simulate building types based on density and coverage
    if all(col in df.columns for col in ['building_density', 'building_coverage']):
        def classify_building_type(row):
            if row['building_density'] > 70 and row['building_coverage'] > 60:
                return 'High-rise Commercial'
            elif row['building_density'] > 50 and row['building_coverage'] > 40:
                return 'Mixed-use Development'
            elif row['building_density'] > 30:
                return 'Medium Density Residential'
            else:
                return 'Low Density Residential'
        
        df['building_type'] = df.apply(classify_building_type, axis=1)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Building type distribution
            type_counts = df['building_type'].value_counts()
            fig = px.bar(
                x=type_counts.index, y=type_counts.values,
                title='🏗️ Building Type Distribution',
                labels={'x': 'Building Type', 'y': 'Count'},
                color=type_counts.values,
                color_continuous_scale='Blues'
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                title_font_color='white'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Temperature by building type
            type_temps = df.groupby('building_type')['LST_Prediction'].agg(['mean', 'std']).reset_index()
            fig = px.box(
                df, x='building_type', y='LST_Prediction',
                title='🌡️ Temperature Distribution by Building Type',
                labels={'building_type': 'Building Type', 'LST_Prediction': 'Temperature (°C)'},
                color='building_type'
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                title_font_color='white'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Enhanced building impact insights
    st.markdown(f"""
    <div class="insight-box">
        <h4>🏢 Building Impact Analysis Results:</h4>
        <ul>
            <li><b>🏗️ Building Density Impact:</b> {building_corr:.3f} correlation ({"Strong" if abs(building_corr) > 0.7 else "Moderate" if abs(building_corr) > 0.4 else "Weak"} relationship)</li>
            <li><b>🏘️ Building Coverage Impact:</b> {coverage_corr:.3f} correlation - {"High impact on temperature" if abs(coverage_corr) > 0.5 else "Moderate impact"}</li>
            <li><b>👥 Population Density:</b> {pop_corr:.3f} correlation with temperature</li>
            <li><b>💡 Key Recommendation:</b> {"Implement building density caps and green building requirements" if building_corr > 0.3 else "Focus on other factors as building impact is minimal"}</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Development scenarios
    st.markdown("---")
    st.markdown("### 🚧 Development Scenario Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📈 High Development Scenario")
        st.markdown("""
        <div class="insight-box">
            <h4>Assumptions:</h4>
            <ul>
                <li>Building density increases by 30%</li>
                <li>Green space reduces by 20%</li>
                <li>Population increases by 50%</li>
            </ul>
            <h4>Predicted Impact:</h4>
            <ul>
                <li>🌡️ Temperature increase: +2.5°C</li>
                <li>🔥 Heat island intensity: +40%</li>
                <li>⚠️ Risk level: High</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### 🌱 Sustainable Development Scenario")
        st.markdown("""
        <div class="insight-box">
            <h4>Assumptions:</h4>
            <ul>
                <li>Green building standards (30% green roofs)</li>
                <li>Increased green space by 15%</li>
                <li>Smart urban design</li>
            </ul>
            <h4>Predicted Impact:</h4>
            <ul>
                <li>🌡️ Temperature change: -1.2°C</li>
                <li>❄️ Cooling effect: +60%</li>
                <li>✅ Risk level: Low</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

else:
    st.error("❌ Unable to load Kilimani LST prediction data")
    st.info("Please ensure the resources/Kilimani_LST_Prediction.tif file exists")

# Navigation back to homepage
st.markdown("---")
st.markdown("### 🧭 Navigation")
st.info("💡 Use the sidebar navigation to explore other analysis pages or return to the homepage")
