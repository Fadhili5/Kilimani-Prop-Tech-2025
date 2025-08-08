import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_lst_prediction_data
from styles import inject_css

inject_css()

# Page configuration
st.set_page_config(
    page_title="Spatial Analysis - Kilimani Heat Island",
    page_icon="🗺️",
    layout="wide"
)

# Page header
st.markdown('<h1 class="main-header">🗺️ Spatial Analysis</h1>', unsafe_allow_html=True)
st.markdown('<h2 class="sub-header">Interactive Mapping & Spatial Correlations</h2>', unsafe_allow_html=True)

# Load data
df = load_lst_prediction_data()

if df is not None:
    # Enhanced map controls
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("🎮 Map Controls")
        map_variable = st.selectbox(
            "Display Variable",
            ['LST_Prediction', 'NDVI', 'building_density', 'building_coverage'],
            help="Select which variable to visualize on the map"
        )
        
        color_scale = st.selectbox(
            "Color Scale",
            ['Viridis', 'Plasma', 'Hot', 'Cool', 'RdYlBu_r']
        )
        
        temp_threshold = st.slider("Heat Island Threshold (°C)", 25, 45, 35)
        show_heat_islands = st.checkbox("🔥 Highlight Heat Islands", True)
        show_building_footprints = st.checkbox("🏢 Show Buildings", False)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col1:
        # Enhanced folium map
        center_lat = df['latitude'].mean()
        center_lon = df['longitude'].mean()
        
        # Create map with modern tiles
        m = folium.Map(
            location=[center_lat, center_lon], 
            zoom_start=13,
            tiles='CartoDB dark_matter'  # Modern dark theme
        )
        
        # Add alternative tile layers
        folium.TileLayer('CartoDB positron').add_to(m)
        folium.TileLayer('OpenStreetMap').add_to(m)
        
        # Add heatmap layer with enhanced styling
        if map_variable in df.columns:
            # Sample data for performance
            sample_df = df.sample(min(300, len(df)))
            
            # Normalize values for color mapping
            min_val = sample_df[map_variable].min()
            max_val = sample_df[map_variable].max()
            
            for idx, row in sample_df.iterrows():
                # Color based on temperature threshold
                if show_heat_islands and row['LST_Prediction'] > temp_threshold:
                    color = '#ff4757'  # Hot red
                    radius = 8
                elif row['LST_Prediction'] > 32:
                    color = '#ffa502'  # Orange
                    radius = 6
                else:
                    color = '#2ed573'  # Cool green
                    radius = 4
                
                # Opacity based on variable value
                norm_val = (row[map_variable] - min_val) / (max_val - min_val)
                opacity = 0.4 + (norm_val * 0.6)
                
                # Enhanced popup with multiple variables
                popup_text = f"""
                <div style="font-family: 'Inter', sans-serif; padding: 10px; min-width: 200px;">
                    <h4 style="margin: 0 0 10px 0; color: #2c3e50;">📍 Location Data</h4>
                    <p><strong>🌡️ LST:</strong> {row['LST_Prediction']:.1f}°C</p>
                    <p><strong>🌱 NDVI:</strong> {row['NDVI']:.3f}</p>
                    <p><strong>🏢 Building Density:</strong> {row['building_density']:.1f}%</p>
                    <p><strong>👥 Population:</strong> {row['population']:.0f}</p>
                    <p><strong>📍 Coordinates:</strong><br>
                    Lat: {row['latitude']:.4f}<br>
                    Lon: {row['longitude']:.4f}</p>
                </div>
                """
                
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=radius,
                    popup=folium.Popup(popup_text, max_width=250),
                    color='white',
                    weight=1,
                    fillColor=color,
                    fillOpacity=opacity,
                    tooltip=f"LST: {row['LST_Prediction']:.1f}°C"
                ).add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Display enhanced map
        map_data = st_folium(m, width=700, height=500, returned_objects=["last_object_clicked"])
    
    # Show clicked point details
    if map_data['last_object_clicked']:
        clicked_lat = map_data['last_object_clicked']['lat']
        clicked_lng = map_data['last_object_clicked']['lng']
        st.success(f"📍 Selected coordinates: {clicked_lat:.4f}, {clicked_lng:.4f}")
    
    # Correlation Analysis
    st.markdown("---")
    st.markdown("### 📊 Spatial Correlation Analysis")
    
    # Calculate correlations
    numeric_cols = ['LST_Prediction', 'NDVI', 'building_density', 'building_coverage', 
                   'population', 'elevation', 'distance_to_water', 'albedo', 'rainfall', 'slope']
    available_cols = [col for col in numeric_cols if col in df.columns]
    
    if len(available_cols) > 1:
        corr_matrix = df[available_cols].corr()
        fig = px.imshow(
            corr_matrix,
            title="🔗 Correlation Matrix - Urban Heat Factors",
            color_continuous_scale='RdBu_r',
            aspect='auto'
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            title_font_color='white'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Enhanced insights
        st.markdown("""
        <div class="insight-box">
            <h4>🔍 Key Insights from Correlations:</h4>
            <ul>
                <li><b>Building Density vs Temperature:</b> Higher building density typically increases surface temperature</li>
                <li><b>NDVI vs Temperature:</b> More vegetation (higher NDVI) generally reduces temperature</li>
                <li><b>Population vs Heat:</b> Areas with higher population density often show elevated temperatures</li>
                <li><b>Distance to Water:</b> Proximity to water bodies provides cooling effect</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Enhanced spatial statistics
    st.markdown("---")
    st.markdown("### 📈 Spatial Patterns & Relationships")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Enhanced temperature vs distance to water
        if all(col in df.columns for col in ['distance_to_water', 'LST_Prediction', 'NDVI']):
            fig = px.scatter(
                df, x='distance_to_water', y='LST_Prediction',
                title='🌊 Temperature vs Distance to Water Bodies',
                labels={'distance_to_water': 'Distance to Water (m)', 'LST_Prediction': 'Temperature (°C)'},
                trendline='ols',
                color='NDVI',
                color_continuous_scale='RdYlGn',
                hover_data=['building_density', 'population']
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                title_font_color='white'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Enhanced elevation vs temperature
        if all(col in df.columns for col in ['elevation', 'LST_Prediction', 'building_coverage']):
            fig = px.scatter(
                df, x='elevation', y='LST_Prediction',
                title='⛰️ Temperature vs Elevation',
                labels={'elevation': 'Elevation (m)', 'LST_Prediction': 'Temperature (°C)'},
                trendline='ols',
                color='building_coverage',
                color_continuous_scale='Reds',
                hover_data=['NDVI', 'population']
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                title_font_color='white'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # 3D Spatial Analysis
    st.markdown("---")
    st.markdown("### 🎯 3D Spatial Visualization")
    
    if all(col in df.columns for col in ['latitude', 'longitude', 'LST_Prediction']):
        # Create 3D surface plot
        sample_df = df.sample(min(500, len(df)))
        
        fig = px.scatter_3d(
            sample_df, 
            x='longitude', 
            y='latitude', 
            z='LST_Prediction',
            color='LST_Prediction',
            size='building_density' if 'building_density' in df.columns else None,
            title='🌍 3D Temperature Distribution',
            labels={
                'longitude': 'Longitude',
                'latitude': 'Latitude',
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
    
    # Spatial statistics summary
    st.markdown("---")
    st.markdown("### 📋 Spatial Statistics Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        hot_spots = len(df[df['LST_Prediction'] > temp_threshold])
        st.metric("🔥 Hot Spots", f"{hot_spots:,}", f"{(hot_spots/len(df)*100):.1f}% of area")
    
    with col2:
        cool_spots = len(df[df['LST_Prediction'] < 25])
        st.metric("❄️ Cool Areas", f"{cool_spots:,}", f"{(cool_spots/len(df)*100):.1f}% of area")
    
    with col3:
        temp_range = df['LST_Prediction'].max() - df['LST_Prediction'].min()
        st.metric("📏 Temperature Range", f"{temp_range:.1f}°C", "Max - Min")
    
    with col4:
        temp_std = df['LST_Prediction'].std()
        st.metric("📊 Temperature Variation", f"{temp_std:.1f}°C", "Standard Deviation")

else:
    st.error("❌ Unable to load Kilimani LST prediction data")
    st.info("Please ensure the resources/Kilimani_LST_Prediction.tif file exists")

# Navigation back to homepage
st.markdown("---")
st.markdown("### 🧭 Navigation")
st.info("💡 Use the sidebar navigation to explore other analysis pages or return to the homepage")
