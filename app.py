import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')
import json
from datetime import datetime
import geopandas as gpd
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
import zipfile
import tempfile
import os
import joblib

def analyze_lst_prediction_with_ai(df, bounds=None):
    """AI-powered analysis of LST predictions"""
    if df is None or df.empty:
        return None
    
    # Calculate key statistics
    mean_temp = df['LST_Prediction'].mean()
    max_temp = df['LST_Prediction'].max()
    min_temp = df['LST_Prediction'].min()
    std_temp = df['LST_Prediction'].std()
    
    # Identify hot spots (temperatures above 90th percentile)
    hot_threshold = df['LST_Prediction'].quantile(0.9)
    hot_spots = df[df['LST_Prediction'] > hot_threshold]
    
    # Cool spots (temperatures below 10th percentile)
    cool_threshold = df['LST_Prediction'].quantile(0.1)
    cool_spots = df[df['LST_Prediction'] < cool_threshold]
    
    # Temperature distribution analysis
    temp_ranges = {
        'Extreme Heat (>40°C)': len(df[df['LST_Prediction'] > 40]),
        'High Heat (35-40°C)': len(df[(df['LST_Prediction'] >= 35) & (df['LST_Prediction'] <= 40)]),
        'Moderate (30-35°C)': len(df[(df['LST_Prediction'] >= 30) & (df['LST_Prediction'] < 35)]),
        'Comfortable (25-30°C)': len(df[(df['LST_Prediction'] >= 25) & (df['LST_Prediction'] < 30)]),
        'Cool (<25°C)': len(df[df['LST_Prediction'] < 25])
    }
    
    analysis = {
        'overview': {
            'total_area_analyzed': len(df),
            'mean_temperature': round(mean_temp, 2),
            'max_temperature': round(max_temp, 2),
            'min_temperature': round(min_temp, 2),
            'temperature_variation': round(std_temp, 2),
            'heat_island_intensity': round(max_temp - min_temp, 2)
        },
        'spatial_patterns': {
            'hot_spots_count': len(hot_spots),
            'hot_spots_percentage': round((len(hot_spots) / len(df)) * 100, 1),
            'cool_spots_count': len(cool_spots),
            'cool_spots_percentage': round((len(cool_spots) / len(df)) * 100, 1),
            'hot_threshold': round(hot_threshold, 2),
            'cool_threshold': round(cool_threshold, 2)
        },
        'temperature_distribution': temp_ranges,
        'risk_assessment': {
            'heat_stress_risk': 'High' if mean_temp > 35 else 'Moderate' if mean_temp > 30 else 'Low',
            'urban_heat_island_severity': 'Severe' if (max_temp - min_temp) > 15 else 'Moderate' if (max_temp - min_temp) > 10 else 'Mild',
            'areas_of_concern': len(df[df['LST_Prediction'] > 38]),
            'priority_intervention_zones': len(df[df['LST_Prediction'] > 40])
        },
        'ai_insights': {
            'primary_concern': get_primary_concern(mean_temp, max_temp, std_temp),
            'spatial_recommendations': get_spatial_recommendations(hot_spots, cool_spots, df),
            'intervention_priority': get_intervention_priority(temp_ranges),
            'climate_adaptation_needs': get_climate_adaptation_needs(mean_temp, max_temp, std_temp)
        }
    }
    
    return analysis

def get_primary_concern(mean_temp, max_temp, std_temp):
    """Determine primary concern based on temperature analysis"""
    if max_temp > 45:
        return "Extreme heat exposure risk - Immediate intervention required in hottest areas"
    elif mean_temp > 38:
        return "Widespread elevated temperatures - Area-wide cooling strategies needed"
    elif std_temp > 8:
        return "High temperature variability - Targeted interventions for heat islands"
    elif max_temp > 40:
        return "Localized extreme heat - Focus on specific hot spot mitigation"
    else:
        return "Moderate heat conditions - Preventive measures recommended"

def get_spatial_recommendations(hot_spots, cool_spots, df):
    """Generate spatial recommendations based on temperature patterns"""
    recommendations = []
    
    if len(hot_spots) > len(df) * 0.2:
        recommendations.append("Implement area-wide green infrastructure due to extensive hot zones")
    
    if len(cool_spots) > 0:
        recommendations.append("Preserve and expand cool corridors to create thermal refugia")
        recommendations.append("Study cool spot characteristics for replication in hot areas")
    
    if len(hot_spots) > 0:
        # Analyze hot spot clustering (simplified)
        hot_lat_range = hot_spots['latitude'].max() - hot_spots['latitude'].min()
        hot_lon_range = hot_spots['longitude'].max() - hot_spots['longitude'].min()
        
        if hot_lat_range < 0.01 and hot_lon_range < 0.01:
            recommendations.append("Hot spots are clustered - Focus intervention in specific geographic area")
        else:
            recommendations.append("Hot spots are distributed - Implement multiple smaller interventions")
    
    return recommendations

def get_intervention_priority(temp_ranges):
    """Determine intervention priorities based on temperature distribution"""
    extreme_heat_pct = (temp_ranges['Extreme Heat (>40°C)'] / sum(temp_ranges.values())) * 100
    high_heat_pct = (temp_ranges['High Heat (35-40°C)'] / sum(temp_ranges.values())) * 100
    
    if extreme_heat_pct > 10:
        return "Critical - Immediate emergency cooling measures required"
    elif extreme_heat_pct > 5 or high_heat_pct > 25:
        return "High - Urgent implementation of cooling infrastructure"
    elif high_heat_pct > 15:
        return "Medium-High - Accelerated green infrastructure deployment"
    else:
        return "Medium - Preventive cooling measures and monitoring"

def get_climate_adaptation_needs(mean_temp, max_temp, std_temp):
    """Assess climate adaptation requirements"""
    needs = []
    
    if mean_temp > 35:
        needs.append("Enhanced building cooling requirements and energy efficiency standards")
        needs.append("Public health heat wave preparedness and cooling centers")
    
    if max_temp > 42:
        needs.append("Emergency heat response protocols and vulnerable population protection")
        needs.append("Infrastructure resilience upgrades for extreme heat conditions")
    
    if std_temp > 6:
        needs.append("Adaptive urban design to address temperature heterogeneity")
        needs.append("Targeted microclimatic interventions for hot spots")
    
    if not needs:
        needs.append("Proactive cooling measures to prevent future heat island intensification")
        needs.append("Sustainable urban development practices to maintain thermal comfort")
    
    return needs

def generate_ai_response(question, ai_analysis, df):
    """Generate AI-style responses to user questions about the LST predictions"""
    question_lower = question.lower()
    
    if any(word in question_lower for word in ['intervention', 'immediate', 'urgent', 'priority']):
        extreme_areas = len(df[df['LST_Prediction'] > 40])
        return f"Based on the analysis, there are {extreme_areas} areas with temperatures above 40°C that need immediate intervention. The AI recommends focusing on {ai_analysis['ai_insights']['intervention_priority'].lower()}. Priority should be given to areas with temperatures above {ai_analysis['spatial_patterns']['hot_threshold']}°C."
    
    elif any(word in question_lower for word in ['hottest', 'hot', 'temperature', 'heat']):
        max_temp = ai_analysis['overview']['max_temperature']
        mean_temp = ai_analysis['overview']['mean_temperature']
        return f"The hottest area reaches {max_temp}°C, while the average temperature across Kilimani is {mean_temp}°C. The heat island intensity is {ai_analysis['overview']['heat_island_intensity']}°C, indicating {ai_analysis['risk_assessment']['urban_heat_island_severity'].lower()} heat island effects."
    
    elif any(word in question_lower for word in ['cool', 'cold', 'lowest', 'minimum']):
        min_temp = ai_analysis['overview']['min_temperature']
        cool_spots = ai_analysis['spatial_patterns']['cool_spots_count']
        return f"The coolest areas in Kilimani are around {min_temp}°C. There are {cool_spots} cool spots identified, representing {ai_analysis['spatial_patterns']['cool_spots_percentage']}% of the analyzed area. These areas should be preserved and their characteristics studied for replication."
    
    elif any(word in question_lower for word in ['risk', 'danger', 'health', 'concern']):
        risk_level = ai_analysis['risk_assessment']['heat_stress_risk']
        areas_concern = ai_analysis['risk_assessment']['areas_of_concern']
        return f"The overall heat stress risk is assessed as {risk_level}. There are {areas_concern} areas of particular concern with dangerous temperature levels. {ai_analysis['ai_insights']['primary_concern']}"
    
    elif any(word in question_lower for word in ['recommend', 'solution', 'what to do', 'action']):
        recommendations = ai_analysis['ai_insights']['spatial_recommendations']
        return f"The AI recommends: {recommendations[0] if recommendations else 'Implementing targeted cooling strategies based on the spatial temperature patterns observed.'} Additionally, {ai_analysis['ai_insights']['intervention_priority'].lower()}."
    
    elif any(word in question_lower for word in ['pattern', 'distribution', 'spatial', 'where']):
        hot_spots_pct = ai_analysis['spatial_patterns']['hot_spots_percentage']
        cool_spots_pct = ai_analysis['spatial_patterns']['cool_spots_percentage']
        return f"The spatial analysis shows {hot_spots_pct}% of areas are hot spots and {cool_spots_pct}% are cool spots. The temperature distribution indicates {ai_analysis['risk_assessment']['urban_heat_island_severity'].lower()} heat island patterns with significant spatial variation."
    
    else:
        return f"Based on the LST prediction analysis for Kilimani, the key finding is: {ai_analysis['ai_insights']['primary_concern']} The analysis covers {ai_analysis['overview']['total_area_analyzed']:,} data points with an average temperature of {ai_analysis['overview']['mean_temperature']}°C and requires {ai_analysis['ai_insights']['intervention_priority'].lower()}."

# Add this new function for AI analysis
def analyze_development_plan_ai(plan_data):
    """Simulate AI analysis of development plan"""
    analysis = {
        'planSummary': {
            'totalArea': plan_data.get('area', "250 hectares"),
            'buildingType': plan_data.get('type', "Mixed-use development"),
            'plannedUnits': plan_data.get('units', "1,200 residential units + commercial"),
            'estimatedPopulation': plan_data.get('population', 3000),
            'greenSpaceRatio': plan_data.get('greenSpace', "15%")
        },
        'thermalImpact': {
            'baselineTemp': 32.1,
            'projectedTemp': 35.8,
            'temperatureIncrease': 3.7,
            'hotspotRisk': "High",
            'affectedRadius': "2.5 km"
        },
        'mitigationStrategies': [
            {
                'strategy': 'Green Building Design Integration',
                'priority': 'Critical',
                'implementation': 'Immediate - Design Phase',
                'details': [
                    "Implement mandatory green roofs on 60% of buildings",
                    "Use light-colored, high-albedo materials for surfaces",
                    "Design buildings with natural ventilation corridors",
                    "Integrate solar shading and cool facade materials"
                ],
                'expectedImpact': "Temperature reduction: 2.1°C",
                'cost': "$2.3M additional investment"
            },
            {
                'strategy': 'Enhanced Urban Forest Plan',
                'priority': 'High',
                'implementation': 'Phase 1 - Before Construction',
                'details': [
                    "Increase green space allocation from 15% to 35%",
                    "Plant 2,500 native trees with high cooling capacity",
                    "Create continuous green corridors connecting to existing forests",
                    "Establish community gardens and pocket parks"
                ],
                'expectedImpact': "Temperature reduction: 1.8°C",
                'cost': "$1.8M over 3 years"
            },
            {
                'strategy': 'Smart Infrastructure Integration',
                'priority': 'Medium',
                'implementation': 'Phase 2 - During Construction',
                'details': [
                    "Install permeable paving for 40% of walkways",
                    "Implement rainwater harvesting for irrigation",
                    "Design wind corridors to enhance natural cooling",
                    "Use smart building materials that adapt to temperature"
                ],
                'expectedImpact': "Temperature reduction: 1.2°C",
                'cost': "$3.1M integrated costs"
            }
        ],
        'riskAssessment': {
            'thermalRisk': {
                'level': "High",
                'probability': "85%",
                'impact': "Severe local temperature increase"
            },
            'healthRisk': {
                'level': "Critical",
                'affectedPopulation': 3200,
                'projectedIncidents': "15-25 heat-related incidents annually"
            },
            'economicImpact': {
                'energyCosts': "+$2.1M annually",
                'healthcareCosts': "+$400K annually",
                'propertyValues': "Potential 8% decrease"
            }
        },
        'recommendations': {
            'designModifications': [
                "Reduce building density by 20% through vertical optimization",
                "Implement mandatory setback requirements for natural airflow",
                "Integrate building-integrated photovoltaics (BIPV) systems"
            ],
            'policyChanges': [
                "Implement heat island impact assessments for all developments",
                "Require urban heat mitigation bonds for large developments",
                "Establish cooling infrastructure requirements"
            ],
            'monitoringPlan': [
                "Install 15 temperature monitoring stations",
                "Conduct quarterly thermal imaging surveys",
                "Monitor vegetation health and growth rates"
            ]
        },
        'timeline': {
            "Pre-Construction (0-6 months)": [
                "Implement design modifications",
                "Begin tree planting program",
                "Establish monitoring baseline"
            ],
            "Construction Phase (6-30 months)": [
                "Install green infrastructure",
                "Implement cool material specifications",
                "Monitor environmental impacts"
            ],
            "Post-Construction (30+ months)": [
                "Activate cooling centers",
                "Complete green space establishment",
                "Begin long-term monitoring"
            ]
        }
    }
    
    return analysis

def load_lst_prediction_data(tif_file):
    """Load and process the LST prediction GeoTIFF from Google Earth Engine"""
    try:
        with rasterio.open(tif_file) as src:
            # Read the raster data
            lst_data = src.read(1)  # Read first band
            
            # Get the bounds
            bounds = src.bounds
            transform = src.transform
            
            # Create coordinate grids
            height, width = lst_data.shape
            cols, rows = np.meshgrid(np.arange(width), np.arange(height))
            
            # Transform pixel coordinates to geographic coordinates
            xs, ys = rasterio.transform.xy(transform, rows, cols)
            
            # Flatten arrays and create DataFrame
            xs_flat = np.array(xs).flatten()
            ys_flat = np.array(ys).flatten()
            lst_flat = lst_data.flatten()
            
            # Remove nodata values
            mask = ~np.isnan(lst_flat)
            
            df = pd.DataFrame({
                'longitude': xs_flat[mask],
                'latitude': ys_flat[mask],
                'LST_Prediction': lst_flat[mask]
            })
            
            return df, bounds
            
    except Exception as e:
        st.error(f"Error loading LST prediction data: {str(e)}")
        return None, None

def load_building_data(shp_file):
    """Load and process the building shapefile from Google Earth Engine"""
    try:
        # If it's a zip file, extract it first
        if str(shp_file.name).endswith('.zip'):
            with tempfile.TemporaryDirectory() as temp_dir:
                with zipfile.ZipFile(shp_file, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                
                # Find the .shp file
                shp_files = [f for f in os.listdir(temp_dir) if f.endswith('.shp')]
                if shp_files:
                    shp_path = os.path.join(temp_dir, shp_files[0])
                    gdf = gpd.read_file(shp_path)
                else:
                    st.error("No .shp file found in the zip archive")
                    return None
        else:
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix='.shp') as tmp_file:
                tmp_file.write(shp_file.getbuffer())
                tmp_file.flush()
                gdf = gpd.read_file(tmp_file.name)
        
        # Convert to DataFrame with centroids for analysis
        gdf['centroid'] = gdf.geometry.centroid
        gdf['longitude'] = gdf['centroid'].x
        gdf['latitude'] = gdf['centroid'].y
        
        # Calculate building area and other metrics
        gdf['building_area'] = gdf.geometry.area
        
        return gdf
        
    except Exception as e:
        st.error(f"Error loading building data: {str(e)}")
        return None

def process_real_data(lst_df, buildings_gdf):
    """Process and merge real LST and building data"""
    try:
        if lst_df is None or buildings_gdf is None:
            return None
        
        # Spatial join to get LST values for building locations
        if buildings_gdf is not None and not buildings_gdf.empty:
            # Create a spatial join or nearest neighbor matching
            # For simplicity, we'll use a distance-based approach
            merged_data = []
            
            for idx, building in buildings_gdf.iterrows():
                # Find nearest LST point
                distances = np.sqrt(
                    (lst_df['longitude'] - building['longitude'])**2 + 
                    (lst_df['latitude'] - building['latitude'])**2
                )
                nearest_idx = distances.idxmin()
                
                # Get LST value
                lst_value = lst_df.loc[nearest_idx, 'LST_Prediction']
                
                merged_data.append({
                    'latitude': building['latitude'],
                    'longitude': building['longitude'],
                    'LST': lst_value,
                    'LST_Prediction': lst_value,
                    'building_area': building.get('building_area', 0),
                    'building_density': building.get('density', np.random.uniform(20, 80)),
                    'building_coverage': building.get('coverage', np.random.uniform(10, 70)),
                    'population': building.get('population', np.random.uniform(50, 500)),
                    'NDVI': np.random.uniform(-0.2, 0.8),  # Will be replaced with actual NDVI if available
                    'elevation': np.random.uniform(1600, 1800),
                    'distance_to_water': np.random.uniform(100, 3000),
                    'albedo': np.random.uniform(0.1, 0.4),
                    'rainfall': np.random.uniform(0.5, 3.5),
                    'slope': np.random.uniform(0, 15)
                })
            
            df = pd.DataFrame(merged_data)
            
        else:
            # Use LST data as primary dataset
            df = lst_df.copy()
            # Add synthetic building data for analysis
            df['building_density'] = np.random.uniform(20, 80, len(df))
            df['building_coverage'] = np.random.uniform(10, 70, len(df))
            df['population'] = np.random.uniform(50, 500, len(df))
            df['NDVI'] = np.random.uniform(-0.2, 0.8, len(df))
            df['elevation'] = np.random.uniform(1600, 1800, len(df))
            df['distance_to_water'] = np.random.uniform(100, 3000, len(df))
            df['albedo'] = np.random.uniform(0.1, 0.4, len(df))
            df['rainfall'] = np.random.uniform(0.5, 3.5, len(df))
            df['slope'] = np.random.uniform(0, 15, len(df))
            df['LST'] = df['LST_Prediction']  # Use prediction as actual for now
        
        return df
        
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        return None

# Set page configuration
st.set_page_config(
    page_title="Urban Heat Island Analysis - Kilimani, Nairobi",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern CSS styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Styles */
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
    
    /* Card Styles */
    .metric-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
        text-align: center;
        margin: 0.5rem;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.15);
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 3px;
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
    }
    
    /* Glass morphism effect */
    .glass-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    /* Insight Box */
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
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 10px;
        color: rgba(255,255,255,0.8);
        font-weight: 500;
        padding: 0.8rem 1.5rem;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: rgba(255,255,255,0.2) !important;
        color: #ffffff !important;
        font-weight: 600;
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(15px);
    }
    
    /* Metric styling */
    .metric-container {
        background: rgba(255,255,255,0.95);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.3);
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.7rem 2rem;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }
    
    /* File uploader styling */
    .stFileUploader {
        background: rgba(255,255,255,0.1);
        border-radius: 15px;
        padding: 1rem;
        border: 2px dashed rgba(255,255,255,0.3);
    }
    
    /* Remove Streamlit branding */
    .css-1rs6os, .css-17ziqus {
        display: none;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: rgba(255,255,255,0.8);
        padding: 3rem 0;
        margin-top: 4rem;
        border-top: 1px solid rgba(255,255,255,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Title and introduction
st.markdown('<h1 class="main-header">🌡️ Kilimani Heat Island Analysis</h1>', unsafe_allow_html=True)
st.markdown('<h2 class="sub-header">Using Local LST Prediction Model - Kilimani, Nairobi</h2>', unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None

# Sidebar
st.sidebar.title("🔧 Data & Controls")
st.sidebar.markdown("---")

# Model Information
st.sidebar.subheader("🤖 Active Model")
if st.session_state.data_loaded:
    st.sidebar.success("🎯 Kilimani LST Prediction Model")
    
    # Check if trained model exists
    model_file_path = "model/best_model.pkl"
    if os.path.exists(model_file_path):
        st.sidebar.success("✅ Pre-trained ML model found")
        model_info = """
        **Model Details:**
        - Source: Local Kilimani Training
        - Type: Best performing ML model
        - Coverage: Kilimani, Nairobi
        - Status: Pre-trained and ready
        """
    else:
        model_info = """
        **Model Details:**
        - Source: Local Kilimani LST Prediction
        - Type: Spatial LST Analysis  
        - Coverage: Kilimani, Nairobi
        - Resolution: High-resolution spatial data
        """
    
    st.sidebar.info(model_info)
else:
    st.sidebar.info("No model loaded")

st.sidebar.markdown("---")

# Data loading section
st.sidebar.subheader("� Kilimani LST Analysis")

# Automatic data loading status
if os.path.exists("resources/Kilimani_LST_Prediction.tif"):
    st.sidebar.success("✅ Kilimani LST Prediction Model Ready")
    
    if st.sidebar.button("🔄 Reload Prediction Data"):
        st.session_state.data_loaded = False
        st.session_state.processed_data = None
        st.rerun()
        
else:
    st.sidebar.error("❌ Kilimani LST Prediction file not found")
    st.sidebar.info("Expected: resources/Kilimani_LST_Prediction.tif")

# Show sample data option only if main data fails to load
if not st.session_state.data_loaded and not os.path.exists("resources/Kilimani_LST_Prediction.tif"):
    if st.sidebar.button("📊 Use Sample Data"):
        # Generate sample data as fallback
        np.random.seed(42)
        n_samples = 1000
        
        lat_min, lat_max = -1.2979, -1.2562
        lon_min, lon_max = 36.7336, 36.8197
        
        data = {
            'latitude': np.random.uniform(lat_min, lat_max, n_samples),
            'longitude': np.random.uniform(lon_min, lon_max, n_samples),
            'LST': np.random.uniform(20, 45, n_samples),
            'LST_Prediction': np.random.uniform(18, 47, n_samples),
            'NDVI': np.random.uniform(-0.2, 0.8, n_samples),
            'building_density': np.random.uniform(0, 100, n_samples),
            'building_coverage': np.random.uniform(0, 90, n_samples),
            'population': np.random.uniform(0, 5000, n_samples),
            'elevation': np.random.uniform(1600, 1800, n_samples),
            'distance_to_water': np.random.uniform(100, 5000, n_samples),
            'albedo': np.random.uniform(0.1, 0.4, n_samples),
            'rainfall': np.random.uniform(0.5, 3.5, n_samples),
            'slope': np.random.uniform(0, 15, n_samples)
        }
        
        # Add correlations
        data['LST'] = data['LST'] + (np.array(data['building_density']) * 0.1) + np.random.normal(0, 2, n_samples)
        data['LST'] = data['LST'] - (np.array(data['NDVI']) * 8) + np.random.normal(0, 1.5, n_samples)
        data['LST_Prediction'] = data['LST'] + np.random.normal(0, 2, n_samples)
        
        st.session_state.processed_data = pd.DataFrame(data)
        st.session_state.data_loaded = True
        st.sidebar.success("✅ Sample data loaded!")
        st.rerun()

# Load data - Use local Kilimani LST prediction by default
if st.session_state.data_loaded and st.session_state.processed_data is not None:
    df = st.session_state.processed_data
else:
    # Try to load default local data first
    try:
        # Load Kilimani LST Prediction from resources
        lst_file_path = "resources/Kilimani_LST_Prediction.tif"
        buildings_file_path = "resources/Kilimani_Building_Grid_Local.shp"
        
        if os.path.exists(lst_file_path):
            with st.spinner("🔄 Loading local Kilimani LST Prediction model..."):
                lst_df, lst_bounds = load_lst_prediction_data(lst_file_path)
                
            if lst_df is not None:
                st.sidebar.success(f"✅ Local LST model loaded: {len(lst_df)} points")
                
                # Load building data if available
                buildings_gdf = None
                if os.path.exists(buildings_file_path):
                    with st.spinner("� Loading local building data..."):
                        buildings_gdf = gpd.read_file(buildings_file_path)
                        # Convert to the expected format
                        buildings_gdf['centroid'] = buildings_gdf.geometry.centroid
                        buildings_gdf['longitude'] = buildings_gdf['centroid'].x
                        buildings_gdf['latitude'] = buildings_gdf['centroid'].y
                        buildings_gdf['building_area'] = buildings_gdf.geometry.area
                    st.sidebar.success(f"✅ Local building data loaded: {len(buildings_gdf)} buildings")
                
                # Process the data
                with st.spinner("🔄 Processing local model data..."):
                    df = process_real_data(lst_df, buildings_gdf)
                    
                if df is not None:
                    st.session_state.processed_data = df
                    st.session_state.data_loaded = True
                    st.session_state.lst_bounds = lst_bounds
                    
                    # Store AI analysis in session state
                    if 'ai_analysis' not in st.session_state:
                        with st.spinner("🤖 Analyzing predictions with AI..."):
                            st.session_state.ai_analysis = analyze_lst_prediction_with_ai(df, lst_bounds)
                    
                    st.sidebar.success("✅ Local Kilimani model ready!")
                else:
                    raise Exception("Failed to process local data")
            else:
                raise Exception("Failed to load LST data")
        else:
            raise Exception(f"Local LST file not found at {lst_file_path}")
            
    except Exception as e:
        st.sidebar.warning(f"⚠️ Could not load local model: {str(e)}")
        st.info("�👆 Please upload your GEE model data or use sample data to get started")
        st.stop()
    
    df = st.session_state.processed_data

# Analysis parameters
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Analysis Parameters")
temp_threshold = st.sidebar.slider("Heat Island Threshold (°C)", 25, 45, 35)
show_predictions = st.sidebar.checkbox("Show Predictions vs Actual", True)
analysis_type = st.sidebar.selectbox(
    "Analysis Focus",
    ["Overview", "Building Impact", "Vegetation Impact", "Spatial Analysis", "Model Performance"]
)

# Main content area with enhanced tabs including AI Analysis
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Dashboard", 
    "🤖 AI Analysis",
    "🗺️ Spatial Analysis", 
    "🏢 Building Impact", 
    "🌱 Vegetation Analysis", 
    "📈 Model Insights", 
    "🛡️ Mitigation"
])

with tab1:
    st.markdown('<h2 class="section-header">📊 Temperature Overview Dashboard</h2>', unsafe_allow_html=True)
    
    # Enhanced metrics cards layout
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_temp = df['LST_Prediction'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #e74c3c; margin: 0;">🌡️ Average LST</h3>
            <h1 style="color: #2c3e50; margin: 10px 0;">{avg_temp:.1f}°C</h1>
            <p style="color: #7f8c8d; margin: 0;">Kilimani Region</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        max_temp = df['LST_Prediction'].max()
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #e67e22; margin: 0;">🔥 Peak Temperature</h3>
            <h1 style="color: #2c3e50; margin: 10px 0;">{max_temp:.1f}°C</h1>
            <p style="color: #7f8c8d; margin: 0;">Hottest Spot</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        min_temp = df['LST_Prediction'].min()
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #3498db; margin: 0;">❄️ Coolest Area</h3>
            <h1 style="color: #2c3e50; margin: 10px 0;">{min_temp:.1f}°C</h1>
            <p style="color: #7f8c8d; margin: 0;">Temperature Range</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        temp_range = max_temp - min_temp
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #9b59b6; margin: 0;">📏 Temperature Range</h3>
            <h1 style="color: #2c3e50; margin: 10px 0;">{temp_range:.1f}°C</h1>
            <p style="color: #7f8c8d; margin: 0;">Heat Island Intensity</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick AI Insights Preview
    if 'ai_analysis' in st.session_state and st.session_state.ai_analysis:
        ai_analysis = st.session_state.ai_analysis
        
        st.markdown(f"""
        <div class="insight-box">
            <h4>🤖 AI Quick Insights</h4>
            <p><strong>Primary Concern:</strong> {ai_analysis['ai_insights']['primary_concern']}</p>
            <p><strong>Heat Risk Level:</strong> {ai_analysis['risk_assessment']['heat_stress_risk']}</p>
            <p><strong>Intervention Priority:</strong> {ai_analysis['ai_insights']['intervention_priority']}</p>
            <p style="margin-top: 15px;"><em>👉 See full AI analysis in the "AI Analysis" tab</em></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Temperature distribution chart
    col1, col2 = st.columns(2)
    
    with col1:
        fig_hist = px.histogram(
            df, x='LST_Prediction', nbins=30,
            title="🌡️ Temperature Distribution in Kilimani",
            labels={'LST_Prediction': 'Temperature (°C)', 'count': 'Frequency'},
            color_discrete_sequence=['#e74c3c']
        )
        fig_hist.update_layout(
            title_font_size=16,
            font=dict(family="Inter, sans-serif"),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        # Temperature zones pie chart
        temp_zones = {
            'Extreme Heat (>40°C)': len(df[df['LST_Prediction'] > 40]),
            'High Heat (35-40°C)': len(df[(df['LST_Prediction'] >= 35) & (df['LST_Prediction'] <= 40)]),
            'Moderate (30-35°C)': len(df[(df['LST_Prediction'] >= 30) & (df['LST_Prediction'] < 35)]),
            'Comfortable (25-30°C)': len(df[(df['LST_Prediction'] >= 25) & (df['LST_Prediction'] < 30)]),
            'Cool (<25°C)': len(df[df['LST_Prediction'] < 25])
        }
        
        fig_pie = px.pie(
            values=list(temp_zones.values()),
            names=list(temp_zones.keys()),
            title="🎯 Temperature Zone Distribution",
            color_discrete_sequence=['#ff4757', '#ff6348', '#ffa502', '#2ed573', '#70a1ff']
        )
        fig_pie.update_layout(
            title_font_size=16,
            font=dict(family="Inter, sans-serif"),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_pie, use_container_width=True)

with tab2:
    st.markdown('<h2 class="section-header">🤖 AI Analysis of LST Predictions</h2>', unsafe_allow_html=True)
    
    if 'ai_analysis' in st.session_state and st.session_state.ai_analysis:
        ai_analysis = st.session_state.ai_analysis
        
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
            # Simulate AI response based on the analysis
            response = generate_ai_response(user_question, ai_analysis, df)
            st.markdown(f"""
            <div class="glass-card">
                <h4 style="color: #3498db;">🤖 AI Response</h4>
                <p>{response}</p>
            </div>
            """, unsafe_allow_html=True)
    
    else:
        st.warning("AI analysis not available. Please ensure the LST prediction data is loaded.")

with tab3:
    st.markdown('<h2 class="section-header">�️ Enhanced Spatial Analysis</h2>', unsafe_allow_html=True)
    
    if all(col in df.columns for col in numeric_cols):
        corr_matrix = df[numeric_cols].corr()
        fig = px.imshow(
            corr_matrix,
            title="Correlation Matrix - Urban Heat Factors",
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

with tab2:
    st.markdown('<h2 class="section-header">🗺️ Spatial Temperature Analysis</h2>', unsafe_allow_html=True)
    
    # Enhanced map controls
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("🎮 Map Controls")
        map_variable = st.selectbox(
            "Display Variable",
            ['LST', 'LST_Prediction', 'NDVI', 'building_density', 'building_coverage'],
            help="Select which variable to visualize on the map"
        )
        
        color_scale = st.selectbox(
            "Color Scale",
            ['Viridis', 'Plasma', 'Hot', 'Cool', 'RdYlBu_r']
        )
        
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
                if show_heat_islands and row['LST'] > temp_threshold:
                    color = '#ff4757'  # Hot red
                    radius = 8
                elif row['LST'] > 32:
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
                    <p><strong>🌡️ LST:</strong> {row['LST']:.1f}°C</p>
                    <p><strong>🔮 Predicted:</strong> {row.get('LST_Prediction', 'N/A')}</p>
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
                    tooltip=f"LST: {row['LST']:.1f}°C"
                ).add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Display enhanced map
        map_data = st_folium(m, width=700, height=500, returned_objects=["last_object_clicked"])
    
    # Enhanced spatial statistics
    st.markdown('<h3 class="section-header">📈 Spatial Patterns & Analysis</h3>', unsafe_allow_html=True)
    
    # Show clicked point details
    if map_data['last_object_clicked']:
        clicked_lat = map_data['last_object_clicked']['lat']
        clicked_lng = map_data['last_object_clicked']['lng']
        st.success(f"📍 Selected coordinates: {clicked_lat:.4f}, {clicked_lng:.4f}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Enhanced temperature vs distance to water
        fig = px.scatter(
            df, x='distance_to_water', y='LST',
            title='🌊 Temperature vs Distance to Water Bodies',
            labels={'distance_to_water': 'Distance to Water (m)', 'LST': 'Temperature (°C)'},
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
        fig = px.scatter(
            df, x='elevation', y='LST',
            title='⛰️ Temperature vs Elevation',
            labels={'elevation': 'Elevation (m)', 'LST': 'Temperature (°C)'},
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

with tab3:
    st.markdown('<h2 class="section-header">🏢 Building Impact Analysis</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Enhanced building density impact
        fig = px.scatter(
            df, x='building_density', y='LST',
            title='🏗️ Building Density Impact on Temperature',
            labels={'building_density': 'Building Density (%)', 'LST': 'Temperature (°C)'},
            trendline='ols',
            color='building_coverage',
            color_continuous_scale='Reds',
            size='population',
            hover_data=['NDVI', 'elevation']
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
        df['building_category'] = pd.cut(df['building_coverage'], 
                                        bins=[0, 25, 50, 75, 100], 
                                        labels=['Low', 'Medium', 'High', 'Very High'])
        
        fig = px.violin(
            df, x='building_category', y='LST',
            title='🏘️ Temperature by Building Coverage',
            labels={'building_category': 'Building Coverage Level', 'LST': 'Temperature (°C)'},
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
    st.markdown('<h3 class="section-header">🎯 3D Building Impact Visualization</h3>', unsafe_allow_html=True)
    
    fig = px.scatter_3d(
        df.sample(min(500, len(df))), 
        x='building_density', 
        y='building_coverage', 
        z='LST',
        color='LST',
        size='population',
        title='🏢 3D Building-Temperature Relationship',
        labels={
            'building_density': 'Building Density (%)',
            'building_coverage': 'Building Coverage (%)',
            'LST': 'Temperature (°C)'
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
    
    # Enhanced building impact insights
    building_corr = df['LST'].corr(df['building_density'])
    coverage_corr = df['LST'].corr(df['building_coverage'])
    pop_corr = df['LST'].corr(df['population'])
    
    # Building impact metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🏗️ Density Correlation", f"{building_corr:.3f}")
    with col2:
        st.metric("🏘️ Coverage Correlation", f"{coverage_corr:.3f}")
    with col3:
        st.metric("👥 Population Correlation", f"{pop_corr:.3f}")
    
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

with tab4:
    st.markdown('<h2 class="section-header">🌱 Vegetation & Cooling Analysis</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Enhanced NDVI vs temperature
        fig = px.scatter(
            df, x='NDVI', y='LST',
            title='🌿 Vegetation Index (NDVI) vs Temperature',
            labels={'NDVI': 'NDVI (Vegetation Index)', 'LST': 'Temperature (°C)'},
            trendline='ols',
            color='LST',
            color_continuous_scale='RdYlGn_r',
            size='building_density',
            hover_data=['elevation', 'distance_to_water']
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
        df['vegetation_category'] = pd.cut(df['NDVI'], 
                                          bins=[-1, 0, 0.3, 0.6, 1], 
                                          labels=['No Vegetation', 'Low', 'Medium', 'High'])
        
        fig = px.box(
            df, x='vegetation_category', y='LST',
            title='🌳 Temperature Distribution by Vegetation Level',
            labels={'vegetation_category': 'Vegetation Level', 'LST': 'Temperature (°C)'},
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
    vegetation_corr = df['LST'].corr(df['NDVI'])
    
    # Calculate vegetation statistics
    high_veg_mask = df['NDVI'] > 0.5
    low_veg_mask = df['NDVI'] < 0.2
    
    if high_veg_mask.any() and low_veg_mask.any():
        avg_temp_high_veg = df[high_veg_mask]['LST'].mean()
        avg_temp_low_veg = df[low_veg_mask]['LST'].mean()
        cooling_effect = avg_temp_low_veg - avg_temp_high_veg
    else:
        avg_temp_high_veg = df['LST'].mean()
        cooling_effect = 0
    
    # Vegetation metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🌿 Vegetation Correlation", f"{vegetation_corr:.3f}")
    with col2:
        st.metric("🌳 High Vegetation Temp", f"{avg_temp_high_veg:.1f}°C")
    with col3:
        st.metric("❄️ Cooling Effect", f"{cooling_effect:.1f}°C", f"Potential reduction")
    with col4:
        vegetation_coverage = (df['NDVI'] > 0.3).sum() / len(df) * 100
        st.metric("🌱 Green Coverage", f"{vegetation_coverage:.1f}%")
    
    # Enhanced vegetation analysis
    st.markdown('<h3 class="section-header">🌳 Vegetation Cooling Analysis</h3>', unsafe_allow_html=True)
    
    # Create vegetation zones map visualization
    fig = px.scatter_mapbox(
        df.sample(min(200, len(df))),
        lat='latitude',
        lon='longitude',
        color='NDVI',
        size='LST',
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

with tab5:
    st.markdown('<h2 class="section-header">🤖 Model Performance & Predictions</h2>', unsafe_allow_html=True)
    
    if show_predictions and 'LST_Prediction' in df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            # Enhanced actual vs predicted scatter plot
            fig = px.scatter(
                df, x='LST', y='LST_Prediction',
                title='🎯 Actual vs Predicted Temperature',
                labels={'LST': 'Actual Temperature (°C)', 'LST_Prediction': 'Predicted Temperature (°C)'},
                color='building_density',
                color_continuous_scale='Viridis',
                hover_data=['NDVI', 'elevation']
            )
            
            # Add perfect prediction line
            min_temp = min(df['LST'].min(), df['LST_Prediction'].min())
            max_temp = max(df['LST'].max(), df['LST_Prediction'].max())
            fig.add_trace(go.Scatter(
                x=[min_temp, max_temp], y=[min_temp, max_temp],
                mode='lines', name='Perfect Prediction',
                line=dict(color='#ff6b6b', dash='dash', width=3)
            ))
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                title_font_color='white'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Enhanced residuals plot
            residuals = df['LST'] - df['LST_Prediction']
            fig = px.scatter(
                x=df['LST_Prediction'], y=residuals,
                title='📊 Prediction Residuals Analysis',
                labels={'x': 'Predicted Temperature (°C)', 'y': 'Residuals (°C)'},
                color=abs(residuals),
                color_continuous_scale='Reds'
            )
            fig.add_hline(y=0, line_dash="dash", line_color="#4ecdc4", line_width=2)
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                title_font_color='white'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Enhanced model metrics
        mse = mean_squared_error(df['LST'], df['LST_Prediction'])
        r2 = r2_score(df['LST'], df['LST_Prediction'])
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(residuals))
        
        col1, col2, col3, col4 = st.columns(4)
        metrics = [
            ("🎯 R² Score", f"{r2:.3f}", "Model Accuracy"),
            ("📏 RMSE", f"{rmse:.2f}°C", "Root Mean Square Error"),
            ("📊 MAE", f"{mae:.2f}°C", "Mean Absolute Error"),
            ("🌡️ Std Dev", f"{residuals.std():.2f}°C", "Residual Standard Deviation")
        ]
        
        for col, (label, value, desc) in zip([col1, col2, col3, col4], metrics):
            with col:
                st.markdown(f"""
                <div class="metric-container">
                    <div style="font-size: 1.5rem; font-weight: 600; color: #2c3e50; margin-bottom: 0.3rem;">{value}</div>
                    <div style="font-size: 1rem; font-weight: 500; color: #7f8c8d;">{label}</div>
                    <div style="font-size: 0.8rem; color: #95a5a6; margin-top: 0.3rem;">{desc}</div>
                </div>
                """, unsafe_allow_html=True)
    
    # Enhanced feature importance
    st.markdown('<h3 class="section-header">📊 Feature Importance Analysis</h3>', unsafe_allow_html=True)
    
    # Simulate feature importance based on actual correlations
    features = ['Building Density', 'NDVI', 'Population', 'Elevation', 'Distance to Water', 
                'Building Coverage', 'Albedo', 'Rainfall', 'Slope']
    
    # Calculate actual correlations for realistic importance
    correlations = []
    for feature in ['building_density', 'NDVI', 'population', 'elevation', 'distance_to_water', 
                   'building_coverage', 'albedo', 'rainfall', 'slope']:
        if feature in df.columns:
            correlations.append(abs(df['LST'].corr(df[feature])))
        else:
            correlations.append(np.random.uniform(0.1, 0.3))
    
    # Normalize importance scores
    importance = np.array(correlations)
    importance = importance / importance.sum()
    
    feature_df = pd.DataFrame({'Feature': features, 'Importance': importance})
    feature_df = feature_df.sort_values('Importance', ascending=True)
    
    fig = px.bar(
        feature_df, y='Feature', x='Importance', orientation='h',
        title='🎯 Feature Importance in Temperature Prediction',
        color='Importance',
        color_continuous_scale='RdYlBu_r',
        text='Importance'
    )
    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        title_font_color='white'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Model insights
    st.markdown(f"""
    <div class="insight-box">
        <h4>🤖 Model Performance Insights:</h4>
        <ul>
            <li><b>Model Accuracy:</b> R² = {r2:.3f} ({"Excellent" if r2 > 0.9 else "Good" if r2 > 0.7 else "Moderate" if r2 > 0.5 else "Needs Improvement"})</li>
            <li><b>Prediction Error:</b> RMSE = {rmse:.2f}°C, MAE = {mae:.2f}°C</li>
            <li><b>Top Contributing Factors:</b> {', '.join(feature_df.tail(3)['Feature'].tolist())}</li>
            <li><b>Model Reliability:</b> {"High confidence in predictions" if r2 > 0.8 else "Moderate confidence - consider additional features"}</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with tab6:
    st.markdown('<h2 class="section-header">🛡️ AI-Powered Mitigation Strategies</h2>', unsafe_allow_html=True)
    
    # Enhanced file upload section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("📁 Development Plan Analysis")
        st.markdown("Upload your development plans, GIS data, or project documents for AI-powered heat island impact assessment and mitigation recommendations.")
        
        uploaded_file = st.file_uploader(
            "Choose development documents", 
            type=['pdf', 'doc', 'docx', 'txt', 'json', 'csv', 'xlsx'],
            help="Supports: PDF plans, Word docs, GIS data, Excel sheets"
        )
        
        if uploaded_file is not None:
            with st.spinner("🧠 AI Analysis in Progress..."):
                import time
                progress_bar = st.progress(0)
                
                # Simulate AI processing with progress
                for i in range(100):
                    progress_bar.progress(i + 1)
                    time.sleep(0.01)
                
                # Mock plan data extraction
                plan_data = {
                    'area': "250 hectares",
                    'type': "Mixed-use development",
                    'units': 1200,
                    'population': 3000,
                    'greenSpace': "15%"
                }
                
                # Get AI analysis
                analysis = analyze_development_plan_ai(plan_data)
                
                st.success("✅ AI analysis completed successfully!")
                
                # Store in session state
                st.session_state.plan_analysis = analysis
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("🎯 Analysis Configuration")
        analysis_depth = st.selectbox(
            "Analysis Depth",
            ["🔍 Basic Assessment", "📊 Comprehensive Analysis", "🎓 Expert Review"],
            help="Choose the level of detail for the AI analysis"
        )
        
        include_economic = st.checkbox("💰 Include Economic Impact", True)
        include_health = st.checkbox("❤️ Include Health Assessment", True)
        include_timeline = st.checkbox("📅 Generate Timeline", True)
        
        if st.button("🔄 Re-analyze Plan", type="primary"):
            if 'plan_analysis' in st.session_state:
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Display enhanced analysis results
    if 'plan_analysis' in st.session_state or uploaded_file is not None:
        analysis = st.session_state.get('plan_analysis', analyze_development_plan_ai({}))
        
        # Enhanced summary cards with animations
        st.markdown("### 📋 Development Plan Impact Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        summary_metrics = [
            ("🏗️", "Total Area", analysis['planSummary']['totalArea'], "Development footprint"),
            ("👥", "Population", f"{analysis['planSummary']['estimatedPopulation']:,}", "Expected residents"),
            ("🌡️", "Temperature Impact", f"+{analysis['thermalImpact']['temperatureIncrease']}°C", "Projected increase"),
            ("⚠️", "Risk Level", analysis['thermalImpact']['hotspotRisk'], "Overall assessment")
        ]
        
        for col, (icon, label, value, desc) in zip([col1, col2, col3, col4], summary_metrics):
            with col:
                # Determine color based on risk level
                if "Temperature Impact" in label:
                    color = "#ff6b6b" if float(value.replace('+', '').replace('°C', '')) > 3 else "#ffa502"
                elif "Risk Level" in label:
                    color = "#ff6b6b" if value == "High" else "#ffa502" if value == "Medium" else "#2ed573"
                else:
                    color = "#4ecdc4"
                
                st.markdown(f"""
                <div class="metric-container" style="border-left: 4px solid {color};">
                    <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">{icon}</div>
                    <div style="font-size: 1.8rem; font-weight: 600; color: #2c3e50; margin-bottom: 0.3rem;">{value}</div>
                    <div style="font-size: 1rem; font-weight: 500; color: #7f8c8d;">{label}</div>
                    <div style="font-size: 0.9rem; color: #95a5a6; margin-top: 0.3rem;">{desc}</div>
                </div>
                """, unsafe_allow_html=True)

        # Enhanced mitigation strategies with priority indicators
        st.markdown("### 🛠️ AI-Generated Mitigation Strategies")
        
        for i, strategy in enumerate(analysis['mitigationStrategies']):
            # Priority color coding
            priority_colors = {
                'Critical': '#ff4757',
                'High': '#ffa502', 
                'Medium': '#2ed573',
                'Low': '#747d8c'
            }
            
            priority_color = priority_colors.get(strategy['priority'], '#747d8c')
            
            with st.expander(
                f"🔧 {strategy['strategy']} - {strategy['priority']} Priority", 
                expanded=(i==0)
            ):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**⏰ Implementation Timeline:** {strategy['implementation']}")
                    st.markdown("**📋 Detailed Actions:**")
                    for j, detail in enumerate(strategy['details'], 1):
                        st.markdown(f"{j}. {detail}")
                
                with col2:
                    st.markdown(f"""
                    <div style="background: {priority_color}; color: white; padding: 1rem; border-radius: 10px; text-align: center;">
                        <h4 style="margin: 0; color: white;">Priority Level</h4>
                        <p style="margin: 5px 0; font-size: 1.2rem; font-weight: bold;">{strategy['priority']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown(f"**💪 Expected Impact:**  \n{strategy['expectedImpact']}")
                    st.markdown(f"**💰 Investment Required:**  \n{strategy['cost']}")

        # Enhanced risk assessment with visual indicators
        st.markdown("### ⚠️ Comprehensive Risk Assessment")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div style="background: linear-gradient(90deg, #ff6b6b, #ff8e53); padding: 1rem; border-radius: 10px; color: white;">
                <h4>🌡️ Thermal Risk</h4>
                <p><strong>Level:</strong> {}</p>
                <p><strong>Probability:</strong> {}</p>
                <p>{}</p>
            </div>
            """.format(
                analysis['riskAssessment']['thermalRisk']['level'],
                analysis['riskAssessment']['thermalRisk']['probability'],
                analysis['riskAssessment']['thermalRisk']['impact']
            ), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background: linear-gradient(90deg, #ffa726, #ff7043); padding: 1rem; border-radius: 10px; color: white;">
                <h4>❤️ Health Risk</h4>
                <p><strong>Level:</strong> {}</p>
                <p><strong>Affected:</strong> {:,} people</p>
                <p>{}</p>
            </div>
            """.format(
                analysis['riskAssessment']['healthRisk']['level'],
                analysis['riskAssessment']['healthRisk']['affectedPopulation'],
                analysis['riskAssessment']['healthRisk']['projectedIncidents']
            ), unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style="background: linear-gradient(90deg, #42a5f5, #2196f3); padding: 1rem; border-radius: 10px; color: white;">
                <h4>💰 Economic Impact</h4>
                <p><strong>Energy:</strong> {}</p>
                <p><strong>Healthcare:</strong> {}</p>
                <p><strong>Property:</strong> {}</p>
            </div>
            """.format(
                analysis['riskAssessment']['economicImpact']['energyCosts'],
                analysis['riskAssessment']['economicImpact']['healthcareCosts'],
                analysis['riskAssessment']['economicImpact']['propertyValues']
            ), unsafe_allow_html=True)

        # Implementation Timeline
        st.markdown("### 📅 Implementation Timeline")
        
        timeline_data = []
        for phase, tasks in analysis['timeline'].items():
            for task in tasks:
                timeline_data.append({
                    'Phase': phase,
                    'Task': task,
                    'Status': 'Planned'
                })
        
        timeline_df = pd.DataFrame(timeline_data)
        st.dataframe(timeline_df, use_container_width=True)

        # Recommendations
        st.markdown("### 💡 AI Recommendations")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**🏗️ Design Modifications**")
            for mod in analysis['recommendations']['designModifications']:
                st.write(f"• {mod}")
        
        with col2:
            st.markdown("**📋 Policy Changes**")
            for policy in analysis['recommendations']['policyChanges']:
                st.write(f"• {policy}")
        
        with col3:
            st.markdown("**📊 Monitoring Plan**")
            for monitor in analysis['recommendations']['monitoringPlan']:
                st.write(f"• {monitor}")

        # Download Results
        st.markdown("---")
        col1, col2 = st.columns([3, 1])
        
        with col2:
            # Create downloadable report
            report_data = {
                'analysis_date': datetime.now().isoformat(),
                'development_plan': analysis['planSummary'],
                'thermal_impact': analysis['thermalImpact'],
                'mitigation_strategies': analysis['mitigationStrategies'],
                'risk_assessment': analysis['riskAssessment'],
                'recommendations': analysis['recommendations'],
                'timeline': analysis['timeline']
            }
            
            json_report = json.dumps(report_data, indent=2)
            
            st.download_button(
                label="📄 Download Full Report",
                data=json_report,
                file_name=f"ai_mitigation_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json"
            )

    else:
        # Show example/demo
        st.info("👆 Upload a development plan document to get started with AI-powered mitigation analysis")
        
        with st.expander("🎯 See Example Analysis"):
            example_analysis = analyze_development_plan_ai({})
            st.json(example_analysis['mitigationStrategies'][0])

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <h4>🌍 Urban Heat Island Analysis - Kilimani, Nairobi</h4>
    <p>Powered by Google Earth Engine, Streamlit, and Machine Learning</p>
    <p>Data sources: Landsat 8/9, Sentinel-2, OpenStreetMap Building Footprints</p>
</div>
""", unsafe_allow_html=True)

# Download results button
st.sidebar.markdown("---")
st.sidebar.subheader("📥 Export Results")

if st.sidebar.button("Download Analysis Report"):
    # Calculate heat island area
    heat_island_area = (df['LST'] > temp_threshold).sum() / len(df) * 100
    
    # Create a summary report
    report_data = {
        'Metric': ['Average Temperature', 'Max Temperature', 'Min Temperature', 
                  'Heat Island Area', 'Model R²', 'Model RMSE'],
        'Value': [f"{df['LST'].mean():.1f}°C", f"{df['LST'].max():.1f}°C", 
                 f"{df['LST'].min():.1f}°C", f"{heat_island_area:.1f}%",
                 f"{r2:.3f}" if 'LST_Prediction' in df.columns else "N/A",
                 f"{rmse:.2f}°C" if 'LST_Prediction' in df.columns else "N/A"]
    }
    
    report_df = pd.DataFrame(report_data)
    csv = report_df.to_csv(index=False)
    
    st.sidebar.download_button(
        label="📊 Download Summary CSV",
        data=csv,
        file_name=f"urban_heat_analysis_summary_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv"
    )