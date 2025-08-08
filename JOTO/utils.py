import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import geopandas as gpd
import rasterio
import tempfile
import zipfile
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

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
            }
        }
    }
    return analysis

def load_lst_prediction_data(tif_file):
    """Load and process the LST prediction GeoTIFF from Google Earth Engine"""
    try:
        with rasterio.open(tif_file) as src:
            lst_data = src.read(1)
            bounds = src.bounds
            transform = src.transform
            
            height, width = lst_data.shape
            cols, rows = np.meshgrid(np.arange(width), np.arange(height))
            
            xs, ys = rasterio.transform.xy(transform, rows, cols)
            
            xs_flat = np.array(xs).flatten()
            ys_flat = np.array(ys).flatten()
            lst_flat = lst_data.flatten()
            
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
        if str(shp_file.name).endswith('.zip'):
            with tempfile.TemporaryDirectory() as temp_dir:
                with zipfile.ZipFile(shp_file, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                
                shp_files = [f for f in os.listdir(temp_dir) if f.endswith('.shp')]
                if shp_files:
                    shp_path = os.path.join(temp_dir, shp_files[0])
                    gdf = gpd.read_file(shp_path)
                else:
                    st.error("No .shp file found in the zip archive")
                    return None
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.shp') as tmp_file:
                tmp_file.write(shp_file.getbuffer())
                tmp_file.flush()
                gdf = gpd.read_file(tmp_file.name)
        
        gdf['centroid'] = gdf.geometry.centroid
        gdf['longitude'] = gdf['centroid'].x
        gdf['latitude'] = gdf['centroid'].y
        gdf['building_area'] = gdf.geometry.area
        
        return gdf
        
    except Exception as e:
        st.error(f"Error loading building data: {str(e)}")
        return None

def process_real_data(lst_df, buildings_gdf):
    """Process and merge real LST and building data"""
    try:
        if lst_df is None or lst_df.empty:
            return None
        
        if buildings_gdf is not None and not buildings_gdf.empty:
            merged_data = []
            
            for idx, building in buildings_gdf.iterrows():
                distances = np.sqrt(
                    (lst_df['longitude'] - building['longitude'])**2 + 
                    (lst_df['latitude'] - building['latitude'])**2
                )
                nearest_idx = distances.idxmin()
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
                    'NDVI': np.random.uniform(-0.2, 0.8),
                    'elevation': np.random.uniform(1600, 1800),
                    'distance_to_water': np.random.uniform(100, 3000),
                    'albedo': np.random.uniform(0.1, 0.4),
                    'rainfall': np.random.uniform(0.5, 3.5),
                    'slope': np.random.uniform(0, 15)
                })
            
            df = pd.DataFrame(merged_data)
            
        else:
            df = lst_df.copy()
            df['building_density'] = np.random.uniform(20, 80, len(df))
            df['building_coverage'] = np.random.uniform(10, 70, len(df))
            df['population'] = np.random.uniform(50, 500, len(df))
            df['NDVI'] = np.random.uniform(-0.2, 0.8, len(df))
            df['elevation'] = np.random.uniform(1600, 1800, len(df))
            df['distance_to_water'] = np.random.uniform(100, 3000, len(df))
            df['albedo'] = np.random.uniform(0.1, 0.4, len(df))
            df['rainfall'] = np.random.uniform(0.5, 3.5, len(df))
            df['slope'] = np.random.uniform(0, 15, len(df))
            df['LST'] = df['LST_Prediction']
        
        return df
        
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        return None

def analyze_lst_prediction_with_ai(df, bounds=None):
    """AI-powered analysis of LST predictions"""
    if df is None or df.empty:
        return None
    
    mean_temp = df['LST_Prediction'].mean()
    max_temp = df['LST_Prediction'].max()
    min_temp = df['LST_Prediction'].min()
    std_temp = df['LST_Prediction'].std()
    
    hot_threshold = df['LST_Prediction'].quantile(0.9)
    hot_spots = df[df['LST_Prediction'] > hot_threshold]
    
    cool_threshold = df['LST_Prediction'].quantile(0.1)
    cool_spots = df[df['LST_Prediction'] < cool_threshold]
    
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
