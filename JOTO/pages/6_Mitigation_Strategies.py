import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import json
from datetime import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_lst_prediction_data, analyze_development_plan_ai

# Page configuration
st.set_page_config(
    page_title="Mitigation Strategies - Kilimani Heat Island",
    page_icon="🛡️",
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
</style>
""", unsafe_allow_html=True)

# Page header
st.markdown('<h1 class="main-header">🛡️ Mitigation Strategies</h1>', unsafe_allow_html=True)
st.markdown('<h2 class="sub-header">AI-Powered Heat Island Mitigation</h2>', unsafe_allow_html=True)

# Load data
df = load_lst_prediction_data()

if df is not None:
    # Enhanced file upload section
    st.markdown("### 📁 Development Plan Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("📄 Upload Development Documents")
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

    # Display analysis results or show general mitigation strategies
    if 'plan_analysis' in st.session_state or uploaded_file is not None:
        analysis = st.session_state.get('plan_analysis', analyze_development_plan_ai({}))
        
        # Enhanced summary cards
        st.markdown("---")
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

        # Enhanced mitigation strategies
        st.markdown("---")
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

        # Enhanced risk assessment
        st.markdown("---")
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
        st.markdown("---")
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
        st.markdown("---")
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

    else:
        # Show general mitigation strategies based on current data
        st.markdown("---")
        st.markdown("### 🌡️ General Heat Island Mitigation Strategies")
        
        # Current conditions analysis
        if df is not None:
            avg_temp = df['LST_Prediction'].mean()
            max_temp = df['LST_Prediction'].max()
            hot_spots = len(df[df['LST_Prediction'] > 35])
            
            st.markdown("#### 📊 Current Conditions Assessment")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-container">
                    <h4>🌡️ Average Temperature</h4>
                    <h2>{avg_temp:.1f}°C</h2>
                    <p>Current conditions</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-container">
                    <h4>🔥 Peak Temperature</h4>
                    <h2>{max_temp:.1f}°C</h2>
                    <p>Hottest areas</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-container">
                    <h4>⚠️ Hot Spots</h4>
                    <h2>{hot_spots:,}</h2>
                    <p>Areas >35°C</p>
                </div>
                """, unsafe_allow_html=True)
        
        # General mitigation strategies
        st.markdown("---")
        st.markdown("### 🛠️ Recommended Mitigation Approaches")
        
        strategies = {
            "🌳 Green Infrastructure": {
                "description": "Increase vegetation coverage and urban forest",
                "actions": [
                    "Plant 1,000+ trees in high-temperature areas",
                    "Create green corridors connecting existing parks",
                    "Implement mandatory green roofs for new buildings",
                    "Establish community gardens in dense areas"
                ],
                "impact": "2-4°C temperature reduction",
                "cost": "$3-5M over 3 years",
                "timeline": "Short to medium term (6-24 months)"
            },
            "🏢 Building Standards": {
                "description": "Implement cool building design requirements",
                "actions": [
                    "Mandate light-colored, high-albedo roofing materials",
                    "Require building orientation for natural ventilation",
                    "Implement green building certification requirements",
                    "Establish maximum building density limits"
                ],
                "impact": "1-3°C local temperature reduction",
                "cost": "$2-4M in additional construction costs",
                "timeline": "Medium to long term (1-5 years)"
            },
            "💧 Water Features": {
                "description": "Strategic water body and cooling system placement",
                "actions": [
                    "Create urban water features and fountains",
                    "Implement stormwater management systems",
                    "Install misting systems in public spaces",
                    "Develop blue-green infrastructure corridors"
                ],
                "impact": "1-2°C localized cooling effect",
                "cost": "$1-3M for infrastructure",
                "timeline": "Medium term (12-36 months)"
            },
            "🚗 Transportation": {
                "description": "Reduce heat from transportation infrastructure",
                "actions": [
                    "Implement cool pavement technologies",
                    "Increase public transit to reduce car dependency",
                    "Create dedicated cycling and walking paths",
                    "Plant street trees for shade coverage"
                ],
                "impact": "0.5-1.5°C reduction along corridors",
                "cost": "$5-8M for comprehensive program",
                "timeline": "Long term (2-7 years)"
            }
        }
        
        for strategy_name, details in strategies.items():
            with st.expander(f"{strategy_name} - {details['impact']}", expanded=False):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**📋 Strategy Overview:**")
                    st.write(details['description'])
                    
                    st.markdown("**🎯 Key Actions:**")
                    for action in details['actions']:
                        st.write(f"• {action}")
                
                with col2:
                    st.markdown(f"""
                    <div class="glass-card">
                        <h4>📊 Strategy Details</h4>
                        <p><strong>Expected Impact:</strong><br>{details['impact']}</p>
                        <p><strong>Investment:</strong><br>{details['cost']}</p>
                        <p><strong>Timeline:</strong><br>{details['timeline']}</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Priority matrix
        st.markdown("---")
        st.markdown("### 🎯 Mitigation Priority Matrix")
        
        # Create priority data
        priority_data = {
            'Strategy': ['Green Infrastructure', 'Building Standards', 'Water Features', 'Transportation'],
            'Impact': [4, 3, 2, 1.5],
            'Cost': [4, 3, 2, 7],
            'Implementation_Ease': [3, 2, 4, 1],
            'Timeline_Months': [12, 36, 24, 60]
        }
        
        priority_df = pd.DataFrame(priority_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Impact vs Cost scatter
            fig = px.scatter(
                priority_df, x='Cost', y='Impact', 
                size='Implementation_Ease',
                color='Timeline_Months',
                hover_name='Strategy',
                title='💰 Impact vs Investment Analysis',
                labels={'Cost': 'Investment Required (Scale 1-10)', 'Impact': 'Temperature Reduction (°C)'},
                color_continuous_scale='RdYlGn_r'
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                title_font_color='white'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Implementation timeline
            fig = px.bar(
                priority_df, x='Strategy', y='Timeline_Months',
                color='Impact',
                title='📅 Implementation Timeline',
                labels={'Timeline_Months': 'Implementation Time (Months)'},
                color_continuous_scale='RdYlGn'
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                title_font_color='white',
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig, use_container_width=True)

else:
    st.error("❌ Unable to load Kilimani LST prediction data")
    st.info("Please ensure the resources/Kilimani_LST_Prediction.tif file exists")

# Navigation back to homepage
st.markdown("---")
st.markdown("### 🧭 Navigation")
st.info("💡 Use the sidebar navigation to explore other analysis pages or return to the homepage")
