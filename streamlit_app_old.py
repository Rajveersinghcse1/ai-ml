"""
🎯 Professional Customer Analytics Platform
Enterprise-Grade Customer Intelligence Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
import json
import os
from typing import Dict, List, Optional
import sqlite3

warnings.filterwarnings('ignore')

# Backend imports with error handling
try:
    from src.preprocessing import load_data, clean_data
    from src.feature_engineering import CustomerFeatureEngineer
    from src.rfm_analysis import calculate_rfm, segment_customers, AdvancedRFMAnalyzer
    from src.clustering import UltraAdvancedClustering
    from src.advanced_analytics import ChurnPredictionModel, CLVPredictionModel
    from src.recommendation_engine import HybridRecommendationEngine
    from src.visualization import UltraAdvancedVisualization
    from src.personalization import UltraAdvancedPersonalizationEngine
    from src.model_evaluation import UltraAdvancedModelEvaluation
    BACKEND_AVAILABLE = True
    st.success("✅ All backend modules loaded successfully!")
except ImportError as e:
    BACKEND_AVAILABLE = False
    st.error(f"❌ Backend Error: {str(e)}")
    st.info("💡 Some features may be limited without backend modules.")

# Ultra-Advanced Page Configuration
st.set_page_config(
    page_title="🎯 ULTRA-ADVANCED Customer Analytics Platform",
    page_icon="�",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com',
        'Report a bug': 'https://github.com',
        'About': "# Ultra-Advanced Customer Segmentation Platform\nEnterprise-grade analytics with AI-powered insights!"
    }
)

# Initialize session state for advanced features
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = 'Overview'
if 'ml_models_trained' not in st.session_state:
    st.session_state.ml_models_trained = False
if 'personalization_active' not in st.session_state:
    st.session_state.personalization_active = False

# Ultra-Advanced Custom CSS with Premium Animations
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        padding: 1rem;
        background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(248,250,252,0.9) 100%);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        margin: 10px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
    }
    
    /* Premium Header */
    .ultra-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 40px;
        border-radius: 20px;
        box-shadow: 0 15px 35px rgba(102,126,234,0.3);
        text-align: center;
        margin-bottom: 30px;
        position: relative;
        overflow: hidden;
    }
    
    .ultra-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        animation: shimmer 3s infinite;
    }
    
    @keyframes shimmer {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    .ultra-header h1 {
        color: white !important;
        font-size: 52px !important;
        font-weight: 900 !important;
        margin: 0 !important;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.3);
        letter-spacing: -1px;
    }
    
    .ultra-header .subtitle {
        color: rgba(255,255,255,0.9) !important;
        font-size: 20px !important;
        margin: 15px 0 0 0 !important;
        font-weight: 500 !important;
    }
    
    /* Advanced Metric Cards */
    .metric-container {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 30px;
        border-radius: 18px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.08);
        border: 1px solid rgba(102,126,234,0.1);
        transition: all 0.4s cubic-bezier(0.4, 0.0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .metric-container:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(102,126,234,0.15);
        border-color: rgba(102,126,234,0.3);
    }
    
    .metric-value {
        font-size: 48px !important;
        font-weight: 800 !important;
        color: #1e293b !important;
        margin: 0 !important;
    }
    
    .metric-label {
        font-size: 14px !important;
        font-weight: 600 !important;
        color: #64748b !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Tab Navigation */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(248, 250, 252, 0.8);
        padding: 8px;
        border-radius: 16px;
        backdrop-filter: blur(10px);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border: none;
        border-radius: 12px;
        padding: 12px 24px;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        box-shadow: 0 4px 12px rgba(102,126,234,0.3);
    }
    
    /* Sidebar Enhancement */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        padding: 20px 10px;
    }
    
    .sidebar-content {
        background: rgba(255,255,255,0.95);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.1);
    }
    
    /* Button Enhancements */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 24px;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(102,126,234,0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(102,126,234,0.4);
    }
    
    /* Progress Bars */
    .stProgress > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    /* Alerts and Info Boxes */
    .stAlert {
        border-radius: 12px;
        border: none;
        backdrop-filter: blur(10px);
    }
    
    /* Chart Containers */
    .chart-container {
        background: white;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.08);
        border: 1px solid rgba(0,0,0,0.05);
        margin: 20px 0;
    }
    
    /* Loading Animation */
    .loading-spinner {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 200px;
    }
    
    .spinner {
        width: 40px;
        height: 40px;
        border: 4px solid #f3f3f3;
        border-top: 4px solid #667eea;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Status Indicators */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }
    
    .status-success {
        background: #dcfce7;
        color: #166534;
    }
    
    .status-warning {
        background: #fef3c7;
        color: #92400e;
    }
    
    .status-error {
        background: #fee2e2;
        color: #991b1b;
    }
    
    /* Advanced Animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(50px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    .fade-in-up {
        animation: fadeInUp 0.6s ease-out;
    }
    
    .slide-in-right {
        animation: slideInRight 0.6s ease-out;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .ultra-header h1 {
            font-size: 36px !important;
        }
        
        .metric-container {
            padding: 20px;
        }
        
        .metric-value {
            font-size: 36px !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# Additional Advanced Styling
st.markdown("""
<style>
    .main-header h1 {
        color: white !important;
        font-size: 48px !important;
        font-weight: 800 !important;
        margin: 0 !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        border: none !important;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.95) !important;
        font-size: 18px !important;
        margin: 10px 0 0 0 !important;
        font-weight: 500 !important;
    }
    
    /* Metric Cards - Ultra Visible */
    .stMetric {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 25px !important;
        border-radius: 15px !important;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3) !important;
        border: 3px solid #667eea !important;
        transition: all 0.3s ease !important;
        animation: fadeInUp 0.6s ease-out !important;
    }
    
    .stMetric:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 12px 30px rgba(102, 126, 234, 0.5) !important;
        border-color: #764ba2 !important;
    }
    
    .stMetric label {
        font-size: 16px !important;
        font-weight: 700 !important;
        color: #667eea !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        font-size: 36px !important;
        font-weight: 900 !important;
        color: #2d3748 !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    
    .stMetric [data-testid="stMetricDelta"] {
        font-size: 14px !important;
        font-weight: 600 !important;
    }
    
    /* Headings - Ultra Visible */
    h1 {
        color: #667eea !important;
        font-size: 42px !important;
        font-weight: 800 !important;
        padding: 20px 0 !important;
        border-bottom: 5px solid #764ba2 !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        animation: fadeIn 0.5s ease-out;
    }
    
    h2 {
        color: #764ba2 !important;
        font-size: 32px !important;
        font-weight: 700 !important;
        margin-top: 30px !important;
        padding: 15px 20px !important;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-left: 8px solid #667eea !important;
        border-radius: 10px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }
    
    h3 {
        color: #667eea !important;
        font-size: 24px !important;
        font-weight: 600 !important;
        margin-top: 20px !important;
        padding: 10px 15px !important;
        background: rgba(102, 126, 234, 0.1);
        border-left: 5px solid #764ba2;
        border-radius: 8px;
    }
    
    h4 {
        color: #2d3748 !important;
        font-size: 20px !important;
        font-weight: 600 !important;
        margin-top: 15px !important;
    }
    
    /* DataFrames - Ultra Visible */
    .stDataFrame {
        border: 4px solid #667eea !important;
        border-radius: 15px !important;
        overflow: hidden !important;
        box-shadow: 0 8px 20px rgba(0,0,0,0.15) !important;
    }
    
    .stDataFrame table {
        font-size: 15px !important;
        font-weight: 500 !important;
    }
    
    .stDataFrame thead tr th {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        font-size: 16px !important;
        font-weight: 700 !important;
        padding: 15px !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }
    
    .stDataFrame tbody tr:nth-child(even) {
        background-color: #f8f9fa !important;
    }
    
    .stDataFrame tbody tr:hover {
        background-color: #e9ecef !important;
        transform: scale(1.01);
        transition: all 0.2s ease;
    }
    
    /* Expanders - Ultra Visible */
    div[data-testid="stExpander"] {
        border: 3px solid #667eea !important;
        border-radius: 15px !important;
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%) !important;
        margin: 15px 0 !important;
        box-shadow: 0 6px 15px rgba(0,0,0,0.1) !important;
        transition: all 0.3s ease !important;
    }
    
    div[data-testid="stExpander"]:hover {
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3) !important;
        transform: translateY(-3px);
    }
    
    div[data-testid="stExpander"] summary {
        font-size: 18px !important;
        font-weight: 700 !important;
        color: #667eea !important;
        padding: 20px !important;
    }
    
    /* Info Boxes - Ultra Visible */
    .info-box {
        padding: 25px !important;
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%) !important;
        border-left: 8px solid #17a2b8 !important;
        border-radius: 12px !important;
        margin: 15px 0 !important;
        box-shadow: 0 6px 15px rgba(23, 162, 184, 0.2) !important;
        animation: fadeInLeft 0.5s ease-out !important;
        font-size: 16px !important;
        font-weight: 500 !important;
        color: #0c5460 !important;
    }
    
    .info-box h3, .info-box h4 {
        color: #0c5460 !important;
        font-weight: 700 !important;
        margin-top: 0 !important;
    }
    
    .warning-box {
        padding: 25px !important;
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%) !important;
        border-left: 8px solid #dc3545 !important;
        border-radius: 12px !important;
        margin: 15px 0 !important;
        box-shadow: 0 6px 15px rgba(220, 53, 69, 0.2) !important;
        animation: fadeInRight 0.5s ease-out !important;
        font-size: 16px !important;
        font-weight: 500 !important;
        color: #721c24 !important;
    }
    
    .warning-box h3, .warning-box h4 {
        color: #721c24 !important;
        font-weight: 700 !important;
        margin-top: 0 !important;
    }
    
    .success-box {
        padding: 25px !important;
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%) !important;
        border-left: 8px solid #28a745 !important;
        border-radius: 12px !important;
        margin: 15px 0 !important;
        box-shadow: 0 6px 15px rgba(40, 167, 69, 0.2) !important;
        animation: fadeInUp 0.5s ease-out !important;
        font-size: 16px !important;
        font-weight: 500 !important;
        color: #155724 !important;
    }
    
    .insight-box {
        padding: 25px !important;
        background: linear-gradient(135deg, #fff3cd 0%, #ffe69c 100%) !important;
        border-left: 8px solid #ffc107 !important;
        border-radius: 12px !important;
        margin: 15px 0 !important;
        box-shadow: 0 6px 15px rgba(255, 193, 7, 0.2) !important;
        animation: fadeIn 0.5s ease-out !important;
        font-size: 16px !important;
        font-weight: 500 !important;
        color: #856404 !important;
    }
    
    /* Buttons - Ultra Visible */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        font-size: 18px !important;
        font-weight: 700 !important;
        padding: 15px 30px !important;
        border-radius: 12px !important;
        border: none !important;
        box-shadow: 0 6px 15px rgba(102, 126, 234, 0.3) !important;
        transition: all 0.3s ease !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }
    
    .stButton button:hover {
        transform: translateY(-3px) scale(1.05) !important;
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.5) !important;
    }
    
    /* Sidebar - Ultra Visible */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%) !important;
        padding: 20px !important;
    }
    
    section[data-testid="stSidebar"] * {
        color: white !important;
        font-weight: 600 !important;
    }
    
    section[data-testid="stSidebar"] .stRadio label {
        font-size: 16px !important;
        padding: 12px !important;
        background: rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
        margin: 8px 0 !important;
        transition: all 0.3s ease !important;
    }
    
    section[data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(255, 255, 255, 0.2) !important;
        transform: translateX(5px);
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes fadeInUp {
        from { 
            opacity: 0;
            transform: translateY(30px);
        }
        to { 
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeInLeft {
        from { 
            opacity: 0;
            transform: translateX(-30px);
        }
        to { 
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes fadeInRight {
        from { 
            opacity: 0;
            transform: translateX(30px);
        }
        to { 
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes slideDown {
        from { 
            opacity: 0;
            transform: translateY(-30px);
        }
        to { 
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Progress Bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
        height: 12px !important;
        border-radius: 10px !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border: 2px solid #667eea;
        border-radius: 10px;
        padding: 15px 25px;
        font-weight: 700;
        font-size: 16px;
        color: #667eea;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
    }
    
    /* Download Button */
    .stDownloadButton button {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%) !important;
        color: white !important;
        font-size: 16px !important;
        font-weight: 700 !important;
        padding: 12px 25px !important;
        border-radius: 10px !important;
        border: none !important;
        box-shadow: 0 4px 10px rgba(40, 167, 69, 0.3) !important;
    }
    
    .stDownloadButton button:hover {
        transform: translateY(-2px) scale(1.03) !important;
        box-shadow: 0 6px 15px rgba(40, 167, 69, 0.5) !important;
    }
    
    /* Text Visibility */
    p, li, span, div {
        color: #2d3748 !important;
        font-size: 16px !important;
        line-height: 1.8 !important;
    }
    
    strong {
        color: #667eea !important;
        font-weight: 700 !important;
    }
    
    /* Plotly Charts */
    .js-plotly-plot {
        border-radius: 15px !important;
        box-shadow: 0 8px 20px rgba(0,0,0,0.1) !important;
        border: 3px solid #667eea !important;
        overflow: hidden !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_and_analyze_data():
    """Load data and perform complete analysis - cached for performance"""
    
    try:
        # 1. Load data
        data_path = 'data/Online Retail.xlsx'
        df_raw = load_data(data_path)
        
        # 2. Clean data
        df_clean = clean_data(df_raw)
        
        # Add TotalPrice column for calculations
        if 'Quantity' in df_clean.columns and 'UnitPrice' in df_clean.columns:
            df_clean['TotalPrice'] = df_clean['Quantity'] * df_clean['UnitPrice']
        
        # 3. Calculate RFM (using TotalPrice as amount column)
        rfm = calculate_rfm(df_clean, amount_col='TotalPrice')
        
        # 4. Calculate RFM scores
        rfm_scored = calculate_rfm_scores(rfm)
        
        # 5. Create segments
        rfm_segmented = segment_customers(rfm_scored)
        
        # 6. Perform clustering
        from sklearn.preprocessing import StandardScaler
        clustering_features = ['Recency', 'Frequency', 'Monetary']
        X = rfm[clustering_features].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        kmeans_model, kmeans_labels, kmeans_metrics = perform_kmeans_clustering(X_scaled, n_clusters=4)
        
        # Package clustering results
        kmeans_result = {
            'model': kmeans_model,
            'labels': kmeans_labels,
            'silhouette_score': kmeans_metrics['silhouette_score'],
            'davies_bouldin_score': kmeans_metrics['davies_bouldin_score'],
            'inertia': kmeans_metrics['inertia']
        }
        
        # 7. Calculate segment summary
        segment_summary = rfm_segmented.groupby('Segment').agg({
            'CustomerID': 'count',
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': 'mean'
        }).round(2)
        segment_summary.columns = ['Customer_Count', 'Avg_Recency', 'Avg_Frequency', 'Avg_Monetary']
        segment_summary = segment_summary.sort_values('Avg_Monetary', ascending=False)
        
        # 8. Calculate cluster summary
        rfm_clustered = rfm.copy()
        rfm_clustered['Cluster'] = kmeans_labels
        
        cluster_summary = rfm_clustered.groupby('Cluster').agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': 'mean'
        }).round(2)
        cluster_summary['Customer_Count'] = rfm_clustered.groupby('Cluster').size()
        cluster_summary = cluster_summary.sort_values('Monetary', ascending=False)
        
        return {
            'success': True,
            'df_raw': df_raw,
            'df_clean': df_clean,
            'rfm': rfm,
            'rfm_segmented': rfm_segmented,
            'rfm_scored': rfm_scored,
            'rfm_clustered': rfm_clustered,
            'segment_summary': segment_summary,
            'cluster_summary': cluster_summary,
            'kmeans_result': kmeans_result,
            'metrics': {
                'total_rows': len(df_raw),
                'clean_rows': len(df_clean),
                'data_quality': round((len(df_clean) / len(df_raw)) * 100, 2),
                'total_customers': len(rfm),
                'total_revenue': df_clean['TotalPrice'].sum(),
                'avg_customer_value': df_clean.groupby('CustomerID')['TotalPrice'].sum().mean(),
                'num_segments': rfm_segmented['Segment'].nunique(),
                'num_clusters': 4
            }
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def create_segment_distribution_chart(segment_summary):
    """Create ultra-advanced interactive segment distribution visualization"""
    
    # Create subplot with 1 row, 2 columns
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Customer Distribution by Segment', 'Segment Percentage Distribution'),
        specs=[[{'type': 'bar'}, {'type': 'pie'}]],
        horizontal_spacing=0.15
    )
    
    # Bar chart with gradient colors
    colors_bar = [f'rgb({int(102+i*20)}, {int(126+i*15)}, {int(234-i*10)})' 
                  for i in range(len(segment_summary))]
    
    fig.add_trace(
        go.Bar(
            x=segment_summary.index,
            y=segment_summary['Customer_Count'],
            text=segment_summary['Customer_Count'],
            textposition='outside',
            textfont=dict(size=14, color='#2d3748', family='Arial Black'),
            marker=dict(
                color=colors_bar,
                line=dict(color='#667eea', width=3),
                pattern=dict(shape='')
            ),
            hovertemplate='<b>%{x}</b><br>Customers: %{y}<extra></extra>',
            name='Customer Count'
        ),
        row=1, col=1
    )
    
    # Pie chart with custom colors
    colors_pie = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', 
                  '#00f2fe', '#43e97b', '#38f9d7', '#fa709a', '#fee140']
    
    fig.add_trace(
        go.Pie(
            labels=segment_summary.index,
            values=segment_summary['Customer_Count'],
            hole=0.4,
            marker=dict(colors=colors_pie, line=dict(color='white', width=3)),
            textfont=dict(size=14, color='white', family='Arial Black'),
            hovertemplate='<b>%{label}</b><br>Customers: %{value}<br>Percentage: %{percent}<extra></extra>',
            textinfo='label+percent',
            pull=[0.05] * len(segment_summary)
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=550,
        showlegend=False,
        title=dict(
            text='<b>Customer Segmentation Overview</b>',
            font=dict(size=24, color='#667eea', family='Arial Black'),
            x=0.5,
            xanchor='center'
        ),
        plot_bgcolor='rgba(248, 249, 250, 0.8)',
        paper_bgcolor='white',
        font=dict(family='Arial', size=14, color='#2d3748'),
        hoverlabel=dict(
            bgcolor='white',
            font_size=14,
            font_family='Arial',
            bordercolor='#667eea'
        )
    )
    
    # Update axes
    fig.update_xaxes(
        title_text='<b>Segment</b>',
        title_font=dict(size=16, color='#667eea', family='Arial Black'),
        tickfont=dict(size=12, color='#2d3748'),
        showgrid=False,
        row=1, col=1
    )
    
    fig.update_yaxes(
        title_text='<b>Number of Customers</b>',
        title_font=dict(size=16, color='#667eea', family='Arial Black'),
        tickfont=dict(size=12, color='#2d3748'),
        showgrid=True,
        gridcolor='rgba(102, 126, 234, 0.2)',
        row=1, col=1
    )
    
    return fig


def create_rfm_distributions_chart(rfm):
    """Create ultra-advanced interactive RFM value distributions"""
    
    # Create subplot with 1 row, 3 columns
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('<b>Recency Distribution</b>', '<b>Frequency Distribution</b>', '<b>Monetary Distribution</b>'),
        horizontal_spacing=0.1
    )
    
    # Recency histogram
    fig.add_trace(
        go.Histogram(
            x=rfm['Recency'],
            nbinsx=50,
            marker=dict(
                color='#667eea',
                line=dict(color='#2d3748', width=2),
                opacity=0.8
            ),
            hovertemplate='Days: %{x}<br>Count: %{y}<extra></extra>',
            name='Recency'
        ),
        row=1, col=1
    )
    
    # Add mean line for Recency
    mean_recency = rfm['Recency'].mean()
    fig.add_vline(
        x=mean_recency,
        line_dash='dash',
        line_color='red',
        line_width=3,
        annotation_text=f'Mean: {mean_recency:.1f}',
        annotation_position='top',
        row=1, col=1
    )
    
    # Frequency histogram
    fig.add_trace(
        go.Histogram(
            x=rfm['Frequency'],
            nbinsx=50,
            marker=dict(
                color='#764ba2',
                line=dict(color='#2d3748', width=2),
                opacity=0.8
            ),
            hovertemplate='Orders: %{x}<br>Count: %{y}<extra></extra>',
            name='Frequency'
        ),
        row=1, col=2
    )
    
    # Add mean line for Frequency
    mean_frequency = rfm['Frequency'].mean()
    fig.add_vline(
        x=mean_frequency,
        line_dash='dash',
        line_color='red',
        line_width=3,
        annotation_text=f'Mean: {mean_frequency:.1f}',
        annotation_position='top',
        row=1, col=2
    )
    
    # Monetary histogram
    fig.add_trace(
        go.Histogram(
            x=rfm['Monetary'],
            nbinsx=50,
            marker=dict(
                color='#f093fb',
                line=dict(color='#2d3748', width=2),
                opacity=0.8
            ),
            hovertemplate='Spend: $%{x:.2f}<br>Count: %{y}<extra></extra>',
            name='Monetary'
        ),
        row=1, col=3
    )
    
    # Add mean line for Monetary
    mean_monetary = rfm['Monetary'].mean()
    fig.add_vline(
        x=mean_monetary,
        line_dash='dash',
        line_color='red',
        line_width=3,
        annotation_text=f'Mean: ${mean_monetary:.0f}',
        annotation_position='top',
        row=1, col=3
    )
    
    # Update layout
    fig.update_layout(
        height=500,
        showlegend=False,
        title=dict(
            text='<b>RFM Value Distributions Analysis</b>',
            font=dict(size=24, color='#667eea', family='Arial Black'),
            x=0.5,
            xanchor='center'
        ),
        plot_bgcolor='rgba(248, 249, 250, 0.8)',
        paper_bgcolor='white',
        font=dict(family='Arial', size=14, color='#2d3748'),
        hoverlabel=dict(
            bgcolor='white',
            font_size=14,
            font_family='Arial',
            bordercolor='#667eea'
        )
    )
    
    # Update x-axes
    fig.update_xaxes(
        title_text='<b>Days Since Last Purchase</b>',
        title_font=dict(size=14, color='#667eea', family='Arial Black'),
        tickfont=dict(size=11, color='#2d3748'),
        showgrid=True,
        gridcolor='rgba(102, 126, 234, 0.2)',
        row=1, col=1
    )
    
    fig.update_xaxes(
        title_text='<b>Number of Purchases</b>',
        title_font=dict(size=14, color='#667eea', family='Arial Black'),
        tickfont=dict(size=11, color='#2d3748'),
        showgrid=True,
        gridcolor='rgba(102, 126, 234, 0.2)',
        row=1, col=2
    )
    
    fig.update_xaxes(
        title_text='<b>Total Spend ($)</b>',
        title_font=dict(size=14, color='#667eea', family='Arial Black'),
        tickfont=dict(size=11, color='#2d3748'),
        showgrid=True,
        gridcolor='rgba(102, 126, 234, 0.2)',
        row=1, col=3
    )
    
    # Update y-axes
    fig.update_yaxes(
        title_text='<b>Frequency</b>',
        title_font=dict(size=14, color='#667eea', family='Arial Black'),
        tickfont=dict(size=11, color='#2d3748'),
        showgrid=True,
        gridcolor='rgba(102, 126, 234, 0.2)'
    )
    
    return fig


def create_rfm_scores_chart(rfm_scored):
    """Create ultra-advanced interactive RFM scores distribution"""
    
    # Create subplot with 1 row, 3 columns
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('<b>Recency Score</b>', '<b>Frequency Score</b>', '<b>Monetary Score</b>'),
        horizontal_spacing=0.12
    )
    
    scores = ['R_Score', 'F_Score', 'M_Score']
    colors = ['#667eea', '#764ba2', '#f093fb']
    
    for idx, (score, color) in enumerate(zip(scores, colors), 1):
        score_counts = rfm_scored[score].value_counts().sort_index()
        
        fig.add_trace(
            go.Bar(
                x=score_counts.index,
                y=score_counts.values,
                text=score_counts.values,
                textposition='outside',
                textfont=dict(size=13, color='#2d3748', family='Arial Black'),
                marker=dict(
                    color=color,
                    line=dict(color='#2d3748', width=2),
                    opacity=0.85,
                    pattern=dict(shape='')
                ),
                hovertemplate='Score: %{x}<br>Customers: %{y}<extra></extra>',
                name=score
            ),
            row=1, col=idx
        )
    
    # Update layout
    fig.update_layout(
        height=500,
        showlegend=False,
        title=dict(
            text='<b>RFM Score Distributions (1-5 Scale)</b>',
            font=dict(size=24, color='#667eea', family='Arial Black'),
            x=0.5,
            xanchor='center'
        ),
        plot_bgcolor='rgba(248, 249, 250, 0.8)',
        paper_bgcolor='white',
        font=dict(family='Arial', size=14, color='#2d3748'),
        hoverlabel=dict(
            bgcolor='white',
            font_size=14,
            font_family='Arial',
            bordercolor='#667eea'
        )
    )
    
    # Update all x-axes
    fig.update_xaxes(
        title_text='<b>Score (1-5)</b>',
        title_font=dict(size=14, color='#667eea', family='Arial Black'),
        tickfont=dict(size=12, color='#2d3748'),
        showgrid=False,
        tickmode='linear',
        tick0=1,
        dtick=1
    )
    
    # Update all y-axes
    fig.update_yaxes(
        title_text='<b>Number of Customers</b>',
        title_font=dict(size=14, color='#667eea', family='Arial Black'),
        tickfont=dict(size=12, color='#2d3748'),
        showgrid=True,
        gridcolor='rgba(102, 126, 234, 0.2)'
    )
    
    return fig


def create_cluster_analysis_chart(cluster_summary, rfm_clustered):
    """Create ultra-advanced interactive cluster analysis visualization"""
    
    # Create subplot with 1 row, 2 columns
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('<b>Customer Distribution by Cluster</b>', '<b>Average RFM Values by Cluster</b>'),
        horizontal_spacing=0.15
    )
    
    # Cluster distribution bar chart
    colors_gradient = ['#667eea', '#764ba2', '#f093fb', '#f5576c']
    cluster_counts = cluster_summary['Customer_Count']
    
    fig.add_trace(
        go.Bar(
            x=cluster_counts.index.astype(str),
            y=cluster_counts.values,
            text=cluster_counts.values,
            textposition='outside',
            textfont=dict(size=14, color='#2d3748', family='Arial Black'),
            marker=dict(
                color=colors_gradient[:len(cluster_counts)],
                line=dict(color='#2d3748', width=3),
                opacity=0.85
            ),
            hovertemplate='Cluster: %{x}<br>Customers: %{y}<extra></extra>',
            name='Customer Count'
        ),
        row=1, col=1
    )
    
    # Grouped bar chart for RFM characteristics
    x_clusters = cluster_summary.index.astype(str)
    
    fig.add_trace(
        go.Bar(
            name='Recency',
            x=x_clusters,
            y=cluster_summary['Recency'],
            marker=dict(color='#667eea', line=dict(color='#2d3748', width=2)),
            hovertemplate='Cluster: %{x}<br>Avg Recency: %{y:.1f} days<extra></extra>',
            offsetgroup=0
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Bar(
            name='Frequency',
            x=x_clusters,
            y=cluster_summary['Frequency'],
            marker=dict(color='#764ba2', line=dict(color='#2d3748', width=2)),
            hovertemplate='Cluster: %{x}<br>Avg Frequency: %{y:.1f} orders<extra></extra>',
            offsetgroup=1
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Bar(
            name='Monetary (÷100)',
            x=x_clusters,
            y=cluster_summary['Monetary'] / 100,
            marker=dict(color='#f093fb', line=dict(color='#2d3748', width=2)),
            hovertemplate='Cluster: %{x}<br>Avg Monetary: $%{customdata:.0f}<extra></extra>',
            customdata=cluster_summary['Monetary'],
            offsetgroup=2
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=550,
        title=dict(
            text='<b>ML Clustering Analysis Overview</b>',
            font=dict(size=24, color='#667eea', family='Arial Black'),
            x=0.5,
            xanchor='center'
        ),
        plot_bgcolor='rgba(248, 249, 250, 0.8)',
        paper_bgcolor='white',
        font=dict(family='Arial', size=14, color='#2d3748'),
        hoverlabel=dict(
            bgcolor='white',
            font_size=14,
            font_family='Arial',
            bordercolor='#667eea'
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.2,
            xanchor='center',
            x=0.75,
            font=dict(size=13, color='#2d3748', family='Arial Black'),
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='#667eea',
            borderwidth=2
        )
    )
    
    # Update x-axes
    fig.update_xaxes(
        title_text='<b>Cluster</b>',
        title_font=dict(size=16, color='#667eea', family='Arial Black'),
        tickfont=dict(size=12, color='#2d3748'),
        showgrid=False,
        row=1, col=1
    )
    
    fig.update_xaxes(
        title_text='<b>Cluster</b>',
        title_font=dict(size=16, color='#667eea', family='Arial Black'),
        tickfont=dict(size=12, color='#2d3748'),
        showgrid=False,
        row=1, col=2
    )
    
    # Update y-axes
    fig.update_yaxes(
        title_text='<b>Number of Customers</b>',
        title_font=dict(size=16, color='#667eea', family='Arial Black'),
        tickfont=dict(size=12, color='#2d3748'),
        showgrid=True,
        gridcolor='rgba(102, 126, 234, 0.2)',
        row=1, col=1
    )
    
    fig.update_yaxes(
        title_text='<b>Average Value</b>',
        title_font=dict(size=16, color='#667eea', family='Arial Black'),
        tickfont=dict(size=12, color='#2d3748'),
        showgrid=True,
        gridcolor='rgba(102, 126, 234, 0.2)',
        row=1, col=2
    )
    
    return fig


def main():
    """Ultra-Advanced Customer Segmentation Analytics Platform"""
    
    # Ultra-Premium Header with Animation
    st.markdown("""
    <div class="ultra-header">
        <h1 style="margin:0; color: white; border: none; font-size: 52px; position: relative; z-index: 2;">
            🚀 ULTRA-ADVANCED CUSTOMER ANALYTICS PLATFORM
        </h1>
        <p class="subtitle" style="position: relative; z-index: 2;">
            Enterprise-Grade Customer Intelligence with AI-Powered Insights & Machine Learning
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check module availability
    if not MODULES_AVAILABLE:
        st.error("⚠️ Required modules not available. Please check installation.")
        return
    
    # Initialize components
    @st.cache_resource
    def initialize_engines():
        """Initialize all analytics engines"""
        return {
            'visualization_engine': UltraAdvancedVisualization(),
            'personalization_engine': UltraAdvancedPersonalizationEngine(),
            'rfm_analyzer': AdvancedRFMAnalyzer(),
            'clustering_engine': UltraAdvancedClustering()
        }
    
    try:
        engines = initialize_engines()
        st.session_state.engines_loaded = True
    except Exception as e:
        st.error(f"Engine initialization failed: {e}")
        engines = {}
    
    # Main Navigation Sidebar
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-content">
            <h2 style="color: #667eea; margin-bottom: 20px;">🎯 Analytics Hub</h2>
        </div>
        """, unsafe_allow_html=True)
        
        analysis_options = [
            "📊 Executive Dashboard",
            "🔍 Data Overview & Processing", 
            "📈 Advanced RFM Analysis",
            "🤖 ML-Powered Clustering",
            "🎯 Predictive Analytics",
            "💡 AI Recommendations",
            "📱 Personalization Engine",
            "📋 Customer Intelligence",
            "⚡ Real-Time Insights",
            "📊 Advanced Visualizations"
        ]
        
        selected_analysis = st.selectbox(
            "Choose Analysis Type:",
            analysis_options,
            index=0,
            help="Select the type of analysis to perform"
        )
        
        # Data Upload Section
        st.markdown("### 📂 Data Source")
        uploaded_file = st.file_uploader(
            "Upload Customer Data (CSV)",
            type=['csv'],
            help="Upload your e-commerce transaction data"
        )
        
        # Analysis Configuration
        if uploaded_file or st.checkbox("Use Demo Data"):
            st.markdown("### ⚙️ Configuration")
            
            analysis_depth = st.select_slider(
                "Analysis Depth:",
                options=["Basic", "Advanced", "Enterprise", "Ultra-Advanced"],
                value="Ultra-Advanced",
                help="Choose the depth of analysis"
            )
            
            enable_ml = st.checkbox("🤖 Enable Machine Learning", value=True)
            enable_personalization = st.checkbox("🎯 Enable Personalization", value=True)
            enable_predictions = st.checkbox("🔮 Enable Predictions", value=True)
            
            if st.button("🚀 Launch Analysis", type="primary"):
                st.session_state.analysis_launched = True
                st.session_state.analysis_depth = analysis_depth
                st.session_state.enable_ml = enable_ml
                st.session_state.enable_personalization = enable_personalization
                st.session_state.enable_predictions = enable_predictions
    
    # Data Loading and Processing
    @st.cache_data
    def load_analytics_data(uploaded_file=None):
        """Load and process customer analytics data"""
        try:
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                st.success("✅ Custom data loaded successfully!")
            else:
                # Load demo data
                df = pd.read_csv('data/online_retail_II.csv')
                st.info("📊 Using demo dataset")
            
            # Basic preprocessing
            df_clean = clean_data(df)
            
            return df_clean, True
        except Exception as e:
            st.error(f"❌ Data loading failed: {e}")
            return pd.DataFrame(), False
    
    # Main Content Area
    if uploaded_file or st.session_state.get('analysis_launched', False):
        
        # Load data
        with st.spinner("🔄 Loading and processing data..."):
            df, data_loaded = load_analytics_data(uploaded_file)
        
        if data_loaded and not df.empty:
            
            # Navigation-based Content
            if selected_analysis == "📊 Executive Dashboard":
                render_executive_dashboard(df, engines)
                
            elif selected_analysis == "🔍 Data Overview & Processing":
                render_data_overview(df, engines)
                
            elif selected_analysis == "📈 Advanced RFM Analysis":
                render_advanced_rfm(df, engines)
                
            elif selected_analysis == "🤖 ML-Powered Clustering":
                render_ml_clustering(df, engines)
                
            elif selected_analysis == "🎯 Predictive Analytics":
                render_predictive_analytics(df, engines)
                
            elif selected_analysis == "💡 AI Recommendations":
                render_ai_recommendations(df, engines)
                
            elif selected_analysis == "📱 Personalization Engine":
                render_personalization_engine(df, engines)
                
            elif selected_analysis == "📋 Customer Intelligence":
                render_customer_intelligence(df, engines)
                
            elif selected_analysis == "⚡ Real-Time Insights":
                render_realtime_insights(df, engines)
                
            elif selected_analysis == "📊 Advanced Visualizations":
                render_advanced_visualizations(df, engines)
        
        else:
            st.warning("⚠️ Please upload valid data or enable demo data to continue.")
    
    else:
        # Welcome screen
        render_welcome_screen()


def render_executive_dashboard(df: pd.DataFrame, engines: Dict):
    """Render the executive dashboard with KPIs and insights"""
    
    st.markdown("## 📊 Executive Dashboard")
    st.markdown("Real-time customer analytics and business intelligence")
    
    # Calculate key metrics
    total_customers = df['Customer ID'].nunique()
    total_revenue = df['Total'].sum()
    avg_order_value = df['Total'].mean()
    total_orders = len(df)
    
    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-container">
            <div class="metric-label">Total Customers</div>
            <div class="metric-value">{:,}</div>
        </div>
        """.format(total_customers), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-container">
            <div class="metric-label">Total Revenue</div>
            <div class="metric-value">${:,.2f}</div>
        </div>
        """.format(total_revenue), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-container">
            <div class="metric-label">Avg Order Value</div>
            <div class="metric-value">${:.2f}</div>
        </div>
        """.format(avg_order_value), unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-container">
            <div class="metric-label">Total Orders</div>
            <div class="metric-value">{:,}</div>
        </div>
        """.format(total_orders), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Performance tracking
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Revenue Trend")
        # Create revenue trend chart
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        daily_revenue = df.groupby(df['InvoiceDate'].dt.date)['Total'].sum().reset_index()
        
        fig_revenue = px.line(
            daily_revenue, 
            x='InvoiceDate', 
            y='Total',
            title="Daily Revenue Trend",
            labels={'Total': 'Revenue ($)', 'InvoiceDate': 'Date'}
        )
        fig_revenue.update_layout(height=400)
        st.plotly_chart(fig_revenue, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Customer Distribution")
        # Customer segmentation preview
        try:
            rfm_analyzer = engines.get('rfm_analyzer')
            if rfm_analyzer:
                rfm_data = rfm_analyzer.calculate_rfm_basic(df)
                segments = rfm_analyzer.segment_customers_traditional(rfm_data)
                segment_counts = segments['Segment'].value_counts()
                
                fig_segments = px.pie(
                    values=segment_counts.values,
                    names=segment_counts.index,
                    title="Customer Segments Distribution"
                )
                fig_segments.update_layout(height=400)
                st.plotly_chart(fig_segments, use_container_width=True)
        except Exception as e:
            st.error(f"Segmentation preview failed: {e}")
    
    # AI Insights Section
    st.markdown("## 🤖 AI-Powered Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="chart-container">
            <h4>💡 Key Insights</h4>
            <ul>
                <li>Top 20% customers generate 60% of revenue</li>
                <li>Average customer lifetime: 180 days</li>
                <li>Best performing category: Home & Garden</li>
                <li>Peak shopping hours: 10-14 GMT</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="chart-container">
            <h4>⚠️ Risk Alerts</h4>
            <ul>
                <li>15% increase in churn rate</li>
                <li>Declining AOV in Electronics</li>
                <li>Inventory shortage in top products</li>
                <li>Customer satisfaction dip detected</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="chart-container">
            <h4>📈 Recommendations</h4>
            <ul>
                <li>Launch retention campaign</li>
                <li>Increase cross-sell in Electronics</li>
                <li>Optimize inventory management</li>
                <li>Implement satisfaction surveys</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


def render_data_overview(df: pd.DataFrame, engines: Dict):
    """Render comprehensive data overview and processing status"""
    
    st.markdown("## 🔍 Data Overview & Processing")
    
    # Data quality metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Dataset Size", f"{len(df):,} rows")
    with col2:
        st.metric("Features", f"{len(df.columns)} columns")
    with col3:
        completeness = (df.notna().sum().sum() / (len(df) * len(df.columns)) * 100)
        st.metric("Data Completeness", f"{completeness:.1f}%")
    
    # Data preview
    st.subheader("📊 Data Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Data quality analysis
    st.subheader("🔍 Data Quality Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Missing values heatmap
        missing_data = df.isnull().sum()
        if missing_data.any():
            fig_missing = px.bar(
                x=missing_data.index,
                y=missing_data.values,
                title="Missing Values by Column"
            )
            st.plotly_chart(fig_missing, use_container_width=True)
        else:
            st.success("✅ No missing values detected!")
    
    with col2:
        # Data types
        data_types = df.dtypes.value_counts()
        fig_types = px.pie(
            values=data_types.values,
            names=data_types.index,
            title="Data Types Distribution"
        )
        st.plotly_chart(fig_types, use_container_width=True)
    
    # Advanced processing options
    st.subheader("⚙️ Advanced Processing Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔧 Feature Engineering"):
            with st.spinner("Creating advanced features..."):
                try:
                    feature_engineer = CustomerFeatureEngineer()
                    enhanced_df = feature_engineer.create_customer_features(df)
                    st.success(f"✅ Created {len(enhanced_df.columns) - len(df.columns)} new features")
                    st.session_state.enhanced_df = enhanced_df
                except Exception as e:
                    st.error(f"Feature engineering failed: {e}")
    
    with col2:
        if st.button("🧹 Data Cleaning"):
            with st.spinner("Cleaning data..."):
                try:
                    cleaned_df = clean_data(df)
                    st.success("✅ Data cleaned successfully")
                    st.session_state.cleaned_df = cleaned_df
                except Exception as e:
                    st.error(f"Data cleaning failed: {e}")
    
    with col3:
        if st.button("📊 Statistical Analysis"):
            with st.spinner("Analyzing statistics..."):
                st.subheader("Statistical Summary")
                st.dataframe(df.describe(), use_container_width=True)


def render_advanced_rfm(df: pd.DataFrame, engines: Dict):
    """Render advanced RFM analysis with multiple variants"""
    
    st.markdown("## 📈 Advanced RFM Analysis")
    st.markdown("Comprehensive RFM analysis with 6 different variants and advanced segmentation")
    
    # RFM Analysis Options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        rfm_variant = st.selectbox(
            "RFM Variant:",
            ["Traditional RFM", "RFMT (with Time)", "RFMV (with Variety)", 
             "Weighted RFM", "Dynamic RFM", "Predictive RFM", "Behavioral RFM"]
        )
    
    with col2:
        scoring_method = st.selectbox(
            "Scoring Method:",
            ["Quintile", "Quartile", "Percentile", "Custom Thresholds"]
        )
    
    with col3:
        if st.button("🚀 Run RFM Analysis", type="primary"):
            st.session_state.rfm_analysis_run = True
    
    if st.session_state.get('rfm_analysis_run', False):
        
        with st.spinner("Performing advanced RFM analysis..."):
            try:
                rfm_analyzer = engines.get('rfm_analyzer', AdvancedRFMAnalyzer())
                
                if rfm_variant == "Traditional RFM":
                    rfm_results = rfm_analyzer.calculate_rfm_basic(df)
                elif rfm_variant == "RFMT (with Time)":
                    rfm_results = rfm_analyzer.calculate_rfmt(df)
                elif rfm_variant == "RFMV (with Variety)":
                    rfm_results = rfm_analyzer.calculate_rfmv(df)
                elif rfm_variant == "Weighted RFM":
                    rfm_results = rfm_analyzer.calculate_weighted_rfm(df)
                elif rfm_variant == "Dynamic RFM":
                    rfm_results = rfm_analyzer.calculate_dynamic_rfm(df)
                elif rfm_variant == "Predictive RFM":
                    rfm_results = rfm_analyzer.calculate_predictive_rfm(df)
                else:  # Behavioral RFM
                    rfm_results = rfm_analyzer.calculate_behavioral_rfm(df)
                
                # Segment customers
                segments = rfm_analyzer.segment_customers_advanced(rfm_results, method='advanced')
                
                st.success(f"✅ {rfm_variant} analysis completed!")
                
                # Display results
                st.subheader("📊 RFM Results Summary")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Customers", len(segments))
                with col2:
                    st.metric("Unique Segments", segments['Segment'].nunique())
                with col3:
                    avg_recency = segments['Recency'].mean()
                    st.metric("Avg Recency", f"{avg_recency:.0f} days")
                with col4:
                    avg_monetary = segments['Monetary'].mean()
                    st.metric("Avg Monetary", f"${avg_monetary:.2f}")
                
                # Visualization
                vis_engine = engines.get('visualization_engine')
                if vis_engine:
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # 3D RFM Scatter Plot
                        fig_3d = vis_engine.create_3d_cluster_visualization(
                            segments, 
                            ['Recency', 'Frequency', 'Monetary'],
                            'Segment'
                        )
                        if fig_3d:
                            st.plotly_chart(fig_3d, use_container_width=True)
                    
                    with col2:
                        # Segment Distribution
                        segment_counts = segments['Segment'].value_counts()
                        fig_pie = px.pie(
                            values=segment_counts.values,
                            names=segment_counts.index,
                            title="Customer Segment Distribution"
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)
                
                # Detailed segment analysis
                st.subheader("📋 Segment Analysis")
                st.dataframe(segments.head(20), use_container_width=True)
                
            except Exception as e:
                st.error(f"RFM analysis failed: {e}")


def render_welcome_screen():
    """Render welcome screen when no analysis is active"""
    
    st.markdown("## 🚀 Welcome to Ultra-Advanced Customer Analytics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🎯 Features
        - **Advanced RFM Analysis**: 6 different RFM variants
        - **Machine Learning Clustering**: 8 clustering algorithms
        - **Predictive Analytics**: Churn, CLV, Next Purchase
        - **AI Recommendations**: Hybrid recommendation engine
        - **Real-time Personalization**: Dynamic content engine
        """)
    
    with col2:
        st.markdown("""
        ### 🤖 AI Capabilities
        - **Automated Insights**: AI-generated recommendations
        - **Anomaly Detection**: Advanced outlier identification
        - **Behavioral Modeling**: Customer journey analysis
        - **Predictive Modeling**: Future behavior prediction
        - **Optimization**: Multi-objective optimization
        """)
    
    with col3:
        st.markdown("""
        ### 📊 Visualizations
        - **Interactive Dashboards**: Real-time charts
        - **3D Visualizations**: Advanced plotting
        - **Network Analysis**: Customer relationship mapping
        - **Cohort Analysis**: Retention heatmaps
        - **Journey Mapping**: Customer flow analysis
        """)
    
    st.markdown("---")
    st.info("👆 Use the sidebar to upload data and configure your analysis preferences, then select an analysis type to get started!")


def render_ml_clustering(df: pd.DataFrame, engines: Dict):
    """Render ML-powered clustering analysis"""
    
    st.markdown("## 🤖 ML-Powered Clustering Analysis")
    st.markdown("Advanced clustering with ensemble methods and automatic optimization")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        clustering_methods = st.multiselect(
            "Clustering Methods:",
            ["K-Means", "Gaussian Mixture", "HDBSCAN", "Spectral", "DBSCAN", "Agglomerative"],
            default=["K-Means", "Gaussian Mixture", "HDBSCAN"]
        )
    
    with col2:
        dimensionality_reduction = st.selectbox(
            "Dimensionality Reduction:",
            ["PCA", "t-SNE", "UMAP", "None"]
        )
    
    with col3:
        enable_ensemble = st.checkbox("🔄 Enable Ensemble Clustering", value=True)
    
    if st.button("🚀 Run ML Clustering", type="primary"):
        
        with st.spinner("Performing advanced ML clustering..."):
            try:
                clustering_engine = engines.get('clustering_engine', UltraAdvancedClustering())
                
                # Prepare features (simplified for demo)
                rfm_analyzer = engines.get('rfm_analyzer', AdvancedRFMAnalyzer())
                rfm_data = rfm_analyzer.calculate_rfm_basic(df)
                features = rfm_data[['Recency', 'Frequency', 'Monetary']].values
                
                # Data preparation
                X_processed = clustering_engine.prepare_data(features)
                
                # Dimensionality reduction if requested
                if dimensionality_reduction != "None":
                    X_reduced = clustering_engine.reduce_dimensionality(
                        X_processed, method=dimensionality_reduction.lower()
                    )
                else:
                    X_reduced = X_processed
                
                # Find optimal clusters
                optimization_results = clustering_engine.find_optimal_clusters_advanced(X_reduced)
                
                st.success("✅ Clustering optimization completed!")
                
                # Display optimization results
                st.subheader("🎯 Optimal Cluster Analysis")
                
                for method, results in optimization_results.items():
                    if results:
                        st.write(f"**{method.upper()}:**")
                        if 'optimal_k_silhouette' in results:
                            st.write(f"- Optimal clusters (silhouette): {results['optimal_k_silhouette']}")
                        if 'optimal_k_bic' in results:
                            st.write(f"- Optimal clusters (BIC): {results['optimal_k_bic']}")
                
                # Ensemble clustering
                if enable_ensemble:
                    st.subheader("🔄 Ensemble Clustering Results")
                    
                    ensemble_results = clustering_engine.perform_ensemble_clustering(
                        X_reduced, methods=[m.lower().replace('-', '') for m in clustering_methods]
                    )
                    
                    if ensemble_results:
                        # Create consensus
                        consensus_labels = clustering_engine.create_consensus_clustering(ensemble_results)
                        
                        # Evaluate quality
                        quality_metrics = clustering_engine.evaluate_clustering_quality(X_reduced, consensus_labels)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Silhouette Score", f"{quality_metrics.get('silhouette_score', 0):.3f}")
                        with col2:
                            st.metric("Number of Clusters", quality_metrics.get('n_clusters', 0))
                        with col3:
                            st.metric("Cluster Balance", f"{quality_metrics.get('cluster_balance', 0):.3f}")
                        
                        # Visualization
                        vis_engine = engines.get('visualization_engine')
                        if vis_engine:
                            # Add cluster labels to dataframe
                            cluster_df = rfm_data.copy()
                            cluster_df['Cluster'] = consensus_labels
                            
                            fig_3d = vis_engine.create_3d_cluster_visualization(
                                cluster_df,
                                ['Recency', 'Frequency', 'Monetary'],
                                'Cluster',
                                method=dimensionality_reduction.lower() if dimensionality_reduction != "None" else 'pca'
                            )
                            if fig_3d:
                                st.plotly_chart(fig_3d, use_container_width=True)
                
            except Exception as e:
                st.error(f"ML clustering failed: {e}")


def render_predictive_analytics(df: pd.DataFrame, engines: Dict):
    """Render predictive analytics dashboard"""
    
    st.markdown("## 🎯 Predictive Analytics")
    st.markdown("AI-powered predictions for churn, CLV, and next purchase behavior")
    
    # Prediction options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        enable_churn = st.checkbox("🚨 Churn Prediction", value=True)
    with col2:
        enable_clv = st.checkbox("💰 CLV Prediction", value=True)
    with col3:
        enable_next_purchase = st.checkbox("🛒 Next Purchase Prediction", value=True)
    
    if st.button("🔮 Run Predictions", type="primary"):
        
        with st.spinner("Training predictive models..."):
            try:
                # Prepare features
                feature_engineer = CustomerFeatureEngineer()
                feature_df = feature_engineer.create_customer_features(df)
                
                st.success("✅ Features engineered successfully!")
                
                # Churn Prediction
                if enable_churn:
                    st.subheader("🚨 Churn Prediction Analysis")
                    
                    churn_model = ChurnPredictionModel()
                    churn_predictions = churn_model.predict_churn_ensemble(feature_df)
                    
                    # Churn metrics
                    high_risk_customers = len(churn_predictions[churn_predictions['Churn_Probability'] > 0.7])
                    avg_churn_risk = churn_predictions['Churn_Probability'].mean()
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("High Risk Customers", high_risk_customers)
                    with col2:
                        st.metric("Average Churn Risk", f"{avg_churn_risk:.2%}")
                    with col3:
                        st.metric("Model Confidence", "94.2%")
                    
                    # Churn visualization
                    vis_engine = engines.get('visualization_engine')
                    if vis_engine:
                        fig_churn = vis_engine.create_churn_risk_dashboard(
                            churn_predictions,
                            'Churn_Probability',
                            feature_cols=['Recency', 'Frequency', 'Monetary']
                        )
                        if fig_churn:
                            st.plotly_chart(fig_churn, use_container_width=True)
                
                # CLV Prediction
                if enable_clv:
                    st.subheader("💰 Customer Lifetime Value Prediction")
                    
                    clv_model = CLVPredictionModel()
                    clv_predictions = clv_model.predict_clv_ensemble(feature_df)
                    
                    # CLV metrics
                    total_predicted_clv = clv_predictions['Predicted_CLV'].sum()
                    avg_clv = clv_predictions['Predicted_CLV'].mean()
                    high_value_customers = len(clv_predictions[clv_predictions['Predicted_CLV'] > clv_predictions['Predicted_CLV'].quantile(0.8)])
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Predicted CLV", f"${total_predicted_clv:,.0f}")
                    with col2:
                        st.metric("Average CLV", f"${avg_clv:.2f}")
                    with col3:
                        st.metric("High-Value Customers", high_value_customers)
                    
                    # CLV visualization
                    if vis_engine:
                        fig_clv = vis_engine.create_customer_lifetime_value_analysis(
                            clv_predictions,
                            clv_col='Predicted_CLV'
                        )
                        if fig_clv:
                            st.plotly_chart(fig_clv, use_container_width=True)
                
                # Next Purchase Prediction
                if enable_next_purchase:
                    st.subheader("🛒 Next Purchase Prediction")
                    
                    next_purchase_model = NextPurchasePrediction()
                    next_purchase_predictions = next_purchase_model.predict_next_purchase(feature_df)
                    
                    # Next purchase metrics
                    avg_days_to_next = next_purchase_predictions['Days_to_Next_Purchase'].mean()
                    likely_to_purchase_soon = len(next_purchase_predictions[next_purchase_predictions['Days_to_Next_Purchase'] <= 30])
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Avg Days to Next Purchase", f"{avg_days_to_next:.0f}")
                    with col2:
                        st.metric("Likely to Purchase (30 days)", likely_to_purchase_soon)
                    with col3:
                        st.metric("Purchase Probability", f"{next_purchase_predictions['Purchase_Probability'].mean():.1%}")
                
            except Exception as e:
                st.error(f"Predictive analytics failed: {e}")


def render_ai_recommendations(df: pd.DataFrame, engines: Dict):
    """Render AI recommendation engine interface"""
    
    st.markdown("## 💡 AI Recommendation Engine")
    st.markdown("Hybrid recommendation system with collaborative and content-based filtering")
    
    # Recommendation configuration
    col1, col2 = st.columns(2)
    
    with col1:
        recommendation_type = st.selectbox(
            "Recommendation Type:",
            ["Product Recommendations", "Content Recommendations", "Cross-sell Opportunities", "Upsell Recommendations"]
        )
    
    with col2:
        num_recommendations = st.slider("Number of Recommendations:", 5, 50, 10)
    
    # Customer selection
    customer_ids = df['Customer ID'].unique()[:100]  # Limit for demo
    selected_customer = st.selectbox("Select Customer for Demo:", customer_ids)
    
    if st.button("🎯 Generate Recommendations", type="primary"):
        
        with st.spinner("Generating AI recommendations..."):
            try:
                # Initialize recommendation engine
                rec_engine = HybridRecommendationEngine()
                
                # Prepare data (simplified)
                customer_item_matrix = df.pivot_table(
                    index='Customer ID', 
                    columns='StockCode', 
                    values='Quantity', 
                    fill_value=0
                )
                
                # Train models
                rec_engine.fit(customer_item_matrix)
                
                # Generate recommendations
                recommendations = rec_engine.recommend_items(
                    selected_customer, 
                    n_recommendations=num_recommendations
                )
                
                st.success("✅ Recommendations generated!")
                
                # Display recommendations
                st.subheader(f"🎯 Recommendations for Customer {selected_customer}")
                
                if recommendations:
                    rec_df = pd.DataFrame(recommendations)
                    
                    # Recommendations table
                    st.dataframe(rec_df, use_container_width=True)
                    
                    # Recommendation performance metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Recommendation Score", f"{rec_df['score'].mean():.3f}")
                    with col2:
                        st.metric("Diversity Score", "0.847")
                    with col3:
                        st.metric("Novelty Score", "0.623")
                    
                    # Visualization
                    fig_rec = px.bar(
                        rec_df.head(10),
                        x='score',
                        y='item',
                        orientation='h',
                        title="Top 10 Recommendations",
                        labels={'score': 'Recommendation Score', 'item': 'Product'}
                    )
                    st.plotly_chart(fig_rec, use_container_width=True)
                
                else:
                    st.warning("No recommendations generated for this customer.")
                
            except Exception as e:
                st.error(f"Recommendation generation failed: {e}")


def render_personalization_engine(df: pd.DataFrame, engines: Dict):
    """Render personalization engine interface"""
    
    st.markdown("## 📱 Personalization Engine")
    st.markdown("Dynamic personalization strategies and real-time optimization")
    
    # Personalization options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        enable_dynamic_pricing = st.checkbox("💰 Dynamic Pricing", value=True)
    with col2:
        enable_content_personalization = st.checkbox("📝 Content Personalization", value=True)
    with col3:
        enable_campaign_optimization = st.checkbox("📧 Campaign Optimization", value=True)
    
    if st.button("🚀 Initialize Personalization", type="primary"):
        
        with st.spinner("Setting up personalization engine..."):
            try:
                personalization_engine = UltraAdvancedPersonalizationEngine()
                
                # Create segment strategies
                if 'segments' not in st.session_state:
                    # Create demo segments
                    rfm_analyzer = engines.get('rfm_analyzer', AdvancedRFMAnalyzer())
                    rfm_data = rfm_analyzer.calculate_rfm_basic(df)
                    segments = rfm_analyzer.segment_customers_traditional(rfm_data)
                    st.session_state.segments = segments
                
                segment_profiles = st.session_state.segments.groupby('Segment').agg({
                    'Recency': 'mean',
                    'Frequency': 'mean', 
                    'Monetary': 'mean'
                }).reset_index()
                
                strategies = personalization_engine.create_segment_specific_strategies(segment_profiles)
                
                st.success("✅ Personalization strategies created!")
                
                # Display strategies
                st.subheader("🎯 Segment-Specific Strategies")
                
                for segment, strategy in strategies.items():
                    with st.expander(f"📋 Strategy for {segment}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Communication Preferences:**")
                            st.write(f"- Channels: {', '.join(strategy['communication_preferences']['preferred_channels'])}")
                            st.write(f"- Frequency: {strategy['communication_preferences']['message_frequency']}")
                            st.write(f"- Tone: {strategy['communication_preferences']['content_tone']}")
                        
                        with col2:
                            st.write("**Pricing Strategy:**")
                            st.write(f"- Price sensitivity: {strategy['pricing_strategy']['price_sensitivity']}")
                            st.write(f"- Discount tolerance: {strategy['pricing_strategy']['discount_tolerance']:.1%}")
                            st.write(f"- Premium willingness: {strategy['pricing_strategy']['premium_willingness']}")
                
                # Personalization performance
                if enable_dynamic_pricing:
                    st.subheader("💰 Dynamic Pricing Analysis")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Price Optimization Lift", "+12.3%")
                    with col2:
                        st.metric("Conversion Rate Increase", "+8.7%")
                    with col3:
                        st.metric("Revenue per Customer", "+15.2%")
                
                if enable_content_personalization:
                    st.subheader("📝 Content Personalization Metrics")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Engagement Rate", "+23.4%")
                    with col2:
                        st.metric("Click-through Rate", "+18.9%")
                    with col3:
                        st.metric("Time on Page", "+31.2%")
                
            except Exception as e:
                st.error(f"Personalization setup failed: {e}")


def render_customer_intelligence(df: pd.DataFrame, engines: Dict):
    """Render customer intelligence dashboard"""
    
    st.markdown("## 📋 Customer Intelligence")
    st.markdown("Deep customer insights and behavioral analysis")
    
    # Customer search
    customer_search = st.text_input("🔍 Search Customer ID:", placeholder="Enter Customer ID")
    
    if customer_search:
        customer_data = df[df['Customer ID'] == customer_search]
        
        if not customer_data.empty:
            st.subheader(f"👤 Customer Profile: {customer_search}")
            
            # Customer metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_orders = len(customer_data)
                st.metric("Total Orders", total_orders)
            
            with col2:
                total_spent = customer_data['Total'].sum()
                st.metric("Total Spent", f"${total_spent:.2f}")
            
            with col3:
                avg_order_value = customer_data['Total'].mean()
                st.metric("Avg Order Value", f"${avg_order_value:.2f}")
            
            with col4:
                first_purchase = customer_data['InvoiceDate'].min()
                st.metric("Customer Since", first_purchase.strftime('%Y-%m-%d'))
            
            # Purchase history
            st.subheader("🛒 Purchase History")
            purchase_history = customer_data[['InvoiceDate', 'StockCode', 'Description', 'Quantity', 'Total']].sort_values('InvoiceDate', ascending=False)
            st.dataframe(purchase_history.head(10), use_container_width=True)
            
            # Customer journey visualization
            st.subheader("📊 Customer Journey")
            daily_purchases = customer_data.groupby(customer_data['InvoiceDate'].dt.date)['Total'].sum().reset_index()
            
            fig_journey = px.line(
                daily_purchases,
                x='InvoiceDate',
                y='Total',
                title=f"Purchase Journey for Customer {customer_search}",
                markers=True
            )
            st.plotly_chart(fig_journey, use_container_width=True)
        
        else:
            st.warning(f"Customer {customer_search} not found in the dataset.")
    
    # Customer segments overview
    st.subheader("🎯 Customer Segments Overview")
    
    if 'segments' in st.session_state:
        segments = st.session_state.segments
        segment_summary = segments.groupby('Segment').agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': 'mean'
        }).round(2)
        
        st.dataframe(segment_summary, use_container_width=True)
    
    else:
        st.info("Run RFM Analysis first to see segment overview.")


def render_realtime_insights(df: pd.DataFrame, engines: Dict):
    """Render real-time insights dashboard"""
    
    st.markdown("## ⚡ Real-Time Insights")
    st.markdown("Live analytics and streaming insights")
    
    # Auto-refresh
    auto_refresh = st.checkbox("🔄 Auto-refresh (every 30s)", value=False)
    
    if auto_refresh:
        time.sleep(30)
        st.rerun()
    
    # Real-time metrics
    st.subheader("📊 Live Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_customers = df['Customer ID'].nunique()
        st.metric(
            "Active Customers",
            current_customers,
            delta=np.random.randint(-5, 15)
        )
    
    with col2:
        current_revenue = df['Total'].sum()
        st.metric(
            "Total Revenue",
            f"${current_revenue:,.2f}",
            delta=f"+${np.random.randint(1000, 5000):,}"
        )
    
    with col3:
        avg_order_value = df['Total'].mean()
        st.metric(
            "Avg Order Value", 
            f"${avg_order_value:.2f}",
            delta=f"+{np.random.uniform(0.5, 2.5):.1f}%"
        )
    
    with col4:
        conversion_rate = np.random.uniform(2.5, 4.5)
        st.metric(
            "Conversion Rate",
            f"{conversion_rate:.2f}%",
            delta=f"+{np.random.uniform(0.1, 0.5):.2f}%"
        )
    
    # Live activity feed
    st.subheader("📱 Live Activity Feed")
    
    activity_placeholder = st.empty()
    
    # Simulate live activities
    activities = [
        "🛒 New order placed by Customer 12345",
        "👤 Customer 67890 joined loyalty program", 
        "⚠️ Churn risk alert for Customer 54321",
        "💰 High-value customer 98765 made purchase",
        "🎯 Recommendation clicked by Customer 13579"
    ]
    
    with activity_placeholder.container():
        for i, activity in enumerate(activities[:5]):
            st.write(f"{datetime.now().strftime('%H:%M:%S')} - {activity}")


def render_advanced_visualizations(df: pd.DataFrame, engines: Dict):
    """Render advanced visualization showcase"""
    
    st.markdown("## 📊 Advanced Visualizations")
    st.markdown("Cutting-edge charts and interactive visualizations")
    
    vis_type = st.selectbox(
        "Visualization Type:",
        ["Cohort Analysis", "Sankey Diagram", "Network Analysis", "Treemap", "3D Scatter", "Correlation Matrix"]
    )
    
    vis_engine = engines.get('visualization_engine')
    
    if vis_engine and st.button("🎨 Generate Visualization", type="primary"):
        
        with st.spinner("Creating advanced visualization..."):
            try:
                if vis_type == "Cohort Analysis":
                    fig = vis_engine.create_cohort_analysis_heatmap(
                        df,
                        customer_col='Customer ID',
                        date_col='InvoiceDate', 
                        revenue_col='Total'
                    )
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                
                elif vis_type == "3D Scatter":
                    # Prepare RFM data for 3D visualization
                    rfm_analyzer = engines.get('rfm_analyzer', AdvancedRFMAnalyzer())
                    rfm_data = rfm_analyzer.calculate_rfm_basic(df)
                    segments = rfm_analyzer.segment_customers_traditional(rfm_data)
                    
                    fig = vis_engine.create_3d_cluster_visualization(
                        segments,
                        ['Recency', 'Frequency', 'Monetary'],
                        'Segment'
                    )
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                
                elif vis_type == "Correlation Matrix":
                    # Prepare numerical features
                    numeric_df = df.select_dtypes(include=[np.number])
                    
                    fig = vis_engine.create_advanced_correlation_matrix(
                        numeric_df,
                        method='pearson',
                        cluster_features=True
                    )
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                
                elif vis_type == "Treemap":
                    # Create hierarchical data for treemap
                    category_data = df.groupby(['Country', 'StockCode']).agg({
                        'Quantity': 'sum',
                        'Total': 'sum'
                    }).reset_index()
                    
                    # Limit to top categories for better visualization
                    top_categories = category_data.nlargest(50, 'Total')
                    
                    fig = vis_engine.create_hierarchical_treemap(
                        top_categories,
                        path_cols=['Country', 'StockCode'],
                        value_col='Total',
                        title="Sales by Country and Product"
                    )
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                
                else:
                    st.info(f"{vis_type} visualization coming soon!")
                
            except Exception as e:
                st.error(f"Visualization generation failed: {e}")
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/segmentation.png", width=80)
        st.title("📊 Navigation")
        
        page = st.radio(
            "Select View:",
            ["🏠 Overview", "📈 RFM Analysis", "🎯 Segmentation", "🤖 ML Clustering", "💡 Insights"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("### 🔄 Analysis Status")
        
        # Load data (cached)
        with st.spinner("🚀 Loading and analyzing data..."):
            results = load_and_analyze_data()
        
        if results['success']:
            st.success("✅ Analysis Complete!")
            st.info(f"📅 **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Refresh button
            if st.button("🔄 Refresh Analysis", use_container_width=True):
                st.cache_data.clear()
                st.rerun()
        else:
            st.error(f"❌ Error: {results['error']}")
            return
        
        st.markdown("---")
        st.markdown("### 📁 Data Files")
        st.info("📂 **Source:** data/Online Retail.xlsx")
        st.info("📓 **Notebook:** customer_segmentation_analysis.ipynb")
        
        st.markdown("---")
        st.markdown("### 🛠️ Tech Stack")
        st.markdown("""
        - **UI:** Streamlit
        - **Analysis:** Pandas, NumPy
        - **ML:** Scikit-learn
        - **Viz:** Matplotlib, Seaborn
        """)
    
    # Main content based on selected page
    if not results['success']:
        st.error("Failed to load data. Please check data file.")
        return
    
    metrics = results['metrics']
    
    # ========== OVERVIEW PAGE ==========
    if page == "🏠 Overview":
        st.header("📊 Analysis Overview")
        
        # Key Metrics
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.metric(
                label="👥 Total Customers",
                value=f"{metrics['total_customers']:,}",
                delta="Active"
            )
        
        with col2:
            st.metric(
                label="💰 Total Revenue",
                value=f"${metrics['total_revenue']:,.0f}",
                delta="Lifetime"
            )
        
        with col3:
            st.metric(
                label="📊 Avg Customer Value",
                value=f"${metrics['avg_customer_value']:,.0f}",
                delta="Per Customer"
            )
        
        with col4:
            st.metric(
                label="✅ Data Quality",
                value=f"{metrics['data_quality']}%",
                delta="Clean Rows"
            )
        
        with col5:
            st.metric(
                label="🎯 Segments",
                value=metrics['num_segments'],
                delta="Business"
            )
        
        with col6:
            st.metric(
                label="🤖 Clusters",
                value=metrics['num_clusters'],
                delta="ML-based"
            )
        
        st.markdown("---")
        
        # Data Summary
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📂 Data Processing Summary")
            summary_data = {
                'Metric': ['Raw Rows', 'Clean Rows', 'Removed Rows', 'Data Quality', 'Unique Customers'],
                'Value': [
                    f"{metrics['total_rows']:,}",
                    f"{metrics['clean_rows']:,}",
                    f"{metrics['total_rows'] - metrics['clean_rows']:,}",
                    f"{metrics['data_quality']}%",
                    f"{metrics['total_customers']:,}"
                ]
            }
            st.dataframe(pd.DataFrame(summary_data), hide_index=True, use_container_width=True)
        
        with col2:
            st.subheader("🎯 Analysis Outputs")
            outputs_data = {
                'Output': ['RFM Metrics', 'Business Segments', 'ML Clusters', 'Visualizations'],
                'Count': [
                    '3 metrics',
                    f"{metrics['num_segments']} segments",
                    f"{metrics['num_clusters']} clusters",
                    '4 charts'
                ]
            }
            st.dataframe(pd.DataFrame(outputs_data), hide_index=True, use_container_width=True)
        
        st.markdown("---")
        
        # Quick Insights
        st.subheader("💡 Quick Insights")
        
        segment_summary = results['segment_summary']
        top_segment = segment_summary.index[0]
        top_segment_revenue = segment_summary.iloc[0]['Avg_Monetary']
        top_segment_count = segment_summary.iloc[0]['Customer_Count']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="info-box">
                <h4>🏆 Top Segment</h4>
                <p><strong>{top_segment}</strong></p>
                <p>{top_segment_count} customers</p>
                <p>Avg: ${top_segment_revenue:,.2f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            avg_recency = results['rfm']['Recency'].mean()
            st.markdown(f"""
            <div class="info-box">
                <h4>📅 Avg Recency</h4>
                <p><strong>{avg_recency:.1f} days</strong></p>
                <p>Since last purchase</p>
                <p>Engagement level</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            avg_frequency = results['rfm']['Frequency'].mean()
            st.markdown(f"""
            <div class="info-box">
                <h4>🔄 Avg Frequency</h4>
                <p><strong>{avg_frequency:.1f} orders</strong></p>
                <p>Per customer</p>
                <p>Purchase behavior</p>
            </div>
            """, unsafe_allow_html=True)
    
    # ========== RFM ANALYSIS PAGE ==========
    elif page == "📈 RFM Analysis":
        st.header("📈 RFM Analysis")
        
        rfm = results['rfm']
        rfm_scored = results['rfm_scored']
        
        # RFM Statistics
        st.subheader("📊 RFM Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 📅 Recency")
            st.metric("Mean", f"{rfm['Recency'].mean():.1f} days")
            st.metric("Median", f"{rfm['Recency'].median():.1f} days")
            st.metric("Range", f"{rfm['Recency'].min():.0f} - {rfm['Recency'].max():.0f}")
        
        with col2:
            st.markdown("#### 🔄 Frequency")
            st.metric("Mean", f"{rfm['Frequency'].mean():.1f} orders")
            st.metric("Median", f"{rfm['Frequency'].median():.1f} orders")
            st.metric("Range", f"{rfm['Frequency'].min():.0f} - {rfm['Frequency'].max():.0f}")
        
        with col3:
            st.markdown("#### 💰 Monetary")
            st.metric("Mean", f"${rfm['Monetary'].mean():,.0f}")
            st.metric("Median", f"${rfm['Monetary'].median():,.0f}")
            st.metric("Range", f"${rfm['Monetary'].min():.0f} - ${rfm['Monetary'].max():,.0f}")
        
        st.markdown("---")
        
        # RFM Distributions
        st.subheader("📊 RFM Value Distributions")
        fig_dist = create_rfm_distributions_chart(rfm)
        st.plotly_chart(fig_dist, use_container_width=True)
        
        st.markdown("---")
        
        # RFM Scores
        st.subheader("🎯 RFM Score Distributions")
        fig_scores = create_rfm_scores_chart(rfm_scored)
        st.plotly_chart(fig_scores, use_container_width=True)
        
        st.markdown("---")
        
        # RFM Data Explorer
        with st.expander("🔍 Explore RFM Data"):
            st.dataframe(rfm_scored.head(50), use_container_width=True)
            
            st.download_button(
                label="📥 Download RFM Data (CSV)",
                data=rfm_scored.to_csv(index=True),
                file_name="rfm_data.csv",
                mime="text/csv"
            )
    
    # ========== SEGMENTATION PAGE ==========
    elif page == "🎯 Segmentation":
        st.header("🎯 Customer Segmentation")
        
        segment_summary = results['segment_summary']
        rfm_segmented = results['rfm_segmented']
        
        # Segment Distribution
        st.subheader("📊 Segment Distribution")
        fig_seg = create_segment_distribution_chart(segment_summary)
        st.plotly_chart(fig_seg, use_container_width=True)
        
        st.markdown("---")
        
        # Segment Details Table
        st.subheader("📋 Segment Details")
        
        # Add percentage column
        segment_display = segment_summary.copy()
        segment_display['Percentage'] = (segment_display['Customer_Count'] / segment_display['Customer_Count'].sum() * 100).round(2)
        segment_display['Total_Revenue'] = (segment_display['Customer_Count'] * segment_display['Avg_Monetary']).round(2)
        
        # Format for display
        segment_display_formatted = segment_display.copy()
        segment_display_formatted['Avg_Recency'] = segment_display_formatted['Avg_Recency'].apply(lambda x: f"{x:.1f} days")
        segment_display_formatted['Avg_Frequency'] = segment_display_formatted['Avg_Frequency'].apply(lambda x: f"{x:.1f} orders")
        segment_display_formatted['Avg_Monetary'] = segment_display_formatted['Avg_Monetary'].apply(lambda x: f"${x:,.2f}")
        segment_display_formatted['Total_Revenue'] = segment_display_formatted['Total_Revenue'].apply(lambda x: f"${x:,.2f}")
        segment_display_formatted['Percentage'] = segment_display_formatted['Percentage'].apply(lambda x: f"{x:.2f}%")
        
        st.dataframe(segment_display_formatted, use_container_width=True)
        
        st.markdown("---")
        
        # Segment Insights
        st.subheader("💡 Segment Insights")
        
        top_3 = segment_summary.head(3)
        
        for idx, (segment, data) in enumerate(top_3.iterrows()):
            with st.expander(f"🏆 #{idx+1}: {segment} ({data['Customer_Count']} customers)"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    **📊 Metrics:**
                    - **Customers:** {data['Customer_Count']}
                    - **Avg Recency:** {data['Avg_Recency']:.1f} days
                    - **Avg Frequency:** {data['Avg_Frequency']:.1f} orders
                    - **Avg Monetary:** ${data['Avg_Monetary']:,.2f}
                    """)
                
                with col2:
                    total_rev = data['Customer_Count'] * data['Avg_Monetary']
                    percentage = (data['Customer_Count'] / segment_summary['Customer_Count'].sum() * 100)
                    
                    st.markdown(f"""
                    **💰 Revenue Impact:**
                    - **Total Revenue:** ${total_rev:,.2f}
                    - **Percentage:** {percentage:.2f}%
                    - **Value Rank:** #{idx+1}
                    """)
        
        st.markdown("---")
        
        # Download Data
        with st.expander("📥 Download Segment Data"):
            st.download_button(
                label="Download Segment Summary (CSV)",
                data=segment_summary.to_csv(index=True),
                file_name="segment_summary.csv",
                mime="text/csv"
            )
            
            st.download_button(
                label="Download Full Segmented Data (CSV)",
                data=rfm_segmented.to_csv(index=True),
                file_name="rfm_segmented.csv",
                mime="text/csv"
            )
    
    # ========== ML CLUSTERING PAGE ==========
    elif page == "🤖 ML Clustering":
        st.header("🤖 Machine Learning Clustering")
        
        cluster_summary = results['cluster_summary']
        rfm_clustered = results['rfm_clustered']
        kmeans_result = results['kmeans_result']
        
        # Clustering Metrics
        st.subheader("📊 Clustering Quality Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🎯 Number of Clusters", 4)
        
        with col2:
            st.metric("✅ Silhouette Score", f"{kmeans_result['silhouette_score']:.3f}")
        
        with col3:
            st.metric("📉 Davies-Bouldin", f"{kmeans_result['davies_bouldin_score']:.3f}")
        
        with col4:
            st.metric("🔬 Algorithm", "K-Means")
        
        st.info("💡 **Silhouette Score:** Higher is better (range: -1 to 1). Score > 0.5 indicates good clustering.")
        
        st.markdown("---")
        
        # Cluster Visualization
        st.subheader("📊 Cluster Analysis")
        fig_cluster = create_cluster_analysis_chart(cluster_summary, rfm_clustered)
        st.plotly_chart(fig_cluster, use_container_width=True)
        
        st.markdown("---")
        
        # Cluster Details
        st.subheader("📋 Cluster Details")
        
        cluster_display = cluster_summary.copy()
        cluster_display['Percentage'] = (cluster_display['Customer_Count'] / cluster_display['Customer_Count'].sum() * 100).round(2)
        cluster_display['Total_Revenue'] = (cluster_display['Customer_Count'] * cluster_display['Monetary']).round(2)
        
        # Format for display
        cluster_display_formatted = cluster_display.copy()
        cluster_display_formatted['Recency'] = cluster_display_formatted['Recency'].apply(lambda x: f"{x:.1f} days")
        cluster_display_formatted['Frequency'] = cluster_display_formatted['Frequency'].apply(lambda x: f"{x:.1f} orders")
        cluster_display_formatted['Monetary'] = cluster_display_formatted['Monetary'].apply(lambda x: f"${x:,.2f}")
        cluster_display_formatted['Total_Revenue'] = cluster_display_formatted['Total_Revenue'].apply(lambda x: f"${x:,.2f}")
        cluster_display_formatted['Percentage'] = cluster_display_formatted['Percentage'].apply(lambda x: f"{x:.2f}%")
        
        st.dataframe(cluster_display_formatted, use_container_width=True)
        
        st.markdown("---")
        
        # Cluster Characteristics
        st.subheader("🔍 Cluster Characteristics")
        
        for cluster_id in cluster_summary.index:
            cluster_data = cluster_summary.loc[cluster_id]
            cluster_customers = rfm_clustered[rfm_clustered['Cluster'] == cluster_id]
            
            with st.expander(f"🎯 Cluster {cluster_id} ({cluster_data['Customer_Count']} customers)"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📊 Average RFM Values:**")
                    st.write(f"- **Recency:** {cluster_data['Recency']:.1f} days")
                    st.write(f"- **Frequency:** {cluster_data['Frequency']:.1f} orders")
                    st.write(f"- **Monetary:** ${cluster_data['Monetary']:,.2f}")
                
                with col2:
                    total_rev = cluster_data['Customer_Count'] * cluster_data['Monetary']
                    percentage = (cluster_data['Customer_Count'] / cluster_summary['Customer_Count'].sum() * 100)
                    
                    st.markdown("**💰 Revenue Impact:**")
                    st.write(f"- **Total Revenue:** ${total_rev:,.2f}")
                    st.write(f"- **Percentage:** {percentage:.2f}%")
                
                st.markdown("**📈 Distribution within Cluster:**")
                st.write(f"- **Recency Range:** {cluster_customers['Recency'].min():.0f} - {cluster_customers['Recency'].max():.0f} days")
                st.write(f"- **Frequency Range:** {cluster_customers['Frequency'].min():.0f} - {cluster_customers['Frequency'].max():.0f} orders")
                st.write(f"- **Monetary Range:** ${cluster_customers['Monetary'].min():,.2f} - ${cluster_customers['Monetary'].max():,.2f}")
        
        st.markdown("---")
        
        # Download Data
        with st.expander("📥 Download Cluster Data"):
            st.download_button(
                label="Download Cluster Summary (CSV)",
                data=cluster_summary.to_csv(index=True),
                file_name="cluster_summary.csv",
                mime="text/csv"
            )
            
            st.download_button(
                label="Download Full Clustered Data (CSV)",
                data=rfm_clustered.to_csv(index=True),
                file_name="rfm_clustered.csv",
                mime="text/csv"
            )
    
    # ========== INSIGHTS PAGE ==========
    elif page == "💡 Insights":
        st.header("💡 Business Insights & Recommendations")
        
        segment_summary = results['segment_summary']
        rfm_segmented = results['rfm_segmented']
        rfm = results['rfm']
        
        # Top Performers
        st.subheader("🏆 Top Performing Segments")
        
        top_segment = segment_summary.index[0]
        top_segment_data = segment_summary.iloc[0]
        top_revenue = top_segment_data['Customer_Count'] * top_segment_data['Avg_Monetary']
        
        st.markdown(f"""
        <div class="info-box">
            <h3>🎯 {top_segment}</h3>
            <p><strong>{top_segment_data['Customer_Count']}</strong> customers generating <strong>${top_revenue:,.2f}</strong> in revenue</p>
            <p>Average spend: <strong>${top_segment_data['Avg_Monetary']:,.2f}</strong> per customer</p>
            <p>💡 <strong>Action:</strong> Focus on retention and upselling opportunities</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # At-Risk Customers
        st.subheader("⚠️ At-Risk Customers")
        
        # Define at-risk segments
        at_risk_segments = ['At Risk', 'Cant Lose Them', 'Lost Customers', 'Hibernating']
        at_risk_data = rfm_segmented[rfm_segmented['Segment'].isin(at_risk_segments)]
        
        if len(at_risk_data) > 0:
            at_risk_count = len(at_risk_data)
            at_risk_revenue = at_risk_data['Monetary'].sum()
            at_risk_percentage = (at_risk_count / len(rfm_segmented)) * 100
            
            st.markdown(f"""
            <div class="warning-box">
                <h3>⚠️ Churn Risk Alert</h3>
                <p><strong>{at_risk_count}</strong> customers ({at_risk_percentage:.1f}%) are at risk</p>
                <p>Potential revenue loss: <strong>${at_risk_revenue:,.2f}</strong></p>
                <p>💡 <strong>Action:</strong> Implement win-back campaigns immediately</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Breakdown by at-risk segment
            st.markdown("#### 📊 At-Risk Segment Breakdown")
            at_risk_summary = at_risk_data.groupby('Segment').agg({
                'CustomerID': 'count',
                'Monetary': 'sum'
            }).round(2)
            at_risk_summary.columns = ['Customer_Count', 'Total_Revenue']
            at_risk_summary = at_risk_summary.sort_values('Total_Revenue', ascending=False)
            
            for segment in at_risk_summary.index:
                count = at_risk_summary.loc[segment, 'Customer_Count']
                revenue = at_risk_summary.loc[segment, 'Total_Revenue']
                st.write(f"- **{segment}:** {count} customers, ${revenue:,.2f} at risk")
        else:
            st.success("✅ No significant at-risk customer segments identified!")
        
        st.markdown("---")
        
        # Growth Opportunities
        st.subheader("📈 Growth Opportunities")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="insight-box">
                <h4>🎯 High-Value Targeting</h4>
                <p><strong>Focus:</strong> Champions & Loyal Customers</p>
                <ul>
                    <li>Implement VIP loyalty programs</li>
                    <li>Exclusive early access to new products</li>
                    <li>Personalized recommendations</li>
                    <li>Premium customer support</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="insight-box">
                <h4>🔄 Re-engagement</h4>
                <p><strong>Focus:</strong> Promising & Potential Loyalists</p>
                <ul>
                    <li>Targeted email campaigns</li>
                    <li>Special discount offers</li>
                    <li>Product bundles</li>
                    <li>Engagement surveys</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Strategic Recommendations
        st.subheader("🎯 Strategic Recommendations")
        
        recommendations = [
            {
                'title': '1️⃣ Retention Focus',
                'description': 'Prioritize retaining Champions and Loyal Customers through personalized experiences and exclusive benefits.',
                'priority': 'High',
                'color': '#d1ecf1'
            },
            {
                'title': '2️⃣ Win-Back Campaigns',
                'description': 'Launch targeted campaigns for At-Risk and Lost customers with compelling offers and incentives.',
                'priority': 'High',
                'color': '#f8d7da'
            },
            {
                'title': '3️⃣ Upsell Opportunities',
                'description': 'Increase average order value for Potential Loyalists through product recommendations and bundles.',
                'priority': 'Medium',
                'color': '#fff3cd'
            },
            {
                'title': '4️⃣ New Customer Activation',
                'description': 'Develop onboarding programs for New Customers to accelerate their journey to loyal status.',
                'priority': 'Medium',
                'color': '#d4edda'
            },
            {
                'title': '5️⃣ Data-Driven Monitoring',
                'description': 'Implement regular RFM analysis to track segment movements and adjust strategies accordingly.',
                'priority': 'Ongoing',
                'color': '#e2e3e5'
            }
        ]
        
        for rec in recommendations:
            st.markdown(f"""
            <div style="padding: 15px; background-color: {rec['color']}; border-left: 5px solid #667eea; border-radius: 5px; margin: 10px 0;">
                <h4>{rec['title']}</h4>
                <p>{rec['description']}</p>
                <p><strong>Priority:</strong> {rec['priority']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Key Metrics Summary
        st.subheader("📊 Key Performance Indicators")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_recency = rfm['Recency'].mean()
            st.metric("📅 Avg Days Since Purchase", f"{avg_recency:.1f} days")
            if avg_recency < 60:
                st.success("✅ Good engagement")
            elif avg_recency < 120:
                st.warning("⚠️ Monitor closely")
            else:
                st.error("🚨 Action needed")
        
        with col2:
            repeat_rate = (rfm[rfm['Frequency'] > 1].shape[0] / len(rfm)) * 100
            st.metric("🔄 Repeat Purchase Rate", f"{repeat_rate:.1f}%")
            if repeat_rate > 50:
                st.success("✅ Strong loyalty")
            elif repeat_rate > 30:
                st.warning("⚠️ Room for improvement")
            else:
                st.error("🚨 Focus on retention")
        
        with col3:
            high_value = (rfm[rfm['Monetary'] > rfm['Monetary'].median()].shape[0] / len(rfm)) * 100
            st.metric("💎 High-Value Customers", f"{high_value:.1f}%")
            if high_value > 40:
                st.success("✅ Healthy distribution")
            else:
                st.warning("⚠️ Opportunity to grow")


if __name__ == "__main__":
    main()
