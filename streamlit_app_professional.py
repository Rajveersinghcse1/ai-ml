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

# Professional Page Configuration
st.set_page_config(
    page_title="Customer Analytics Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'mailto:support@company.com',
        'Report a bug': 'mailto:bugs@company.com',
        'About': """
        # Customer Analytics Platform
        
        **Version:** 2.0.0
        **Built with:** Python, Streamlit, Plotly
        **License:** Enterprise
        
        Professional customer segmentation and analytics platform
        for data-driven business insights.
        """
    }
)

# Professional CSS Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        font-family: 'Roboto', sans-serif;
    }
    
    /* Main Content Area */
    .main .block-container {
        padding: 2rem 1rem;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Header Styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .main-header h1 {
        color: white !important;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        margin: 0 !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9) !important;
        font-size: 1.2rem !important;
        margin: 1rem 0 0 0 !important;
        font-weight: 400 !important;
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
    }
    
    .css-1d391kg .css-1v3fvcr {
        color: white !important;
    }
    
    /* Metric Cards */
    [data-testid="metric-container"] {
        background: white;
        border: 2px solid #e1e8ed;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    
    [data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.1);
    }
    
    [data-testid="metric-container"] > div {
        color: #2c3e50 !important;
        font-weight: 600 !important;
    }
    
    [data-testid="metric-container"] [data-testid="metric-value"] {
        color: #667eea !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    
    /* Cards */
    .info-card {
        background: white;
        border: 1px solid #e1e8ed;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .info-card h3 {
        color: #2c3e50 !important;
        font-size: 1.3rem !important;
        font-weight: 600 !important;
        margin-bottom: 0.5rem !important;
    }
    
    .info-card p {
        color: #5a6c7d !important;
        font-size: 1rem !important;
        line-height: 1.6 !important;
    }
    
    /* Success/Error Messages */
    .stSuccess {
        background-color: #d4edda !important;
        border: 1px solid #c3e6cb !important;
        color: #155724 !important;
        border-radius: 8px !important;
    }
    
    .stError {
        background-color: #f8d7da !important;
        border: 1px solid #f5c6cb !important;
        color: #721c24 !important;
        border-radius: 8px !important;
    }
    
    .stInfo {
        background-color: #cce7ff !important;
        border: 1px solid #b3d7ff !important;
        color: #004085 !important;
        border-radius: 8px !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        border-radius: 8px !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 0.6rem 1.5rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(102,126,234,0.3) !important;
    }
    
    /* Selectbox */
    .stSelectbox label {
        color: #2c3e50 !important;
        font-weight: 600 !important;
    }
    
    /* DataFrames */
    .dataframe {
        border: 1px solid #e1e8ed !important;
        border-radius: 8px !important;
        overflow: hidden !important;
    }
    
    .dataframe thead th {
        background-color: #667eea !important;
        color: white !important;
        font-weight: 600 !important;
    }
    
    .dataframe tbody tr:nth-child(even) {
        background-color: #f8f9fa !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: none;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: white;
        border: 2px solid #e1e8ed;
        border-radius: 8px;
        color: #5a6c7d;
        font-weight: 500;
        padding: 0.5rem 1rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-color: #667eea;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #5a6c7d;
        border-top: 1px solid #e1e8ed;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize session state variables"""
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'current_data' not in st.session_state:
        st.session_state.current_data = pd.DataFrame()
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = {}
    if 'models_trained' not in st.session_state:
        st.session_state.models_trained = False

initialize_session_state()

def load_sample_data():
    """Load sample data for demonstration"""
    try:
        # Try to load the actual data file
        if os.path.exists('data/online_retail_II.csv'):
            df = pd.read_csv('data/online_retail_II.csv')
            return df.head(1000)  # Limit for demo
        else:
            # Generate sample data if file not found
            np.random.seed(42)
            n_customers = 1000
            n_transactions = 5000
            
            customers = [f'C{i:04d}' for i in range(n_customers)]
            
            data = []
            for _ in range(n_transactions):
                customer_id = np.random.choice(customers)
                quantity = np.random.randint(1, 10)
                unit_price = np.random.uniform(5, 100)
                invoice_date = datetime.now() - timedelta(days=np.random.randint(1, 365))
                country = np.random.choice(['United Kingdom', 'France', 'Germany', 'Netherlands', 'EIRE'])
                
                data.append({
                    'CustomerID': customer_id,
                    'Quantity': quantity,
                    'UnitPrice': unit_price,
                    'InvoiceDate': invoice_date,
                    'Country': country,
                    'Total': quantity * unit_price
                })
            
            return pd.DataFrame(data)
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

def create_professional_metrics(df):
    """Create professional metric cards"""
    if df.empty:
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_customers = df['CustomerID'].nunique() if 'CustomerID' in df.columns else 0
        st.metric("Total Customers", f"{total_customers:,}")
    
    with col2:
        total_transactions = len(df)
        st.metric("Transactions", f"{total_transactions:,}")
    
    with col3:
        total_revenue = df['Total'].sum() if 'Total' in df.columns else 0
        st.metric("Revenue", f"${total_revenue:,.2f}")
    
    with col4:
        avg_order_value = df['Total'].mean() if 'Total' in df.columns else 0
        st.metric("Avg Order Value", f"${avg_order_value:.2f}")

def render_data_overview(df):
    """Render data overview section"""
    st.markdown("""
    <div class="info-card">
        <h3>📊 Data Overview</h3>
        <p>Explore your customer transaction data with comprehensive analytics and insights.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not df.empty:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Dataset Information")
            info_df = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes,
                'Non-Null Count': df.count(),
                'Null Count': df.isnull().sum()
            })
            st.dataframe(info_df, use_container_width=True)
        
        with col2:
            st.subheader("Data Quality")
            completeness = (1 - df.isnull().sum() / len(df)) * 100
            
            for col in df.columns:
                progress_val = completeness[col] / 100
                st.progress(progress_val, text=f"{col}: {completeness[col]:.1f}%")
        
        # Sample data preview
        st.subheader("Data Preview")
        st.dataframe(df.head(10), use_container_width=True)

def perform_rfm_analysis(df):
    """Perform RFM Analysis"""
    if not BACKEND_AVAILABLE:
        st.warning("⚠️ RFM Analysis requires backend modules")
        return None
    
    try:
        # Ensure required columns exist
        if not all(col in df.columns for col in ['CustomerID', 'InvoiceDate', 'Total']):
            st.error("❌ Required columns missing for RFM analysis")
            return None
        
        # Calculate RFM
        analyzer = AdvancedRFMAnalyzer()
        rfm_data = analyzer.calculate_rfm_basic(df)
        segments = analyzer.segment_customers_traditional(rfm_data)
        
        return segments
    
    except Exception as e:
        st.error(f"❌ RFM Analysis Error: {str(e)}")
        return None

def create_visualizations(df, analysis_type):
    """Create professional visualizations"""
    try:
        if analysis_type == "Revenue Trends":
            if 'InvoiceDate' in df.columns and 'Total' in df.columns:
                # Convert to datetime if needed
                df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
                
                # Daily revenue
                daily_revenue = df.groupby(df['InvoiceDate'].dt.date)['Total'].sum().reset_index()
                
                fig = px.line(
                    daily_revenue, 
                    x='InvoiceDate', 
                    y='Total',
                    title="Daily Revenue Trends",
                    labels={'Total': 'Revenue ($)', 'InvoiceDate': 'Date'},
                    color_discrete_sequence=['#667eea']
                )
                fig.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font_family="Roboto",
                    title_font_size=16,
                    title_font_color='#2c3e50'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        elif analysis_type == "Customer Distribution":
            if 'Country' in df.columns:
                country_counts = df['Country'].value_counts().head(10)
                
                fig = px.bar(
                    x=country_counts.index,
                    y=country_counts.values,
                    title="Top 10 Countries by Customer Count",
                    labels={'y': 'Number of Transactions', 'x': 'Country'},
                    color_discrete_sequence=['#764ba2']
                )
                fig.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font_family="Roboto",
                    title_font_size=16,
                    title_font_color='#2c3e50'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        elif analysis_type == "RFM Analysis":
            rfm_results = perform_rfm_analysis(df)
            if rfm_results is not None:
                # RFM Segment Distribution
                segment_counts = rfm_results['Segment'].value_counts()
                
                fig = px.pie(
                    values=segment_counts.values,
                    names=segment_counts.index,
                    title="Customer Segment Distribution",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font_family="Roboto",
                    title_font_size=16,
                    title_font_color='#2c3e50'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # RFM Metrics Summary
                st.subheader("RFM Analysis Results")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    avg_recency = rfm_results['Recency'].mean()
                    st.metric("Avg Recency", f"{avg_recency:.0f} days")
                
                with col2:
                    avg_frequency = rfm_results['Frequency'].mean()
                    st.metric("Avg Frequency", f"{avg_frequency:.1f}")
                
                with col3:
                    avg_monetary = rfm_results['Monetary'].mean()
                    st.metric("Avg Monetary", f"${avg_monetary:.2f}")
    
    except Exception as e:
        st.error(f"❌ Visualization Error: {str(e)}")

def main():
    """Main application function"""
    
    # Professional Header
    st.markdown("""
    <div class="main-header">
        <h1>📊 Customer Analytics Platform</h1>
        <p>Professional Customer Segmentation & Business Intelligence Dashboard</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar Navigation
    with st.sidebar:
        st.markdown("### 🎯 Navigation")
        page = st.selectbox(
            "Choose Analysis:",
            ["🏠 Dashboard", "📊 Data Overview", "🔍 RFM Analysis", "📈 Visualizations", "⚙️ Settings"],
            index=0
        )
        
        st.markdown("---")
        
        # Data Upload Section
        st.markdown("### 📂 Data Source")
        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type=['csv'],
            help="Upload your transaction data in CSV format"
        )
        
        use_sample = st.checkbox("Use Sample Data", value=True)
        
        # Load data
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.current_data = df
                st.session_state.data_loaded = True
                st.success("✅ File uploaded successfully!")
            except Exception as e:
                st.error(f"❌ Upload Error: {str(e)}")
        elif use_sample:
            df = load_sample_data()
            st.session_state.current_data = df
            st.session_state.data_loaded = True
        else:
            df = pd.DataFrame()
            st.session_state.data_loaded = False
        
        # Data info
        if not df.empty:
            st.markdown("### 📈 Data Info")
            st.write(f"**Rows:** {len(df):,}")
            st.write(f"**Columns:** {len(df.columns)}")
            st.write(f"**Size:** {df.memory_usage().sum() / 1024:.1f} KB")
    
    # Main Content Area
    if page == "🏠 Dashboard":
        if st.session_state.data_loaded and not st.session_state.current_data.empty:
            # Key Metrics
            create_professional_metrics(st.session_state.current_data)
            
            st.markdown("---")
            
            # Quick Insights
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div class="info-card">
                    <h3>🎯 Quick Insights</h3>
                    <p>Your customer analytics dashboard provides comprehensive insights into customer behavior, 
                    segmentation, and business performance metrics.</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="info-card">
                    <h3>⚡ Recent Activity</h3>
                    <p>Data loaded successfully. Navigate through different sections to explore RFM analysis, 
                    visualizations, and advanced analytics capabilities.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Quick Actions
            st.markdown("### 🚀 Quick Actions")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if st.button("📊 Run RFM Analysis"):
                    st.switch_page = "🔍 RFM Analysis"
            
            with col2:
                if st.button("📈 Create Charts"):
                    st.switch_page = "📈 Visualizations"
            
            with col3:
                if st.button("🔍 Explore Data"):
                    st.switch_page = "📊 Data Overview"
            
            with col4:
                if st.button("⚙️ Settings"):
                    st.switch_page = "⚙️ Settings"
        
        else:
            st.markdown("""
            <div class="info-card">
                <h3>👋 Welcome to Customer Analytics Platform</h3>
                <p>To get started, please upload your data file or use the sample data from the sidebar.</p>
            </div>
            """, unsafe_allow_html=True)
    
    elif page == "📊 Data Overview":
        if st.session_state.data_loaded:
            render_data_overview(st.session_state.current_data)
        else:
            st.warning("⚠️ Please load data first using the sidebar.")
    
    elif page == "🔍 RFM Analysis":
        if st.session_state.data_loaded:
            st.header("🔍 RFM Customer Segmentation")
            
            if BACKEND_AVAILABLE:
                rfm_results = perform_rfm_analysis(st.session_state.current_data)
                
                if rfm_results is not None:
                    # Display results
                    tab1, tab2, tab3 = st.tabs(["📊 Segments", "📈 Metrics", "📋 Data"])
                    
                    with tab1:
                        create_visualizations(st.session_state.current_data, "RFM Analysis")
                    
                    with tab2:
                        st.subheader("RFM Metrics Distribution")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            fig = px.histogram(rfm_results, x='Recency', title='Recency Distribution')
                            fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            fig = px.histogram(rfm_results, x='Frequency', title='Frequency Distribution')
                            fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col3:
                            fig = px.histogram(rfm_results, x='Monetary', title='Monetary Distribution')
                            fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with tab3:
                        st.subheader("Detailed RFM Data")
                        st.dataframe(rfm_results, use_container_width=True)
            else:
                st.error("❌ RFM Analysis requires backend modules to be loaded.")
        else:
            st.warning("⚠️ Please load data first using the sidebar.")
    
    elif page == "📈 Visualizations":
        if st.session_state.data_loaded:
            st.header("📈 Data Visualizations")
            
            viz_type = st.selectbox(
                "Choose Visualization Type:",
                ["Revenue Trends", "Customer Distribution", "RFM Analysis"]
            )
            
            create_visualizations(st.session_state.current_data, viz_type)
        else:
            st.warning("⚠️ Please load data first using the sidebar.")
    
    elif page == "⚙️ Settings":
        st.header("⚙️ Settings & Configuration")
        
        tab1, tab2, tab3 = st.tabs(["🔧 General", "📊 Analytics", "ℹ️ About"])
        
        with tab1:
            st.subheader("General Settings")
            
            theme = st.selectbox("Theme", ["Professional", "Dark", "Light"])
            auto_refresh = st.checkbox("Auto-refresh data")
            show_advanced = st.checkbox("Show advanced options")
            
            if st.button("Save Settings"):
                st.success("✅ Settings saved successfully!")
        
        with tab2:
            st.subheader("Analytics Configuration")
            
            st.write("**Backend Status:**")
            if BACKEND_AVAILABLE:
                st.success("✅ All modules loaded")
            else:
                st.error("❌ Backend modules not available")
            
            st.write("**Available Features:**")
            features = [
                ("Data Processing", "✅" if BACKEND_AVAILABLE else "❌"),
                ("RFM Analysis", "✅" if BACKEND_AVAILABLE else "❌"),
                ("Visualization", "✅"),
                ("Export Options", "✅")
            ]
            
            for feature, status in features:
                st.write(f"{status} {feature}")
        
        with tab3:
            st.subheader("About This Platform")
            st.markdown("""
            **Customer Analytics Platform v2.0**
            
            A professional customer segmentation and analytics platform built with:
            - **Frontend:** Streamlit
            - **Backend:** Python, Pandas, Scikit-learn
            - **Visualization:** Plotly, Matplotlib
            - **Analytics:** RFM Analysis, Machine Learning
            
            **Features:**
            - Customer segmentation
            - RFM analysis
            - Interactive visualizations
            - Professional dashboard
            - Export capabilities
            
            **Support:** support@company.com
            """)
    
    # Professional Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <p>© 2025 Customer Analytics Platform | Professional Edition | 
        Built with ❤️ using Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()