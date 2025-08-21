"""
Main Streamlit dashboard application for insurance risk analysis.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from typing import Optional

from src.config.settings import get_settings
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.data.validator import DataValidator
from src.models.risk_predictor import RiskPredictor
from .components import (
    DataExplorer,
    ModelTrainer,
    RiskAnalyzer,
    PremiumOptimizer,
    ExplainabilityViewer,
    ReportGenerator
)


def create_app():
    """Create and configure the Streamlit dashboard application."""
    
    # Page configuration
    st.set_page_config(
        page_title="Insurance Risk Analysis Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown('<h1 class="main-header">📊 Insurance Risk Analysis Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Initialize session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'current_data' not in st.session_state:
        st.session_state.current_data = None
    if 'current_model' not in st.session_state:
        st.session_state.current_model = None
    
    # Sidebar navigation
    st.sidebar.markdown('<p class="sidebar-header">Navigation</p>', unsafe_allow_html=True)
    
    page = st.sidebar.selectbox(
        "Select Page",
        [
            "🏠 Dashboard Overview",
            "📈 Data Explorer", 
            "🤖 Model Training",
            "🎯 Risk Analysis",
            "💰 Premium Optimization",
            "🔍 Explainability",
            "📋 Reports & Model Cards"
        ]
    )
    
    # Sidebar data upload
    st.sidebar.markdown("---")
    st.sidebar.markdown('<p class="sidebar-header">Data Management</p>', unsafe_allow_html=True)
    
    uploaded_file = st.sidebar.file_uploader(
        "Upload Insurance Data (CSV)",
        type=['csv'],
        help="Upload your insurance data file"
    )
    
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.session_state.current_data = data
            st.session_state.data_loaded = True
            st.sidebar.success(f"✅ Data loaded: {len(data)} rows, {len(data.columns)} columns")
        except Exception as e:
            st.sidebar.error(f"❌ Error loading data: {str(e)}")
    
    # Load default data if available
    elif not st.session_state.data_loaded:
        settings = get_settings()
        data_path = settings.data_dir / settings.data_file
        
        if data_path.exists():
            try:
                loader = DataLoader(settings)
                data = loader.load_data()
                st.session_state.current_data = data
                st.session_state.data_loaded = True
                st.sidebar.success(f"✅ Default data loaded: {len(data)} rows")
            except Exception as e:
                st.sidebar.warning(f"⚠️ Could not load default data: {str(e)}")
    
    # Page routing
    if page == "🏠 Dashboard Overview":
        show_dashboard_overview()
    elif page == "📈 Data Explorer":
        show_data_explorer()
    elif page == "🤖 Model Training":
        show_model_training()
    elif page == "🎯 Risk Analysis":
        show_risk_analysis()
    elif page == "💰 Premium Optimization":
        show_premium_optimization()
    elif page == "🔍 Explainability":
        show_explainability()
    elif page == "📋 Reports & Model Cards":
        show_reports()


def show_dashboard_overview():
    """Display the main dashboard overview."""
    
    st.header("🏠 Dashboard Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.session_state.data_loaded:
            data = st.session_state.current_data
            st.metric(
                label="Total Policies",
                value=f"{len(data):,}",
                delta=None
            )
        else:
            st.metric(label="Total Policies", value="No data")
    
    with col2:
        if st.session_state.data_loaded:
            data = st.session_state.current_data
            if 'TotalPremium' in data.columns:
                total_premium = data['TotalPremium'].sum()
                st.metric(
                    label="Total Premium",
                    value=f"${total_premium:,.0f}",
                    delta=None
                )
            else:
                st.metric(label="Total Premium", value="N/A")
        else:
            st.metric(label="Total Premium", value="No data")
    
    with col3:
        if st.session_state.data_loaded:
            data = st.session_state.current_data
            if 'TotalClaims' in data.columns:
                total_claims = data['TotalClaims'].sum()
                st.metric(
                    label="Total Claims",
                    value=f"${total_claims:,.0f}",
                    delta=None
                )
            else:
                st.metric(label="Total Claims", value="N/A")
        else:
            st.metric(label="Total Claims", value="No data")
    
    with col4:
        if st.session_state.model_trained:
            st.metric(label="Model Status", value="✅ Trained")
        else:
            st.metric(label="Model Status", value="❌ Not Trained")
    
    st.markdown("---")
    
    # Quick actions
    st.subheader("🚀 Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Explore Data", use_container_width=True):
            st.switch_page("📈 Data Explorer")
    
    with col2:
        if st.button("🤖 Train Model", use_container_width=True):
            st.switch_page("🤖 Model Training")
    
    with col3:
        if st.button("🎯 Analyze Risk", use_container_width=True):
            st.switch_page("🎯 Risk Analysis")
    
    # Recent activity
    st.markdown("---")
    st.subheader("📈 Recent Activity")
    
    if st.session_state.data_loaded:
        data = st.session_state.current_data
        
        # Data summary
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Data Summary**")
            summary_df = pd.DataFrame({
                'Metric': ['Rows', 'Columns', 'Missing Values', 'Data Types'],
                'Value': [
                    len(data),
                    len(data.columns),
                    data.isnull().sum().sum(),
                    len(data.dtypes.unique())
                ]
            })
            st.dataframe(summary_df, use_container_width=True)
        
        with col2:
            st.markdown("**Column Overview**")
            col_info = pd.DataFrame({
                'Column': data.columns,
                'Type': data.dtypes.astype(str),
                'Missing': data.isnull().sum(),
                'Unique': data.nunique()
            })
            st.dataframe(col_info, use_container_width=True)
    
    else:
        st.info("📁 Please upload data or load default data to see dashboard metrics.")


def show_data_explorer():
    """Display the data explorer component."""
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first to explore.")
        return
    
    explorer = DataExplorer(st.session_state.current_data)
    explorer.render()


def show_model_training():
    """Display the model training component."""
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first to train models.")
        return
    
    trainer = ModelTrainer(st.session_state.current_data)
    trainer.render()


def show_risk_analysis():
    """Display the risk analysis component."""
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first to analyze risk.")
        return
    
    analyzer = RiskAnalyzer(st.session_state.current_data)
    analyzer.render()


def show_premium_optimization():
    """Display the premium optimization component."""
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data first to optimize premiums.")
        return
    
    optimizer = PremiumOptimizer(st.session_state.current_data)
    optimizer.render()


def show_explainability():
    """Display the explainability component."""
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train a model first to view explainability.")
        return
    
    explainer = ExplainabilityViewer(st.session_state.current_model)
    explainer.render()


def show_reports():
    """Display the reports and model cards component."""
    generator = ReportGenerator()
    generator.render()


if __name__ == "__main__":
    create_app()
