"""
Dashboard components for insurance risk analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
from typing import Optional, Dict, Any
import json
from datetime import datetime

from src.data.preprocessor import DataPreprocessor
from src.models.risk_predictor import RiskPredictor
from src.config.settings import get_settings


class DataExplorer:
    """Interactive data exploration component."""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.preprocessor = DataPreprocessor()
    
    def render(self):
        """Render the data explorer interface."""
        st.header("📈 Data Explorer")
        
        # Data overview
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Data Overview")
            st.dataframe(self.data.head(10), use_container_width=True)
        
        with col2:
            st.subheader("Quick Stats")
            st.metric("Total Rows", f"{len(self.data):,}")
            st.metric("Total Columns", len(self.data.columns))
            st.metric("Memory Usage", f"{self.data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Data quality analysis
        st.subheader("🔍 Data Quality Analysis")
        
        # Missing values
        missing_data = self.data.isnull().sum()
        missing_pct = (missing_data / len(self.data)) * 100
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=missing_data.index,
            y=missing_pct,
            name="Missing %",
            marker_color='red'
        ))
        fig.update_layout(
            title="Missing Values by Column",
            xaxis_title="Columns",
            yaxis_title="Missing Percentage",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Data types distribution
        st.subheader("📊 Data Types Distribution")
        dtype_counts = self.data.dtypes.value_counts()
        
        fig = px.pie(
            values=dtype_counts.values,
            names=dtype_counts.index,
            title="Data Types Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Numerical columns analysis
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            st.subheader("📈 Numerical Columns Analysis")
            
            selected_num_col = st.selectbox("Select Numerical Column", numerical_cols)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Histogram
                fig = px.histogram(
                    self.data, 
                    x=selected_num_col,
                    title=f"Distribution of {selected_num_col}",
                    nbins=50
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Box plot
                fig = px.box(
                    self.data,
                    y=selected_num_col,
                    title=f"Box Plot of {selected_num_col}"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            st.markdown(f"**Statistics for {selected_num_col}**")
            stats = self.data[selected_num_col].describe()
            st.dataframe(stats, use_container_width=True)
        
        # Categorical columns analysis
        categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            st.subheader("📊 Categorical Columns Analysis")
            
            selected_cat_col = st.selectbox("Select Categorical Column", categorical_cols)
            
            # Value counts
            value_counts = self.data[selected_cat_col].value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    x=value_counts.index,
                    y=value_counts.values,
                    title=f"Value Counts for {selected_cat_col}"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.pie(
                    values=value_counts.values,
                    names=value_counts.index,
                    title=f"Distribution of {selected_cat_col}"
                )
                st.plotly_chart(fig, use_container_width=True)


class ModelTrainer:
    """Interactive model training component."""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.preprocessor = DataPreprocessor()
        self.settings = get_settings()
    
    def render(self):
        """Render the model training interface."""
        st.header("🤖 Model Training")
        
        # Data preprocessing
        st.subheader("📋 Data Preprocessing")
        
        with st.expander("Data Preprocessing Options", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Preprocessing Steps**")
                st.checkbox("Clean Data", value=True, key="clean_data")
                st.checkbox("Engineer Features", value=True, key="engineer_features")
                st.checkbox("Handle Missing Values", value=True, key="handle_missing")
            
            with col2:
                st.markdown("**Feature Engineering**")
                st.checkbox("Add Date Features", value=True, key="date_features")
                st.checkbox("Add Risk Categories", value=True, key="risk_categories")
                st.checkbox("Add Interaction Features", value=True, key="interaction_features")
        
        # Preprocess data
        if st.button("🔄 Preprocess Data", type="primary"):
            with st.spinner("Preprocessing data..."):
                try:
                    # Clean data
                    if st.session_state.get("clean_data", True):
                        self.cleaned_data = self.preprocessor.clean_data(self.data)
                        st.success(f"✅ Data cleaned: {len(self.cleaned_data)} rows")
                    
                    # Engineer features
                    if st.session_state.get("engineer_features", True):
                        self.engineered_data = self.preprocessor.engineer_features(self.cleaned_data)
                        st.success(f"✅ Features engineered: {len(self.engineered_data.columns)} columns")
                    
                    st.session_state.preprocessed_data = self.engineered_data
                    st.session_state.data_preprocessed = True
                    
                except Exception as e:
                    st.error(f"❌ Error preprocessing data: {str(e)}")
        
        # Model training
        if st.session_state.get("data_preprocessed", False):
            st.subheader("🎯 Model Training")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Model Configuration**")
                model_type = st.selectbox(
                    "Model Type",
                    ["random_forest", "gradient_boosting", "logistic", "svm"],
                    help="Select the type of model to train"
                )
                
                n_estimators = st.slider("Number of Estimators", 10, 500, 100)
                max_depth = st.slider("Max Depth", 3, 20, 10)
                test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
            
            with col2:
                st.markdown("**Training Parameters**")
                random_state = st.number_input("Random State", value=42, min_value=0)
                cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5)
                
                # Feature selection
                st.markdown("**Feature Selection**")
                use_all_features = st.checkbox("Use All Features", value=True)
                
                if not use_all_features:
                    available_features = list(self.engineered_data.columns)
                    selected_features = st.multiselect(
                        "Select Features",
                        available_features,
                        default=available_features[:10]
                    )
                else:
                    selected_features = None
            
            # Train model
            if st.button("🚀 Train Model", type="primary"):
                with st.spinner("Training model..."):
                    try:
                        # Prepare features
                        X, feature_names = self.preprocessor.prepare_features(self.engineered_data)
                        y = self.engineered_data['HadClaim']
                        
                        # Feature selection
                        if selected_features is not None:
                            X = X[selected_features]
                            feature_names = selected_features
                        
                        # Split data
                        X_train, X_test, y_train, y_test = self.preprocessor.split_data(X, y)
                        
                        # Train model
                        model = RiskPredictor(
                            model_type=model_type,
                            n_estimators=n_estimators,
                            max_depth=max_depth
                        )
                        
                        model.fit(X_train, y_train)
                        
                        # Evaluate model
                        train_score = model.training_metrics['accuracy']
                        test_predictions = model.predict(X_test)
                        
                        from sklearn.metrics import accuracy_score, classification_report
                        test_score = accuracy_score(y_test, test_predictions)
                        
                        # Store model and results
                        st.session_state.current_model = model
                        st.session_state.model_trained = True
                        st.session_state.model_results = {
                            'train_accuracy': train_score,
                            'test_accuracy': test_score,
                            'model_type': model_type,
                            'feature_names': feature_names
                        }
                        
                        st.success("✅ Model trained successfully!")
                        
                        # Display results
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Training Accuracy", f"{train_score:.4f}")
                        with col2:
                            st.metric("Test Accuracy", f"{test_score:.4f}")
                        with col3:
                            st.metric("Model Type", model_type.replace("_", " ").title())
                        
                        # Classification report
                        st.markdown("**Classification Report**")
                        report = classification_report(y_test, test_predictions, output_dict=True)
                        report_df = pd.DataFrame(report).transpose()
                        st.dataframe(report_df, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"❌ Error training model: {str(e)}")


class RiskAnalyzer:
    """Interactive risk analysis component."""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.preprocessor = DataPreprocessor()
    
    def render(self):
        """Render the risk analysis interface."""
        st.header("🎯 Risk Analysis")
        
        # Risk metrics overview
        st.subheader("📊 Risk Metrics Overview")
        
        if 'TotalClaims' in self.data.columns and 'TotalPremium' in self.data.columns:
            # Calculate risk metrics
            self.data['LossRatio'] = self.data['TotalClaims'] / self.data['TotalPremium']
            self.data['RiskCategory'] = pd.cut(
                self.data['LossRatio'],
                bins=[0, 0.5, 1.0, 2.0, np.inf],
                labels=['Low', 'Medium', 'High', 'Very High']
            )
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_loss_ratio = self.data['LossRatio'].mean()
                st.metric("Average Loss Ratio", f"{avg_loss_ratio:.3f}")
            
            with col2:
                high_risk_pct = (self.data['RiskCategory'] == 'High').sum() / len(self.data) * 100
                st.metric("High Risk %", f"{high_risk_pct:.1f}%")
            
            with col3:
                very_high_risk_pct = (self.data['RiskCategory'] == 'Very High').sum() / len(self.data) * 100
                st.metric("Very High Risk %", f"{very_high_risk_pct:.1f}%")
            
            with col4:
                total_risk_exposure = self.data[self.data['RiskCategory'].isin(['High', 'Very High'])]['TotalPremium'].sum()
                st.metric("Risk Exposure", f"${total_risk_exposure:,.0f}")
            
            # Risk distribution
            st.subheader("📈 Risk Distribution")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Loss ratio distribution
                fig = px.histogram(
                    self.data,
                    x='LossRatio',
                    nbins=50,
                    title="Loss Ratio Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Risk category distribution
                risk_counts = self.data['RiskCategory'].value_counts()
                fig = px.pie(
                    values=risk_counts.values,
                    names=risk_counts.index,
                    title="Risk Category Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Risk by demographic factors
            st.subheader("👥 Risk by Demographics")
            
            demographic_cols = ['Gender', 'Province', 'VehicleType']
            available_demographics = [col for col in demographic_cols if col in self.data.columns]
            
            if available_demographics:
                selected_demo = st.selectbox("Select Demographic Factor", available_demographics)
                
                # Risk analysis by demographic
                demo_risk = self.data.groupby(selected_demo).agg({
                    'LossRatio': 'mean',
                    'TotalPremium': 'sum',
                    'TotalClaims': 'sum'
                }).reset_index()
                
                demo_risk['RiskScore'] = demo_risk['LossRatio'] * demo_risk['TotalPremium']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.bar(
                        demo_risk,
                        x=selected_demo,
                        y='LossRatio',
                        title=f"Average Loss Ratio by {selected_demo}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.bar(
                        demo_risk,
                        x=selected_demo,
                        y='RiskScore',
                        title=f"Risk Score by {selected_demo}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Risk table
                st.markdown(f"**Risk Analysis by {selected_demo}**")
                st.dataframe(demo_risk, use_container_width=True)
        
        # Risk trends over time
        if 'TransactionMonth' in self.data.columns:
            st.subheader("📅 Risk Trends Over Time")
            
            # Convert to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(self.data['TransactionMonth']):
                self.data['TransactionMonth'] = pd.to_datetime(self.data['TransactionMonth'])
            
            # Monthly risk trends
            monthly_risk = self.data.groupby(self.data['TransactionMonth'].dt.to_period('M')).agg({
                'LossRatio': 'mean',
                'TotalPremium': 'sum',
                'TotalClaims': 'sum'
            }).reset_index()
            
            monthly_risk['TransactionMonth'] = monthly_risk['TransactionMonth'].astype(str)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=monthly_risk['TransactionMonth'],
                y=monthly_risk['LossRatio'],
                mode='lines+markers',
                name='Loss Ratio'
            ))
            fig.update_layout(
                title="Monthly Loss Ratio Trends",
                xaxis_title="Month",
                yaxis_title="Loss Ratio",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)


class PremiumOptimizer:
    """Interactive premium optimization component."""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.preprocessor = DataPreprocessor()
    
    def render(self):
        """Render the premium optimization interface."""
        st.header("💰 Premium Optimization")
        
        st.info("💡 This tool helps optimize premium pricing based on risk analysis and market conditions.")
        
        # Current premium analysis
        st.subheader("📊 Current Premium Analysis")
        
        if 'TotalPremium' in self.data.columns:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_premium = self.data['TotalPremium'].mean()
                st.metric("Average Premium", f"${avg_premium:,.2f}")
            
            with col2:
                premium_std = self.data['TotalPremium'].std()
                st.metric("Premium Std Dev", f"${premium_std:,.2f}")
            
            with col3:
                premium_range = self.data['TotalPremium'].max() - self.data['TotalPremium'].min()
                st.metric("Premium Range", f"${premium_range:,.2f}")
            
            # Premium distribution
            fig = px.histogram(
                self.data,
                x='TotalPremium',
                nbins=50,
                title="Premium Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Premium optimization parameters
        st.subheader("⚙️ Optimization Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Risk-Based Pricing**")
            risk_multiplier = st.slider("Risk Multiplier", 0.5, 2.0, 1.0, 0.1)
            base_premium = st.number_input("Base Premium", value=1000.0, min_value=0.0)
            
        with col2:
            st.markdown("**Market Conditions**")
            market_factor = st.slider("Market Factor", 0.8, 1.5, 1.0, 0.05)
            competition_discount = st.slider("Competition Discount %", 0, 20, 5)
        
        # Optimization strategy
        st.subheader("🎯 Optimization Strategy")
        
        strategy = st.selectbox(
            "Select Optimization Strategy",
            [
                "Risk-Based Pricing",
                "Market-Based Pricing", 
                "Competitive Pricing",
                "Profit Maximization",
                "Custom Strategy"
            ]
        )
        
        if strategy == "Custom Strategy":
            st.text_area(
                "Define Custom Strategy",
                placeholder="Enter your custom pricing strategy...",
                height=100
            )
        
        # Run optimization
        if st.button("🚀 Run Premium Optimization", type="primary"):
            with st.spinner("Optimizing premiums..."):
                try:
                    # Create optimization results
                    optimized_data = self.data.copy()
                    
                    if 'LossRatio' in optimized_data.columns:
                        # Apply risk-based pricing
                        optimized_data['OptimizedPremium'] = base_premium * (
                            1 + (optimized_data['LossRatio'] - optimized_data['LossRatio'].mean()) * risk_multiplier
                        ) * market_factor * (1 - competition_discount / 100)
                        
                        # Ensure minimum premium
                        optimized_data['OptimizedPremium'] = optimized_data['OptimizedPremium'].clip(lower=base_premium * 0.5)
                        
                        # Calculate impact
                        premium_change = optimized_data['OptimizedPremium'] - optimized_data['TotalPremium']
                        total_impact = premium_change.sum()
                        
                        st.success(f"✅ Premium optimization completed!")
                        
                        # Display results
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Total Premium Change", f"${total_impact:,.0f}")
                        
                        with col2:
                            avg_change = premium_change.mean()
                            st.metric("Average Change", f"${avg_change:,.2f}")
                        
                        with col3:
                            change_pct = (total_impact / optimized_data['TotalPremium'].sum()) * 100
                            st.metric("Change %", f"{change_pct:.1f}%")
                        
                        # Premium comparison
                        st.subheader("📊 Premium Comparison")
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=optimized_data['TotalPremium'],
                            y=optimized_data['OptimizedPremium'],
                            mode='markers',
                            name='Premium Comparison'
                        ))
                        fig.add_trace(go.Scatter(
                            x=[optimized_data['TotalPremium'].min(), optimized_data['TotalPremium'].max()],
                            y=[optimized_data['TotalPremium'].min(), optimized_data['TotalPremium'].max()],
                            mode='lines',
                            name='No Change',
                            line=dict(dash='dash')
                        ))
                        fig.update_layout(
                            title="Current vs Optimized Premiums",
                            xaxis_title="Current Premium",
                            yaxis_title="Optimized Premium"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Risk-based premium analysis
                        if 'LossRatio' in optimized_data.columns:
                            st.subheader("🎯 Risk-Based Premium Analysis")
                            
                            risk_premium = optimized_data.groupby('RiskCategory').agg({
                                'TotalPremium': 'mean',
                                'OptimizedPremium': 'mean',
                                'LossRatio': 'mean'
                            }).reset_index()
                            
                            fig = px.bar(
                                risk_premium,
                                x='RiskCategory',
                                y=['TotalPremium', 'OptimizedPremium'],
                                title="Premium by Risk Category",
                                barmode='group'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Download optimized data
                        csv = optimized_data.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Optimized Data",
                            data=csv,
                            file_name="optimized_premiums.csv",
                            mime="text/csv"
                        )
                        
                except Exception as e:
                    st.error(f"❌ Error optimizing premiums: {str(e)}")


class ExplainabilityViewer:
    """Interactive explainability component using SHAP."""
    
    def __init__(self, model: RiskPredictor):
        self.model = model
    
    def render(self):
        """Render the explainability interface."""
        st.header("🔍 Model Explainability")
        
        st.info("💡 This section provides insights into how the model makes predictions using SHAP (SHapley Additive exPlanations).")
        
        if not hasattr(self.model, 'model') or self.model.model is None:
            st.error("❌ No trained model available for explainability analysis.")
            return
        
        # SHAP analysis
        st.subheader("📊 SHAP Analysis")
        
        # Get sample data for SHAP analysis
        if hasattr(st.session_state, 'preprocessed_data'):
            sample_data = st.session_state.preprocessed_data.sample(min(100, len(st.session_state.preprocessed_data)))
            X_sample, _ = self.preprocessor.prepare_features(sample_data)
            
            # Calculate SHAP values
            try:
                explainer = shap.TreeExplainer(self.model.model)
                shap_values = explainer.shap_values(X_sample)
                
                # SHAP summary plot
                st.markdown("**SHAP Summary Plot**")
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
                st.pyplot(fig)
                plt.close()
                
                # SHAP waterfall plot for a sample prediction
                st.markdown("**SHAP Waterfall Plot (Sample Prediction)**")
                sample_idx = st.slider("Select Sample", 0, len(X_sample)-1, 0)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.waterfall_plot(
                    shap.Explanation(
                        values=shap_values[sample_idx],
                        base_values=explainer.expected_value,
                        data=X_sample.iloc[sample_idx]
                    ),
                    show=False
                )
                st.pyplot(fig)
                plt.close()
                
                # Feature importance comparison
                st.markdown("**Feature Importance Comparison**")
                
                # SHAP importance
                shap_importance = np.abs(shap_values).mean(0)
                feature_importance = dict(zip(X_sample.columns, shap_importance))
                
                # Model importance (if available)
                if hasattr(self.model.model, 'feature_importances_'):
                    model_importance = dict(zip(X_sample.columns, self.model.model.feature_importances_))
                    
                    # Compare importance
                    importance_df = pd.DataFrame({
                        'Feature': X_sample.columns,
                        'SHAP_Importance': shap_importance,
                        'Model_Importance': [model_importance.get(f, 0) for f in X_sample.columns]
                    })
                    
                    fig = px.scatter(
                        importance_df,
                        x='SHAP_Importance',
                        y='Model_Importance',
                        text='Feature',
                        title="SHAP vs Model Feature Importance"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Top features by SHAP importance
                st.markdown("**Top Features by SHAP Importance**")
                top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
                
                fig = px.bar(
                    x=[f[1] for f in top_features],
                    y=[f[0] for f in top_features],
                    orientation='h',
                    title="Top 10 Features by SHAP Importance"
                )
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Error calculating SHAP values: {str(e)}")
                st.info("💡 SHAP analysis requires tree-based models (Random Forest, Gradient Boosting).")
        
        # Local explanations
        st.subheader("🎯 Local Explanations")
        
        st.markdown("**Individual Prediction Explanations**")
        
        # Create sample input for explanation
        if hasattr(st.session_state, 'preprocessed_data'):
            sample_data = st.session_state.preprocessed_data.sample(1)
            X_sample, _ = self.preprocessor.prepare_features(sample_data)
            
            # Display sample features
            st.markdown("**Sample Input Features**")
            st.dataframe(X_sample, use_container_width=True)
            
            # Get prediction and probability
            prediction = self.model.predict(X_sample)[0]
            probability = self.model.predict_risk_probability(X_sample)[0]
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Prediction", "High Risk" if prediction == 1 else "Low Risk")
            with col2:
                st.metric("Risk Probability", f"{probability:.3f}")
            
            # Feature contribution analysis
            st.markdown("**Feature Contributions**")
            
            try:
                explainer = shap.TreeExplainer(self.model.model)
                shap_values = explainer.shap_values(X_sample)
                
                # Create feature contribution plot
                contribution_data = pd.DataFrame({
                    'Feature': X_sample.columns,
                    'Value': X_sample.iloc[0].values,
                    'SHAP_Value': shap_values[0]
                })
                contribution_data = contribution_data.sort_values('SHAP_Value', key=abs, ascending=False)
                
                fig = px.bar(
                    contribution_data,
                    x='SHAP_Value',
                    y='Feature',
                    orientation='h',
                    title="Feature Contributions to Prediction"
                )
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Error calculating feature contributions: {str(e)}")


class ReportGenerator:
    """Report and model card generation component."""
    
    def __init__(self):
        self.settings = get_settings()
    
    def render(self):
        """Render the report generation interface."""
        st.header("📋 Reports & Model Cards")
        
        # Report types
        st.subheader("📄 Report Types")
        
        report_type = st.selectbox(
            "Select Report Type",
            [
                "Executive Summary",
                "Technical Report", 
                "Model Card",
                "Risk Assessment Report",
                "Premium Optimization Report",
                "Compliance Report"
            ]
        )
        
        # Report parameters
        st.subheader("⚙️ Report Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            include_charts = st.checkbox("Include Charts", value=True)
            include_data = st.checkbox("Include Data Tables", value=True)
            include_recommendations = st.checkbox("Include Recommendations", value=True)
        
        with col2:
            report_format = st.selectbox("Report Format", ["PDF", "HTML", "Markdown"])
            include_appendix = st.checkbox("Include Appendix", value=False)
            custom_title = st.text_input("Custom Report Title", value="Insurance Risk Analysis Report")
        
        # Generate report
        if st.button("📊 Generate Report", type="primary"):
            with st.spinner("Generating report..."):
                try:
                    report = self.generate_report(
                        report_type=report_type,
                        include_charts=include_charts,
                        include_data=include_data,
                        include_recommendations=include_recommendations,
                        format=report_format,
                        title=custom_title
                    )
                    
                    st.success("✅ Report generated successfully!")
                    
                    # Display report preview
                    st.subheader("📄 Report Preview")
                    
                    with st.expander("View Report", expanded=True):
                        st.markdown(report)
                    
                    # Download report
                    if report_format == "PDF":
                        # Note: PDF generation would require additional libraries
                        st.info("📥 PDF download requires additional setup (e.g., weasyprint, reportlab)")
                    else:
                        st.download_button(
                            label=f"📥 Download {report_format} Report",
                            data=report,
                            file_name=f"insurance_risk_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{report_format.lower()}",
                            mime="text/plain"
                        )
                
                except Exception as e:
                    st.error(f"❌ Error generating report: {str(e)}")
        
        # Model card generation
        st.subheader("🃏 Model Card Generation")
        
        if st.session_state.get("model_trained", False):
            if st.button("🃏 Generate Model Card", type="primary"):
                with st.spinner("Generating model card..."):
                    try:
                        model_card = self.generate_model_card()
                        st.success("✅ Model card generated successfully!")
                        
                        st.markdown("**Model Card Preview**")
                        st.json(model_card)
                        
                        # Download model card
                        st.download_button(
                            label="📥 Download Model Card (JSON)",
                            data=json.dumps(model_card, indent=2),
                            file_name=f"model_card_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                    
                    except Exception as e:
                        st.error(f"❌ Error generating model card: {str(e)}")
        else:
            st.info("💡 Train a model first to generate a model card.")
    
    def generate_report(self, **kwargs) -> str:
        """Generate a report based on the specified parameters."""
        
        report_type = kwargs.get('report_type', 'Executive Summary')
        title = kwargs.get('title', 'Insurance Risk Analysis Report')
        
        # Base report template
        report = f"""
# {title}

**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Report Type:** {report_type}

## Executive Summary

This report provides a comprehensive analysis of insurance risk patterns and premium optimization strategies.

### Key Findings

- **Data Overview:** Analysis of insurance policy data
- **Risk Patterns:** Identification of high-risk segments
- **Model Performance:** Evaluation of predictive models
- **Recommendations:** Strategic recommendations for risk management

### Methodology

The analysis employs advanced machine learning techniques including:
- Data preprocessing and feature engineering
- Risk prediction models (Random Forest, Gradient Boosting)
- SHAP explainability analysis
- Premium optimization algorithms

## Technical Details

### Data Processing
- Data cleaning and validation
- Feature engineering and selection
- Missing value handling
- Outlier detection and treatment

### Model Development
- Model selection and training
- Hyperparameter optimization
- Cross-validation and evaluation
- Performance metrics analysis

### Risk Analysis
- Loss ratio analysis
- Risk category classification
- Demographic risk profiling
- Temporal trend analysis

## Recommendations

1. **Risk Management:** Implement targeted risk mitigation strategies
2. **Premium Optimization:** Adjust pricing based on risk profiles
3. **Monitoring:** Establish continuous monitoring systems
4. **Compliance:** Ensure regulatory compliance in all processes

---

*This report was generated automatically by the Insurance Risk Analysis Dashboard.*
        """
        
        return report
    
    def generate_model_card(self) -> Dict[str, Any]:
        """Generate a model card for the trained model."""
        
        if not st.session_state.get("model_trained", False):
            raise ValueError("No trained model available")
        
        model = st.session_state.current_model
        results = st.session_state.model_results
        
        model_card = {
            "model_card_version": "1.0",
            "model_details": {
                "name": f"Insurance Risk Predictor - {results['model_type']}",
                "version": "1.0.0",
                "description": "Machine learning model for predicting insurance risk",
                "author": "AlphaCare Insurance Solutions",
                "date": datetime.now().isoformat(),
                "license": "MIT",
                "citation": "Insurance Risk Analysis Dashboard"
            },
            "intended_use": {
                "primary_uses": ["Risk assessment", "Premium pricing", "Underwriting decisions"],
                "primary_users": ["Insurance underwriters", "Risk managers", "Actuaries"],
                "out_of_scope_use_cases": ["Legal decisions", "Medical diagnosis", "Credit scoring"]
            },
            "factors": {
                "relevant_factors": results.get('feature_names', []),
                "evaluation_factors": ["Accuracy", "Precision", "Recall", "F1-Score"]
            },
            "metrics": {
                "model_performance": {
                    "training_accuracy": results['train_accuracy'],
                    "test_accuracy": results['test_accuracy'],
                    "model_type": results['model_type']
                },
                "thresholds": {
                    "accuracy_threshold": 0.8,
                    "precision_threshold": 0.7,
                    "recall_threshold": 0.7
                }
            },
            "evaluation_data": {
                "dataset": "Insurance policy data",
                "motivation": "Risk assessment and premium optimization",
                "preprocessing": "Standardized and cleaned insurance data"
            },
            "training_data": {
                "training_data": "Historical insurance policy data",
                "training_data_size": len(st.session_state.preprocessed_data) if hasattr(st.session_state, 'preprocessed_data') else "Unknown",
                "training_data_distribution": "Insurance policies across different demographics and risk profiles"
            },
            "quantitative_analyses": {
                "unitary_results": {
                    "accuracy": results['test_accuracy'],
                    "precision": "Calculated during training",
                    "recall": "Calculated during training"
                }
            },
            "ethical_considerations": {
                "data": "Insurance data may contain sensitive personal information",
                "human_ai_interaction": "Model should be used as decision support, not replacement",
                "mitigations": "Regular model monitoring and bias detection"
            },
            "caveats_and_recommendations": {
                "caveats": [
                    "Model performance may vary with data distribution shifts",
                    "Regular retraining recommended",
                    "Human oversight required for critical decisions"
                ],
                "recommendations": [
                    "Monitor model performance regularly",
                    "Validate predictions with domain experts",
                    "Update model with new data periodically"
                ]
            }
        }
        
        return model_card
