"""
Turtle Trading Strategy Streamlit Dashboard
==========================================

An interactive Streamlit dashboard for the Turtle Trading Strategy project.
Provides user-friendly interface for predictions, analysis, and visualization.

Author: Panwei Hu
Date: August 27, 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import project modules
from src.model_utils import TurtleModelManager, load_latest_model, create_sample_prediction_data
from src.config import get_key

# Page configuration
st.set_page_config(
    page_title="Turtle Trading Strategy Dashboard",
    page_icon="🐢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E8B57;
    }
    .prediction-box {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 2px solid #2E8B57;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if 'model_manager' not in st.session_state:
        st.session_state.model_manager = load_latest_model()
    if 'predictions' not in st.session_state:
        st.session_state.predictions = None
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None

def main():
    """Main dashboard function."""
    
    # Initialize
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🐢 Turtle Trading Strategy Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("Navigation")
        page = st.selectbox(
            "Choose a page:",
            ["🏠 Overview", "📊 Predictions", "🔬 Analysis", "📈 Performance", "⚙️ Settings"]
        )
        
        st.header("Model Status")
        if st.session_state.model_manager and len(st.session_state.model_manager.models) > 0:
            st.success("✅ Models Loaded")
            st.write(f"**Available Models:** {len(st.session_state.model_manager.models)}")
        else:
            st.warning("⚠️ No Models Loaded")
            st.write("Run analysis to train models")
    
    # Page routing
    if page == "🏠 Overview":
        show_overview()
    elif page == "📊 Predictions":
        show_predictions()
    elif page == "🔬 Analysis":
        show_analysis()
    elif page == "📈 Performance":
        show_performance()
    elif page == "⚙️ Settings":
        show_settings()

def show_overview():
    """Show dashboard overview."""
    st.header("📊 Dashboard Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Models Available",
            value=len(st.session_state.model_manager.models) if st.session_state.model_manager else 0
        )
    
    with col2:
        if st.session_state.model_manager and st.session_state.model_manager.performance_history:
            latest_perf = st.session_state.model_manager.performance_history[-1]
            if 'regression' in latest_perf['task']:
                best_r2 = max([v['R²'] for v in latest_perf['results'].values()])
                st.metric(label="Best R² Score", value=f"{best_r2:.4f}")
            else:
                st.metric(label="Best R² Score", value="N/A")
        else:
            st.metric(label="Best R² Score", value="N/A")
    
    with col3:
        if st.session_state.model_manager and st.session_state.model_manager.performance_history:
            latest_perf = st.session_state.model_manager.performance_history[-1]
            if 'regression' in latest_perf['task']:
                best_dir_acc = max([v['Dir_Acc'] for v in latest_perf['results'].values()])
                st.metric(label="Best Directional Accuracy", value=f"{best_dir_acc:.1%}")
            else:
                st.metric(label="Best Directional Accuracy", value="N/A")
        else:
            st.metric(label="Best Directional Accuracy", value="N/A")
    
    with col4:
        if st.session_state.model_manager and st.session_state.model_manager.metadata:
            timestamp = st.session_state.model_manager.metadata.get('timestamp', 'Unknown')
            st.metric(label="Last Updated", value=timestamp[:8])
        else:
            st.metric(label="Last Updated", value="Never")
    
    # Strategy description
    st.subheader("🎯 Turtle Trading Strategy")
    st.markdown("""
    The Turtle Trading Strategy is a systematic trend-following approach that:
    
    - **Identifies trends** using Donchian Channels and moving averages
    - **Manages risk** with ATR-based position sizing
    - **Generates signals** for entry and exit points
    - **Diversifies** across multiple asset classes
    
    This dashboard provides tools for:
    - Making real-time predictions
    - Running comprehensive analysis
    - Monitoring model performance
    - Generating stakeholder reports
    """)
    
    # Quick actions
    st.subheader("🚀 Quick Actions")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Run Full Analysis", type="primary"):
            st.info("Use the Analysis page to run full analysis pipeline")
    
    with col2:
        if st.button("📊 Make Prediction"):
            st.info("Use the Predictions page to make model predictions")

def show_predictions():
    """Show prediction interface."""
    st.header("📊 Model Predictions")
    
    if not st.session_state.model_manager or len(st.session_state.model_manager.models) == 0:
        st.warning("⚠️ No models available. Please run analysis first.")
        return
    
    # Prediction method selection
    prediction_method = st.radio(
        "Choose prediction method:",
        ["Manual Input", "Sample Data", "File Upload"]
    )
    
    if prediction_method == "Manual Input":
        show_manual_prediction()
    elif prediction_method == "Sample Data":
        show_sample_prediction()
    elif prediction_method == "File Upload":
        show_file_prediction()

def show_manual_prediction():
    """Show manual prediction interface."""
    st.subheader("📝 Manual Feature Input")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Technical Indicators:**")
        sma_20 = st.number_input("SMA 20", value=100.0, step=0.1)
        sma_50 = st.number_input("SMA 50", value=98.0, step=0.1)
        donchian_high = st.number_input("Donchian High 20", value=105.0, step=0.1)
        donchian_low = st.number_input("Donchian Low 20", value=95.0, step=0.1)
    
    with col2:
        st.write("**Risk Metrics:**")
        atr_20 = st.number_input("ATR 20", value=2.0, min_value=0.1, step=0.1)
        volatility = st.number_input("Volatility 20", value=0.025, min_value=0.001, step=0.001)
        trend_strength = st.number_input("Trend Strength", value=0.3, min_value=-1.0, max_value=1.0, step=0.1)
        price_position = st.number_input("Price Position", value=0.6, min_value=0.0, max_value=1.0, step=0.1)
    
    # Make prediction
    if st.button("🔮 Make Prediction", type="primary"):
        features = [sma_20, sma_50, donchian_high, donchian_low, atr_20, volatility, trend_strength, price_position]
        
        try:
            # Create feature DataFrame
            feature_names = ['sma_20', 'sma_50', 'donchian_high_20', 'donchian_low_20', 
                           'atr_20', 'volatility_20', 'trend_strength', 'price_position']
            X = pd.DataFrame([features], columns=feature_names)
            
            # Get predictions
            predictions = {}
            
            # Regression predictions
            for model_name in ['regression_Ridge', 'regression_Linear', 'regression_Random Forest']:
                if model_name in st.session_state.model_manager.models:
                    pred = st.session_state.model_manager.predict(X, model_name)[0]
                    predictions[model_name.replace('regression_', '')] = float(pred)
            
            # Classification predictions
            for model_name in ['classification_binary_Random Forest', 'classification_binary_Logistic']:
                if model_name in st.session_state.model_manager.models:
                    pred = st.session_state.model_manager.predict(X, model_name)[0]
                    proba = st.session_state.model_manager.predict_proba(X, model_name)[0]
                    predictions[model_name.replace('classification_binary_', '')] = {
                        'prediction': int(pred),
                        'probability': proba.tolist() if proba is not None else None
                    }
            
            st.session_state.predictions = predictions
            
            # Display results
            st.success("✅ Predictions generated successfully!")
            
            # Show regression results
            if any('regression' in k for k in predictions.keys()):
                st.subheader("📈 Return Predictions")
                reg_cols = st.columns(len([k for k in predictions.keys() if 'regression' in k or k in ['Ridge', 'Linear', 'Random Forest']]))
                
                for i, (model, pred) in enumerate(predictions.items()):
                    if isinstance(pred, (int, float)):
                        with reg_cols[i]:
                            st.metric(
                                label=f"{model} Model",
                                value=f"{pred:.4f}",
                                delta=f"{pred*100:.2f}%" if pred > 0 else f"{pred*100:.2f}%"
                            )
            
            # Show classification results
            if any('classification' in k for k in predictions.keys()):
                st.subheader("🎯 Direction Predictions")
                for model, result in predictions.items():
                    if isinstance(result, dict):
                        direction = "🟢 LONG" if result['prediction'] == 1 else "🔴 SHORT"
                        prob = result['probability'][1] if result['probability'] else 0.5
                        st.metric(
                            label=f"{model} Model",
                            value=direction,
                            delta=f"{prob:.1%} confidence"
                        )
        
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")

def show_sample_prediction():
    """Show sample prediction interface."""
    st.subheader("🎲 Sample Data Prediction")
    
    # Generate sample data
    n_samples = st.slider("Number of samples", 1, 10, 5)
    
    if st.button("🎲 Generate Sample Predictions", type="primary"):
        try:
            sample_data = create_sample_prediction_data(n_samples=n_samples)
            
            # Make predictions
            predictions = {}
            for i, row in sample_data.iterrows():
                X = pd.DataFrame([row.values], columns=row.index)
                
                # Regression predictions
                for model_name in ['regression_Ridge', 'regression_Linear']:
                    if model_name in st.session_state.model_manager.models:
                        pred = st.session_state.model_manager.predict(X, model_name)[0]
                        predictions[f'sample_{i}_{model_name.replace("regression_", "")}'] = float(pred)
            
            # Display results
            st.success(f"✅ Generated predictions for {n_samples} samples!")
            
            # Create visualization
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Sample Features', 'Predictions'),
                vertical_spacing=0.1
            )
            
            # Feature plot
            for col in sample_data.columns:
                fig.add_trace(
                    go.Scatter(y=sample_data[col], name=col, mode='lines+markers'),
                    row=1, col=1
                )
            
            # Prediction plot
            pred_values = [v for v in predictions.values()]
            fig.add_trace(
                go.Bar(y=pred_values, name='Predictions', marker_color='lightgreen'),
                row=2, col=1
            )
            
            fig.update_layout(height=600, title_text="Sample Data Analysis")
            st.plotly_chart(fig, use_container_width=True)
            
            # Show data table
            st.subheader("📊 Sample Data")
            st.dataframe(sample_data)
            
        except Exception as e:
            st.error(f"❌ Sample prediction error: {str(e)}")

def show_file_prediction():
    """Show file upload prediction interface."""
    st.subheader("📁 File Upload Prediction")
    
    uploaded_file = st.file_uploader(
        "Upload CSV file with features",
        type=['csv'],
        help="File should contain columns: sma_20, sma_50, donchian_high_20, donchian_low_20, atr_20, volatility_20, trend_strength, price_position"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("📊 Uploaded data preview:")
            st.dataframe(df.head())
            
            if st.button("🔮 Predict on Uploaded Data", type="primary"):
                # Validate columns
                required_cols = ['sma_20', 'sma_50', 'donchian_high_20', 'donchian_low_20', 
                               'atr_20', 'volatility_20', 'trend_strength', 'price_position']
                
                missing_cols = [col for col in required_cols if col not in df.columns]
                if missing_cols:
                    st.error(f"❌ Missing columns: {missing_cols}")
                    return
                
                # Make predictions
                predictions = []
                for _, row in df.iterrows():
                    X = pd.DataFrame([row[required_cols].values], columns=required_cols)
                    
                    row_pred = {}
                    for model_name in ['regression_Ridge', 'regression_Linear']:
                        if model_name in st.session_state.model_manager.models:
                            pred = st.session_state.model_manager.predict(X, model_name)[0]
                            row_pred[model_name.replace('regression_', '')] = float(pred)
                    
                    predictions.append(row_pred)
                
                # Display results
                pred_df = pd.DataFrame(predictions)
                st.success(f"✅ Generated predictions for {len(df)} rows!")
                
                st.subheader("📈 Prediction Results")
                st.dataframe(pred_df)
                
                # Visualization
                fig = px.line(pred_df, title="Predictions Over Time")
                st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"❌ File processing error: {str(e)}")

def show_analysis():
    """Show analysis interface."""
    st.header("🔬 Analysis Pipeline")
    
    st.subheader("📊 Run Full Analysis")
    st.markdown("""
    This will run the complete Turtle Trading analysis pipeline:
    1. Load and preprocess data
    2. Engineer features
    3. Train regression and classification models
    4. Generate predictions and performance metrics
    5. Save models and results
    """)
    
    # Data path input
    data_path = st.text_input(
        "Data Path",
        value="data/processed/turtle_enhanced_features_20250825_140536.parquet",
        help="Path to processed data file"
    )
    
    output_dir = st.text_input(
        "Output Directory",
        value="reports",
        help="Directory to save analysis results"
    )
    
    if st.button("🚀 Run Full Analysis", type="primary"):
        if not Path(data_path).exists():
            st.error(f"❌ Data file not found: {data_path}")
            return
        
        try:
            with st.spinner("🔄 Running analysis..."):
                # Initialize model manager if needed
                if not st.session_state.model_manager:
                    st.session_state.model_manager = TurtleModelManager()
                
                # Run analysis
                results = st.session_state.model_manager.run_full_analysis(data_path, output_dir)
                st.session_state.analysis_results = results
            
            st.success("✅ Analysis completed successfully!")
            
            # Show results summary
            st.subheader("📊 Analysis Results")
            
            if 'model_performance' in results:
                # Regression performance
                if 'regression' in results['model_performance']:
                    st.write("**Regression Models:**")
                    reg_df = pd.DataFrame(results['model_performance']['regression']).T
                    st.dataframe(reg_df)
                
                # Classification performance
                if 'binary_classification' in results['model_performance']:
                    st.write("**Binary Classification Models:**")
                    bin_df = pd.DataFrame(results['model_performance']['binary_classification']).T
                    st.dataframe(bin_df)
            
            # Show outputs
            if 'outputs' in results:
                st.write("**Generated Outputs:**")
                for key, value in results['outputs'].items():
                    st.write(f"- {key}: {value}")
        
        except Exception as e:
            st.error(f"❌ Analysis error: {str(e)}")

def show_performance():
    """Show performance monitoring."""
    st.header("📈 Performance Monitoring")
    
    if not st.session_state.model_manager:
        st.warning("⚠️ No models available. Run analysis first.")
        return
    
    # Performance history
    if st.session_state.model_manager.performance_history:
        st.subheader("📊 Performance History")
        
        # Create performance timeline
        history_data = []
        for entry in st.session_state.model_manager.performance_history:
            timestamp = entry['timestamp']
            task = entry['task']
            results = entry['results']
            
            for model, metrics in results.items():
                for metric, value in metrics.items():
                    history_data.append({
                        'timestamp': timestamp,
                        'task': task,
                        'model': model,
                        'metric': metric,
                        'value': value
                    })
        
        if history_data:
            perf_df = pd.DataFrame(history_data)
            perf_df['timestamp'] = pd.to_datetime(perf_df['timestamp'])
            
            # Performance visualization
            fig = px.line(
                perf_df, 
                x='timestamp', 
                y='value', 
                color='model',
                facet_col='metric',
                title="Model Performance Over Time"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Performance table
            st.dataframe(perf_df.pivot_table(
                index=['task', 'model'], 
                columns='metric', 
                values='value'
            ))
    else:
        st.info("📊 No performance history available. Run analysis to generate performance data.")

def show_settings():
    """Show settings and configuration."""
    st.header("⚙️ Settings & Configuration")
    
    st.subheader("🔧 Model Configuration")
    
    # Model summary
    if st.session_state.model_manager:
        summary = st.session_state.model_manager.get_model_summary()
        
        st.write("**Model Summary:**")
        st.json(summary)
        
        # Model management
        st.subheader("🗂️ Model Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Save Models"):
                try:
                    path = st.session_state.model_manager.save_models()
                    st.success(f"✅ Models saved to: {path}")
                except Exception as e:
                    st.error(f"❌ Save error: {str(e)}")
        
        with col2:
            if st.button("🔄 Reload Models"):
                try:
                    st.session_state.model_manager = load_latest_model()
                    st.success("✅ Models reloaded successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Reload error: {str(e)}")
    
    # Configuration
    st.subheader("⚙️ System Configuration")
    
    # Environment variables
    st.write("**Environment Variables:**")
    env_vars = {
        'API_KEY': get_key('API_KEY'),
        'DATA_DIR_RAW': get_key('DATA_DIR_RAW'),
        'DATA_DIR_PROCESSED': get_key('DATA_DIR_PROCESSED')
    }
    
    for key, value in env_vars.items():
        st.write(f"- {key}: {'✅ Set' if value else '❌ Not set'}")
    
    # System info
    st.write("**System Information:**")
    st.write(f"- Python version: {st.get_option('server.enableCORS')}")
    st.write(f"- Streamlit version: {st.__version__}")
    st.write(f"- Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main() 