import streamlit as st
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime
import plotly.graph_objects as go
from io import BytesIO

# Import our custom modules
from churn_model import ChurnPredictionModel
from data_utils import DataHandler
from visualization_utils import ChurnVisualizer

# Page configuration
st.set_page_config(
    page_title="Lunarv - AI Churn Prediction Platform",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern, clean styling
st.markdown("""
<style>
    /* Modern, clean styling matching your website design */
    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        text-align: center;
        color: #212529;
        margin-bottom: 2rem;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    .metric-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.75rem;
        border: 1px solid #e9ecef;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        transition: all 0.2s ease;
    }
    
    .metric-card:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }
    
    .success-message {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #bee5eb;
        border-left: 4px solid #17a2b8;
    }
    
    .warning-message {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ffeaa7;
        border-left: 4px solid #ffc107;
    }
    
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #f5c6cb;
        border-left: 4px solid #dc3545;
    }
    
    .upload-area {
        border: 2px dashed #dee2e6;
        border-radius: 0.75rem;
        padding: 2rem;
        text-align: center;
        background-color: #f8f9fa;
        transition: all 0.2s ease;
        margin: 1rem 0;
    }
    
    .upload-area:hover {
        border-color: #007bff;
        background-color: #f0f8ff;
    }
    
    .nav-button {
        background-color: #007bff;
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 0.5rem;
        font-weight: 500;
        transition: all 0.2s ease;
        text-decoration: none;
        display: inline-block;
        margin: 0.25rem;
    }
    
    .nav-button:hover {
        background-color: #0056b3;
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,123,255,0.3);
    }
    
    .stProgress > div > div > div > div {
        background-color: #007bff;
    }
    
    .stButton > button {
        border-radius: 0.5rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    /* Hide Streamlit's default styling */
    .stApp {
        background-color: #ffffff;
    }
    
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_uploaded' not in st.session_state:
    st.session_state.data_uploaded = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'predictions_made' not in st.session_state:
    st.session_state.predictions_made = False
if 'current_data' not in st.session_state:
    st.session_state.current_data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_performance' not in st.session_state:
    st.session_state.model_performance = None
if 'customer_id_col' not in st.session_state:
    st.session_state.customer_id_col = None
if 'churn_col' not in st.session_state:
    st.session_state.churn_col = None
if 'selected_page' not in st.session_state:
    st.session_state.selected_page = "🏠 Home"

# Initialize classes
data_handler = DataHandler()
visualizer = ChurnVisualizer()

def get_safe_column_display_name(column_name, fallback="Column"):
    """Get a safe display name for a column, handling None and empty values"""
    if column_name is None or column_name == "":
        return fallback
    return str(column_name)

def validate_column_selections(df, customer_id_col, churn_col):
    """Validate that the selected columns exist and are appropriate"""
    errors = []
    warnings = []
    
    if df is None:
        errors.append("No data available")
        return errors, warnings
    
    if customer_id_col is None or customer_id_col not in df.columns:
        errors.append(f"Customer ID column '{customer_id_col}' not found in data")
    elif df[customer_id_col].isnull().sum() > 0:
        warnings.append(f"Customer ID column '{customer_id_col}' contains missing values")
    
    if churn_col is None or churn_col not in df.columns:
        errors.append(f"Churn column '{churn_col}' not found in data")
    elif df[churn_col].isnull().sum() > 0:
        warnings.append(f"Churn column '{churn_col}' contains missing values")
    
    return errors, warnings

def suggest_columns_dynamically(df):
    """Dynamically suggest columns based on data content and patterns"""
    suggestions = {}
    
    if df is None or len(df.columns) == 0:
        return suggestions
    
    # Look for customer ID columns
    customer_id_candidates = ['customer_id', 'customerid', 'id', 'user_id', 'userid', 'client_id', 'clientid', 'customer']
    for col in df.columns:
        col_lower = col.lower()
        for candidate in customer_id_candidates:
            if candidate in col_lower:
                suggestions['customer_id'] = col
                break
        if 'customer_id' in suggestions:
            break
    
    # Look for churn columns
    churn_candidates = ['churn', 'churn_status', 'status', 'is_churned', 'churned', 'attrition', 'left']
    for col in df.columns:
        col_lower = col.lower()
        for candidate in churn_candidates:
            if candidate in col_lower:
                suggestions['churn'] = col
                break
        if 'churn' in suggestions:
            break
    
    # If no specific matches, use heuristics
    if 'customer_id' not in suggestions:
        # Look for columns with unique values (potential IDs)
        for col in df.columns:
            if df[col].nunique() == len(df) and df[col].dtype in ['object', 'int64']:
                suggestions['customer_id'] = col
                break
    
    if 'churn' not in suggestions:
        # Look for binary or categorical columns (potential churn)
        for col in df.columns:
            if df[col].nunique() <= 5 and df[col].dtype == 'object':
                suggestions['churn'] = col
                break
    
    # Fallback to first and last columns if no suggestions found
    if 'customer_id' not in suggestions:
        suggestions['customer_id'] = df.columns[0]
    if 'churn' not in suggestions:
        suggestions['churn'] = df.columns[-1]
    
    return suggestions

def analyze_columns_for_user(df):
    """Analyze columns and provide user-friendly insights"""
    analysis = {
        'total_columns': len(df.columns),
        'total_rows': len(df),
        'column_types': {},
        'potential_customer_id': [],
        'potential_churn': [],
        'numeric_features': [],
        'categorical_features': []
    }
    
    if df is None or len(df.columns) == 0:
        return analysis
    
    for col in df.columns:
        col_type = str(df[col].dtype)
        unique_count = df[col].nunique()
        missing_count = df[col].isnull().sum()
        
        analysis['column_types'][col] = {
            'type': col_type,
            'unique_values': unique_count,
            'missing_values': missing_count,
            'sample_values': df[col].dropna().head(3).tolist()
        }
        
        # Categorize columns
        if col_type in ['int64', 'float64']:
            analysis['numeric_features'].append(col)
        else:
            analysis['categorical_features'].append(col)
        
        # Identify potential customer ID columns
        if unique_count == len(df) and missing_count == 0:
            analysis['potential_customer_id'].append(col)
        
        # Identify potential churn columns
        if unique_count <= 5 and missing_count == 0:
            analysis['potential_churn'].append(col)
    
    return analysis

def main():
    # Header
    st.markdown('<h1 class="main-header">🌙 Lunarv</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">AI-Powered Customer Churn Prediction Platform</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Navigation")
        page = st.selectbox(
            "Choose a section:",
            ["🏠 Home", "📁 Data Upload", "🤖 Model Training", "📈 Results & Analytics", "💾 Export Results"],
            key="page_selector"
        )
        
        # Update session state when page changes
        if page != st.session_state.selected_page:
            st.session_state.selected_page = page
        
        st.markdown("---")
        st.header("ℹ️ About")
        st.markdown("""
        Lunarv uses advanced machine learning to predict customer churn with high accuracy.
        
        **Features:**
        - Multiple ML algorithms
        - Real-time analytics
        - Beautiful visualizations
        - Export capabilities
        
        **Supported formats:**
        - CSV files
        - Excel files
        """)
        
        if st.button("🔄 Reset Session"):
            st.session_state.clear()
            st.success("🔄 Session reset! The app will refresh automatically.")
        
        # Debug section
        if st.checkbox("🐛 Show Debug Info", key="sidebar_debug_info"):
            st.subheader("🔍 Debug Information")
            st.write("**Session State:**")
            st.json({
                'data_uploaded': st.session_state.data_uploaded,
                'current_data_shape': st.session_state.current_data.shape if st.session_state.current_data is not None else None,
                'customer_id_col': st.session_state.customer_id_col,
                'churn_col': st.session_state.churn_col,
                'selected_page': st.session_state.selected_page
            })
            
            if st.session_state.current_data is not None:
                st.write("**Available Columns:**")
                st.write(list(st.session_state.current_data.columns))
    
    # Main content based on selected page
    if st.session_state.selected_page == "🏠 Home":
        show_home_page()
    elif st.session_state.selected_page == "📁 Data Upload":
        show_data_upload_page()
    elif st.session_state.selected_page == "🤖 Model Training":
        show_model_training_page()
    elif st.session_state.selected_page == "📈 Results & Analytics":
        show_results_page()
    elif st.session_state.selected_page == "💾 Export Results":
        show_export_page()

def show_home_page():
    # Modern navigation header like your website
    st.markdown("""
    <div style="background: white; padding: 1rem 0; border-bottom: 1px solid #e9ecef; margin-bottom: 2rem;">
        <div style="display: flex; justify-content: space-between; align-items: center; max-width: 1200px; margin: 0 auto;">
            <div style="font-size: 1.5rem; font-weight: 600; color: #212529;">🌙 Lunarv</div>
            <div style="display: flex; gap: 1rem;">
                <a href="#" class="nav-button" style="background-color: #007bff; color: white; padding: 0.5rem 1rem; border-radius: 0.5rem; text-decoration: none;">Input</a>
                <a href="#" class="nav-button" style="background-color: transparent; color: #6c757d; border: 1px solid #dee2e6;">Output</a>
                <a href="#" class="nav-button" style="background-color: transparent; color: #6c757d; border: 1px solid #dee2e6;">History</a>
                <a href="#" class="nav-button" style="background-color: transparent; color: #6c757d; border: 1px solid #dee2e6;">Help</a>
                <span style="color: #6c757d; margin-left: 1rem;">❤️</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.header("🎯 AI-Powered Customer Churn Prediction")
    st.markdown("### Transform your customer data into actionable retention insights")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 What is Customer Churn?
        Customer churn occurs when customers stop doing business with a company. 
        Predicting churn helps businesses:
        
        - **Retain valuable customers** before they leave
        - **Improve customer satisfaction** through targeted interventions
        - **Increase revenue** by reducing customer acquisition costs
        - **Optimize marketing strategies** for better ROI
        
        ### 🚀 How It Works
        1. **Upload Data**: Import your customer data (CSV/Excel)
        2. **Train Model**: Our AI learns patterns from your data
        3. **Get Predictions**: Identify at-risk customers
        4. **Take Action**: Implement retention strategies
        
        ### 📊 Key Benefits
        - **High Accuracy**: 95%+ prediction accuracy
        - **Fast Processing**: Real-time analysis
        - **Actionable Insights**: Clear risk assessments
        - **Easy Integration**: Simple data upload process
        """)
    
    with col2:
        st.markdown("### 📈 Sample Results")
        
        # Create sample metrics
        sample_metrics = {
            'accuracy': 0.98,
            'f1_score': 0.96,
            'precision': 0.94,
            'recall': 0.97
        }
        
        fig = visualizer.create_performance_metrics_card(sample_metrics)
        st.plotly_chart(fig, use_container_width=True)
    
    # Platform Preview with modern design
    st.markdown("---")
    st.markdown("""
    <div style="background: #f8f9fa; padding: 2rem; border-radius: 0.75rem; margin: 2rem 0;">
        <h2 style="color: #212529; margin-bottom: 1.5rem;">👀 Platform Preview</h2>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem;">
            <div>
                <h3 style="color: #495057; margin-bottom: 1rem;">📊 What You'll Get</h3>
                <ul style="color: #6c757d; line-height: 1.6;">
                    <li>Customer risk assessment</li>
                    <li>Churn probability scores</li>
                    <li>Feature importance ranking</li>
                    <li>Performance metrics</li>
                    <li>Actionable insights</li>
                </ul>
                
                <h3 style="color: #495057; margin-top: 1.5rem; margin-bottom: 1rem;">🎯 Key Features</h3>
                <ul style="color: #6c757d; line-height: 1.6;">
                    <li><strong>Smart Column Detection</strong>: Automatically finds customer ID and churn columns</li>
                    <li><strong>Multiple ML Models</strong>: Tests Random Forest, Gradient Boosting, and Logistic Regression</li>
                    <li><strong>Real-time Analytics</strong>: Instant insights and visualizations</li>
                    <li><strong>Export Capabilities</strong>: Download results in CSV or Excel format</li>
                </ul>
            </div>
            
            <div>
                <h3 style="color: #495057; margin-bottom: 1rem;">📈 Sample Dashboard</h3>
                <div style="background: white; padding: 1rem; border-radius: 0.5rem; border: 1px solid #e9ecef;">
                    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 0.5rem; text-align: center;">
                        <div style="padding: 0.5rem; background: #e8f5e8; border-radius: 0.25rem;">
                            <div style="font-size: 0.875rem; color: #6c757d;">Total Customers</div>
                            <div style="font-weight: 600; color: #212529;">10,000</div>
                        </div>
                        <div style="padding: 0.5rem; background: #fff3cd; border-radius: 0.25rem;">
                            <div style="font-size: 0.875rem; color: #6c757d;">Churn Rate</div>
                            <div style="font-weight: 600; color: #212529;">15.2%</div>
                        </div>
                        <div style="padding: 0.5rem; background: #d1ecf1; border-radius: 0.25rem;">
                            <div style="font-size: 0.875rem; color: #6c757d;">Accuracy</div>
                            <div style="font-weight: 600; color: #212529;">94.8%</div>
                        </div>
                    </div>
                </div>
                
                <h3 style="color: #495057; margin-top: 1.5rem; margin-bottom: 1rem;">🔮 Prediction Example</h3>
                <div style="background: #fff3cd; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #ffc107;">
                    <strong>Customer #12345:</strong><br>
                    • Churn Probability: 78.5%<br>
                    • Risk Level: High Risk<br>
                    • Recommendation: Immediate retention action needed
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick start section
    st.markdown("---")
    st.header("🚀 Quick Start")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("**Step 1: Upload Data**\n\nPrepare your customer data with columns for customer ID and churn status.")
    
    with col2:
        st.info("**Step 2: Train Model**\n\nOur AI will automatically select the best algorithm for your data.")
    
    with col3:
        st.info("**Step 3: Get Insights**\n\nView predictions, risk levels, and actionable recommendations.")
    
    if st.button("🚀 Start Now", type="primary"):
        # Update the page selection in session state to trigger navigation
        st.session_state.selected_page = "📁 Data Upload"
        st.success("🎯 Navigating to Data Upload... Please use the sidebar to continue.")

def show_data_upload_page():
    # Modern navigation header
    st.markdown("""
    <div style="background: white; padding: 1rem 0; border-bottom: 1px solid #e9ecef; margin-bottom: 2rem;">
        <div style="display: flex; justify-content: space-between; align-items: center; max-width: 1200px; margin: 0 auto;">
            <div style="font-size: 1.5rem; font-weight: 600; color: #212529;">🌙 Lunarv</div>
            <div style="display: flex; gap: 1rem;">
                <a href="#" class="nav-button" style="background-color: #007bff; color: white; padding: 0.5rem 1rem; border-radius: 0.5rem; text-decoration: none;">Input</a>
                <a href="#" class="nav-button" style="background-color: transparent; color: #6c757d; border: 1px solid #dee2e6;">Output</a>
                <a href="#" class="nav-button" style="background-color: transparent; color: #6c757d; border: 1px solid #dee2e6;">History</a>
                <a href="#" class="nav-button" style="background-color: transparent; color: #6c757d; border: 1px solid #dee2e6;">Help</a>
                <span style="color: #6c757d; margin-left: 1rem;">❤️</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.header("📁 Data Upload & Validation")
    
    # Modern file upload section like your website
    st.markdown("""
    <div style="background: #f8f9fa; padding: 2rem; border-radius: 0.75rem; margin: 2rem 0;">
        <h3 style="color: #212529; margin-bottom: 1.5rem;">📤 Upload Your Input File(s)</h3>
        
        <div style="border: 2px dashed #dee2e6; border-radius: 0.75rem; padding: 3rem 2rem; text-align: center; background: white; margin-bottom: 1.5rem;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">☁️⬆️</div>
            <div style="font-size: 1.25rem; font-weight: 500; color: #212529; margin-bottom: 0.5rem;">Upload Your Input File(s)</div>
            <div style="color: #6c757d; margin-bottom: 1rem;">Supported: .csv, .xls, .xlsx</div>
            <div style="color: #6c757d;">Max 20MB</div>
        </div>
        
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 1rem; margin-bottom: 1.5rem;">
            <button style="background: white; border: 1px solid #dee2e6; padding: 0.75rem; border-radius: 0.5rem; display: flex; align-items: center; gap: 0.5rem; cursor: pointer; transition: all 0.2s ease;">
                🔗 URL
            </button>
            <button style="background: white; border: 1px solid #dee2e6; padding: 0.75rem; border-radius: 0.5rem; display: flex; align-items: center; gap: 0.5rem; cursor: pointer; transition: all 0.2s ease;">
                ☁️ Drive
            </button>
            <button style="background: white; border: 1px solid #dee2e6; padding: 0.75rem; border-radius: 0.5rem; display: flex; align-items: center; gap: 0.5rem; cursor: pointer; transition: all 0.2s ease;">
                📦 Dropbox
            </button>
            <button style="background: white; border: 1px solid #dee2e6; padding: 0.75rem; border-radius: 0.5rem; display: flex; align-items: center; gap: 0.5rem; cursor: pointer; transition: all 0.2s ease;">
                ☁️ OneDrive
            </button>
            <button style="background: white; border: 1px solid #dee2e6; padding: 0.75rem; border-radius: 0.5rem; display: flex; align-items: center; gap: 0.5rem; cursor: pointer; transition: all 0.2s ease;">
                🐙 GitHub
            </button>
            <button style="background: white; border: 1px solid #dee2e6; padding: 0.75rem; border-radius: 0.5rem; display: flex; align-items: center; gap: 0.5rem; cursor: pointer; transition: all 0.2s ease;">
                ➕ More
            </button>
        </div>
        
        <div style="text-align: center;">
            <button style="background: #007bff; color: white; border: none; padding: 1rem 2rem; border-radius: 0.5rem; font-weight: 500; cursor: pointer; transition: all 0.2s ease;">
                📄 Load Sample Input Data
            </button>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Actual file uploader (hidden but functional)
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload your customer data file (CSV or Excel format)",
        label_visibility="collapsed"
    )
        
        if uploaded_file is not None:
            # Validate file
            is_valid, message = data_handler.validate_file(uploaded_file)
            
            if is_valid:
                st.success(f"✅ {message}")
                
                # Load data
                with st.spinner("Loading data..."):
                    df = data_handler.load_data(uploaded_file)
                
                if df is not None:
                    st.session_state.current_data = df
                    st.session_state.data_uploaded = True
                    
                    # Initialize column selections for uploaded data using dynamic suggestions
                    suggestions = suggest_columns_dynamically(df)
                    st.session_state.customer_id_col = suggestions.get('customer_id', df.columns[0])
                    st.session_state.churn_col = suggestions.get('churn', df.columns[-1])
                    
                    # Show data preview
                    st.subheader("📋 Data Preview")
                    st.dataframe(df.head(10), use_container_width=True)
                    
                    # Data summary
                    st.subheader("📊 Data Summary")
                    summary = data_handler.get_data_summary(df)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Rows", f"{summary['shape'][0]:,}")
                        st.metric("Total Columns", summary['shape'][1])
                    
                    with col2:
                        st.metric("Numeric Columns", len(summary['numeric_summary']))
                        st.metric("Categorical Columns", len(summary['categorical_summary']))
                    
                    # Manual column mapping section
                    st.subheader("🔧 Manual Column Mapping")
                    st.info("💡 **Skip automatic validation** and manually map your columns below:")
                    
                    # Column selection
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Primary Key Column (Customer ID):**")
                        customer_id_col = st.selectbox(
                            "Select the column that contains unique customer identifiers:",
                            options=df.columns.tolist(),
                            index=0,
                            key="upload_customer_id_col"
                        )
                        
                        # Show sample values
                        if customer_id_col:
                            st.write(f"**Sample values:** {df[customer_id_col].head(3).tolist()}")
                    
                    with col2:
                        st.markdown("**Churn Column:**")
                        churn_col = st.selectbox(
                            "Select the column that indicates churn status:",
                            options=df.columns.tolist(),
                            index=len(df.columns)-1,
                            key="upload_churn_col"
                        )
                        
                        # Show churn column info and conversion options
                        if churn_col:
                            st.write(f"**Current values:** {df[churn_col].value_counts().head(5).to_dict()}")
                            
                            # Check if churn column needs conversion
                            if df[churn_col].dtype == 'object' or df[churn_col].nunique() > 10:
                                st.warning("⚠️ **Categorical churn column detected!**")
                                
                                # Conversion options
                                conversion_method = st.selectbox(
                                    "How to convert to binary (0/1):",
                                    ["Auto-detect (Yes/No → 1/0)", "Custom mapping", "Keep as-is"],
                                    key="churn_conversion_method"
                                )
                                
                                if conversion_method == "Custom mapping":
                                    st.write("**Map your values to binary:**")
                                    unique_values = df[churn_col].unique()
                                    col_a, col_b = st.columns(2)
                                    
                                    with col_a:
                                        st.write("**Churn values (1):**")
                                        churn_values = st.multiselect(
                                            "Select values that mean 'churned':",
                                            options=unique_values,
                                            key="churn_values"
                                        )
                                    
                                    with col_b:
                                        st.write("**Loyal values (0):**")
                                        loyal_values = st.multiselect(
                                            "Select values that mean 'loyal':",
                                            options=unique_values,
                                            key="loyal_values"
                                        )
                    
                    # Manual validation and proceed button
                    if st.button("✅ Proceed with Manual Mapping", type="primary"):
                        if customer_id_col and churn_col:
                            # Store column selections
                            st.session_state.customer_id_col = customer_id_col
                            st.session_state.churn_col = churn_col
                            st.session_state.data_uploaded = True
                            
                            # Convert churn column if needed
                            if 'conversion_method' in locals() and conversion_method == "Auto-detect (Yes/No → 1/0)":
                                # Auto-convert common patterns
                                df_converted = df.copy()
                                df_converted[churn_col] = df_converted[churn_col].map({
                                    'Yes': 1, 'No': 0,
                                    'yes': 1, 'no': 0,
                                    'YES': 1, 'NO': 0,
                                    'True': 1, 'False': 0,
                                    'true': 1, 'false': 0,
                                    'TRUE': 1, 'FALSE': 0,
                                    'Churned': 1, 'Loyal': 0,
                                    'churned': 1, 'loyal': 0
                                })
                                
                                # Check if conversion was successful
                                if df_converted[churn_col].dtype in ['int64', 'float64']:
                                    st.session_state.current_data = df_converted
                                    st.success("✅ Churn column auto-converted to binary format!")
                                else:
                                    st.session_state.current_data = df
                                    st.warning("⚠️ Auto-conversion failed. Using original format.")
                            
                            elif 'conversion_method' in locals() and conversion_method == "Custom mapping" and 'churn_values' in locals() and 'loyal_values' in locals() and churn_values and loyal_values:
                                # Custom conversion
                                df_converted = df.copy()
                                df_converted[churn_col] = df_converted[churn_col].map(
                                    {val: 1 for val in churn_values}
                                ).fillna(0)
                                
                                st.session_state.current_data = df_converted
                                st.success("✅ Churn column converted using custom mapping!")
                            
                            else:
                                # Keep as-is
                                st.session_state.current_data = df
                                st.info("ℹ️ Using original churn column format.")
                            
                            # Show converted data preview
                            if 'df_converted' in locals() and df_converted is not None:
                                st.subheader("📋 Converted Data Preview")
                                st.write("**Churn column after conversion:**")
                                st.write(f"**Unique values:** {df_converted[churn_col].unique()}")
                                st.write(f"**Value counts:** {df_converted[churn_col].value_counts().to_dict()}")
                                st.dataframe(df_converted[[customer_id_col, churn_col]].head(10))
                            
                            st.success("🎉 Data loaded successfully! You can now proceed to model training.")
                            st.experimental_rerun()
                        
                        else:
                            st.error("❌ Please select both Customer ID and Churn columns.")
                    
                    # Skip validation option
                    if st.button("🚀 Skip Validation & Continue"):
                        st.session_state.customer_id_col = df.columns[0]
                        st.session_state.churn_col = df.columns[-1]
                        st.session_state.current_data = df
                        st.session_state.data_uploaded = True
                        st.success("✅ Skipped validation. Using first and last columns as defaults.")
                        st.experimental_rerun()
                
            else:
                st.error(f"❌ {message}")
    
    with col2:
        st.markdown("### 📋 Data Requirements")
        st.markdown("""
        **Required Columns:**
        - Customer ID (unique identifier)
        - Churn status (0 = loyal, 1 = churned)
        
        **Recommended Features:**
        - Demographics (age, gender, location)
        - Usage patterns (tenure, frequency)
        - Financial data (revenue, charges)
        - Service preferences
        
        **File Formats:**
        - CSV (UTF-8 encoding)
        - Excel (.xlsx, .xls)
        - Max size: 100MB
        """)
        
        # Sample data option
        if st.button("📊 Try Sample Data"):
            with st.spinner("Generating sample data..."):
                sample_df = data_handler.create_sample_data()
                st.session_state.current_data = sample_df
                st.session_state.data_uploaded = True
                # Set default column selections for sample data
                # Use the actual column names from the sample data
                st.session_state.customer_id_col = 'customer_id' if 'customer_id' in sample_df.columns else sample_df.columns[0]
                st.session_state.churn_col = 'churn' if 'churn' in sample_df.columns else sample_df.columns[-1]
                st.success("✅ Sample data loaded! You can now proceed to model training.")
                st.info("💡 Use the sidebar to navigate to Model Training when ready.")
                st.experimental_rerun()
    
    # Column mapping section
    if st.session_state.data_uploaded and st.session_state.current_data is not None:
        df = st.session_state.current_data
        
        # Safety check: ensure dataframe has columns
        if df is None or len(df.columns) == 0:
            st.error("❌ No columns found in the uploaded data!")
            return
        
        st.markdown("---")
        st.subheader("🔧 Column Configuration")
        
        # Suggest column mapping using dynamic suggestions
        suggestions = suggest_columns_dynamically(df)
        
        # Initialize session state for column selections if not exists
        if 'customer_id_col' not in st.session_state:
            st.session_state.customer_id_col = suggestions.get('customer_id', df.columns[0])
        if 'churn_col' not in st.session_state:
            st.session_state.churn_col = suggestions.get('churn', df.columns[-1])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Customer ID Column:**")
            # Safe index calculation for customer ID column
            customer_id_index = 0
            if st.session_state.customer_id_col and st.session_state.customer_id_col in df.columns:
                try:
                    customer_id_index = df.columns.get_loc(st.session_state.customer_id_col)
                except (ValueError, KeyError):
                    customer_id_index = 0
            
            selected_customer_id = st.selectbox(
                "Select customer ID column:",
                options=df.columns.tolist(),
                index=customer_id_index,
                key="customer_id_col_selector"
            )
            
            # Update session state when selection changes
            if selected_customer_id != st.session_state.customer_id_col:
                st.session_state.customer_id_col = selected_customer_id
        
        with col2:
            st.markdown("**Churn Column:**")
            # Safe index calculation for churn column
            churn_index = len(df.columns) - 1
            if st.session_state.churn_col and st.session_state.churn_col in df.columns:
                try:
                    churn_index = df.columns.get_loc(st.session_state.churn_col)
                except (ValueError, KeyError):
                    churn_index = len(df.columns) - 1
            
            selected_churn = st.selectbox(
                "Select churn column:",
                options=df.columns.tolist(),
                index=churn_index,
                key="churn_col_selector"
            )
            
            # Update session state when selection changes
            if selected_churn != st.session_state.churn_col:
                st.session_state.churn_col = selected_churn
        
        # Show current column selections
        st.subheader("📋 Current Column Selections")
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Customer ID:** {get_safe_column_display_name(st.session_state.customer_id_col, 'Not set')}")
        with col2:
            st.info(f"**Churn Column:** {get_safe_column_display_name(st.session_state.churn_col, 'Not set')}")
         
        # Add column analysis section
        if st.checkbox("🔍 Show Detailed Column Analysis", key="show_column_analysis"):
            st.subheader("📊 Column Analysis")
            analysis = analyze_columns_for_user(df)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Total Rows:** {analysis['total_rows']:,}")
                st.write(f"**Total Columns:** {analysis['total_columns']}")
                st.write(f"**Numeric Features:** {len(analysis['numeric_features'])}")
                st.write(f"**Categorical Features:** {len(analysis['categorical_features'])}")
            
            with col2:
                st.write("**Potential Customer ID Columns:**")
                if analysis['potential_customer_id']:
                    for col in analysis['potential_customer_id']:
                        st.write(f"  • {col}")
                else:
                    st.write("  None identified")
                
                st.write("**Potential Churn Columns:**")
                if analysis['potential_churn']:
                    for col in analysis['potential_churn']:
                        st.write(f"  • {col}")
                else:
                    st.write("  None identified")
            
            # Show detailed column information
            st.subheader("📋 Column Details")
            for col, info in analysis['column_types'].items():
                with st.expander(f"📊 {col} ({info['type']})"):
                    st.write(f"**Unique Values:** {info['unique_values']:,}")
                    st.write(f"**Missing Values:** {info['missing_values']:,}")
                    st.write(f"**Sample Values:** {info['sample_values']}")
        
        if st.session_state.customer_id_col and st.session_state.churn_col:
            st.success("✅ Column configuration complete! You can now proceed to model training.")
        else:
            st.warning("⚠️ Please select both Customer ID and Churn columns before proceeding.")
        
        # Manual save button as backup
        if st.button("💾 Save Column Selections", type="primary"):
            if selected_customer_id and selected_churn:
                st.session_state.customer_id_col = selected_customer_id
                st.session_state.churn_col = selected_churn
                st.success("✅ Column selections manually saved!")
                st.experimental_rerun()
            else:
                st.error("❌ Please select both columns before saving.")

def show_model_training_page():
    st.header("🤖 Model Training & Evaluation")
    
    if not st.session_state.data_uploaded:
        st.warning("⚠️ Please upload data first before training the model.")
        return
    
    df = st.session_state.current_data
    customer_id_col = st.session_state.customer_id_col
    churn_col = st.session_state.churn_col
    
    # Debug section to help diagnose issues
    if st.checkbox("🐛 Show Debug Info", key="model_training_debug_info"):
        st.subheader("🔍 Debug Information")
        st.write("**Session State:**")
        st.json({
            'data_uploaded': st.session_state.data_uploaded,
            'current_data_shape': df.shape if df is not None else None,
            'customer_id_col': customer_id_col,
            'churn_col': churn_col,
            'selected_page': st.session_state.selected_page
        })
        
        if df is not None:
            st.write("**Available Columns:**")
            st.write(list(df.columns))
            
            if customer_id_col and customer_id_col in df.columns:
                st.write(f"**Customer ID Column '{customer_id_col}' sample values:**")
                st.write(df[customer_id_col].head(5).tolist())
            
            if churn_col and churn_col in df.columns:
                st.write(f"**Churn Column '{churn_col}' sample values:**")
                st.write(df[churn_col].head(5).tolist())
                st.write(f"**Churn Column dtype:** {df[churn_col].dtype}")
                st.write(f"**Churn Column unique values:** {df[churn_col].unique()[:10].tolist()}")
    
    # Validate column selections
    if customer_id_col is None or churn_col is None:
        st.error("❌ Column configuration is incomplete!")
        st.info("💡 Please go back to Data Upload and configure your columns first.")
        
        # Show current column selections and available columns
        st.subheader("🔧 Current Column Configuration")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Customer ID Column:** {customer_id_col or 'Not set'}")
        with col2:
            st.write(f"**Churn Column:** {churn_col or 'Not set'}")
        
        # Show available columns
        st.subheader("📋 Available Columns in Your Data")
        st.write("**All columns:** " + ", ".join(df.columns.tolist()))
        
        # Provide quick fix options
        st.subheader("🚀 Quick Fix Options")
        
        # Auto-detect columns
        if st.button("🔍 Auto-Detect Columns"):
            # Smart column detection - look for common patterns
            customer_id_candidates = ['customer_id', 'customerid', 'id', 'user_id', 'userid', 'client_id', 'clientid']
            churn_candidates = ['churn', 'churn_status', 'status', 'is_churned', 'churned']
            
            # Find customer ID column
            customer_id_col = None
            for candidate in customer_id_candidates:
                if candidate in df.columns:
                    customer_id_col = candidate
                    break
            if customer_id_col is None:
                customer_id_col = df.columns[0]  # Default to first column
            
            # Find churn column
            churn_col = None
            for candidate in churn_candidates:
                if candidate in df.columns:
                    churn_col = candidate
                    break
            if churn_col is None:
                churn_col = df.columns[-1]  # Default to last column
            
            st.session_state.customer_id_col = customer_id_col
            st.session_state.churn_col = churn_col
            
            st.success(f"✅ Columns auto-detected! Customer ID: {customer_id_col}, Churn: {churn_col}")
            
            # Try to refresh the page
            try:
                st.experimental_rerun()
            except:
                st.info("💡 Please refresh the page manually or use the sidebar to navigate back to Model Training.")
        
        # Manual column selection
        st.subheader("📝 Manual Column Selection")
        col1, col2 = st.columns(2)
        
        with col1:
            # Find the index of customer ID column, default to first column if not found
            customer_id_index = 0
            if st.session_state.customer_id_col and st.session_state.customer_id_col in df.columns:
                customer_id_index = df.columns.get_loc(st.session_state.customer_id_col)
            elif 'customer_id' in df.columns:
                customer_id_index = df.columns.get_loc('customer_id')
            
            new_customer_id_col = st.selectbox(
                "Select Customer ID Column:",
                options=df.columns.tolist(),
                index=customer_id_index,
                key="temp_customer_id"
            )
        
        with col2:
            # Find the index of 'churn' column, default to last column if not found
            churn_index = len(df.columns) - 1
            if 'churn' in df.columns:
                churn_index = df.columns.get_loc('churn')
            
            new_churn_col = st.selectbox(
                "Select Churn Column:",
                options=df.columns.tolist(),
                index=churn_index,
                key="temp_churn_col"
            )
        
        if st.button("💾 Save Column Selection"):
            st.session_state.customer_id_col = new_customer_id_col
            st.session_state.churn_col = new_churn_col
            st.success("✅ Column selection saved!")
            
            # Try to refresh the page
            try:
                st.experimental_rerun()
            except:
                st.info("💡 Please refresh the page manually or use the sidebar to navigate back to Model Training.")
        
        return
    
    # Validate that columns exist in the dataframe
    if customer_id_col is None or customer_id_col not in df.columns:
        st.error(f"❌ Customer ID column '{customer_id_col}' not found in data!")
        st.info("💡 Please check your column configuration.")
        
        # Show debug info
        st.subheader("🔍 Debug Information")
        st.write(f"**Expected Customer ID Column:** {customer_id_col}")
        st.write(f"**Available Columns:** {list(df.columns)}")
        st.write(f"**Session State Customer ID:** {st.session_state.customer_id_col}")
        
        # Auto-fix option
        if st.button("🔧 Auto-Fix Column Selection"):
            # Try to find reasonable defaults using common patterns
            customer_id_candidates = ['customer_id', 'customerid', 'id', 'user_id', 'userid', 'client_id', 'clientid']
            churn_candidates = ['churn', 'churn_status', 'status', 'is_churned', 'churned']
            
            # Find customer ID column
            customer_id_col = None
            for candidate in customer_id_candidates:
                if candidate in df.columns:
                    customer_id_col = candidate
                    break
            if customer_id_col is None:
                customer_id_col = df.columns[0]  # Default to first column
            
            # Find churn column
            churn_col = None
            for candidate in churn_candidates:
                if candidate in df.columns:
                    churn_col = candidate
                    break
            if churn_col is None:
                churn_col = df.columns[-1]  # Default to last column
            
            st.session_state.customer_id_col = customer_id_col
            st.session_state.churn_col = churn_col
            
            st.success(f"✅ Auto-fixed! Customer ID: {st.session_state.customer_id_col}, Churn: {st.session_state.churn_col}")
            st.experimental_rerun()
        
        return
    
    if churn_col is None or churn_col not in df.columns:
        st.error(f"❌ Churn column '{churn_col}' not found in data!")
        st.info("💡 Please check your column configuration.")
        
        # Show debug info
        st.subheader("🔍 Debug Information")
        st.write(f"**Expected Churn Column:** {churn_col}")
        st.write(f"**Available Columns:** {list(df.columns)}")
        st.write(f"**Session State Churn:** {st.session_state.churn_col}")
        
        # Auto-fix option
        if st.button("🔧 Auto-Fix Churn Column"):
            # Try to find reasonable defaults
            if 'churn' in df.columns:
                st.session_state.churn_col = 'churn'
            elif 'churn_status' in df.columns:
                st.session_state.churn_col = 'churn_status'
            elif 'status' in df.columns:
                st.session_state.churn_col = 'status'
            else:
                st.session_state.churn_col = df.columns[-1]
            
            st.success(f"✅ Auto-fixed! Churn column: {st.session_state.churn_col}")
            st.experimental_rerun()
        
        return
    
    st.subheader("📊 Data Overview")
    
    # Check churn column format and warn user if needed
    try:
        if df[churn_col].dtype == 'object':
            st.warning(f"⚠️ **Churn column '{churn_col}' contains text values.** For best results, consider converting to binary (0/1) format.")
            st.info(f"**Current churn values:** {df[churn_col].value_counts().head(10).to_dict()}")
            
            # Quick conversion option
            if st.button("🔄 Quick Convert to Binary", type="secondary"):
                try:
                    # Auto-convert common patterns
                    df_converted = df.copy()
                    df_converted[churn_col] = df_converted[churn_col].map({
                        'Yes': 1, 'No': 0,
                        'yes': 1, 'no': 0,
                        'YES': 1, 'NO': 0,
                        'True': 1, 'False': 0,
                        'true': 1, 'false': 0,
                        'TRUE': 1, 'FALSE': 0,
                        'Churned': 1, 'Loyal': 0,
                        'churned': 1, 'loyal': 0,
                        '1': 1, '0': 0
                    }).fillna(0)
                    
                    # Update session state
                    st.session_state.current_data = df_converted
                    st.success("✅ Churn column converted to binary format! The page will refresh automatically.")
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"❌ Conversion failed: {str(e)}")
                    st.info("💡 Please use the Data Upload section for manual conversion.")
    except Exception as e:
        st.error(f"❌ Error accessing churn column '{churn_col}': {str(e)}")
        st.info("💡 Please check your column configuration in the Data Upload section.")
        return
    
    # Show data info
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Customers", f"{len(df):,}")
    
    with col2:
        # Handle different churn column data types
        try:
            if df[churn_col].dtype in ['int64', 'float64']:
                churn_count = df[churn_col].sum()
            else:
                # For categorical columns, count non-zero/True/Yes values
                churn_count = len(df[df[churn_col].isin([1, '1', 'Yes', 'yes', 'YES', 'True', 'true', 'TRUE', 'Churned', 'churned'])])
        except Exception as e:
            st.error(f"❌ Error calculating churn count: {str(e)}")
            churn_count = 0
        
        st.metric("Churned Customers", f"{churn_count:,}")
    
    with col3:
        try:
            churn_rate = (churn_count / len(df)) * 100
            st.metric("Churn Rate", f"{churn_rate:.1f}%")
        except Exception as e:
            st.error(f"❌ Error calculating churn rate: {str(e)}")
            st.metric("Churn Rate", "N/A")
    
    with col4:
        try:
            loyal_count = len(df) - churn_count
            st.metric("Loyal Customers", f"{loyal_count:,}")
        except Exception as e:
            st.error(f"❌ Error calculating loyal count: {str(e)}")
            st.metric("Loyal Customers", "N/A")
    
    # Model training section
    st.subheader("🚀 Train Churn Prediction Model")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🎯 Start Training", type="primary", disabled=st.session_state.model_trained):
            with st.spinner("Training models..."):
                # Initialize model
                model = ChurnPredictionModel()
                
                # Train models
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, step in enumerate(['Preprocessing data...', 'Training Random Forest...', 
                                        'Training Gradient Boosting...', 'Training Logistic Regression...', 
                                        'Evaluating models...', 'Selecting best model...']):
                    status_text.text(step)
                    progress_bar.progress((i + 1) / 6)
                    time.sleep(0.5)
                
                # Train the model
                model_performance = model.train_models(df, churn_col)
                
                # Store results
                st.session_state.model = model
                st.session_state.model_performance = model_performance
                st.session_state.model_trained = True
                
                progress_bar.progress(1.0)
                status_text.text("✅ Training complete!")
                
                st.success("🎉 Model training completed successfully!")
                st.info("💡 Use the sidebar to navigate to Results & Analytics to view predictions.")
    
    with col2:
        st.markdown("### 🎯 Training Process")
        st.markdown("""
        **Algorithms Used:**
        1. Random Forest
        2. Gradient Boosting
        3. Logistic Regression
        
        **Automatic Selection:**
        - Best model chosen by F1 score
        - Feature importance analysis
        - Performance comparison
        
        **Expected Time:**
        - Small datasets: < 1 minute
        - Large datasets: 2-5 minutes
        """)
    
    # Show training results
    if st.session_state.model_trained and st.session_state.model_performance:
        st.markdown("---")
        st.subheader("📊 Training Results")
        
        model_performance = st.session_state.model_performance
        model = st.session_state.model
        
        # Performance metrics
        st.markdown("### 🎯 Model Performance")
        
        # Get best model
        best_model_name = max(model_performance.keys(), 
                             key=lambda x: model_performance[x]['f1_score'])
        best_metrics = model_performance[best_model_name]
        
        # Create performance visualization
        fig = visualizer.create_performance_metrics_card(best_metrics)
        st.plotly_chart(fig, use_container_width=True)
        
        # Add simple explanation of metrics
        with st.expander("📚 What Do These Metrics Mean?"):
            st.markdown("""
            **🔍 Accuracy**: How often the model is correct overall
            - **{:.1%}** means the model is right {:.0f} out of 100 times
            
            **🎯 F1 Score**: The balance between precision and recall
            - **{:.1%}** indicates good balance between finding churners and avoiding false alarms
            
            **✅ Precision**: When we predict churn, how often are we right?
            - **{:.1%}** means {:.0f}% of our "churn alerts" are accurate
            
            **🔄 Recall**: How many actual churners did we catch?
            - **{:.1%}** means we identified {:.0f}% of customers who actually churned
            """.format(
                best_metrics.get('accuracy', 0),
                best_metrics.get('accuracy', 0) * 100,
                best_metrics.get('f1_score', 0),
                best_metrics.get('precision', 0),
                best_metrics.get('precision', 0) * 100,
                best_metrics.get('recall', 0),
                best_metrics.get('recall', 0) * 100
            ))
        
        # Model comparison
        st.markdown("### 🔍 Model Comparison")
        comparison_fig = visualizer.create_performance_comparison_chart(model_performance)
        st.plotly_chart(comparison_fig, use_container_width=True)
        
        # Feature importance
        if model.feature_importance is not None:
            st.markdown("### 🎯 Feature Importance")
            feature_fig = visualizer.create_feature_importance_chart(model.feature_importance, top_n=10)
            st.plotly_chart(feature_fig, use_container_width=True)
        
        st.success("✅ Model is ready for predictions! You can now proceed to results and analytics.")

def show_results_page():
    st.header("📈 Results & Analytics")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train the model first before viewing results.")
        return
    
    df = st.session_state.current_data
    model = st.session_state.model
    model_performance = st.session_state.model_performance
    churn_col = st.session_state.churn_col
    
    # Validate column selections
    if churn_col is None:
        st.error("❌ Churn column is not configured!")
        st.info("💡 Please go back to Data Upload and configure your columns first.")
        return
    
    if churn_col not in df.columns:
        st.error(f"❌ Churn column '{churn_col}' not found in data!")
        st.info("💡 Please check your column configuration.")
        return
    
    # Generate predictions
    if not st.session_state.predictions_made:
        st.subheader("🔮 Generating Predictions")
        
        with st.spinner("Generating churn predictions..."):
            try:
                # Make predictions
                results_df = model.predict_churn(df, churn_col)
                st.session_state.results_df = results_df
                st.session_state.predictions_made = True
                st.success("✅ Predictions generated successfully!")
            except Exception as e:
                st.error(f"❌ Error generating predictions: {str(e)}")
                return
    
    if st.session_state.predictions_made:
        results_df = st.session_state.results_df
        
        # Key metrics
        st.subheader("📊 Key Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_prob = results_df['churn_probability'].mean() * 100
            st.metric("Avg Churn Probability", f"{avg_prob:.1f}%")
        
        with col2:
            high_risk = len(results_df[results_df['churn_probability'] > 0.7])
            st.metric("High Risk Customers", f"{high_risk:,}")
        
        with col3:
            medium_risk = len(results_df[(results_df['churn_probability'] > 0.3) & (results_df['churn_probability'] <= 0.7)])
            st.metric("Medium Risk Customers", f"{medium_risk:,}")
        
        with col4:
            low_risk = len(results_df[results_df['churn_probability'] <= 0.3])
            st.metric("Low Risk Customers", f"{low_risk:,}")
        
        # Visualizations
        st.subheader("📈 Visualizations")
        
        # Churn distribution
        col1, col2 = st.columns(2)
        
        with col1:
            churn_dist_fig = visualizer.create_churn_distribution_chart(results_df, churn_col)
            st.plotly_chart(churn_dist_fig, use_container_width=True)
        
        with col2:
            risk_level_fig = visualizer.create_risk_level_chart(results_df)
            st.plotly_chart(risk_level_fig, use_container_width=True)
        
        # Add explanation of risk levels
        with st.expander("📊 Understanding Risk Levels"):
            st.markdown("""
            **🚨 High Risk (70%+ churn probability)**
            - **Action Required**: Immediate intervention needed
            - **What to do**: Contact within 24-48 hours, personalized offers
            - **Priority**: Highest - these customers are likely to leave soon
            
            **⚠️ Medium Risk (30-70% churn probability)**
            - **Action Required**: Preventive measures
            - **What to do**: Regular check-ins, satisfaction surveys, retention programs
            - **Priority**: Medium - proactive retention efforts
            
            **✅ Low Risk (0-30% churn probability)**
            - **Action Required**: Maintenance and satisfaction
            - **What to do**: Continue good service, occasional check-ins
            - **Priority**: Low - focus on keeping them happy
            """)
        
        # Churn probability distribution
        prob_dist_fig = visualizer.create_churn_probability_distribution(results_df)
        st.plotly_chart(prob_dist_fig, use_container_width=True)
        
        # Customer segmentation
        segment_fig = visualizer.create_customer_segments_chart(results_df)
        st.plotly_chart(segment_fig, use_container_width=True)
        
        # Detailed results table
        st.subheader("📋 Detailed Results")
        
        # Filter options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            risk_filter = st.selectbox(
                "Filter by Risk Level:",
                ["All", "Low Risk", "Medium Risk", "High Risk"]
            )
        
        with col2:
            min_prob = st.slider("Min Churn Probability:", 0.0, 1.0, 0.0, 0.01)
        
        with col3:
            max_prob = st.slider("Max Churn Probability:", 0.0, 1.0, 1.0, 0.01)
        
        # Apply filters
        filtered_df = results_df[
            (results_df['churn_probability'] >= min_prob) &
            (results_df['churn_probability'] <= max_prob)
        ]
        
        if risk_filter != "All":
            filtered_df = filtered_df[filtered_df['risk_level'] == risk_filter]
        
        # Show filtered results
        # Use dynamic column names from session state
        display_columns = [st.session_state.customer_id_col, 'churn_probability', 'churn_prediction', 'risk_level']
        # Filter out None values and ensure columns exist
        display_columns = [col for col in display_columns if col is not None and col in filtered_df.columns]
        
        st.dataframe(
            filtered_df[display_columns].head(20),
            use_container_width=True
        )
        
        st.info(f"Showing {len(filtered_df)} customers out of {len(results_df)} total customers.")
        
        # Summary report
        st.markdown("---")
        st.subheader("📊 Summary Report")
        
        summary_report = visualizer.create_summary_report(
            results_df, 
            model.get_model_summary(),
            churn_col
        )
        
        st.markdown(summary_report, unsafe_allow_html=True)

def show_export_page():
    st.header("💾 Export Results")
    
    if not st.session_state.predictions_made:
        st.warning("⚠️ Please generate predictions first before exporting results.")
        return
    
    results_df = st.session_state.results_df
    
    st.subheader("📤 Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Export Format")
        export_format = st.selectbox(
            "Choose export format:",
            ["CSV", "Excel"]
        )
        
        if st.button("💾 Export Results", type="primary"):
            with st.spinner("Preparing export..."):
                if export_format == "CSV":
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv,
                        file_name=f"churn_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                else:
                    # Excel export
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        results_df.to_excel(writer, index=False, sheet_name='Churn_Predictions')
                    
                    excel_data = output.getvalue()
                    st.download_button(
                        label="📥 Download Excel",
                        data=excel_data,
                        file_name=f"churn_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
    
    with col2:
        st.markdown("### 📋 Export Contents")
        # Dynamic column description based on actual data
        customer_id_name = st.session_state.customer_id_col or "Customer ID"
        churn_name = st.session_state.churn_col or "Churn Status"
        
        st.markdown(f"""
        **Included Data:**
        - {customer_id_name}
        - Churn Probability
        - Churn Prediction
        - Risk Level
        - All Original Features
        
        **File Naming:**
        - Automatic timestamp
        - Descriptive filename
        - Ready for analysis
        """)
    
    # Export summary
    st.markdown("---")
    st.subheader("📊 Export Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Records", f"{len(results_df):,}")
    
    with col2:
        st.metric("Columns Exported", f"{len(results_df.columns):,}")
    
    with col3:
        st.metric("File Size (est.)", f"{len(results_df) * len(results_df.columns) * 8 / 1024:.1f} KB")
    
    # Sample export preview
    st.markdown("### 👀 Export Preview")
    st.dataframe(results_df.head(10), use_container_width=True)
    
    st.success("✅ Your results are ready for export! Use the download button above to save your data.")

if __name__ == "__main__":
    main()
