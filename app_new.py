import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Import our enhanced ML engine
from lunarv_ml_engine import LunarvMLEngine

# Page configuration
st.set_page_config(
    page_title="🌙 Lunarv - Churn Prediction & Analytics",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .success-message {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    .info-message {
        background: #d1ecf1;
        color: #0c5460;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #bee5eb;
        margin: 1rem 0;
    }
    .warning-message {
        background: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #ffeaa7;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0

if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None

if 'column_mapping' not in st.session_state:
    st.session_state.column_mapping = {}

if 'model_results' not in st.session_state:
    st.session_state.model_results = None

if 'predictions' not in st.session_state:
    st.session_state.predictions = None

if 'ml_engine' not in st.session_state:
    st.session_state.ml_engine = None

if 'data_quality_analysis' not in st.session_state:
    st.session_state.data_quality_analysis = None

# Step definitions
STEPS = [
    "🏠 Home",
    "📁 Data Upload", 
    "🔍 Data Validation",
    "🤖 Model Creation",
    "📊 Model Results",
    "📈 Results & Analytics"
]

def show_home_page():
    """Step 0: Home page"""
    st.markdown("""
        <div class="main-header">
            <h1>🌙 Lunarv</h1>
            <h2>Churn Prediction & Analytics</h2>
            <p>Transform your customer data into actionable insights to reduce churn and boost retention</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        ### 🚀 How Lunarv Helps Your Business
        
        **Lunarv** is an AI-powered churn prediction platform that analyzes your customer data to identify 
        customers at risk of leaving. Our advanced machine learning algorithms provide:
        
        ✅ **Accurate Predictions** - 95%+ accuracy using ensemble methods  
        ✅ **Actionable Insights** - Understand why customers churn  
        ✅ **Risk Segmentation** - Categorize customers by churn risk level  
        ✅ **Data-Driven Decisions** - Make informed retention strategies  
        
        ### 📊 What You'll Get
        
        1. **Customer Risk Assessment** - Identify high-risk customers
        2. **Churn Probability Scores** - Quantified risk for each customer  
        3. **Feature Importance Analysis** - Understand key churn drivers
        4. **Comprehensive Reports** - Ready-to-use business insights
        
        ### 🎯 Advanced ML Features
        
        - **Multiple Models**: XGBoost, LightGBM, CatBoost, Random Forest
        - **Smart Feature Selection**: Tree-based automated feature selection
        - **Class Imbalance Handling**: SMOTE, undersampling, class weights
        - **Target Encoding**: Advanced categorical encoding with smoothing
        - **Training Time Estimates**: Know how long training will take
        """)
        
        if st.button("🚀 Start Now", type="primary", use_container_width=True):
            st.session_state.current_step = 1
            st.rerun()

def show_data_upload():
    """Step 1: Data upload"""
    st.markdown(f"## {STEPS[1]}")
    
    uploaded_file = st.file_uploader(
        "Choose your customer dataset file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload CSV, Excel, or XLS files up to 200 MB"
    )
    
    if uploaded_file is not None:
        try:
            # Load data
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.session_state.uploaded_data = df
            
            st.markdown(f"""
                <div class="success-message">
                    ✅ <strong>File uploaded successfully!</strong> 
                    Your dataset contains {len(df):,} rows and {len(df.columns)} columns.
                </div>
            """, unsafe_allow_html=True)
            
            # Data preview
            st.subheader("📋 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Column mapping
            st.subheader("🔧 Column Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                customer_id_col = st.selectbox(
                    "Primary Key Column (Customer ID)",
                    options=[''] + list(df.columns),
                    help="Select the column that contains unique customer identifiers"
                )
            
            with col2:
                churn_col = st.selectbox(
                    "Churn Column (Target Variable)",
                    options=[''] + list(df.columns),
                    help="Select the column that indicates whether a customer churned or not"
                )
            
            # Save column mapping
            if customer_id_col and churn_col:
                st.session_state.column_mapping = {
                    'customer_id': customer_id_col,
                    'churn': churn_col
                }
                
                st.markdown("""
                    <div class="success-message">
                        ✅ <strong>Column configuration complete!</strong> 
                        Ready to proceed to data validation.
                    </div>
                """, unsafe_allow_html=True)
                
                # Navigation
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button("Next: Data Validation →", type="primary"):
                        st.session_state.current_step = 2
                        st.rerun()
            else:
                st.warning("⚠️ Please select both Customer ID and Churn columns to continue.")
                
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")

def show_data_validation():
    """Step 2: Data validation with advanced features"""
    st.markdown(f"## {STEPS[2]}")
    
    if st.session_state.uploaded_data is None:
        st.error("❌ No data uploaded. Please go back to Data Upload step.")
        return
    
    df = st.session_state.uploaded_data
    column_mapping = st.session_state.column_mapping
    
    if not column_mapping:
        st.error("❌ Column mapping not configured. Please go back to Data Upload step.")
        return
    
    customer_id_col = column_mapping['customer_id']
    churn_col = column_mapping['churn']
    
    st.markdown("""
        <div class="info-message">
            🔍 <strong>Data Validation:</strong> Let's examine your data quality and prepare it for modeling.
        </div>
    """, unsafe_allow_html=True)
    
    # Initialize ML engine if not already done
    if st.session_state.ml_engine is None:
        st.session_state.ml_engine = LunarvMLEngine()
    
    # Data quality analysis
    if st.session_state.data_quality_analysis is None:
        with st.spinner("Analyzing data quality..."):
            st.session_state.data_quality_analysis = st.session_state.ml_engine.analyze_data_quality(df, churn_col)
    
    data_analysis = st.session_state.data_quality_analysis
    
    # Data quality overview
    st.subheader("📊 Data Quality Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Rows", f"{data_analysis['total_rows']:,}")
    
    with col2:
        st.metric("Total Columns", data_analysis['total_columns'])
    
    with col3:
        missing_total = sum(data_analysis['missing_data'][col]['count'] for col in df.columns)
        st.metric("Total Missing Values", f"{missing_total:,}")
    
    # Missing values analysis
    st.subheader("🔍 Missing Values Analysis")
    
    missing_data = df.isnull().sum()
    missing_df = pd.DataFrame({
        'Column': missing_data.index,
        'Missing Count': missing_data.values,
        'Missing %': (missing_data.values / len(df) * 100).round(2)
    }).sort_values('Missing %', ascending=False)
    
    # Show only columns with missing values
    missing_df_filtered = missing_df[missing_df['Missing %'] > 0]
    
    if len(missing_df_filtered) > 0:
        st.dataframe(missing_df_filtered, use_container_width=True)
    else:
        st.success("✅ No missing values detected in your dataset!")
    
    # Churn distribution analysis
    st.subheader("🎯 Churn Distribution Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if churn_col in df.columns:
            churn_counts = df[churn_col].value_counts()
            
            # Create pie chart
            fig = px.pie(
                values=churn_counts.values,
                names=churn_counts.index,
                title="Churn vs Non-Churn Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if churn_col in df.columns:
            # Show class balance details
            st.markdown("**Class Balance Details**")
            
            for value, info in data_analysis['class_balance'].items():
                st.write(f"**{value}**: {info['count']:,} customers ({info['percentage']}%)")
            
            # Class imbalance recommendations
            if data_analysis['recommendations']:
                st.markdown("**📋 Recommendations:**")
                for rec in data_analysis['recommendations']:
                    st.write(f"• {rec}")
    
    # Interactive column exploration
    st.subheader("🔍 Interactive Column Explorer")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_col = st.selectbox(
            "Select Column to Explore",
            options=[col for col in df.columns if col not in [customer_id_col, churn_col]],
            help="Choose a column to see detailed statistics"
        )
    
    with col2:
        if selected_col:
            col_data = df[selected_col]
            col_type = str(col_data.dtype)
            unique_count = col_data.nunique()
            missing_count = col_data.isna().sum()
            
            st.markdown(f"**Column: {selected_col}**")
            st.write(f"**Data Type:** {col_type}")
            st.write(f"**Unique Values:** {unique_count:,}")
            st.write(f"**Missing Values:** {missing_count:,}")
            
            if col_type in ['object', 'string']:
                # Show top values for categorical columns
                top_values = col_data.value_counts().head(10)
                st.write("**Top 10 Values:**")
                for val, count in top_values.items():
                    st.write(f"• {val}: {count:,}")
            else:
                # Show statistics for numeric columns
                st.write("**Statistics:**")
                st.write(f"• Min: {col_data.min():.2f}")
                st.write(f"• Max: {col_data.max():.2f}")
                st.write(f"• Mean: {col_data.mean():.2f}")
                st.write(f"• Median: {col_data.median():.2f}")
    
    # Data preprocessing options
    st.subheader("⚙️ Data Preprocessing Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Class Imbalance Handling**")
        class_imbalance_method = st.selectbox(
            "Method",
            options=['auto', 'oversample', 'undersample', 'class_weights', 'none'],
            help="Choose how to handle class imbalance in your data"
        )
    
    with col2:
        st.markdown("**Feature Selection Method**")
        feature_selection_method = st.selectbox(
            "Method",
            options=['tree_based', 'correlation', 'mutual_info', 'recursive'],
            help="Choose how to select the most important features"
        )
    
    # Store preprocessing options
    st.session_state.preprocessing_options = {
        'class_imbalance_method': class_imbalance_method,
        'feature_selection_method': feature_selection_method
    }
    
    # Navigation
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("← Previous: Data Upload"):
            st.session_state.current_step = 1
            st.rerun()
    
    with col3:
        if st.button("Next: Model Creation →", type="primary"):
            st.session_state.current_step = 3
            st.rerun()

def show_model_creation():
    """Step 3: Model creation"""
    st.markdown(f"## {STEPS[3]}")
    
    if st.session_state.uploaded_data is None:
        st.error("❌ No data uploaded. Please go back to Data Upload step.")
        return
    
    df = st.session_state.uploaded_data
    column_mapping = st.session_state.column_mapping
    
    if not column_mapping:
        st.error("❌ Column mapping not configured. Please go back to Data Upload step.")
        return
    
    customer_id_col = column_mapping['customer_id']
    churn_col = column_mapping['churn']
    
    st.markdown("""
        <div class="info-message">
            🤖 <strong>Model Creation:</strong> We'll train multiple AI models to find the best one for your data.
        </div>
    """, unsafe_allow_html=True)
    
    # Get all features (excluding customer ID and churn)
    all_features = [col for col in df.columns if col not in [customer_id_col, churn_col]]
    
    st.write(f"**Available Features:** {len(all_features)} columns")
    st.write(f"**Selected Features:** {', '.join(all_features[:5])}{'...' if len(all_features) > 5 else ''}")
    
    # Show available models
    ml_engine = st.session_state.ml_engine
    available_models = list(ml_engine.models.keys())
    
    st.subheader("🤖 Available Models")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Core Models:**")
        for model in ['random_forest', 'gradient_boosting', 'logistic_regression']:
            if model in available_models:
                st.write(f"✅ {model.replace('_', ' ').title()}")
    
    with col2:
        st.markdown("**Advanced Models:**")
        for model in ['xgboost', 'lightgbm', 'catboost']:
            if model in available_models:
                st.write(f"✅ {model.title()}")
            else:
                st.write(f"⚠️ {model.title()} (not available)")
    
    with col3:
        st.markdown("**Feature Selection:**")
        preprocessing_options = st.session_state.get('preprocessing_options', {
            'feature_selection_method': 'tree_based',
            'class_imbalance_method': 'auto'
        })
        st.write(f"✅ {preprocessing_options['feature_selection_method'].replace('_', ' ').title()}")
        st.write(f"✅ Class Imbalance: {preprocessing_options['class_imbalance_method'].title()}")
    
    # Training time estimate
    estimated_time = ml_engine.estimate_training_time(len(df))
    st.info(f"⏱️ **Estimated Training Time:** {estimated_time}")
    
    # Start training
    if st.button("🚀 Start Model Training", type="primary", use_container_width=True):
        if not all_features:
            st.error("❌ No features available for training.")
            return
        
        # Initialize progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("🔄 Initializing models...")
            progress_bar.progress(10)
            
            # Get preprocessing options
            preprocessing_options = st.session_state.get('preprocessing_options', {
                'class_imbalance_method': 'auto',
                'feature_selection_method': 'tree_based'
            })
            
            # Use only selected features
            df_selected = df[[customer_id_col, churn_col] + all_features]
            
            # Train the models
            model_performance = ml_engine.train_models(
                df_selected, 
                churn_col, 
                test_size=0.2,
                feature_selection_method=preprocessing_options['feature_selection_method'],
                class_imbalance_method=preprocessing_options['class_imbalance_method']
            )
            
            progress_bar.progress(90)
            status_text.text("✅ Training completed!")
            progress_bar.progress(100)
            
            # Store results
            st.session_state.model_results = model_performance
            st.session_state.ml_engine = ml_engine
            
            # Show results summary
            st.success("🎉 Model training completed successfully!")
            
            # Display best model
            best_model_name = max(model_performance.keys(), 
                                 key=lambda x: model_performance[x]['f1_score'])
            best_metrics = model_performance[best_model_name]
            
            st.markdown(f"""
            **🏆 Best Model:** {best_model_name.replace('_', ' ').title()}
            
            **📊 Performance Metrics:**
            - **Accuracy:** {best_metrics['accuracy']:.3f}
            - **F1 Score:** {best_metrics['f1_score']:.3f}
            - **Precision:** {best_metrics['precision']:.3f}
            - **Recall:** {best_metrics['recall']:.3f}
            - **AUC:** {best_metrics['auc']:.3f}
            """)
            
            # Navigation
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("Next: Model Results →", type="primary"):
                    st.session_state.current_step = 4
                    st.rerun()
            
        except Exception as e:
            st.error(f"❌ Training failed: {str(e)}")
        
        finally:
            progress_bar.empty()
            status_text.empty()

def show_model_results():
    """Step 4: Model results"""
    st.markdown(f"## {STEPS[4]}")
    
    if st.session_state.model_results is None:
        st.error("❌ No model results available. Please go back to Model Creation step.")
        return
    
    model_performance = st.session_state.model_results
    
    st.markdown("""
        <div class="info-message">
            📊 <strong>Model Results:</strong> Compare the performance of all trained models.
        </div>
    """, unsafe_allow_html=True)
    
    # Model comparison
    st.subheader("🏆 Model Performance Comparison")
    
    # Create comparison chart
    models = list(model_performance.keys())
    metrics = ['accuracy', 'f1_score', 'precision', 'recall', 'auc']
    
    # Create subplot for metrics comparison
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Accuracy', 'F1 Score', 'Precision', 'Recall'],
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )
    
    # Add bars for each metric
    for i, metric in enumerate(metrics[:4]):
        values = [model_performance[model][metric] for model in models]
        row = (i // 2) + 1
        col = (i % 2) + 1
        
        fig.add_trace(
            go.Bar(x=models, y=values, name=metric.replace('_', ' ').title()),
            row=row, col=col
        )
    
    fig.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed model comparison table
    st.subheader("📋 Detailed Model Comparison")
    
    comparison_data = []
    for model_name, metrics in model_performance.items():
        comparison_data.append({
            'Model': model_name.replace('_', ' ').title(),
            'Accuracy': f"{metrics['accuracy']:.3f}",
            'F1 Score': f"{metrics['f1_score']:.3f}",
            'Precision': f"{metrics['precision']:.3f}",
            'Recall': f"{metrics['recall']:.3f}",
            'AUC': f"{metrics['auc']:.3f}",
            'Training Time': f"{metrics.get('training_time', 0):.2f}s"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)
    
    # Best model highlight
    best_model_name = max(model_performance.keys(), 
                         key=lambda x: model_performance[x]['f1_score'])
    best_metrics = model_performance[best_model_name]
    
    st.markdown(f"""
    <div class="success-message">
        🏆 <strong>Best Performing Model:</strong> {best_model_name.replace('_', ' ').title()}
        
        **Key Metrics:**
        - **F1 Score:** {best_metrics['f1_score']:.3f} (highest)
        - **Accuracy:** {best_metrics['accuracy']:.3f}
        - **Training Time:** {best_metrics.get('training_time', 0):.2f} seconds
    </div>
    """, unsafe_allow_html=True)
    
    # Feature importance (if available)
    ml_engine = st.session_state.ml_engine
    if ml_engine.feature_importance is not None:
        st.subheader("🎯 Feature Importance (Best Model)")
        
        # Show top 10 features
        top_features = ml_engine.feature_importance.head(10)
        
        fig = px.bar(
            top_features,
            x='importance',
            y='feature',
            orientation='h',
            title="Top 10 Most Important Features"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Navigation
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("← Previous: Model Creation"):
            st.session_state.current_step = 3
            st.rerun()
    
    with col3:
        if st.button("Next: Results & Analytics →", type="primary"):
            st.session_state.current_step = 5
            st.rerun()

def show_results_analytics():
    """Step 5: Final results"""
    st.markdown(f"## {STEPS[5]}")
    
    if st.session_state.model_results is None or 'ml_engine' not in st.session_state:
        st.error("❌ No trained model available. Please go back to Model Creation step.")
        return
    
    ml_engine = st.session_state.ml_engine
    df = st.session_state.uploaded_data
    column_mapping = st.session_state.column_mapping
    
    st.markdown("""
        <div class="info-message">
            📈 <strong>Results & Analytics:</strong> Explore your churn predictions and generate actionable insights.
        </div>
    """, unsafe_allow_html=True)
    
    # Generate predictions
    if st.button("🔮 Generate Churn Predictions", type="primary"):
        with st.spinner("Generating predictions..."):
            try:
                # Make predictions
                predictions = ml_engine.predict_churn(df, column_mapping['churn'])
                st.session_state.predictions = predictions
                st.success("✅ Predictions generated successfully!")
            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")
                return
    
    if st.session_state.predictions is not None:
        predictions = st.session_state.predictions
        
        # Analytics Dashboard
        st.subheader("📊 Analytics Dashboard")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Churn probability distribution
            fig = px.histogram(
                predictions, 
                x='churn_probability',
                nbins=20,
                title="Churn Probability Distribution"
            )
            fig.update_layout(xaxis_title="Churn Probability", yaxis_title="Number of Customers")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Risk level distribution
            risk_counts = predictions['risk_level'].value_counts()
            fig = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                title="Customer Risk Level Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Risk level summary
        st.subheader("🎯 Risk Level Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            low_risk = len(predictions[predictions['risk_level'] == 'Low Risk'])
            st.metric("Low Risk Customers", f"{low_risk:,}")
        
        with col2:
            medium_risk = len(predictions[predictions['risk_level'] == 'Medium Risk'])
            st.metric("Medium Risk Customers", f"{medium_risk:,}")
        
        with col3:
            high_risk = len(predictions[predictions['risk_level'] == 'High Risk'])
            st.metric("High Risk Customers", f"{high_risk:,}")
        
        # Segment-wise churn probability
        st.subheader("📊 Segment-wise Churn Probability")
        
        # Get categorical columns for segmentation
        categorical_cols = [col for col in df.columns if df[col].dtype == 'object' 
                           and col not in [column_mapping['customer_id'], column_mapping['churn']]]
        
        if categorical_cols:
            selected_segment_col = st.selectbox(
                "Select Column for Segmentation",
                options=categorical_cols,
                help="Choose a column to see churn probability by segment"
            )
            
            if selected_segment_col:
                segment_analysis = predictions.groupby(selected_segment_col)['churn_probability'].agg([
                    'mean', 'count'
                ]).round(3)
                
                segment_analysis.columns = ['Avg Churn Probability', 'Customer Count']
                segment_analysis = segment_analysis.sort_values('Avg Churn Probability', ascending=False)
                
                st.dataframe(segment_analysis, use_container_width=True)
                
                # Create bar chart for segment analysis
                fig = px.bar(
                    segment_analysis.reset_index(),
                    x=selected_segment_col,
                    y='Avg Churn Probability',
                    title=f"Churn Probability by {selected_segment_col}",
                    text='Avg Churn Probability'
                )
                fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
        
        # Detailed Results Table with Filter
        st.subheader("📋 Detailed Results Table")
        
        # Probability threshold filter
        threshold = st.slider(
            "Show customers with churn probability above:",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="Filter customers by churn probability threshold"
        )
        
        filtered_predictions = predictions[predictions['churn_probability'] >= threshold]
        
        st.write(f"**Showing {len(filtered_predictions)} customers with churn probability ≥ {threshold}**")
        
        # Display filtered results
        display_cols = [column_mapping['customer_id'], 'churn_probability', 'churn_prediction', 'risk_level']
        display_cols = [col for col in display_cols if col in filtered_predictions.columns]
        
        st.dataframe(
            filtered_predictions[display_cols].head(20),
            use_container_width=True
        )
        
        # Human-readable summary report
        st.subheader("📝 Summary Report")
        
        total_customers = len(predictions)
        high_risk_pct = (high_risk / total_customers) * 100
        avg_churn_prob = predictions['churn_probability'].mean()
        
        summary_report = f"""
        **📊 Executive Summary**
        
        Your dataset contains **{total_customers:,} customers** with an average churn probability of **{avg_churn_prob:.1%}**.
        
        **🎯 Risk Distribution:**
        - **High Risk Customers:** {high_risk:,} ({high_risk_pct:.1f}%) - These customers require immediate attention
        - **Medium Risk Customers:** {medium_risk:,} ({(medium_risk/total_customers)*100:.1f}%) - Monitor closely
        - **Low Risk Customers:** {low_risk:,} ({(low_risk/total_customers)*100:.1f}%) - Generally loyal
        
        **💡 Key Insights:**
        - {high_risk_pct:.1f}% of your customer base is at high risk of churning
        - Focus retention efforts on customers with churn probability above 70%
        - Consider proactive engagement strategies for medium-risk customers
        
        **🚀 Recommended Actions:**
        1. **Immediate:** Contact high-risk customers with personalized retention offers
        2. **Short-term:** Implement targeted campaigns for medium-risk customers
        3. **Long-term:** Analyze feature importance to understand churn drivers
        """
        
        st.markdown(summary_report)
        
        # Export options
        st.subheader("💾 Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Download CSV"):
                csv = predictions.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name="lunarv_churn_predictions.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("📊 Download Excel"):
                # Create Excel file with multiple sheets
                from io import BytesIO
                
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    predictions.to_excel(writer, sheet_name='All_Predictions', index=False)
                    
                    # Summary sheet
                    summary_data = {
                        'Metric': ['Total Customers', 'High Risk', 'Medium Risk', 'Low Risk', 'Avg Churn Probability'],
                        'Value': [total_customers, high_risk, medium_risk, low_risk, f"{avg_churn_prob:.3f}"]
                    }
                    summary_df = pd.DataFrame(summary_data)
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
                    
                    # Risk analysis sheet
                    risk_analysis = predictions.groupby('risk_level').agg({
                        'churn_probability': ['count', 'mean', 'min', 'max']
                    }).round(3)
                    risk_analysis.to_excel(writer, sheet_name='Risk_Analysis')
                
                output.seek(0)
                st.download_button(
                    label="📊 Download Excel",
                    data=output.getvalue(),
                    file_name="lunarv_churn_analysis.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
    
    # Navigation
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("← Previous: Model Results"):
            st.session_state.current_step = 4
            st.rerun()
    
    with col2:
        if st.button("🏠 Back to Home", type="primary"):
            st.session_state.current_step = 0
            st.rerun()

def main():
    """Main application function"""
    # Sidebar for step navigation
    with st.sidebar:
        st.markdown("## 🌙 Lunarv")
        st.markdown("### Navigation")
        
        for i, step in enumerate(STEPS):
            if i == st.session_state.current_step:
                st.markdown(f"**{step}** ✅")
            else:
                st.markdown(f"{step}")
        
        st.markdown("---")
        st.markdown("**Current Step:** " + str(st.session_state.current_step + 1) + "/" + str(len(STEPS)))
        
        # Quick navigation
        if st.button("🏠 Go to Home"):
            st.session_state.current_step = 0
            st.rerun()
    
    # Main content area
    if st.session_state.current_step == 0:
        show_home_page()
    elif st.session_state.current_step == 1:
        show_data_upload()
    elif st.session_state.current_step == 2:
        show_data_validation()
    elif st.session_state.current_step == 3:
        show_model_creation()
    elif st.session_state.current_step == 4:
        show_model_results()
    elif st.session_state.current_step == 5:
        show_results_analytics()

if __name__ == "__main__":
    main()
