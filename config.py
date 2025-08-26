# Configuration file for Churn Prediction Tool

# Model Configuration
MODEL_CONFIG = {
    'test_size': 0.2,  # Test set size for model evaluation
    'random_state': 42,  # Random seed for reproducibility
    'min_features': 5,  # Minimum number of features to select
    'max_features': 50,  # Maximum number of features to select
}

# Random Forest Configuration
RANDOM_FOREST_CONFIG = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'class_weight': 'balanced',
    'random_state': 42
}

# Gradient Boosting Configuration
GRADIENT_BOOSTING_CONFIG = {
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 6,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'random_state': 42
}

# Logistic Regression Configuration
LOGISTIC_REGRESSION_CONFIG = {
    'max_iter': 1000,
    'class_weight': 'balanced',
    'random_state': 42,
    'solver': 'liblinear'
}

# Data Processing Configuration
DATA_CONFIG = {
    'max_file_size_mb': 100,  # Maximum file size in MB
    'supported_formats': ['.csv', '.xlsx', '.xls'],
    'encoding_options': ['utf-8', 'latin-1', 'cp1252'],
    'min_rows': 50,  # Minimum number of rows required
    'max_missing_percentage': 50,  # Maximum percentage of missing values allowed
}

# Visualization Configuration
VISUALIZATION_CONFIG = {
    'color_palette': {
        'primary': '#1f77b4',
        'secondary': '#ff7f0e',
        'success': '#2ca02c',
        'danger': '#d62728',
        'warning': '#ff7f0e',
        'info': '#17a2b8',
        'light': '#f8f9fa',
        'dark': '#343a40'
    },
    'chart_height': 500,
    'gauge_height': 600,
    'font_size': 14
}

# Risk Assessment Configuration
RISK_CONFIG = {
    'low_risk_threshold': 0.3,    # Below this = Low Risk
    'high_risk_threshold': 0.7,   # Above this = High Risk
    'risk_levels': ['Low Risk', 'Medium Risk', 'High Risk']
}

# Export Configuration
EXPORT_CONFIG = {
    'csv_encoding': 'utf-8',
    'excel_sheet_name': 'Churn_Predictions',
    'include_timestamp': True,
    'timestamp_format': '%Y%m%d_%H%M%S'
}

# Performance Thresholds
PERFORMANCE_THRESHOLDS = {
    'excellent_accuracy': 0.95,   # 95%+
    'good_accuracy': 0.85,        # 85-94%
    'acceptable_accuracy': 0.75,   # 75-84%
    'poor_accuracy': 0.75,        # Below 75%
    
    'excellent_f1': 0.90,         # 90%+
    'good_f1': 0.80,              # 80-89%
    'acceptable_f1': 0.70,        # 70-79%
    'poor_f1': 0.70,              # Below 70%
}

# Feature Selection Configuration
FEATURE_SELECTION_CONFIG = {
    'correlation_threshold': 0.95,  # Remove highly correlated features
    'variance_threshold': 0.01,     # Remove low variance features
    'mutual_info_threshold': 0.01,  # Minimum mutual information score
}

# Model Persistence Configuration
MODEL_PERSISTENCE_CONFIG = {
    'model_save_path': 'models/',
    'model_filename_prefix': 'churn_model_',
    'auto_save': True,
    'save_format': 'joblib'
}

# Logging Configuration
LOGGING_CONFIG = {
    'log_level': 'INFO',
    'log_file': 'churn_prediction.log',
    'log_format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
}

# Sample Data Configuration
SAMPLE_DATA_CONFIG = {
    'n_customers': 1000,
    'features': [
        'age', 'tenure_months', 'monthly_charges', 'total_charges',
        'contract_type', 'payment_method', 'internet_service',
        'online_security', 'online_backup', 'device_protection',
        'tech_support', 'streaming_tv', 'streaming_movies',
        'paperless_billing', 'gender', 'partner', 'dependents',
        'phone_service', 'multiple_lines'
    ],
    'churn_probability_factors': {
        'tenure_months': 0.3,      # Weight for tenure < 12 months
        'monthly_charges': 0.2,    # Weight for charges > $100
        'contract_type': 0.3,      # Weight for month-to-month
        'payment_method': 0.1,     # Weight for electronic check
        'online_security': 0.1     # Weight for no security
    }
}
