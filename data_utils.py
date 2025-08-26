import pandas as pd
import numpy as np
import streamlit as st
from io import StringIO, BytesIO
import base64
import json
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class DataHandler:
    """
    Handles data upload, validation, and processing for the churn prediction tool
    """
    
    def __init__(self):
        self.supported_formats = ['.csv', '.xlsx', '.xls']
        self.max_file_size = 100 * 1024 * 1024  # 100MB
        
    def validate_file(self, uploaded_file) -> Tuple[bool, str]:
        """
        Validate uploaded file format and size
        """
        if uploaded_file is None:
            return False, "No file uploaded"
        
        # Check file size
        if uploaded_file.size > self.max_file_size:
            return False, f"File size ({uploaded_file.size / 1024 / 1024:.1f}MB) exceeds limit (100MB)"
        
        # Check file format
        file_extension = uploaded_file.name.lower().split('.')[-1]
        if file_extension not in ['csv', 'xlsx', 'xls']:
            return False, f"Unsupported file format: {file_extension}. Supported: CSV, Excel"
        
        return True, "File is valid"
    
    def load_data(self, uploaded_file) -> Optional[pd.DataFrame]:
        """
        Load data from uploaded file
        """
        try:
            if uploaded_file.name.lower().endswith('.csv'):
                # Try different encodings
                encodings = ['utf-8', 'latin-1', 'cp1252']
                for encoding in encodings:
                    try:
                        df = pd.read_csv(uploaded_file, encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    st.error("Could not read CSV file with any encoding")
                    return None
            else:
                # Excel file
                df = pd.read_excel(uploaded_file, engine='openpyxl')
            
            return df
            
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return None
    
    def validate_data_structure(self, df: pd.DataFrame) -> Tuple[bool, List[str], List[str]]:
        """
        Validate data structure for churn prediction
        """
        errors = []
        warnings = []
        
        # Check if dataframe is empty
        if df.empty:
            errors.append("Dataset is empty")
            return False, errors, warnings
        
        # Check minimum required rows
        if len(df) < 50:
            warnings.append("Dataset has fewer than 50 rows, which may affect model performance")
        
        # Check for required columns
        required_columns = []
        if 'customer_id' in df.columns or 'customerid' in df.columns or 'id' in df.columns:
            required_columns.append('customer_id')
        else:
            errors.append("No customer ID column found. Expected: customer_id, customerid, or id")
        
        # Check for churn column
        churn_columns = [col for col in df.columns if 'churn' in col.lower()]
        if churn_columns:
            required_columns.append('churn')
        else:
            errors.append("No churn column found. Expected column name containing 'churn'")
        
        # Check data types
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) < 3:
            warnings.append("Dataset has fewer than 3 numeric columns, which may limit model performance")
        
        # Check for missing values
        missing_percentages = df.isnull().sum() / len(df) * 100
        high_missing_cols = missing_percentages[missing_percentages > 50].index.tolist()
        if high_missing_cols:
            warnings.append(f"Columns with >50% missing values: {', '.join(high_missing_cols)}")
        
        # Check churn distribution if churn column exists
        if churn_columns:
            churn_col = churn_columns[0]
            churn_counts = df[churn_col].value_counts()
            if len(churn_counts) < 2:
                errors.append("Churn column must have at least 2 unique values (0 and 1)")
            elif len(churn_counts) == 2:
                # Check for balanced dataset
                min_count = churn_counts.min()
                max_count = churn_counts.max()
                if min_count / max_count < 0.1:
                    warnings.append("Dataset is highly imbalanced, which may affect model performance")
        
        return len(errors) == 0, errors, warnings
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict:
        """
        Generate comprehensive data summary
        """
        summary = {
            'shape': df.shape,
            'columns': list(df.columns),
            'data_types': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'numeric_summary': {},
            'categorical_summary': {}
        }
        
        # Numeric columns summary
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            summary['numeric_summary'][col] = {
                'mean': df[col].mean(),
                'median': df[col].median(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'unique_values': df[col].nunique()
            }
        
        # Categorical columns summary
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            summary['categorical_summary'][col] = {
                'unique_values': df[col].nunique(),
                'top_values': df[col].value_counts().head(5).to_dict(),
                'missing_count': df[col].isnull().sum()
            }
        
        return summary
    
    def suggest_column_mapping(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Suggest column mapping for customer ID and churn
        """
        suggestions = {}
        
        # Suggest customer ID column
        customer_id_candidates = ['customer_id', 'customerid', 'id', 'user_id', 'userid', 'client_id', 'clientid']
        for candidate in customer_id_candidates:
            if candidate in df.columns:
                suggestions['customer_id'] = candidate
                break
        
        # Suggest churn column
        churn_candidates = [col for col in df.columns if 'churn' in col.lower()]
        if churn_candidates:
            suggestions['churn'] = churn_candidates[0]
        
        return suggestions
    
    def clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean column names for consistency
        """
        df_clean = df.copy()
        
        # Convert to lowercase and replace spaces with underscores
        df_clean.columns = df_clean.columns.str.lower().str.replace(' ', '_')
        
        # Remove special characters
        df_clean.columns = df_clean.columns.str.replace('[^a-zA-Z0-9_]', '', regex=True)
        
        return df_clean
    
    def create_sample_data(self) -> pd.DataFrame:
        """
        Create sample data for demonstration
        """
        np.random.seed(42)
        n_customers = 1000
        
        # Generate sample customer data
        data = {
            'customer_id': range(1, n_customers + 1),
            'age': np.random.randint(18, 80, n_customers),
            'tenure_months': np.random.randint(1, 60, n_customers),
            'monthly_charges': np.random.uniform(20, 150, n_customers),
            'total_charges': np.random.uniform(100, 5000, n_customers),
            'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_customers),
            'payment_method': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_customers),
            'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], n_customers),
            'online_security': np.random.choice(['Yes', 'No', 'No internet service'], n_customers),
            'online_backup': np.random.choice(['Yes', 'No', 'No internet service'], n_customers),
            'device_protection': np.random.choice(['Yes', 'No', 'No internet service'], n_customers),
            'tech_support': np.random.choice(['Yes', 'No', 'No internet service'], n_customers),
            'streaming_tv': np.random.choice(['Yes', 'No', 'No internet service'], n_customers),
            'streaming_movies': np.random.choice(['Yes', 'No', 'No internet service'], n_customers),
            'paperless_billing': np.random.choice(['Yes', 'No'], n_customers),
            'gender': np.random.choice(['Male', 'Female'], n_customers),
            'partner': np.random.choice(['Yes', 'No'], n_customers),
            'dependents': np.random.choice(['Yes', 'No'], n_customers),
            'phone_service': np.random.choice(['Yes', 'No'], n_customers),
            'multiple_lines': np.random.choice(['Yes', 'No', 'No phone service'], n_customers)
        }
        
        df = pd.DataFrame(data)
        
        # Create churn based on some business logic
        churn_prob = (
            (df['tenure_months'] < 12) * 0.3 +
            (df['monthly_charges'] > 100) * 0.2 +
            (df['contract_type'] == 'Month-to-month') * 0.3 +
            (df['payment_method'] == 'Electronic check') * 0.1 +
            (df['online_security'] == 'No') * 0.1
        )
        
        df['churn'] = (np.random.random(n_customers) < churn_prob).astype(int)
        
        return df
    
    def export_results(self, df: pd.DataFrame, format: str = 'csv') -> str:
        """
        Export results to different formats
        """
        if format == 'csv':
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="churn_predictions.csv">Download CSV</a>'
            return href
        elif format == 'excel':
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Churn_Predictions')
            excel_data = output.getvalue()
            b64 = base64.b64encode(excel_data).decode()
            href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="churn_predictions.xlsx">Download Excel</a>'
            return href
        else:
            return "Unsupported format"
