#!/usr/bin/env python3
"""
Debug script to help identify column configuration issues
"""

import pandas as pd
import numpy as np

def debug_column_issues():
    """Debug common column configuration issues"""
    
    print("🔍 Debugging Column Configuration Issues")
    print("=" * 50)
    
    # Create sample data similar to what might cause issues
    print("\n📊 Sample Data Structure:")
    sample_data = {
        'customer_id': ['C001', 'C002', 'C003', 'C004', 'C005'],
        'age': [25, 30, 35, 40, 45],
        'tenure': [12, 24, 36, 48, 60],
        'monthly_charges': [29.85, 56.95, 53.85, 42.30, 70.35],
        'churn_status': ['No', 'Yes', 'No', 'Yes', 'No']  # Note: different from 'churn'
    }
    
    df = pd.DataFrame(sample_data)
    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Data types: {df.dtypes.to_dict()}")
    
    print("\n🔍 Potential Issues Found:")
    
    # Check 1: Column name mismatches
    expected_customer_col = 'customer_id'
    expected_churn_col = 'churn'
    
    if expected_customer_col not in df.columns:
        print(f"❌ Expected customer column '{expected_customer_col}' not found!")
        print(f"   Available: {[col for col in df.columns if 'id' in col.lower() or 'customer' in col.lower()]}")
    
    if expected_churn_col not in df.columns:
        print(f"❌ Expected churn column '{expected_churn_col}' not found!")
        print(f"   Available: {[col for col in df.columns if 'churn' in col.lower() or 'status' in col.lower()]}")
    
    # Check 2: Data type issues
    print("\n📊 Data Type Analysis:")
    for col in df.columns:
        print(f"  {col}: {df[col].dtype} - Sample: {df[col].head(3).tolist()}")
    
    # Check 3: Missing values
    print("\n📊 Missing Values:")
    for col in df.columns:
        missing = df[col].isnull().sum()
        if missing > 0:
            print(f"  {col}: {missing} missing values")
    
    # Check 4: Churn column analysis
    print("\n🎯 Churn Column Analysis:")
    churn_candidates = [col for col in df.columns if 'churn' in col.lower() or 'status' in col.lower()]
    
    for col in churn_candidates:
        print(f"\n  Column: {col}")
        print(f"    Data type: {df[col].dtype}")
        print(f"    Unique values: {df[col].unique()}")
        print(f"    Value counts: {df[col].value_counts().to_dict()}")
        
        # Test the churn counting logic
        try:
            if df[col].dtype in ['int64', 'float64']:
                churn_count = df[col].sum()
                print(f"    Numeric churn count: {churn_count}")
            else:
                churn_count = len(df[df[col].isin([1, '1', 'Yes', 'yes', 'YES', 'True', 'true', 'TRUE', 'Churned', 'churned'])])
                print(f"    Categorical churn count: {churn_count}")
        except Exception as e:
            print(f"    ❌ Error counting churn: {e}")
    
    print("\n💡 Recommendations:")
    print("1. Ensure your data has columns named 'customer_id' and 'churn'")
    print("2. Or manually map your existing columns in the Data Upload section")
    print("3. Check that churn column contains binary values (0/1) or text (Yes/No)")
    print("4. Use the debug checkbox in the Model Training page to see detailed info")

if __name__ == "__main__":
    debug_column_issues()
