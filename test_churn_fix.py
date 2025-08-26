#!/usr/bin/env python3
"""
Test script to verify churn column handling fix
"""

import pandas as pd
import numpy as np

def test_churn_handling():
    """Test churn column handling with different data types"""
    
    print("🧪 Testing Churn Column Handling Fix")
    print("=" * 50)
    
    # Test 1: Binary churn (0/1)
    print("\n📊 Test 1: Binary Churn (0/1)")
    df1 = pd.DataFrame({
        'customer_id': ['C001', 'C002', 'C003', 'C004', 'C005'],
        'age': [25, 30, 35, 40, 45],
        'churn': [0, 1, 0, 1, 0]
    })
    
    try:
        if df1['churn'].dtype in ['int64', 'float64']:
            churn_count = df1['churn'].sum()
        else:
            churn_count = len(df1[df1['churn'].isin([1, '1', 'Yes', 'yes', 'YES', 'True', 'true', 'TRUE', 'Churned', 'churned'])])
        print(f"✅ Binary churn: {churn_count} churned customers")
    except Exception as e:
        print(f"❌ Binary churn failed: {e}")
    
    # Test 2: Text churn (Yes/No)
    print("\n📊 Test 2: Text Churn (Yes/No)")
    df2 = pd.DataFrame({
        'customer_id': ['C001', 'C002', 'C003', 'C004', 'C005'],
        'age': [25, 30, 35, 40, 45],
        'churn': ['No', 'Yes', 'No', 'Yes', 'No']
    })
    
    try:
        if df2['churn'].dtype in ['int64', 'float64']:
            churn_count = df2['churn'].sum()
        else:
            churn_count = len(df2[df2['churn'].isin([1, '1', 'Yes', 'yes', 'YES', 'True', 'true', 'TRUE', 'Churned', 'churned'])])
        print(f"✅ Text churn: {churn_count} churned customers")
    except Exception as e:
        print(f"❌ Text churn failed: {e}")
    
    # Test 3: Custom churn (Churned/Loyal)
    print("\n📊 Test 3: Custom Churn (Churned/Loyal)")
    df3 = pd.DataFrame({
        'customer_id': ['C001', 'C002', 'C003', 'C004', 'C005'],
        'age': [25, 30, 35, 40, 45],
        'churn': ['Loyal', 'Churned', 'Loyal', 'Churned', 'Loyal']
    })
    
    try:
        if df3['churn'].dtype in ['int64', 'float64']:
            churn_count = df3['churn'].sum()
        else:
            churn_count = len(df3[df3['churn'].isin([1, '1', 'Yes', 'yes', 'YES', 'True', 'true', 'TRUE', 'Churned', 'churned'])])
        print(f"✅ Custom churn: {churn_count} churned customers")
    except Exception as e:
        print(f"❌ Custom churn failed: {e}")
    
    # Test 4: Mixed churn values
    print("\n📊 Test 4: Mixed Churn Values")
    df4 = pd.DataFrame({
        'customer_id': ['C001', 'C002', 'C003', 'C004', 'C005'],
        'age': [25, 30, 35, 40, 45],
        'churn': [0, 'Yes', 1, 'No', 'Churned']
    })
    
    try:
        if df4['churn'].dtype in ['int64', 'float64']:
            churn_count = df4['churn'].sum()
        else:
            churn_count = len(df4[df4['churn'].isin([1, '1', 'Yes', 'yes', 'YES', 'True', 'true', 'TRUE', 'Churned', 'churned'])])
        print(f"✅ Mixed churn: {churn_count} churned customers")
    except Exception as e:
        print(f"❌ Mixed churn failed: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Test completed!")

if __name__ == "__main__":
    test_churn_handling()
