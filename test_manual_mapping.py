#!/usr/bin/env python3
"""
Test script to demonstrate manual column mapping and churn conversion
"""

import streamlit as st
import pandas as pd
import numpy as np

def test_manual_mapping():
    """Test manual column mapping and churn conversion features"""
    
    st.title("🧪 Manual Column Mapping Test")
    
    # Initialize session state
    if 'test_data' not in st.session_state:
        st.session_state.test_data = None
    if 'customer_id_col' not in st.session_state:
        st.session_state.customer_id_col = None
    if 'churn_col' not in st.session_state:
        st.session_state.churn_col = None
    
    # Create test data with different churn formats
    if st.session_state.test_data is None:
        st.subheader("📊 Create Test Data")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Binary Churn (0/1)"):
                create_binary_churn_data()
        
        with col2:
            if st.button("📊 Text Churn (Yes/No)"):
                create_text_churn_data()
        
        with col3:
            if st.button("📊 Custom Churn (Churned/Loyal)"):
                create_custom_churn_data()
    
    # Test manual mapping
    if st.session_state.test_data is not None:
        df = st.session_state.test_data
        
        st.subheader("📊 Test Data")
        st.write(f"**Shape:** {df.shape}")
        st.write(f"**Columns:** {', '.join(df.columns.tolist())}")
        
        # Show data preview
        st.subheader("📋 Data Preview")
        st.dataframe(df.head(10))
        
        # Manual column mapping section
        st.subheader("🔧 Manual Column Mapping")
        st.info("💡 **Manually map your columns below:**")
        
        # Column selection
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Primary Key Column (Customer ID):**")
            customer_id_col = st.selectbox(
                "Select the column that contains unique customer identifiers:",
                options=df.columns.tolist(),
                index=0,
                key="test_customer_id_col"
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
                key="test_churn_col"
            )
            
            # Show churn column info and conversion options
            if churn_col:
                st.write(f"**Current values:** {df[churn_col].value_counts().head(5).to_dict()}")
                st.write(f"**Data type:** {df[churn_col].dtype}")
                st.write(f"**Unique values:** {df[churn_col].nunique()}")
                
                # Check if churn column needs conversion
                if df[churn_col].dtype == 'object' or df[churn_col].nunique() > 10:
                    st.warning("⚠️ **Categorical churn column detected!**")
                    
                    # Conversion options
                    conversion_method = st.selectbox(
                        "How to convert to binary (0/1):",
                        ["Auto-detect (Yes/No → 1/0)", "Custom mapping", "Keep as-is"],
                        key="test_churn_conversion_method"
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
                                key="test_churn_values"
                            )
                        
                        with col_b:
                            st.write("**Loyal values (0):**")
                            loyal_values = st.multiselect(
                                "Select values that mean 'loyal':",
                                options=unique_values,
                                key="test_loyal_values"
                            )
        
        # Manual validation and proceed button
        if st.button("✅ Test Manual Mapping", type="primary"):
            if customer_id_col and churn_col:
                # Store column selections
                st.session_state.customer_id_col = customer_id_col
                st.session_state.churn_col = churn_col
                
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
                
                st.success("🎉 Manual mapping successful!")
                
            else:
                st.error("❌ Please select both Customer ID and Churn columns.")
        
        # Show current session state
        st.subheader("📋 Current Session State")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Customer ID Column:** {st.session_state.customer_id_col or 'Not set'}")
        with col2:
            st.write(f"**Churn Column:** {st.session_state.churn_col or 'Not set'}")
        
        # Reset button
        if st.button("🔄 Reset Test"):
            st.session_state.clear()
            st.success("🔄 Test reset!")
            st.experimental_rerun()

def create_binary_churn_data():
    """Create test data with binary churn (0/1)"""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'customer_id': [f'CUST_{i:03d}' for i in range(1, n_samples + 1)],
        'age': np.random.randint(18, 80, n_samples),
        'tenure': np.random.randint(1, 60, n_samples),
        'monthly_charges': np.random.uniform(20, 150, n_samples),
        'churn': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    }
    
    df = pd.DataFrame(data)
    st.session_state.test_data = df
    st.success("✅ Binary churn data created!")
    st.experimental_rerun()

def create_text_churn_data():
    """Create test data with text churn (Yes/No)"""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'customer_id': [f'CUST_{i:03d}' for i in range(1, n_samples + 1)],
        'age': np.random.randint(18, 80, n_samples),
        'tenure': np.random.randint(1, 60, n_samples),
        'monthly_charges': np.random.uniform(20, 150, n_samples),
        'churn': np.random.choice(['No', 'Yes'], n_samples, p=[0.8, 0.2])
    }
    
    df = pd.DataFrame(data)
    st.session_state.test_data = df
    st.success("✅ Text churn data created!")
    st.experimental_rerun()

def create_custom_churn_data():
    """Create test data with custom churn values"""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'customer_id': [f'CUST_{i:03d}' for i in range(1, n_samples + 1)],
        'age': np.random.randint(18, 80, n_samples),
        'tenure': np.random.randint(1, 60, n_samples),
        'monthly_charges': np.random.uniform(20, 150, n_samples),
        'churn': np.random.choice(['Loyal', 'Churned'], n_samples, p=[0.8, 0.2])
    }
    
    df = pd.DataFrame(data)
    st.session_state.test_data = df
    st.success("✅ Custom churn data created!")
    st.experimental_rerun()

if __name__ == "__main__":
    test_manual_mapping()
