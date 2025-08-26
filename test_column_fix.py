#!/usr/bin/env python3
"""
Test script to verify column selection fix
"""

import streamlit as st
import pandas as pd
import numpy as np

def test_column_fix():
    """Test the fixed column selection logic"""
    
    st.title("🧪 Column Selection Fix Test")
    
    # Initialize session state
    if 'test_data' not in st.session_state:
        st.session_state.test_data = None
    if 'customer_id_col' not in st.session_state:
        st.session_state.customer_id_col = None
    if 'churn_col' not in st.session_state:
        st.session_state.churn_col = None
    
    # Create test data similar to user's data
    if st.session_state.test_data is None:
        if st.button("📊 Create Test Data"):
            np.random.seed(42)
            n_samples = 100
            
            data = {
                'customer_id': [f'CUST_{i:03d}' for i in range(1, n_samples + 1)],
                'age': np.random.randint(18, 80, n_samples),
                'tenure_months': np.random.randint(1, 60, n_samples),
                'monthly_charges': np.random.uniform(20, 150, n_samples),
                'total_charges': np.random.uniform(100, 5000, n_samples),
                'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
                'payment_method': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples),
                'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
                'online_security': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
                'online_backup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
                'device_protection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
                'tech_support': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
                'streaming_tv': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
                'streaming_movies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
                'paperless_billing': np.random.choice(['Yes', 'No'], n_samples),
                'gender': np.random.choice(['Male', 'Female'], n_samples),
                'partner': np.random.choice(['Yes', 'No'], n_samples),
                'dependents': np.random.choice(['Yes', 'No'], n_samples),
                'phone_service': np.random.choice(['Yes', 'No'], n_samples),
                'multiple_lines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples),
                'churn': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
            }
            
            df = pd.DataFrame(data)
            st.session_state.test_data = df
            st.success("✅ Test data created!")
            st.experimental_rerun()
    
    # Test column selection
    if st.session_state.test_data is not None:
        df = st.session_state.test_data
        
        st.subheader("📊 Test Data")
        st.write(f"**Shape:** {df.shape}")
        st.write(f"**Columns:** {', '.join(df.columns.tolist())}")
        
        st.subheader("🔧 Column Selection Test")
        
        # Test 1: Auto-detect columns
        if st.button("🔍 Auto-Detect Columns"):
            # Smart column detection (same logic as fixed app.py)
            customer_id_col = 'customer_id' if 'customer_id' in df.columns else df.columns[0]
            churn_col = 'churn' if 'churn' in df.columns else df.columns[-1]
            
            st.session_state.customer_id_col = customer_id_col
            st.session_state.churn_col = churn_col
            
            st.success(f"✅ Columns auto-detected! Customer ID: {customer_id_col}, Churn: {churn_col}")
            st.experimental_rerun()
        
        # Test 2: Manual column selection
        st.subheader("📝 Manual Column Selection")
        col1, col2 = st.columns(2)
        
        with col1:
            # Find the index of 'customer_id' column, default to first column if not found
            customer_id_index = 0
            if 'customer_id' in df.columns:
                customer_id_index = df.columns.get_loc('customer_id')
            
            new_customer_id_col = st.selectbox(
                "Select Customer ID Column:",
                options=df.columns.tolist(),
                index=customer_id_index,
                key="test_customer_id"
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
                key="test_churn_col"
            )
        
        if st.button("💾 Save Column Selection"):
            st.session_state.customer_id_col = new_customer_id_col
            st.session_state.churn_col = new_churn_col
            st.success("✅ Column selection saved!")
            st.experimental_rerun()
        
        # Show current state
        st.subheader("📋 Current Column Configuration")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Customer ID Column:** {st.session_state.customer_id_col or 'Not set'}")
        with col2:
            st.write(f"**Churn Column:** {st.session_state.churn_col or 'Not set'}")
        
        # Test column access
        if st.session_state.customer_id_col and st.session_state.churn_col:
            st.subheader("✅ Column Access Test")
            try:
                customer_id_col = st.session_state.customer_id_col
                churn_col = st.session_state.churn_col
                
                # Test calculations
                total_customers = len(df)
                churn_count = df[churn_col].sum()
                churn_rate = (churn_count / total_customers) * 100
                loyal_count = total_customers - churn_count
                
                st.success("🎉 Column access working perfectly!")
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Customers", f"{total_customers:,}")
                with col2:
                    st.metric("Churned Customers", f"{churn_count:,}")
                with col3:
                    st.metric("Churn Rate", f"{churn_rate:.1f}%")
                with col4:
                    st.metric("Loyal Customers", f"{loyal_count:,}")
                
                # Show sample data
                st.subheader("📋 Sample Data Preview")
                st.dataframe(df[[customer_id_col, churn_col]].head(10))
                
            except Exception as e:
                st.error(f"❌ Error accessing columns: {e}")
        
        # Reset button
        if st.button("🔄 Reset Test"):
            st.session_state.clear()
            st.success("🔄 Test reset!")
            st.experimental_rerun()

if __name__ == "__main__":
    test_column_fix()
