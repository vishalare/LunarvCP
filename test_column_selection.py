#!/usr/bin/env python3
"""
Test script to verify column selection and saving works correctly
"""

import streamlit as st
import pandas as pd
import numpy as np

def test_column_selection():
    """Test column selection and saving functionality"""
    
    st.title("🧪 Column Selection Test")
    
    # Initialize session state
    if 'current_data' not in st.session_state:
        st.session_state.current_data = None
    if 'customer_id_col' not in st.session_state:
        st.session_state.customer_id_col = None
    if 'churn_col' not in st.session_state:
        st.session_state.churn_col = None
    
    # Create sample data if not exists
    if st.session_state.current_data is None:
        if st.button("📊 Create Sample Data"):
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
            st.session_state.current_data = df
            st.success("✅ Sample data created!")
            st.experimental_rerun()
    
    # Show current state
    if st.session_state.current_data is not None:
        df = st.session_state.current_data
        
        st.subheader("📊 Current Data")
        st.write(f"**Shape:** {df.shape}")
        st.write(f"**Columns:** {', '.join(df.columns.tolist())}")
        
        st.subheader("🔧 Current Column Configuration")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Customer ID Column:** {st.session_state.customer_id_col or 'Not set'}")
        with col2:
            st.write(f"**Churn Column:** {st.session_state.churn_col or 'Not set'}")
        
        # Test column selection
        if st.session_state.customer_id_col is None or st.session_state.churn_col is None:
            st.subheader("🚀 Column Selection Test")
            
            # Auto-detect
            if st.button("🔍 Auto-Detect Columns"):
                st.session_state.customer_id_col = 'customer_id'
                st.session_state.churn_col = 'churn'
                st.success("✅ Columns auto-detected!")
                st.experimental_rerun()
            
            # Manual selection
            st.subheader("📝 Manual Selection")
            col1, col2 = st.columns(2)
            
            with col1:
                new_customer_id_col = st.selectbox(
                    "Select Customer ID Column:",
                    options=df.columns.tolist(),
                    index=0,
                    key="test_customer_id"
                )
            
            with col2:
                new_churn_col = st.selectbox(
                    "Select Churn Column:",
                    options=df.columns.tolist(),
                    index=len(df.columns)-1,
                    key="test_churn_col"
                )
            
            if st.button("💾 Save Column Selection"):
                st.session_state.customer_id_col = new_customer_id_col
                st.session_state.churn_col = new_churn_col
                st.success("✅ Column selection saved!")
                st.experimental_rerun()
        
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
    test_column_selection()
