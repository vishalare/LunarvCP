#!/usr/bin/env python3
"""
Test script to verify column initialization works correctly
"""

import streamlit as st
import pandas as pd
import numpy as np

def test_column_initialization():
    """Test column initialization and validation"""
    
    st.title("🧪 Column Initialization Test")
    
    # Initialize session state
    if 'data_uploaded' not in st.session_state:
        st.session_state.data_uploaded = False
    if 'current_data' not in st.session_state:
        st.session_state.current_data = None
    if 'customer_id_col' not in st.session_state:
        st.session_state.customer_id_col = None
    if 'churn_col' not in st.session_state:
        st.session_state.churn_col = None
    
    st.write("**Current Session State:**")
    st.write(f"- Data uploaded: {st.session_state.data_uploaded}")
    st.write(f"- Customer ID column: {st.session_state.customer_id_col}")
    st.write(f"- Churn column: {st.session_state.churn_col}")
    
    # Test 1: Create sample data
    if st.button("📊 Create Sample Data"):
        # Create sample data
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
        st.session_state.data_uploaded = True
        st.session_state.customer_id_col = 'customer_id'
        st.session_state.churn_col = 'churn'
        
        st.success("✅ Sample data created and columns initialized!")
        st.dataframe(df.head())
    
    # Test 2: Validate column access
    if st.session_state.data_uploaded and st.session_state.current_data is not None:
        st.markdown("---")
        st.subheader("🔍 Column Validation Test")
        
        df = st.session_state.current_data
        customer_id_col = st.session_state.customer_id_col
        churn_col = st.session_state.churn_col
        
        # Test column access
        try:
            if customer_id_col is None or churn_col is None:
                st.error("❌ Columns not initialized!")
                return
            
            if customer_id_col not in df.columns:
                st.error(f"❌ Customer ID column '{customer_id_col}' not found!")
                return
            
            if churn_col not in df.columns:
                st.error(f"❌ Churn column '{churn_col}' not found!")
                return
            
            # Test calculations
            total_customers = len(df)
            churn_count = df[churn_col].sum()
            churn_rate = (churn_count / total_customers) * 100
            loyal_count = total_customers - churn_count
            
            st.success("✅ Column access working correctly!")
            
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
                
        except Exception as e:
            st.error(f"❌ Error accessing columns: {e}")
            st.info("💡 This is the error we're trying to fix!")
    
    # Test 3: Reset session
    if st.button("🔄 Reset Session"):
        st.session_state.clear()
        st.success("🔄 Session reset!")
        st.experimental_rerun()

if __name__ == "__main__":
    test_column_initialization()
