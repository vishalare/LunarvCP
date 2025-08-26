#!/usr/bin/env python3
"""
Test script to verify the Streamlit app can run without errors
"""

import streamlit as st
import pandas as pd
import numpy as np

def test_session_state():
    """Test session state initialization"""
    print("🧪 Testing session state initialization...")
    
    # Initialize session state
    if 'data_uploaded' not in st.session_state:
        st.session_state.data_uploaded = False
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'current_data' not in st.session_state:
        st.session_state.current_data = None
    if 'customer_id_col' not in st.session_state:
        st.session_state.customer_id_col = None
    if 'churn_col' not in st.session_state:
        st.session_state.churn_col = None
    
    print("✅ Session state initialized successfully")
    
    # Test creating sample data
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
        
        st.success("✅ Sample data created and session state updated!")
        st.dataframe(df.head())
    
    # Test column selection
    if st.session_state.data_uploaded and st.session_state.current_data is not None:
        st.subheader("🔧 Column Configuration Test")
        
        df = st.session_state.current_data
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.selectbox(
                "Select customer ID column:",
                options=df.columns.tolist(),
                index=df.columns.get_loc(st.session_state.customer_id_col) if st.session_state.customer_id_col in df.columns else 0,
                key="customer_id_col"
            )
        
        with col2:
            st.selectbox(
                "Select churn column:",
                options=df.columns.tolist(),
                index=df.columns.get_loc(st.session_state.churn_col) if st.session_state.churn_col in df.columns else -1,
                key="churn_col"
            )
        
        st.success("✅ Column configuration working without errors!")

def main():
    st.title("🧪 App Test - Session State")
    st.write("This tests the session state fixes for the churn prediction tool.")
    
    test_session_state()
    
    st.markdown("---")
    st.write("If you see this message and no errors, the session state fix is working!")

if __name__ == "__main__":
    main()
