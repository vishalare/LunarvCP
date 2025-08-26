#!/usr/bin/env python3
"""
Test script to verify column selections are properly saved to session state
"""

import streamlit as st
import pandas as pd
import numpy as np

def test_column_save():
    """Test that column selections are properly saved to session state"""
    
    st.title("🧪 Column Save Test")
    
    # Initialize session state
    if 'test_data' not in st.session_state:
        st.session_state.test_data = None
    if 'customer_id_col' not in st.session_state:
        st.session_state.customer_id_col = None
    if 'churn_col' not in st.session_state:
        st.session_state.churn_col = None
    
    # Create test data
    if st.session_state.test_data is None:
        if st.button("📊 Create Test Data"):
            np.random.seed(42)
            n_samples = 50
            
            data = {
                'customer_id': [f'CUST_{i:03d}' for i in range(1, n_samples + 1)],
                'age': np.random.randint(18, 80, n_samples),
                'tenure_months': np.random.randint(1, 60, n_samples),
                'monthly_charges': np.random.uniform(20, 150, n_samples),
                'churn': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
            }
            
            df = pd.DataFrame(data)
            st.session_state.test_data = df
            st.success("✅ Test data created!")
            st.experimental_rerun()
    
    # Test column selection and saving
    if st.session_state.test_data is not None:
        df = st.session_state.test_data
        
        st.subheader("📊 Test Data")
        st.write(f"**Shape:** {df.shape}")
        st.write(f"**Columns:** {', '.join(df.columns.tolist())}")
        
        st.subheader("🔧 Column Selection Test")
        
        # Column selection with automatic saving
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Customer ID Column:**")
            selected_customer_id = st.selectbox(
                "Select customer ID column:",
                options=df.columns.tolist(),
                index=0,
                key="test_customer_id_selector"
            )
            
            # Update session state when selection changes
            if selected_customer_id != st.session_state.customer_id_col:
                st.session_state.customer_id_col = selected_customer_id
        
        with col2:
            st.markdown("**Churn Column:**")
            selected_churn = st.selectbox(
                "Select churn column:",
                options=df.columns.tolist(),
                index=len(df.columns)-1,
                key="test_churn_selector"
            )
            
            # Update session state when selection changes
            if selected_churn != st.session_state.churn_col:
                st.session_state.churn_col = selected_churn
        
        # Show current session state
        st.subheader("📋 Session State Status")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Customer ID:** {st.session_state.customer_id_col or 'Not set'}")
        with col2:
            st.write(f"**Churn Column:** {st.session_state.churn_col or 'Not set'}")
        
        # Manual save button as backup
        if st.button("💾 Manual Save", type="primary"):
            if selected_customer_id and selected_churn:
                st.session_state.customer_id_col = selected_customer_id
                st.session_state.churn_col = selected_churn
                st.success("✅ Column selections manually saved!")
                st.experimental_rerun()
            else:
                st.error("❌ Please select both columns before saving.")
        
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
        
        # Debug info
        st.subheader("🐛 Debug Information")
        st.write("**Session State:**")
        st.json({
            'customer_id_col': st.session_state.customer_id_col,
            'churn_col': st.session_state.churn_col,
            'selected_customer_id': selected_customer_id,
            'selected_churn': selected_churn
        })
        
        # Reset button
        if st.button("🔄 Reset Test"):
            st.session_state.clear()
            st.success("🔄 Test reset!")
            st.experimental_rerun()

if __name__ == "__main__":
    test_column_save()
