#!/usr/bin/env python3
"""
Test script to verify safe selectbox usage without index errors
"""

import streamlit as st
import pandas as pd
import numpy as np

def test_safe_selectbox():
    """Test safe selectbox usage with proper index bounds"""
    
    st.title("🧪 Safe Selectbox Test")
    
    # Initialize session state
    if 'test_data' not in st.session_state:
        st.session_state.test_data = None
    if 'selected_col1' not in st.session_state:
        st.session_state.selected_col1 = None
    if 'selected_col2' not in st.session_state:
        st.session_state.selected_col2 = None
    
    # Create test data
    if st.session_state.test_data is None:
        if st.button("📊 Create Test Data"):
            # Create sample data with different column counts
            np.random.seed(42)
            n_samples = 50
            
            data = {
                'id': [f'ID_{i:03d}' for i in range(1, n_samples + 1)],
                'value': np.random.randint(1, 100, n_samples),
                'category': np.random.choice(['A', 'B', 'C'], n_samples),
                'status': np.random.choice([0, 1], n_samples)
            }
            
            df = pd.DataFrame(data)
            st.session_state.test_data = df
            st.success("✅ Test data created!")
            st.experimental_rerun()
    
    # Test safe selectbox usage
    if st.session_state.test_data is not None:
        df = st.session_state.test_data
        
        st.subheader("📊 Test Data")
        st.write(f"**Shape:** {df.shape}")
        st.write(f"**Columns:** {', '.join(df.columns.tolist())}")
        
        st.subheader("🔧 Safe Selectbox Test")
        
        # Test 1: Safe index calculation for first column
        st.write("**Test 1: Customer ID Column Selection**")
        col1_index = 0
        if st.session_state.selected_col1 and st.session_state.selected_col1 in df.columns:
            try:
                col1_index = df.columns.get_loc(st.session_state.selected_col1)
            except (ValueError, KeyError):
                col1_index = 0
        
        col1_selection = st.selectbox(
            "Select first column:",
            options=df.columns.tolist(),
            index=col1_index,
            key="safe_col1"
        )
        
        # Test 2: Safe index calculation for second column
        st.write("**Test 2: Churn Column Selection**")
        col2_index = len(df.columns) - 1
        if st.session_state.selected_col2 and st.session_state.selected_col2 in df.columns:
            try:
                col2_index = df.columns.get_loc(st.session_state.selected_col2)
            except (ValueError, KeyError):
                col2_index = len(df.columns) - 1
        
        col2_selection = st.selectbox(
            "Select second column:",
            options=df.columns.tolist(),
            index=col2_index,
            key="safe_col2"
        )
        
        # Save selections
        if st.button("💾 Save Selections"):
            st.session_state.selected_col1 = col1_selection
            st.session_state.selected_col2 = col2_selection
            st.success("✅ Selections saved!")
            st.experimental_rerun()
        
        # Show current selections
        st.subheader("📋 Current Selections")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Column 1:** {st.session_state.selected_col1 or 'Not set'}")
        with col2:
            st.write(f"**Column 2:** {st.session_state.selected_col2 or 'Not set'}")
        
        # Test column access
        if st.session_state.selected_col1 and st.session_state.selected_col2:
            st.subheader("✅ Column Access Test")
            try:
                # Safe column access
                col1_data = df[st.session_state.selected_col1]
                col2_data = df[st.session_state.selected_col2]
                
                st.success("🎉 Column access working perfectly!")
                
                # Show sample data
                st.subheader("📋 Sample Data Preview")
                preview_df = df[[st.session_state.selected_col1, st.session_state.selected_col2]].head(10)
                st.dataframe(preview_df)
                
            except Exception as e:
                st.error(f"❌ Error accessing columns: {e}")
        
        # Reset button
        if st.button("🔄 Reset Test"):
            st.session_state.clear()
            st.success("🔄 Test reset!")
            st.experimental_rerun()

if __name__ == "__main__":
    test_safe_selectbox()
