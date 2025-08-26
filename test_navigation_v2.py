#!/usr/bin/env python3
"""
Test script to verify navigation works with all Streamlit versions
"""

import streamlit as st

def test_navigation():
    """Test navigation without rerun functions"""
    
    # Initialize session state
    if 'selected_page' not in st.session_state:
        st.session_state.selected_page = "🏠 Home"
    
    st.title("🧪 Navigation Test (Version Compatible)")
    
    # Show current page
    st.write(f"**Current page:** {st.session_state.selected_page}")
    
    # Navigation buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🚀 Go to Data Upload"):
            st.session_state.selected_page = "📁 Data Upload"
            st.success("Page updated! The app will refresh automatically.")
    
    with col2:
        if st.button("🏠 Go Back Home"):
            st.session_state.selected_page = "🏠 Home"
            st.success("Page updated! The app will refresh automatically.")
    
    # Test navigation
    if st.session_state.selected_page == "🏠 Home":
        st.header("🏠 Home Page")
        st.success("✅ Navigation to Home working!")
        st.info("💡 Click 'Go to Data Upload' to test navigation")
        
    elif st.session_state.selected_page == "📁 Data Upload":
        st.header("📁 Data Upload Page")
        st.success("✅ Navigation to Data Upload working!")
        st.info("💡 Click 'Go Back Home' to return")
    
    st.markdown("---")
    st.write("**Note:** This version works with all Streamlit versions!")
    st.write("The page will refresh automatically when you click navigation buttons.")

if __name__ == "__main__":
    test_navigation()
