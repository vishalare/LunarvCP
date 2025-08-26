#!/usr/bin/env python3
"""
Test script to verify navigation works without switch_page errors
"""

import streamlit as st

def test_navigation():
    """Test navigation without switch_page"""
    
    # Initialize session state
    if 'selected_page' not in st.session_state:
        st.session_state.selected_page = "🏠 Home"
    
    st.title("🧪 Navigation Test")
    
    # Test button that updates session state
    if st.button("🚀 Go to Data Upload"):
        st.session_state.selected_page = "📁 Data Upload"
        st.experimental_rerun()
    
    # Show current page
    st.write(f"**Current page:** {st.session_state.selected_page}")
    
    # Test navigation
    if st.session_state.selected_page == "🏠 Home":
        st.header("🏠 Home Page")
        st.success("✅ Navigation to Home working!")
        
    elif st.session_state.selected_page == "📁 Data Upload":
        st.header("📁 Data Upload Page")
        st.success("✅ Navigation to Data Upload working!")
        
        # Test going back
        if st.button("🏠 Go Back Home"):
            st.session_state.selected_page = "🏠 Home"
            st.experimental_rerun()
    
    st.markdown("---")
    st.write("If you see this and can navigate between pages, the fix is working!")

if __name__ == "__main__":
    test_navigation()
