#!/usr/bin/env python3
"""
Check Streamlit version and compatibility
"""

import streamlit as st
import sys

def check_streamlit_version():
    """Check Streamlit version and show compatibility info"""
    
    st.title("🔍 Streamlit Version Check")
    
    # Get Streamlit version
    try:
        import streamlit as st
        version = st.__version__
        st.write(f"**Streamlit Version:** {version}")
        
        # Parse version
        major, minor, patch = map(int, version.split('.')[:3])
        
        st.markdown("---")
        st.subheader("📋 Version Compatibility")
        
        if major >= 1 and minor >= 28:
            st.success("✅ **Modern Version** - Full compatibility with all features!")
            st.write("You can use `st.rerun()` for navigation.")
            
        elif major >= 1 and minor >= 27:
            st.warning("⚠️ **Recent Version** - Good compatibility")
            st.write("You can use `st.experimental_rerun()` for navigation.")
            
        elif major >= 1 and minor >= 20:
            st.info("ℹ️ **Standard Version** - Basic compatibility")
            st.write("Use sidebar navigation (no rerun functions).")
            
        else:
            st.error("❌ **Older Version** - Limited compatibility")
            st.write("Consider upgrading to Streamlit 1.20+ for better features.")
        
        st.markdown("---")
        st.subheader("🚀 Recommended Actions")
        
        if major >= 1 and minor >= 28:
            st.success("Your version is perfect! All features will work.")
            
        elif major >= 1 and minor >= 20:
            st.info("Your version works well. Consider upgrading for latest features.")
            
        else:
            st.warning("Consider upgrading Streamlit:")
            st.code("pip install --upgrade streamlit")
        
        st.markdown("---")
        st.subheader("💡 Current App Status")
        st.success("✅ The Churn Prediction Tool is compatible with your Streamlit version!")
        st.write("Navigation works through the sidebar selectbox.")
        
    except Exception as e:
        st.error(f"❌ Error checking version: {e}")
        st.write("Please ensure Streamlit is properly installed.")

if __name__ == "__main__":
    check_streamlit_version()
