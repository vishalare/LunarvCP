#!/usr/bin/env python3
"""
Quick Setup Script for Churn Prediction Tool
Alternative setup method with better error handling
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    print("🚀 Churn Prediction Tool - Quick Setup")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required. Current:", sys.version)
        return False
    
    print(f"✅ Python {sys.version.split()[0]} detected")
    
    # Create directories
    print("\n📁 Creating directories...")
    for dir_name in ['models', 'data', 'logs', 'exports']:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"✅ Created {dir_name}/")
    
    # Try different installation methods
    print("\n📦 Installing packages...")
    
    # Method 1: Try pip install --user
    try:
        print("🔄 Method 1: User installation...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", 
                             "pandas", "numpy", "scikit-learn", "streamlit", "plotly", "openpyxl", "joblib"])
        print("✅ Installation successful!")
        return True
    except subprocess.CalledProcessError:
        print("❌ User installation failed")
    
    # Method 2: Try without version constraints
    try:
        print("🔄 Method 2: Latest versions...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", 
                             "pandas", "numpy", "scikit-learn", "streamlit", "plotly", "openpyxl", "joblib"])
        print("✅ Installation successful!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Latest versions installation failed")
    
    # Method 3: Try conda
    try:
        print("🔄 Method 3: Using conda...")
        subprocess.check_call(["conda", "install", "-y", 
                             "pandas", "numpy", "scikit-learn", "streamlit", "plotly", "openpyxl", "joblib"])
        print("✅ Conda installation successful!")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Conda not available or failed")
    
    print("\n❌ All installation methods failed!")
    print("\n💡 Manual installation required:")
    print("1. pip install --user pandas numpy scikit-learn streamlit plotly openpyxl joblib")
    print("2. Or use conda: conda install pandas numpy scikit-learn streamlit plotly openpyxl joblib")
    print("3. Check your internet connection and try again")
    
    return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Quick setup completed!")
        print("Run: streamlit run app.py")
    else:
        print("\n❌ Quick setup failed. Try manual installation.")
