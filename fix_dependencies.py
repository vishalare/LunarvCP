#!/usr/bin/env python3
"""
Quick Fix Script for Missing Dependencies
Installs all required packages for the Churn Prediction Tool
"""

import subprocess
import sys
import os

def install_package(package_name, version=None):
    """Install a specific package"""
    try:
        if version:
            package_spec = f"{package_name}=={version}"
        else:
            package_spec = package_name
            
        print(f"📦 Installing {package_spec}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_spec])
        print(f"✅ {package_name} installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package_name}: {e}")
        return False

def main():
    print("🔧 Fixing Missing Dependencies")
    print("=" * 40)
    
    # Core packages that might be missing
    packages = [
        ("seaborn", "0.12.2"),
        ("matplotlib", "3.7.2"),
        ("pandas", "2.0.3"),
        ("numpy", "1.24.3"),
        ("scikit-learn", "1.3.0"),
        ("streamlit", "1.25.0"),
        ("plotly", "5.15.0"),
        ("openpyxl", "3.1.2"),
        ("joblib", "1.3.1")
    ]
    
    print("🔍 Checking and installing missing packages...")
    
    success_count = 0
    for package, version in packages:
        if install_package(package, version):
            success_count += 1
    
    print(f"\n📊 Installation Summary:")
    print(f"✅ Successfully installed: {success_count}/{len(packages)} packages")
    
    if success_count == len(packages):
        print("\n🎉 All dependencies installed successfully!")
        print("You can now run: python -m streamlit run app.py")
    else:
        print("\n⚠️ Some packages failed to install.")
        print("Try running: pip install --upgrade pip")
        print("Then run this script again.")
    
    # Test imports
    print("\n🧪 Testing imports...")
    try:
        import seaborn as sns
        print("✅ seaborn imported successfully")
        
        import matplotlib.pyplot as plt
        print("✅ matplotlib imported successfully")
        
        import pandas as pd
        print("✅ pandas imported successfully")
        
        import streamlit as st
        print("✅ streamlit imported successfully")
        
        print("\n🎯 All critical imports working! You're ready to go!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please run this script again or install packages manually.")

if __name__ == "__main__":
    main()
