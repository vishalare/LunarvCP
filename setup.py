#!/usr/bin/env python3
"""
Setup script for Churn Prediction Tool
"""

import subprocess
import sys
import os
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        print("\n💡 Solutions:")
        print("1. Download Python 3.8+ from https://python.org")
        print("2. Or use conda: conda create -n churn python=3.9")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def check_pip():
    """Check if pip is available and working"""
    try:
        import pip
        print("✅ pip is available")
        return True
    except ImportError:
        print("❌ pip not found")
        print("\n💡 Install pip:")
        print("1. Download get-pip.py from https://bootstrap.pypa.io/get-pip.py")
        print("2. Run: python get-pip.py")
        return False

def install_requirements():
    """Install required packages"""
    print("\n📦 Installing required packages...")
    
    # Try multiple installation methods
    methods = [
        ("requirements.txt", lambda: subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])),
        ("individual packages", lambda: install_packages_individually()),
        ("conda (if available)", lambda: install_with_conda()),
        ("user installation", lambda: subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "-r", "requirements.txt"]))
    ]
    
    for method_name, method_func in methods:
        try:
            print(f"🔄 Trying {method_name}...")
            method_func()
            print(f"✅ Installation successful using {method_name}!")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"❌ {method_name} failed: {e}")
            continue
    
    print("\n❌ All installation methods failed!")
    print("\n💡 Manual installation required:")
    print("1. Run: pip install --upgrade pip")
    print("2. Run: pip install pandas numpy scikit-learn streamlit plotly openpyxl joblib")
    print("3. Or use conda: conda install pandas numpy scikit-learn streamlit plotly openpyxl joblib")
    print("\n🔧 Alternative solutions:")
    print("4. Try: python -m pip install --user -r requirements.txt")
    print("5. Check your internet connection and firewall settings")
    print("6. On Windows, run Command Prompt as Administrator")
    print("7. On Mac/Linux, try: sudo pip install -r requirements.txt")
    return False

def install_packages_individually():
    """Install packages one by one"""
    packages = [
        "pandas==2.0.3",
        "numpy==1.24.3", 
        "scikit-learn==1.3.0",
        "streamlit==1.25.0",
        "plotly==5.15.0",
        "openpyxl==3.1.2",
        "joblib==1.3.1"
    ]
    
    for package in packages:
        print(f"  Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def install_with_conda():
    """Try installing with conda if available"""
    try:
        subprocess.check_call(["conda", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        packages = ["pandas", "numpy", "scikit-learn", "streamlit", "plotly", "openpyxl", "joblib"]
        subprocess.check_call(["conda", "install", "-y"] + packages)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise Exception("conda not available")

def check_windows_requirements():
    """Check Windows-specific requirements"""
    if os.name == 'nt':  # Windows
        print("🪟 Windows detected - checking requirements...")
        
        # Check if Visual C++ Redistributable might be needed
        print("💡 Note: If you get compilation errors, you may need:")
        print("   Microsoft Visual C++ Redistributable for Visual Studio 2015-2022")
        print("   Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe")
        
        # Try to upgrade pip with user flag on Windows
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "--user", "pip"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("✅ Pip upgraded successfully!")
        except subprocess.CalledProcessError:
            print("⚠️ Could not upgrade pip automatically")

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    
    directories = ['models', 'data', 'logs', 'exports']
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}/")
    
    return True

def test_imports():
    """Test if all modules can be imported"""
    print("\n🧪 Testing imports...")
    
    try:
        import pandas as pd
        print("✅ pandas imported successfully")
        
        import numpy as np
        print("✅ numpy imported successfully")
        
        import sklearn
        print("✅ scikit-learn imported successfully")
        
        import streamlit as st
        print("✅ streamlit imported successfully")
        
        import plotly
        print("✅ plotly imported successfully")
        
        # Test our custom modules
        from churn_model import ChurnPredictionModel
        print("✅ ChurnPredictionModel imported successfully")
        
        from data_utils import DataHandler
        print("✅ DataHandler imported successfully")
        
        from visualization_utils import ChurnVisualizer
        print("✅ ChurnVisualizer imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def run_quick_test():
    """Run a quick functionality test"""
    print("\n🚀 Running quick functionality test...")
    
    try:
        from demo import quick_test
        success = quick_test()
        
        if success:
            print("✅ Quick test passed!")
            return True
        else:
            print("❌ Quick test failed!")
            return False
            
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

def create_sample_data():
    """Create a sample dataset for testing"""
    print("\n📊 Creating sample dataset...")
    
    try:
        from data_utils import DataHandler
        data_handler = DataHandler()
        sample_df = data_handler.create_sample_data()
        
        # Save sample data
        sample_df.to_csv('data/sample_customer_data.csv', index=False)
        print(f"✅ Sample data created: data/sample_customer_data.csv ({len(sample_df):,} records)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating sample data: {e}")
        return False

def show_next_steps():
    """Show next steps for the user"""
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next Steps:")
    print("1. 🚀 Start the web application:")
    print("   streamlit run app.py")
    print("\n2. 🧪 Test the demo script:")
    print("   python demo.py")
    print("\n3. 📁 Upload your data:")
    print("   - Use the web interface")
    print("   - Or place CSV/Excel files in the data/ folder")
    print("\n4. ⚙️ Customize settings:")
    print("   - Edit config.py for model parameters")
    print("   - Modify visualization colors and styles")
    print("\n5. 🔗 Integrate with your business:")
    print("   - Use the trained models in your applications")
    print("   - Export results for further analysis")
    print("\n📚 Documentation:")
    print("   - README.md: Complete guide")
    print("   - config.py: Configuration options")
    print("   - demo.py: Example usage")

def show_troubleshooting_help():
    """Show troubleshooting help when setup fails"""
    print("\n🆘 SETUP FAILED - TROUBLESHOOTING HELP")
    print("=" * 50)
    print("\n🔍 Common Issues & Solutions:")
    print("\n1. ❌ Permission Denied:")
    print("   Windows: Run Command Prompt as Administrator")
    print("   Mac/Linux: Use 'sudo python setup.py'")
    print("   Or try: pip install --user -r requirements.txt")
    print("\n2. ❌ Network/Firewall Issues:")
    print("   Check your internet connection")
    print("   Disable antivirus/firewall temporarily")
    print("   Try using a different network")
    print("\n3. ❌ Python Version Issues:")
    print("   Ensure Python 3.8+ is installed")
    print("   Check PATH environment variable")
    print("   Try: python3 setup.py instead")
    print("\n4. ❌ Package Conflicts:")
    print("   Create virtual environment:")
    print("   python -m venv churn_env")
    print("   churn_env\\Scripts\\activate (Windows)")
    print("   source churn_env/bin/activate (Mac/Linux)")
    print("\n5. ❌ Missing Dependencies:")
    print("   Install system requirements:")
    print("   Windows: Visual C++ Redistributable")
    print("   Mac: Xcode Command Line Tools")
    print("   Linux: build-essential package")
    print("\n💡 Quick Fix Commands:")
    print("   pip install --upgrade pip")
    print("   pip install --user pandas numpy scikit-learn streamlit plotly openpyxl joblib")
    print("   python -m pip install -r requirements.txt")
    print("\n📞 Still having issues? Check:")
    print("   - Python version: python --version")
    print("   - pip version: pip --version")
    print("   - System architecture (32-bit vs 64-bit)")
    print("   - Available disk space")

def main():
    """Main setup function"""
    print("🚀 Churn Prediction Tool - Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check pip availability
    if not check_pip():
        print("\n❌ pip is required for package installation")
        print("Please install pip first and try again")
        sys.exit(1)
    
    # Check Windows-specific requirements
    check_windows_requirements()
    
    # Install requirements
    if not install_requirements():
        print("\n❌ Setup failed during package installation")
        show_troubleshooting_help()
        sys.exit(1)
    
    # Create directories
    if not create_directories():
        print("\n❌ Setup failed during directory creation")
        sys.exit(1)
    
    # Test imports
    if not test_imports():
        print("\n❌ Setup failed during import testing")
        print("Please check the error messages above")
        show_troubleshooting_help()
        sys.exit(1)
    
    # Run quick test
    if not run_quick_test():
        print("\n❌ Setup failed during functionality testing")
        print("Please check the error messages above")
        show_troubleshooting_help()
        sys.exit(1)
    
    # Create sample data
    if not create_sample_data():
        print("\n⚠️ Warning: Could not create sample data")
        print("You can still use the tool with your own data")
    
    # Show next steps
    show_next_steps()
    
    print("\n🎯 Setup completed! You're ready to predict customer churn!")

if __name__ == "__main__":
    main()
