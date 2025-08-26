# 🆘 Troubleshooting Guide

## 🚨 Common Setup Issues & Solutions

### **1. ❌ Permission Denied Errors**

#### **Windows:**
```bash
# Run Command Prompt as Administrator
# Right-click Command Prompt → "Run as administrator"
cd "path\to\your\project"
python setup.py
```

#### **Mac/Linux:**
```bash
# Use sudo (not recommended for pip, but sometimes necessary)
sudo python setup.py

# Better alternative - user installation
pip install --user -r requirements.txt
```

### **2. ❌ Network/Firewall Issues**

#### **Check Internet Connection:**
```bash
# Test basic connectivity
ping google.com

# Test pip connectivity
pip install --upgrade pip
```

#### **Firewall/Antivirus:**
- Temporarily disable antivirus/firewall
- Add Python/pip to firewall exceptions
- Try different network (mobile hotspot)

#### **Corporate Networks:**
```bash
# Use proxy if required
pip install --proxy http://proxy.company.com:8080 -r requirements.txt
```

### **3. ❌ Python Version Issues**

#### **Check Python Version:**
```bash
python --version
python3 --version
```

#### **Install Python 3.8+:**
- **Windows**: Download from [python.org](https://python.org)
- **Mac**: `brew install python@3.9`
- **Linux**: `sudo apt install python3.9`

#### **PATH Issues:**
```bash
# Check if Python is in PATH
where python  # Windows
which python  # Mac/Linux

# Add Python to PATH manually if needed
```

### **4. ❌ Package Conflicts**

#### **Create Virtual Environment:**
```bash
# Create virtual environment
python -m venv churn_env

# Activate (Windows)
churn_env\Scripts\activate

# Activate (Mac/Linux)
source churn_env/bin/activate

# Install packages in clean environment
pip install -r requirements.txt
```

#### **Clean Installation:**
```bash
# Uninstall conflicting packages
pip uninstall pandas numpy scikit-learn streamlit plotly openpyxl joblib

# Reinstall fresh
pip install -r requirements.txt
```

### **5. ❌ Missing System Dependencies**

#### **Windows:**
- Install [Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)
- Install [Microsoft Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

#### **Mac:**
```bash
# Install Xcode Command Line Tools
xcode-select --install
```

#### **Linux:**
```bash
# Ubuntu/Debian
sudo apt-get install build-essential python3-dev

# CentOS/RHEL
sudo yum groupinstall "Development Tools"
sudo yum install python3-devel
```

### **6. ❌ Specific Package Errors**

#### **scikit-learn Issues:**
```bash
# Try conda instead
conda install scikit-learn

# Or install pre-compiled wheel
pip install --only-binary=scikit-learn scikit-learn
```

#### **streamlit Issues:**
```bash
# Install streamlit separately
pip install streamlit

# Check if it's in PATH
streamlit --version
```

#### **plotly Issues:**
```bash
# Install plotly with dependencies
pip install plotly kaleido
```

### **7. ❌ Import Errors After Installation**

#### **Check Installation:**
```bash
# Verify packages are installed
pip list | grep pandas
pip list | grep streamlit

# Check import in Python
python -c "import pandas; print('pandas OK')"
python -c "import streamlit; print('streamlit OK')"
```

#### **Reinstall Packages:**
```bash
# Force reinstall
pip install --force-reinstall pandas numpy scikit-learn streamlit plotly openpyxl joblib
```

### **8. ❌ Disk Space Issues**

#### **Check Available Space:**
```bash
# Windows
dir

# Mac/Linux
df -h
```

#### **Clean Up:**
```bash
# Clear pip cache
pip cache purge

# Remove old packages
pip uninstall -y $(pip list | grep -v "pip\|setuptools\|wheel" | awk '{print $1}')
```

## 🔧 Alternative Installation Methods

### **Method 1: Quick Setup Script**
```bash
python quick_setup.py
```

### **Method 2: Manual Package Installation**
```bash
# Install core packages
pip install pandas numpy scikit-learn

# Install web framework
pip install streamlit plotly

# Install data processing
pip install openpyxl joblib
```

### **Method 3: Using Conda**
```bash
# Create environment
conda create -n churn python=3.9

# Activate environment
conda activate churn

# Install packages
conda install pandas numpy scikit-learn streamlit plotly openpyxl joblib
```

### **Method 4: Docker (Advanced)**
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## 📋 Diagnostic Commands

### **System Information:**
```bash
# Python info
python --version
pip --version
which python
which pip

# System info
uname -a  # Linux/Mac
systeminfo  # Windows

# Disk space
df -h  # Linux/Mac
dir  # Windows
```

### **Package Information:**
```bash
# List installed packages
pip list

# Check specific package
pip show pandas
pip show streamlit

# Check package compatibility
pip check
```

### **Network Diagnostics:**
```bash
# Test connectivity
ping google.com
curl -I https://pypi.org

# Test pip
pip install --upgrade pip
```

## 🆘 Getting Help

### **1. Check Error Messages**
- Copy the **exact error message**
- Note the **Python version** and **operating system**
- Check if it's a **permission**, **network**, or **package** issue

### **2. Common Error Types**
- **ModuleNotFoundError**: Package not installed
- **PermissionError**: Need admin rights
- **ConnectionError**: Network/firewall issue
- **CompilationError**: Missing system dependencies

### **3. Still Stuck?**
- Try the **quick_setup.py** script
- Use **manual installation** commands
- Check if **conda** works better than pip
- Create a **virtual environment** for isolation

### **4. Emergency Run**
If nothing works, you can still run the demo:
```bash
# Try running demo directly (may fail but will show specific errors)
python demo.py

# Or try the web app (may fail but will show specific errors)
streamlit run app.py
```

## 🎯 Quick Fix Checklist

- [ ] **Python 3.8+ installed?**
- [ ] **pip working?** (`pip --version`)
- [ ] **Running as admin?** (Windows) or **using --user flag?**
- [ ] **Internet connection working?**
- [ ] **Enough disk space?**
- [ ] **Virtual environment created?** (if having conflicts)
- [ ] **System dependencies installed?** (Visual C++, Xcode, build tools)

---

**💡 Remember**: Most issues can be solved by using the right installation method for your system. Try the automated setup first, then quick setup, then manual installation as a last resort.
