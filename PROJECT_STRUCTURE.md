# 📁 Project Structure Overview

## 🚀 Core Application Files

```
churn-prediction-tool/
├── app.py                          # 🎯 Main Streamlit application
├── churn_model.py                  # 🤖 Core ML model implementation
├── data_utils.py                   # 📊 Data handling and validation
├── visualization_utils.py          # 📈 Chart and report generation
├── config.py                       # ⚙️ Configuration and parameters
├── demo.py                         # 🧪 Demo and testing script
├── setup.py                        # 🛠️ Installation and setup script
├── requirements.txt                # 📦 Python dependencies
├── README.md                       # 📚 Complete documentation
├── run_app.bat                     # 🪟 Windows launcher
└── PROJECT_STRUCTURE.md            # 📁 This file
```

## 🗂️ Generated Directories (after setup)

```
churn-prediction-tool/
├── models/                         # 💾 Saved trained models
├── data/                          # 📁 Sample and user data
├── logs/                          # 📝 Application logs
└── exports/                       # 📤 Exported results
```

## 🔧 File Descriptions

### **Core Application (`app.py`)**
- **Purpose**: Main web interface using Streamlit
- **Features**: 
  - Data upload and validation
  - Model training interface
  - Results visualization
  - Export functionality
- **Navigation**: 5 main sections with sidebar navigation

### **Machine Learning Model (`churn_model.py`)**
- **Purpose**: Core churn prediction algorithms
- **Algorithms**: 
  - Random Forest
  - Gradient Boosting
  - Logistic Regression
- **Features**: 
  - Automatic model selection
  - Feature importance analysis
  - Model persistence

### **Data Utilities (`data_utils.py`)**
- **Purpose**: Data handling and validation
- **Features**:
  - File upload support (CSV, Excel)
  - Data structure validation
  - Column mapping suggestions
  - Sample data generation

### **Visualization (`visualization_utils.py`)**
- **Purpose**: Beautiful charts and reports
- **Charts**:
  - Performance metrics (gauge charts)
  - Churn distribution (pie charts)
  - Risk assessment (bar charts)
  - Feature importance (horizontal bars)

### **Configuration (`config.py`)**
- **Purpose**: Centralized configuration
- **Sections**:
  - Model parameters
  - Data processing settings
  - Visualization options
  - Risk thresholds

### **Demo Script (`demo.py`)**
- **Purpose**: Test functionality without web interface
- **Options**:
  - Full demo with sample data
  - Quick functionality test
  - Command-line interface

### **Setup Script (`setup.py`)**
- **Purpose**: Automated installation and setup
- **Tasks**:
  - Package installation
  - Directory creation
  - Import testing
  - Functionality testing

## 🚀 Getting Started

### **1. Quick Start (Windows)**
```bash
# Double-click run_app.bat
# Or run in command prompt:
run_app.bat
```

### **2. Manual Setup**
```bash
# Install dependencies
python setup.py

# Run the application
streamlit run app.py

# Test functionality
python demo.py
```

### **3. Development Setup**
```bash
# Install in development mode
pip install -e .

# Run tests
python -m pytest tests/

# Format code
black *.py
```

## 🔄 Workflow

```
Data Upload → Validation → Model Training → Predictions → Analytics → Export
     ↓              ↓            ↓            ↓           ↓         ↓
  CSV/Excel    Structure    ML Algorithms  Risk Levels  Charts   CSV/Excel
```

## 🎯 Key Features

### **Data Processing**
- ✅ Multiple file formats (CSV, Excel)
- ✅ Automatic column detection
- ✅ Data validation and cleaning
- ✅ Missing value handling

### **Machine Learning**
- ✅ Ensemble methods (3 algorithms)
- ✅ Automatic feature selection
- ✅ Model performance comparison
- ✅ Feature importance analysis

### **Visualization**
- ✅ Interactive charts (Plotly)
- ✅ Performance metrics
- ✅ Risk assessment
- ✅ Customer segmentation

### **Export & Integration**
- ✅ Multiple export formats
- ✅ Model persistence
- ✅ API-ready architecture
- ✅ Business intelligence ready

## 🛠️ Customization

### **Model Parameters**
Edit `config.py` to adjust:
- Algorithm hyperparameters
- Feature selection thresholds
- Risk assessment levels
- Visualization styles

### **Adding New Algorithms**
1. Add new model to `churn_model.py`
2. Update configuration in `config.py`
3. Test with `demo.py`

### **Custom Visualizations**
1. Add new methods to `visualization_utils.py`
2. Integrate in `app.py`
3. Update configuration

## 🔍 Troubleshooting

### **Common Issues**
- **Import errors**: Run `python setup.py`
- **Missing packages**: Check `requirements.txt`
- **Data format issues**: See data requirements in README
- **Performance issues**: Adjust parameters in `config.py`

### **Support**
- Check `README.md` for detailed instructions
- Run `python demo.py` to test functionality
- Review error messages in the application

## 📚 Next Steps

1. **Run the application**: `streamlit run app.py`
2. **Upload your data**: Use the web interface
3. **Train models**: Let AI find the best algorithm
4. **Analyze results**: Explore insights and visualizations
5. **Export findings**: Save results for business use
6. **Customize**: Adjust parameters for your needs
7. **Integrate**: Use models in your business processes

---

**🎉 You're ready to predict customer churn like a pro!**
