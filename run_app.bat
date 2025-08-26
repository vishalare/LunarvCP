@echo off
echo.
echo ========================================
echo    Churn Prediction Tool Launcher
echo ========================================
echo.
echo Starting the application...
echo.
echo If this is your first time running the app:
echo 1. Make sure you have Python installed
echo 2. Run: python setup.py
echo.
echo Press any key to continue...
pause >nul

echo.
echo Starting Streamlit application...
echo.
echo The app will open in your default browser
echo If it doesn't open automatically, go to: http://localhost:8501
echo.
echo To stop the app, press Ctrl+C in this window
echo.

streamlit run app.py

echo.
echo Application stopped.
echo Press any key to exit...
pause >nul
