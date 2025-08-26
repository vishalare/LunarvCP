@echo off
echo.
echo ========================================
echo    Churn Prediction Tool - Fix & Run
echo ========================================
echo.

echo 🔧 Fixing missing dependencies...
python fix_dependencies.py

echo.
echo 🚀 Starting the application...
echo.

echo If the app opens in your browser, you're all set!
echo If you see errors, check the output above.
echo.
echo To stop the app, press Ctrl+C in this window
echo.

python -m streamlit run app.py

echo.
echo Application stopped.
echo Press any key to exit...
pause >nul
