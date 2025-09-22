@echo off
:: Enhanced AI Rockfall Prediction System - Complete Startup Script
:: This script starts all components including new features

echo ========================================
echo AI Rockfall Prediction System v2.0
echo Enhanced with Soil Classification, 3D Visualization, Email Reporting
echo ========================================
echo.

:: Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

:: Check if Node.js is available
node --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Node.js is not installed or not in PATH
    echo Please install Node.js 16+ and try again
    pause
    exit /b 1
)

:: Create log directory
if not exist "logs" mkdir logs

echo [1/6] Installing Python dependencies...
echo ----------------------------------------
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo WARNING: Some Python packages may have failed to install
    echo Continuing with startup...
)

echo.
echo [2/6] Installing frontend dependencies...
echo ----------------------------------------
cd frontend
call npm install --silent
if errorlevel 1 (
    echo ERROR: Failed to install frontend dependencies
    cd ..
    pause
    exit /b 1
)
cd ..

echo.
echo [3/6] Setting up environment...
echo ----------------------------------------

:: Create .env file if it doesn't exist
if not exist ".env" (
    echo Creating default .env file...
    echo DATABASE_URL=sqlite:///rockfall_system.db > .env
    echo FLASK_ENV=development >> .env
    echo # Email configuration ^(optional^) >> .env
    echo # SMTP_SERVER=smtp.gmail.com >> .env
    echo # SMTP_PORT=587 >> .env
    echo # SMTP_USERNAME=your_email@gmail.com >> .env
    echo # SMTP_PASSWORD=your_app_password >> .env
    echo # FROM_EMAIL=your_email@gmail.com >> .env
    echo # FROM_NAME=Rockfall Alert System >> .env
    echo.
    echo Default .env file created. You can edit it to configure email alerts.
)

:: Create uploads directory for soil/rock images
if not exist "backend\uploads" mkdir backend\uploads
if not exist "backend\uploads\soil_images" mkdir backend\uploads\soil_images

:: Create models directory
if not exist "ml_models" mkdir ml_models

echo.
echo [4/6] Initializing database and models...
echo ----------------------------------------
cd backend
python -c "
try:
    from app import app, db
    with app.app_context():
        db.create_all()
        print('✅ Database initialized successfully')
except Exception as e:
    print(f'⚠️ Database initialization warning: {e}')
"
cd ..

echo.
echo [5/6] Starting backend services...
echo ----------------------------------------
echo Starting Flask backend on port 5000...
start "Rockfall Backend" cmd /k "cd backend && python app.py"

:: Wait for backend to start
echo Waiting for backend to initialize...
timeout /t 5 /nobreak >nul

:: Test backend health
python -c "
import requests
import time
for i in range(10):
    try:
        response = requests.get('http://localhost:5000/api/health', timeout=2)
        if response.status_code == 200:
            print('✅ Backend is healthy and ready')
            break
    except:
        pass
    time.sleep(1)
else:
    print('⚠️ Backend may still be starting up')
" >nul 2>&1

echo.
echo [6/6] Starting frontend application...
echo ----------------------------------------
echo Starting React frontend on port 3000...
start "Rockfall Frontend" cmd /k "cd frontend && npm start"

:: Wait a moment for frontend to start
timeout /t 3 /nobreak >nul

echo.
echo ========================================
echo 🚀 ENHANCED SYSTEM STARTUP COMPLETE
echo ========================================
echo.
echo 🌐 Frontend: http://localhost:3000
echo 🔗 Backend API: http://localhost:5000/api
echo 📊 API Health: http://localhost:5000/api/health
echo.
echo NEW FEATURES AVAILABLE:
echo 🧠 Enhanced Dashboard with India mining data
echo 🪨 Soil/Rock Classification (AI-powered)
echo 📧 Email Risk Reports
echo 🗺️ 3D Holographic Mine Visualization
echo 📡 Real-time sensor monitoring
echo.
echo QUICK ACCESS ROUTES:
echo • Enhanced Dashboard: http://localhost:3000/
echo • Soil Classifier: http://localhost:3000/soil-classifier
echo • 3D Mine View: http://localhost:3000/mine-3d
echo • Classic Dashboard: http://localhost:3000/dashboard
echo.
echo 📋 To run system tests: python test_enhanced_system.py
echo 📄 Check logs in the logs/ directory
echo.
echo ⚠️ NOTES:
echo • Email alerts require SMTP configuration in .env
echo • Soil classification requires image uploads
echo • 3D visualization may require WebGL support
echo.
echo Press Ctrl+C in any terminal window to stop services
echo ========================================

:: Optional: Start sensor simulator
choice /c YN /m "Start sensor data simulator? (Y/N)"
if errorlevel 2 goto :skip_simulator
if errorlevel 1 (
    echo.
    echo Starting sensor data simulator...
    start "Sensor Simulator" cmd /k "python start_simulator.py"
    echo ✅ Sensor simulator started
)

:skip_simulator

echo.
echo 🎉 All services are now running!
echo Open http://localhost:3000 in your browser to access the system.
echo.

:: Keep this window open
echo Press any key to open the system in your default browser...
pause >nul

:: Open browser
start http://localhost:3000

echo.
echo System is running. Close this window to stop monitoring.
echo Press any key to exit...
pause >nul