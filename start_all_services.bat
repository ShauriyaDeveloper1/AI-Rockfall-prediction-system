@echo off
echo ===============================================
echo   AI Rockfall Prediction System - Full Stack
echo ===============================================
echo.
echo Starting all services...
echo.

echo [1/3] Starting Authentication Backend (Port 5002)...
start "Auth Backend" cmd /k "python auth_backend.py"
timeout /t 3 /nobreak >nul

echo [2/3] Starting Main Backend (Port 5000)...
start "Main Backend" cmd /k "python simple_backend.py"
timeout /t 3 /nobreak >nul

echo [3/3] Starting Frontend (Port 3001)...
start "Frontend" cmd /k "cd frontend && npm start"
timeout /t 5 /nobreak >nul

echo.
echo ===============================================
echo   All Services Started Successfully!
echo ===============================================
echo.
echo Services:
echo   - Authentication: http://localhost:5002
echo   - Main Backend:   http://localhost:5000  
echo   - Frontend:       http://localhost:3001
echo.
echo Demo Login Credentials:
echo   Email:    admin@rockfall.com
echo   Password: Admin123!
echo.
echo Press any key to exit...
pause >nul