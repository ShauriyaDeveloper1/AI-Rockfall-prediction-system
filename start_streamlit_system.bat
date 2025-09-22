@echo off
echo 🚀 Starting AI Rockfall Prediction System with Streamlit
echo =========================================================

echo 🔧 Starting Backend Server...
start "Rockfall Backend" cmd /k "cd backend && python app.py"

echo ⏳ Waiting for backend to start...
timeout /t 10 /nobreak > nul

echo 🌐 Starting Streamlit Dashboard...
streamlit run streamlit_app.py --server.port 8501

echo.
echo ✅ System started successfully!
echo 🌐 Dashboard: http://localhost:8501
echo 🔧 Backend API: http://localhost:5000
echo.
pause