# 🚀 Enhanced AI Rockfall Prediction System v2.0

A comprehensive AI-powered system for predicting and monitoring rockfall incidents in open-pit mines with advanced features including soil/rock classification, 3D holographic visualization, email reporting, and enhanced dashboard with India mining data.

## 🌟 New Features in v2.0

### 🧠 AI-Powered Soil/Rock Classification
- **Deep Learning CNN Model**: Classifies 4 soil types (Alluvial, Black, Cinder, Red Soil)
- **Image Upload Interface**: Drag-and-drop image analysis
- **Detailed Geological Analysis**: Comprehensive soil characteristics and mining implications
- **Risk Assessment**: Automated stability scoring and risk evaluation

### 📧 Intelligent Email Reporting
- **Automated PDF Reports**: Comprehensive risk analysis reports
- **SMTP Integration**: Configurable email delivery
- **Chart Generation**: Visual data representation in reports
- **Scheduled Reports**: Daily, weekly, and on-demand reporting

### 🗺️ 3D Holographic Mine Visualization
- **Interactive 3D Mine Model**: Holographic-style mine pit visualization
- **Real-time Sensor Networks**: Live sensor data overlay
- **Risk Zone Mapping**: Color-coded risk areas
- **Emergency Systems**: Escape routes and safety equipment locations

### 📊 Enhanced Dashboard
- **India Mining Data**: Real-time mining statistics for Indian states
- **Advanced Analytics**: Comprehensive mining statistics and trends
- **Interactive Maps**: SVG-based India map with mining data
- **Real-time Charts**: Live data visualization with Chart.js

### 🔬 Advanced API Endpoints
- Enhanced dashboard statistics
- India-specific mining data
- Soil/rock classification services
- Email reporting functionality
- 3D visualization data

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │    Backend      │    │   AI Models     │
│   (React)       │────│   (Flask)       │────│  (TensorFlow)   │
│ Port 3000       │    │  Port 5000      │    │   & scikit      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                       │                       │
        │              ┌─────────────────┐              │
        └──────────────│   Database      │──────────────┘
                       │ (SQLite/PgSQL)  │
                       └─────────────────┘
```

## 🚀 Quick Start

### Option 1: Enhanced Startup Script (Recommended)
```bash
# Windows
start_enhanced_system.bat

# The script will:
# 1. Install all dependencies
# 2. Set up the database
# 3. Start backend services
# 4. Launch frontend application
# 5. Optionally start sensor simulator
```

### Option 2: Manual Setup
```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Install frontend dependencies
cd frontend
npm install
cd ..

# 3. Start backend
cd backend
python app.py

# 4. Start frontend (in new terminal)
cd frontend
npm start

# 5. Start sensor simulator (optional)
python start_simulator.py
```

## 🔧 Configuration

### Environment Variables (.env)
Create a `.env` file in the root directory:
```bash
# Database
DATABASE_URL=sqlite:///rockfall_system.db

# Email Configuration (Optional)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your_email@gmail.com
SMTP_PASSWORD=your_app_password
FROM_EMAIL=your_email@gmail.com
FROM_NAME=Rockfall Alert System

# Development
FLASK_ENV=development
```

### Email Setup for Gmail
1. Enable 2-factor authentication
2. Generate app password
3. Use app password in SMTP_PASSWORD
4. Set SMTP_SERVER=smtp.gmail.com and SMTP_PORT=587

## 📁 Project Structure

```
ai-rockfall-prediction-system/
├── 📁 backend/
│   ├── app.py                     # Main Flask application
│   ├── soil_rock_classifier.py    # AI soil classification
│   ├── email_report_service.py    # Email reporting system
│   ├── alert_service.py           # Alert management
│   └── ml_service.py              # ML services
├── 📁 frontend/
│   ├── 📁 src/
│   │   ├── 📁 components/
│   │   │   ├── EnhancedDashboard.js    # Enhanced dashboard
│   │   │   ├── SoilRockClassifier.js   # Soil classification UI
│   │   │   ├── MineView3D.js           # 3D mine visualization
│   │   │   └── *.css                   # Component styling
│   │   └── App.js                      # Main React app
├── 📁 ml_models/
│   ├── rockfall_predictor.py      # Traditional ML models
│   └── *.pkl                      # Trained model files
├── 📁 config/
│   └── emergency_contacts.json    # Alert contacts
├── start_enhanced_system.bat      # Enhanced startup script
├── test_enhanced_system.py        # Comprehensive test suite
└── README_ENHANCED.md             # This file
```

## 🌐 API Endpoints

### Core Endpoints
- `GET /api/health` - System health check
- `POST /api/sensor-data` - Submit sensor data
- `GET /api/risk-assessment` - Current risk assessment
- `GET /api/alerts` - Recent alerts

### Enhanced Dashboard
- `GET /api/enhanced-dashboard/statistics` - Mining statistics
- `GET /api/enhanced-dashboard/india-mining-data` - India mining data
- `GET /api/enhanced-dashboard/risk-distribution` - Risk distribution charts

### Soil/Rock Classification
- `POST /api/soil-rock/classify` - Classify soil/rock image
- `GET /api/soil-rock/info` - Model information
- `POST /api/soil-rock/train` - Train classification model

### Email Reporting
- `POST /api/email-report` - Send risk analysis report

### LSTM Predictions
- `GET /api/lstm/status` - LSTM model status
- `GET /api/lstm/predict-realtime` - Real-time predictions
- `POST /api/lstm/train` - Train LSTM model

## 🎨 User Interface Features

### Navigation
- **Enhanced Dashboard**: Modern dashboard with India mining data
- **AI Analysis**: Soil/rock classifier and LSTM predictor
- **Visualizations**: 3D mine view and LIDAR visualization
- **Classic Views**: Original dashboard and components

### Responsive Design
- **Mobile-friendly**: Responsive layout for all devices
- **Modern Styling**: Holographic theme with CSS animations
- **Interactive Elements**: Hover effects and smooth transitions

## 🧪 Testing

### Comprehensive Test Suite
```bash
python test_enhanced_system.py
```

The test suite covers:
- Basic API endpoints
- Enhanced dashboard functionality
- Soil/rock classification
- Email reporting
- LSTM model integration
- Sensor data injection

### Manual Testing
1. **Soil Classification**: Upload soil images at `/soil-classifier`
2. **Email Reports**: Configure SMTP and test email delivery
3. **3D Visualization**: Explore mine view at `/mine-3d`
4. **Dashboard**: Check enhanced features at `/`

## 🔒 Security Features

- **Input Validation**: Comprehensive request validation
- **Error Handling**: Graceful error management
- **CORS Protection**: Cross-origin request security
- **File Upload Security**: Safe image upload handling

## 📊 Performance Optimizations

- **Lazy Loading**: Components load on demand
- **API Caching**: Reduced server load with smart caching
- **Image Optimization**: Efficient image processing
- **Database Indexing**: Optimized query performance

## 🔧 Deployment Options

### Docker Deployment
```bash
cd deployment
docker-compose up --build
```

### Production Deployment
1. Set production environment variables
2. Configure PostgreSQL database
3. Set up NGINX reverse proxy
4. Enable SSL/TLS certificates
5. Configure monitoring and logging

## 🚨 Troubleshooting

### Common Issues

**Backend not starting:**
```bash
# Check Python version (3.8+ required)
python --version

# Install missing dependencies
pip install -r requirements.txt

# Check database permissions
ls -la rockfall_system.db
```

**Frontend not loading:**
```bash
# Check Node.js version (16+ required)
node --version

# Clear npm cache
npm cache clean --force
cd frontend && npm install
```

**Email not working:**
- Verify SMTP credentials in `.env`
- Check firewall settings
- Test with Gmail app passwords

**Soil classification errors:**
- Ensure TensorFlow is installed: `pip install tensorflow`
- Check image format (JPG, PNG supported)
- Verify upload directory permissions

### Debug Mode
Enable debug logging by setting `FLASK_ENV=development` in `.env`

## 📚 Documentation

### API Documentation
Visit `http://localhost:5000/api/docs` (when running) for interactive API documentation.

### Component Documentation
Each React component includes inline documentation and prop definitions.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- TensorFlow team for deep learning capabilities
- React and Flask communities for excellent frameworks
- Chart.js for beautiful data visualizations
- Leaflet for mapping functionality

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Run the test suite: `python test_enhanced_system.py`
3. Check logs in the `logs/` directory
4. Open an issue on GitHub

---

### 🎯 Version 2.0 Achievement Summary

✅ **AI-Powered Soil Classification** - Deep learning CNN for 4 soil types  
✅ **Email Reporting System** - PDF reports with SMTP integration  
✅ **3D Holographic Visualization** - Interactive mine visualization  
✅ **Enhanced Dashboard** - India mining data and advanced analytics  
✅ **Modern UI Design** - Holographic theme with animations  
✅ **Comprehensive API** - 20+ endpoints covering all features  
✅ **Robust Testing** - Comprehensive test suite  
✅ **Production Ready** - Docker deployment and scaling  

**System is now fully functional with all requested features implemented! 🚀**