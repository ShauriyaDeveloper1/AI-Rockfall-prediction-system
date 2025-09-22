# AI Rockfall Prediction System - Copilot Instructions

## System Architecture Overview

This is a **multi-service AI system** for predicting rockfall incidents in open-pit mines with real-time monitoring, ML predictions, and alert management.

### Core Components
- **Frontend**: React dashboard (`frontend/`) with real-time visualization using Leaflet maps and Chart.js
- **Backend**: Flask API (`backend/app.py`) serving REST endpoints and managing SQLite/PostgreSQL database
- **ML Engine**: scikit-learn models (`ml_models/rockfall_predictor.py`) with Random Forest + Gradient Boosting
- **Sensor Simulation**: Realistic data generator (`data_processing/sensor_simulator.py`) mimicking geotechnical sensors
- **Alert System**: Multi-channel notifications (`backend/alert_service.py`) via Twilio SMS and SendGrid email

### Critical Data Flow
1. Sensor data → Backend API (`/api/sensor-data` POST) → Database storage
2. ML predictions triggered every 30 seconds via `/api/risk-assessment` → Real-time risk calculations
3. Alert escalation based on risk levels (LOW/MEDIUM/HIGH/CRITICAL) → Contact hierarchy in `config/emergency_contacts.json`

## Development Workflows

### System Startup
**Use batch scripts on Windows:**
- `start_complete_system.bat` - Full system with sensor simulation (recommended for development)
- `run_system.py --with-simulator` - Python alternative with dependency checking

**Manual startup order:**
1. Backend: `cd backend && python app.py` (port 5000)
2. Frontend: `cd frontend && npm start` (port 3000)  
3. Sensor sim: `python start_simulator.py` (optional but recommended)

### Database Management
- Models defined inline in `backend/app.py` (SensorData, RiskAssessment, Alert)
- Auto-initialization on first run via `db.create_all()`
- SQLite default (`rockfall_system.db`), PostgreSQL via `DATABASE_URL` env var

### ML Model Training
- Models auto-train with synthetic data on first run if no saved models exist
- Saved to: `ml_models/rockfall_classifier.pkl`, `ml_models/rockfall_regressor.pkl`, `ml_models/scaler.pkl`
- Retrain manually: delete .pkl files and restart backend

## Project-Specific Patterns

### Sensor Data Structure
```python
# Standard sensor types with specific units and ranges
sensor_types = {
    'displacement': {'unit': 'mm', 'normal_range': [0.1, 2.0]},
    'strain': {'unit': 'microstrain', 'normal_range': [50, 200]},
    'pore_pressure': {'unit': 'kPa', 'normal_range': [20, 100]},
    'temperature': {'unit': 'celsius', 'normal_range': [-10, 40]}
}
```

### Alert Escalation Logic
Risk levels trigger different contact groups (see `config/emergency_contacts.json`):
- **CRITICAL** (>80% probability): All contacts immediately
- **HIGH** (60-80%): Primary + secondary + technical contacts
- **MEDIUM** (40-60%): Primary + secondary contacts
- **LOW** (<40%): Primary contact only

### API Endpoint Conventions
- All endpoints prefixed with `/api/`
- RESTful patterns: GET for data retrieval, POST for creation
- Consistent error responses with `{"error": "message", "status": "error"}` format
- Real-time updates via polling (no WebSockets currently implemented)

### Frontend Component Structure
- Route-based navigation using React Router
- Bootstrap 5 for consistent styling
- Leaflet for interactive risk mapping with heat layers
- Chart.js for time-series sensor data visualization
- Components fetch data independently via axios calls to backend API

## Configuration Management

### Environment Variables
Set in `.env` file (create if missing):
```
DATABASE_URL=sqlite:///rockfall_system.db
TWILIO_ACCOUNT_SID=your_sid
TWILIO_AUTH_TOKEN=your_token
SENDGRID_API_KEY=your_key
```

### Contact Management
Edit `config/emergency_contacts.json` for production deployments. Structure includes role-based escalation and department categorization.

## Testing & Debugging

### Health Checks
- Backend health: `GET /api/health`
- Frontend connectivity: Check console for API call errors
- Sensor simulation: Monitor `POST /api/sensor-data` frequency (every 10-30 seconds)

### Common Issues
- **Models not loading**: Check if .pkl files exist in `ml_models/`, restart backend to retrain
- **No sensor data**: Verify sensor simulator is running and backend is accessible
- **Alert failures**: Check Twilio/SendGrid credentials in environment variables
- **Database errors**: Delete `rockfall_system.db` for fresh start with auto-recreation

## Integration Points

### External Services
- **Twilio**: SMS alerts via REST API
- **SendGrid**: Email notifications via Web API
- **Optional**: Weather APIs for enhanced predictions (not currently implemented)

### Docker Deployment
Use `deployment/docker-compose.yml` for production:
```bash
docker-compose up --build
```
Includes separate containers for frontend, backend, and PostgreSQL database.