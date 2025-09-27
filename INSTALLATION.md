# Installation Guide - AI-Based Rockfall Prediction System

## Quick Start

### Prerequisites
- Python 3.8 or higher
- Node.js 16 or higher
- Git (optional)

### Automated Setup
```bash
# 1. Clone or download the project
git clone <repository-url>
cd rockfall-prediction-system

# 2. Run automated setup
python scripts/setup.py

# 3. Start the system
python run_system.py --with-simulator
```

### Manual Setup

#### 1. Python Environment Setup
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Unix/Linux/macOS:
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

#### 2. Frontend Setup
```bash
cd frontend
npm install
cd ..
```

#### 3. Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your settings
# Update database URL, API keys, etc.
```

#### 4. Database Initialization
```bash
# Initialize database tables
python -c "
import sys
sys.path.append('backend')
from app import app, db
with app.app_context():
    db.create_all()
    print('Database initialized')
"
```

#### 5. ML Models Setup
```bash
# Initialize ML models
python -c "
import sys
sys.path.append('ml_models')
from rockfall_predictor import RockfallPredictor
predictor = RockfallPredictor()
print('ML models initialized')
"
```

## Running the System

### Option 1: Using the Runner Script (Recommended)
```bash
# Start complete system with sensor simulator
python run_system.py --with-simulator

# Start without simulator
python run_system.py
```

### Option 2: Manual Start
```bash
# Terminal 1: Start backend
python backend/app.py

# Terminal 2: Start frontend
cd frontend
npm start

# Terminal 3: Start sensor simulator (optional)
python data_processing/sensor_simulator.py
```

### Option 3: Docker Deployment
```bash
# Build and start with Docker Compose
cd deployment
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Access Points

- **Dashboard**: http://localhost:3000
- **API**: http://localhost:5000
- **API Health Check**: http://localhost:5000/api/health

## Configuration

### Environment Variables (.env)
```bash
# Database
DATABASE_URL=sqlite:///rockfall_system.db

# Twilio (SMS Alerts)
TWILIO_ACCOUNT_SID=your_account_sid
TWILIO_AUTH_TOKEN=your_auth_token
TWILIO_PHONE_NUMBER=+1234567890

# SendGrid (Email Alerts)
SENDGRID_API_KEY=your_sendgrid_api_key
FROM_EMAIL=alerts@yourcompany.com

# Flask
FLASK_ENV=development
SECRET_KEY=your_secret_key
```

### Emergency Contacts (config/emergency_contacts.json)
```json
{
  "emergency_contacts": [
    {
      "name": "Safety Manager",
      "phone": "+1234567890",
      "email": "safety@mine.com",
      "role": "primary"
    }
  ]
}
```

## Testing the System

### 1. Health Check
```bash
curl http://localhost:5000/api/health
```

### 2. Send Test Sensor Data
```bash
curl -X POST http://localhost:5000/api/sensor-data \
  -H "Content-Type: application/json" \
  -d '{
    "sensor_id": "TEST-001",
    "sensor_type": "displacement",
    "location_x": -23.5505,
    "location_y": -46.6333,
    "value": 1.5,
    "unit": "mm",
    "timestamp": "2024-01-01T12:00:00Z"
  }'
```

### 3. Generate Test Alert
```bash
# Use the sensor simulator to generate emergency scenario
python data_processing/sensor_simulator.py
``'
```

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend API   │    │   ML Models     │
│   (React)       │◄──►│   (Flask)       │◄──►│   (sklearn)     │
│   Port: 3000    │    │   Port: 5000    │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       ▼                       │
         │              ┌─────────────────┐              │
         │              │   Database      │              │
         │              │   (SQLite/      │              │
         │              │   PostgreSQL)   │              │
         │              └─────────────────┘              │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Sensor Data   │    │   Alert System  │    │   Monitoring    │
│   Simulator     │    │   (SMS/Email)   │    │   & Logging     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review system logs in the `logs/` directory
3. Check API health endpoint
4. Verify all dependencies are installed correctly

## Next Steps

After successful installation:
1. Configure your specific mine coordinates in the risk map
2. Set up real sensor integrations
3. Configure alert contacts and escalation rules
4. Train ML models with historical data
5. Set up production monitoring and backups