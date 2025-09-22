#!/usr/bin/env python3
"""
Simplified AI Rockfall Prediction System Backend
Starts with core functionality only
"""

from flask import Flask, request, jsonify, session
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import JWTManager, jwt_required, create_access_token, get_jwt_identity
from werkzeug.security import generate_password_hash, check_password_hash
import os
import re
import dns.resolver
from datetime import datetime, timedelta
import json
import smtplib
import random
import numpy as np
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from werkzeug.utils import secure_filename
import base64
import requests

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-change-in-production'
app.config['JWT_SECRET_KEY'] = 'jwt-secret-change-in-production'
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=24)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///rockfall_system.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize extensions
db = SQLAlchemy(app)
jwt = JWTManager(app)
CORS(app, origins=['http://localhost:3000'], supports_credentials=True)

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    first_name = db.Column(db.String(50), nullable=False)
    last_name = db.Column(db.String(50), nullable=False)
    company = db.Column(db.String(100))
    role = db.Column(db.String(20), default='user')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)

class SensorData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    sensor_id = db.Column(db.String(50), nullable=False)
    sensor_type = db.Column(db.String(50), nullable=False)
    location = db.Column(db.String(100))
    value = db.Column(db.Float, nullable=False)
    unit = db.Column(db.String(20))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    latitude = db.Column(db.Float)
    longitude = db.Column(db.Float)

class RiskAssessment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    location = db.Column(db.String(100), nullable=False)
    risk_level = db.Column(db.String(20), nullable=False)
    probability = db.Column(db.Float, nullable=False)
    factors = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    latitude = db.Column(db.Float)
    longitude = db.Column(db.Float)

class Alert(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    alert_type = db.Column(db.String(50), nullable=False)
    message = db.Column(db.Text, nullable=False)
    risk_level = db.Column(db.String(20), nullable=False)
    location = db.Column(db.String(100))
    latitude = db.Column(db.Float)
    longitude = db.Column(db.Float)
    status = db.Column(db.String(20), default='active')
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))

# Validation Functions
def validate_email_domain(email):
    try:
        domain = email.split('@')[1]
        dns.resolver.resolve(domain, 'MX')
        return True
    except:
        try:
            dns.resolver.resolve(domain, 'A')
            return True
        except:
            return False

def validate_password_strength(password):
    errors = []
    if len(password) < 8:
        errors.append("Password must be at least 8 characters long")
    if not re.search(r"[A-Z]", password):
        errors.append("Password must contain at least one uppercase letter")
    if not re.search(r"[a-z]", password):
        errors.append("Password must contain at least one lowercase letter")
    if not re.search(r"\d", password):
        errors.append("Password must contain at least one number")
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        errors.append("Password must contain at least one special character")
    return errors

def validate_name(name, field_name):
    errors = []
    if len(name) < 2:
        errors.append(f"{field_name} must be at least 2 characters long")
    if len(name) > 50:
        errors.append(f"{field_name} must be less than 50 characters long")
    if not re.match(r"^[a-zA-Z\s\-']+$", name):
        errors.append(f"{field_name} can only contain letters, spaces, hyphens, and apostrophes")
    return errors

# Email Configuration
def get_email_config():
    """Get email configuration for sending reports"""
    return {
        'smtp_server': 'smtp.gmail.com',
        'smtp_port': 587,
        'email': 'rockfall.alerts.system@gmail.com',  # Default system email
        'password': 'your-app-password'  # You would set this in production
    }

def send_risk_report_email(user_email, risk_data, alert_data=None):
    """Send risk analysis report via email"""
    try:
        config = get_email_config()
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = config['email']
        msg['To'] = user_email
        msg['Subject'] = 'AI Rockfall Risk Analysis Report'
        
        # Create HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                .container {{ background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
                .header {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; margin-bottom: 20px; }}
                .risk-high {{ color: #e74c3c; font-weight: bold; }}
                .risk-medium {{ color: #f39c12; font-weight: bold; }}
                .risk-low {{ color: #27ae60; font-weight: bold; }}
                .risk-critical {{ color: #8e44ad; font-weight: bold; }}
                .section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #3498db; background-color: #f8f9fa; }}
                .timestamp {{ color: #7f8c8d; font-size: 0.9em; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🔍 AI Rockfall Risk Analysis Report</h1>
                    <p class="timestamp">Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <div class="section">
                    <h3>📊 Risk Assessment Summary</h3>
                    <p><strong>Location:</strong> {risk_data.get('location', 'Unknown')}</p>
                    <p><strong>Risk Level:</strong> <span class="risk-{risk_data.get('risk_level', 'unknown').lower()}">{risk_data.get('risk_level', 'UNKNOWN')}</span></p>
                    <p><strong>Probability:</strong> {risk_data.get('probability', 0):.1%}</p>
                    <p><strong>Coordinates:</strong> {risk_data.get('latitude', 'N/A')}, {risk_data.get('longitude', 'N/A')}</p>
                </div>
                
                <div class="section">
                    <h3>⚠️ Risk Factors</h3>
                    <ul>
        """
        
        # Add risk factors
        factors = risk_data.get('factors', [])
        if isinstance(factors, str):
            factors = json.loads(factors) if factors.startswith('[') else [factors]
        
        for factor in factors:
            html_content += f"<li>{factor}</li>"
        
        html_content += """
                    </ul>
                </div>
        """
        
        # Add alert information if provided
        if alert_data:
            html_content += f"""
                <div class="section">
                    <h3>🚨 Alert Information</h3>
                    <p><strong>Alert Type:</strong> {alert_data.get('alert_type', 'Risk Assessment')}</p>
                    <p><strong>Message:</strong> {alert_data.get('message', 'Risk level updated')}</p>
                    <p><strong>Status:</strong> {alert_data.get('status', 'Active').upper()}</p>
                </div>
            """
        
        html_content += """
                <div class="section">
                    <h3>📋 Recommendations</h3>
                    <ul>
                        <li>Monitor sensor readings continuously</li>
                        <li>Implement safety protocols based on risk level</li>
                        <li>Consider evacuation procedures if risk level is CRITICAL</li>
                        <li>Regular geological assessments recommended</li>
                    </ul>
                </div>
                
                <div class="section">
                    <p><em>This report was automatically generated by the AI Rockfall Prediction System.</em></p>
                    <p><em>For immediate assistance, contact emergency services.</em></p>
                </div>
            </div>
        </body>
        </html>
        """
        
        msg.attach(MIMEText(html_content, 'html'))
        
        # For demo purposes, we'll just log the email instead of actually sending
        print(f"📧 EMAIL REPORT SENT TO: {user_email}")
        print(f"📋 SUBJECT: AI Rockfall Risk Analysis Report")
        print(f"📄 CONTENT: Risk Level {risk_data.get('risk_level')} at {risk_data.get('location')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Email sending failed: {str(e)}")
        return False

def get_wikipedia_info(rock_type):
    """Get Wikipedia information about rock type"""
    try:
        # Wikipedia API endpoint
        wiki_url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + rock_type.replace(' ', '_')
        
        response = requests.get(wiki_url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return {
                'title': data.get('title', rock_type),
                'description': data.get('extract', f'{rock_type} is a type of rock formation.'),
                'url': data.get('content_urls', {}).get('desktop', {}).get('page', ''),
                'thumbnail': data.get('thumbnail', {}).get('source', '') if data.get('thumbnail') else ''
            }
        else:
            return get_fallback_rock_info(rock_type)
    except Exception as e:
        return get_fallback_rock_info(rock_type)

def get_fallback_rock_info(rock_type):
    """Fallback rock information when Wikipedia is not available"""
    rock_info = {
        'Granite': {
            'description': 'Granite is a light-colored igneous rock with grains large enough to be visible with the unaided eye. It forms from the slow crystallization of magma below Earth\'s surface.',
            'properties': ['High strength', 'Durable', 'Weather resistant'],
            'uses': ['Construction', 'Monuments', 'Countertops']
        },
        'Limestone': {
            'description': 'Limestone is a carbonate sedimentary rock that is often composed of the skeletal fragments of marine organisms such as coral, foraminifera, and molluscs.',
            'properties': ['Moderate strength', 'Soluble in acid', 'Fossil-bearing'],
            'uses': ['Building stone', 'Cement production', 'Road aggregate']
        },
        'Sandstone': {
            'description': 'Sandstone is a clastic sedimentary rock composed mainly of sand-sized minerals or rock grains, most commonly quartz.',
            'properties': ['Variable strength', 'Porous', 'Layered structure'],
            'uses': ['Building material', 'Paving', 'Filtration']
        },
        'Shale': {
            'description': 'Shale is a fine-grained sedimentary rock formed from mud that is a mix of flakes of clay minerals and tiny fragments of other minerals.',
            'properties': ['Low strength', 'Fissile', 'Impermeable'],
            'uses': ['Oil shale', 'Brick making', 'Cement production']
        }
    }
    
    info = rock_info.get(rock_type, {
        'description': f'{rock_type} is a geological formation with specific properties important for mining operations.',
        'properties': ['Variable properties'],
        'uses': ['Various applications']
    })
    
    return {
        'title': rock_type,
        'description': info['description'],
        'properties': info.get('properties', []),
        'uses': info.get('uses', []),
        'url': f'https://en.wikipedia.org/wiki/{rock_type.replace(" ", "_")}',
        'thumbnail': ''
    }

def classify_rock_image(image_data):
    """Simulate rock/geological image classification with Wikipedia integration"""
    try:
        # Simulate AI classification with random results for demo
        classifications = [
            {'type': 'Granite', 'stability': 'Stable', 'risk': 'LOW', 'confidence': 0.92},
            {'type': 'Limestone', 'stability': 'Moderate', 'risk': 'MEDIUM', 'confidence': 0.85},
            {'type': 'Sandstone', 'stability': 'Unstable', 'risk': 'HIGH', 'confidence': 0.78},
            {'type': 'Shale', 'stability': 'Very Unstable', 'risk': 'CRITICAL', 'confidence': 0.89},
            {'type': 'Basalt', 'stability': 'Very Stable', 'risk': 'LOW', 'confidence': 0.94},
            {'type': 'Quartzite', 'stability': 'Very Stable', 'risk': 'LOW', 'confidence': 0.91}
        ]
        
        # Random selection for demo
        result = random.choice(classifications)
        
        # Get Wikipedia information
        wiki_info = get_wikipedia_info(result['type'])
        
        return {
            'success': True,
            'predicted_class': result['type'],
            'confidence': int(result['confidence'] * 100),
            'classification': result,
            'analysis': {
                'rock_type': result['type'],
                'stability_assessment': result['stability'],
                'risk_level': result['risk'],
                'confidence_score': result['confidence'],
                'recommendations': f"Monitor {result['type']} formation for stability changes"
            },
            'wikipedia_info': wiki_info,
            'geological_details': {
                'formation_type': 'Sedimentary' if result['type'] in ['Limestone', 'Sandstone', 'Shale'] else 'Igneous',
                'hardness': 'High' if result['risk'] == 'LOW' else 'Medium' if result['risk'] == 'MEDIUM' else 'Low',
                'porosity': 'Low' if result['type'] in ['Granite', 'Basalt'] else 'Medium' if result['type'] == 'Limestone' else 'High',
                'mining_suitability': 'Excellent' if result['risk'] == 'LOW' else 'Good' if result['risk'] == 'MEDIUM' else 'Poor'
            }
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def generate_risk_assessment(latitude, longitude, location_name="Unknown Location"):
    """Generate risk assessment for given coordinates"""
    try:
        # Simulate risk calculation based on location
        base_risk = random.uniform(0.1, 0.9)
        
        # Determine risk level
        if base_risk >= 0.8:
            risk_level = "CRITICAL"
        elif base_risk >= 0.6:
            risk_level = "HIGH"
        elif base_risk >= 0.4:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        factors = []
        if base_risk >= 0.7:
            factors.extend(["High displacement detected", "Recent seismic activity"])
        if base_risk >= 0.5:
            factors.extend(["Elevated strain levels", "Weather conditions"])
        if base_risk >= 0.3:
            factors.extend(["Minor geological changes"])
        else:
            factors.append("Normal conditions")
        
        return {
            'location': location_name,
            'latitude': latitude,
            'longitude': longitude,
            'risk_level': risk_level,
            'probability': base_risk,
            'factors': factors,
            'timestamp': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        return None

# Routes
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'timestamp': datetime.utcnow().isoformat()})

@app.route('/api/auth/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        
        # Validation
        errors = []
        
        # Email validation
        email = data.get('email', '').strip().lower()
        if not email:
            errors.append("Email is required")
        elif not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
            errors.append("Invalid email format")
        elif not validate_email_domain(email):
            errors.append("Email domain does not exist")
        elif User.query.filter_by(email=email).first():
            errors.append("Email already exists")
        
        # Password validation
        password = data.get('password', '')
        password_errors = validate_password_strength(password)
        errors.extend(password_errors)
        
        # Name validation
        first_name = data.get('first_name', '').strip()
        last_name = data.get('last_name', '').strip()
        errors.extend(validate_name(first_name, "First name"))
        errors.extend(validate_name(last_name, "Last name"))
        
        if errors:
            return jsonify({'error': 'Validation failed', 'details': errors}), 400
        
        # Create user
        user = User(
            email=email,
            password_hash=generate_password_hash(password),
            first_name=first_name,
            last_name=last_name,
            company=data.get('company', '').strip(),
            role=data.get('role', 'user')
        )
        
        db.session.add(user)
        db.session.commit()
        
        return jsonify({'message': 'User registered successfully'}), 201
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/auth/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        
        user = User.query.filter_by(email=email).first()
        
        if user and check_password_hash(user.password_hash, password):
            access_token = create_access_token(identity=user.id)
            return jsonify({
                'access_token': access_token,
                'user': {
                    'id': user.id,
                    'email': user.email,
                    'first_name': user.first_name,
                    'last_name': user.last_name,
                    'role': user.role
                }
            })
        
        return jsonify({'error': 'Invalid credentials'}), 401
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/sensor-data', methods=['GET', 'POST'])
def handle_sensor_data():
    if request.method == 'POST':
        try:
            data = request.get_json()
            sensor_data = SensorData(
                sensor_id=data['sensor_id'],
                sensor_type=data['sensor_type'],
                location=data.get('location'),
                value=data['value'],
                unit=data.get('unit'),
                latitude=data.get('latitude'),
                longitude=data.get('longitude')
            )
            db.session.add(sensor_data)
            db.session.commit()
            return jsonify({'message': 'Sensor data recorded'}), 201
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    # GET request
    sensors = SensorData.query.order_by(SensorData.timestamp.desc()).limit(100).all()
    return jsonify([{
        'id': s.id,
        'sensor_id': s.sensor_id,
        'sensor_type': s.sensor_type,
        'location': s.location,
        'value': s.value,
        'unit': s.unit,
        'timestamp': s.timestamp.isoformat(),
        'latitude': s.latitude,
        'longitude': s.longitude
    } for s in sensors])

@app.route('/api/risk-assessment', methods=['GET'])
def get_risk_assessment():
    # Get location parameters if provided
    lat = request.args.get('lat', type=float)
    lng = request.args.get('lng', type=float)
    
    # Default risk zones for London if no coordinates provided
    if lat is None or lng is None:
        lat, lng = 51.5074, -0.1278  # London
    
    # Generate risk zones around the specified location
    risk_zones = []
    risk_levels = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
    colors = ['#28a745', '#ffc107', '#fd7e14', '#dc3545']
    
    for i in range(8):  # Generate 8 risk zones
        # Create zones around the center point
        lat_offset = random.uniform(-0.02, 0.02)  # ~2km radius
        lng_offset = random.uniform(-0.02, 0.02)
        
        zone_lat = lat + lat_offset
        zone_lng = lng + lng_offset
        
        risk_level = random.choice(risk_levels)
        probability = {
            'LOW': random.uniform(0.1, 0.3),
            'MEDIUM': random.uniform(0.3, 0.6),
            'HIGH': random.uniform(0.6, 0.8),
            'CRITICAL': random.uniform(0.8, 0.95)
        }[risk_level]
        
        risk_zones.append({
            'id': i + 1,
            'location': f'Zone {i + 1}',
            'risk_level': risk_level,
            'probability': round(probability, 2),
            'latitude': round(zone_lat, 6),
            'longitude': round(zone_lng, 6),
            'factors': get_risk_factors(risk_level),
            'color': colors[risk_levels.index(risk_level)]
        })
    
    return jsonify(risk_zones)

def get_risk_factors(risk_level):
    """Get risk factors based on risk level"""
    factors_map = {
        'LOW': ['Stable geological conditions', 'Normal weather patterns'],
        'MEDIUM': ['Moderate displacement detected', 'Recent temperature changes'],
        'HIGH': ['High displacement readings', 'Heavy rainfall detected', 'Increased seismic activity'],
        'CRITICAL': ['Critical displacement levels', 'Extreme weather conditions', 'Multiple sensor alerts']
    }
    return factors_map.get(risk_level, ['Unknown factors'])

@app.route('/api/risk-map', methods=['GET'])
def get_risk_map():
    """Get risk map data for visualization"""
    try:
        # Generate sample risk map data
        risk_points = []
        
        # Base coordinates around a mining area
        base_lat = 51.5074
        base_lng = -0.1278
        
        for i in range(20):
            lat_offset = random.uniform(-0.01, 0.01)
            lng_offset = random.uniform(-0.01, 0.01)
            risk_level = random.choice(['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'])
            probability = random.uniform(0.1, 0.9)
            
            risk_points.append({
                'id': i + 1,
                'latitude': base_lat + lat_offset,
                'longitude': base_lng + lng_offset,
                'risk_level': risk_level,
                'probability': probability,
                'location': f'Zone {i + 1}',
                'factors': ['Geological analysis', 'Sensor data']
            })
        
        return jsonify({
            'success': True,
            'data': risk_points,
            'center': {'latitude': base_lat, 'longitude': base_lng}
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/location-search', methods=['POST'])
def search_location():
    """Search for location coordinates using OpenStreetMap Nominatim API"""
    try:
        data = request.get_json()
        location_name = data.get('location', '').strip()
        
        if not location_name:
            return jsonify({'error': 'Location name is required'}), 400
        
        # Use OpenStreetMap Nominatim API for geocoding
        import urllib.parse
        import urllib.request
        
        # URL encode the location name
        encoded_location = urllib.parse.quote(location_name)
        nominatim_url = f"https://nominatim.openstreetmap.org/search?format=json&q={encoded_location}&limit=5&addressdetails=1"
        
        try:
            # Make request to Nominatim API
            with urllib.request.urlopen(nominatim_url) as response:
                data = json.loads(response.read().decode())
            
            # Format results for our frontend
            results = []
            for item in data:
                result = {
                    'name': item.get('display_name', location_name),
                    'lat': float(item.get('lat', 0)),
                    'lng': float(item.get('lon', 0))
                }
                results.append(result)
            
            # If we have results, return them
            if results:
                return jsonify({
                    'success': True,
                    'results': results
                })
        
        except Exception as api_error:
            print(f"Nominatim API error: {api_error}")
        
        # Fallback to mock data if API fails
        mock_locations = {
            'london': [
                {'name': 'London, UK', 'lat': 51.5074, 'lng': -0.1278},
                {'name': 'London, Ontario, Canada', 'lat': 42.9849, 'lng': -81.2453}
            ],
            'new york': [
                {'name': 'New York City, NY, USA', 'lat': 40.7128, 'lng': -74.0060}
            ],
            'paris': [
                {'name': 'Paris, France', 'lat': 48.8566, 'lng': 2.3522}
            ],
            'tokyo': [
                {'name': 'Tokyo, Japan', 'lat': 35.6762, 'lng': 139.6503}
            ],
            'mumbai': [
                {'name': 'Mumbai, India', 'lat': 19.0760, 'lng': 72.8777}
            ],
            'delhi': [
                {'name': 'Delhi, India', 'lat': 28.7041, 'lng': 77.1025}
            ],
            'sydney': [
                {'name': 'Sydney, Australia', 'lat': -33.8688, 'lng': 151.2093}
            ],
            'berlin': [
                {'name': 'Berlin, Germany', 'lat': 52.5200, 'lng': 13.4050}
            ]
        }
        
        # Search for location in mock data
        search_key = location_name.lower()
        results = []
        
        for key, locations in mock_locations.items():
            if search_key in key or key in search_key:
                results.extend(locations)
        
        # If still no results, try partial matches
        if not results:
            for key, locations in mock_locations.items():
                if any(word in key for word in search_key.split()) or any(word in search_key for word in key.split()):
                    results.extend(locations)
        
        # Return whatever we found
        if results:
            return jsonify({
                'success': True,
                'results': results
            })
        else:
            return jsonify({
                'success': False,
                'error': f'No results found for "{location_name}". Try searching for major cities like London, New York, Paris, etc.'
            })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/classify-image', methods=['POST'])
def classify_image():
    """Classify uploaded rock/geological image"""
    try:
        # Check if image was uploaded
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Save uploaded file
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Classify the image
            classification_result = classify_rock_image(filepath)
            
            if classification_result['success']:
                return jsonify({
                    'success': True,
                    'classification': classification_result['classification'],
                    'analysis': classification_result['analysis']
                })
            else:
                return jsonify({'error': 'Classification failed'}), 500
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/send-alert', methods=['POST'])
def send_alert():
    """Send risk alert and email report"""
    try:
        data = request.get_json()
        
        # Get or generate risk assessment
        risk_data = data.get('risk_data')
        if not risk_data:
            # Generate risk assessment for current location
            latitude = data.get('latitude', 51.5074)
            longitude = data.get('longitude', -0.1278)
            location = data.get('location', 'Selected Location')
            
            risk_data = generate_risk_assessment(latitude, longitude, location)
        
        if not risk_data:
            return jsonify({'error': 'Could not generate risk assessment'}), 500
        
        # Try to get current user if authenticated
        user_email = 'demo@rockfall.com'  # Default email for demo
        current_user_id = None
        
        try:
            # Try to get authenticated user
            current_user_id = get_jwt_identity() if 'Authorization' in request.headers else None
            if current_user_id:
                user = User.query.get(current_user_id)
                if user:
                    user_email = user.email
        except:
            pass  # Continue with demo email if authentication fails
        
        # Create alert record
        alert = Alert(
            alert_type='RISK_ASSESSMENT',
            message=f"Risk level {risk_data['risk_level']} detected at {risk_data['location']}",
            risk_level=risk_data['risk_level'],
            location=risk_data['location'],
            latitude=risk_data.get('latitude'),
            longitude=risk_data.get('longitude'),
            user_id=current_user_id
        )
        
        db.session.add(alert)
        db.session.commit()
        
        # Send email report
        alert_data = {
            'alert_type': alert.alert_type,
            'message': alert.message,
            'status': alert.status
        }
        
        email_sent = send_risk_report_email(user_email, risk_data, alert_data)
        
        return jsonify({
            'success': True,
            'message': 'Alert sent and email report generated',
            'alert_id': alert.id,
            'email_sent': email_sent,
            'risk_data': risk_data
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    """Get recent alerts"""
    try:
        alerts = Alert.query.order_by(Alert.timestamp.desc()).limit(10).all()
        
        return jsonify([{
            'id': alert.id,
            'alert_type': alert.alert_type,
            'message': alert.message,
            'risk_level': alert.risk_level,
            'location': alert.location,
            'latitude': alert.latitude,
            'longitude': alert.longitude,
            'status': alert.status,
            'timestamp': alert.timestamp.isoformat()
        } for alert in alerts])
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    
    print("🚀 Starting AI Rockfall Prediction System Backend...")
    print("📊 Dashboard: http://localhost:3000")
    print("🔌 API: http://localhost:5000")
    print("❤️  Health Check: http://localhost:5000/api/health")
    
    app.run(debug=True, host='0.0.0.0', port=5000)