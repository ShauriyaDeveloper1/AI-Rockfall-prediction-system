"""
AI Rockfall Prediction System - Streamlit Dashboard
Complete web application with LIDAR 3D visualization, LSTM AI, and automatic model training
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import time
import os
from pathlib import Path
import base64
from io import BytesIO
import zipfile

# Configure Streamlit page
st.set_page_config(
    page_title="AI Rockfall Prediction System",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    color: #1f4e7a;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
}
.danger-alert {
    background-color: #f8d7da;
    border: 1px solid #f5c6cb;
    color: #721c24;
    padding: 1rem;
    border-radius: 5px;
    margin: 1rem 0;
}
.success-alert {
    background-color: #d4edda;
    border: 1px solid #c3e6cb;
    color: #155724;
    padding: 1rem;
    border-radius: 5px;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# Backend API configuration
API_BASE_URL = "http://localhost:5000/api"

class RockfallDashboard:
    def __init__(self):
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'selected_page' not in st.session_state:
            st.session_state.selected_page = "Dashboard"
        if 'auto_refresh' not in st.session_state:
            st.session_state.auto_refresh = False
        if 'last_update' not in st.session_state:
            st.session_state.last_update = datetime.now()
    
    def make_api_request(self, endpoint, method="GET", data=None, files=None):
        """Make API request to backend"""
        try:
            url = f"{API_BASE_URL}/{endpoint}"
            if method == "GET":
                response = requests.get(url, timeout=10)
            elif method == "POST":
                if files:
                    response = requests.post(url, data=data, files=files, timeout=30)
                else:
                    response = requests.post(url, json=data, timeout=30)
            
            if response.status_code == 200:
                return response.json(), None
            else:
                return None, f"API Error: {response.status_code} - {response.text}"
        except Exception as e:
            return None, f"Connection Error: {str(e)}"
    
    def render_sidebar(self):
        """Render sidebar navigation"""
        with st.sidebar:
            st.image("https://via.placeholder.com/150x80/1f4e7a/ffffff?text=Rockfall+AI", 
                    caption="AI Rockfall Prediction System")
            
            st.markdown("### 🎛️ Navigation")
            pages = [
                ("🏠 Dashboard", "Dashboard"),
                ("🤖 LSTM AI Predictions", "LSTM"),
                ("🌍 LIDAR 3D Visualization", "LIDAR"),
                ("📊 Sensor Data", "Sensors"),
                ("⚠️ Alert Management", "Alerts"),
                ("📈 Risk Assessment", "Risk"),
                ("⚙️ System Settings", "Settings")
            ]
            
            for page_name, page_key in pages:
                if st.button(page_name, key=page_key, use_container_width=True):
                    st.session_state.selected_page = page_key
            
            st.markdown("---")
            st.markdown("### ⚡ Auto Refresh")
            st.session_state.auto_refresh = st.checkbox("Enable Auto Refresh (30s)")
            
            if st.session_state.auto_refresh:
                st.markdown(f"Last Update: {st.session_state.last_update.strftime('%H:%M:%S')}")
                time.sleep(1)
                st.rerun()
    
    def render_dashboard(self):
        """Render main dashboard page"""
        st.markdown('<div class="main-header">🏔️ AI Rockfall Prediction System</div>', 
                   unsafe_allow_html=True)
        
        # Get risk assessment data
        risk_data, error = self.make_api_request("risk-assessment")
        if error:
            st.error(f"Failed to load risk data: {error}")
            return
        
        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_risk = risk_data.get('risk_level', 'UNKNOWN')
            risk_color = {
                'LOW': '#28a745', 'MEDIUM': '#ffc107', 
                'HIGH': '#fd7e14', 'CRITICAL': '#dc3545'
            }.get(current_risk, '#6c757d')
            
            st.markdown(f"""
            <div style="background:{risk_color};padding:1rem;border-radius:10px;color:white;text-align:center;">
                <h3>Current Risk Level</h3>
                <h2>{current_risk}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            probability = risk_data.get('probability', 0) * 100
            st.metric("Risk Probability", f"{probability:.1f}%", 
                     delta=f"{probability-50:.1f}%" if probability > 50 else None)
        
        with col3:
            st.metric("Active Sensors", "24", delta="2")
        
        with col4:
            st.metric("System Status", "Online", delta="Healthy")
        
        # Real-time charts
        st.markdown("### 📈 Real-Time Monitoring")
        
        col1, col2 = st.columns(2)
        
        with col1:
            self.render_risk_gauge(probability)
        
        with col2:
            self.render_sensor_trends()
        
        # Recent alerts
        st.markdown("### ⚠️ Recent Alerts")
        alerts_data, error = self.make_api_request("alerts")
        if not error and alerts_data:
            self.render_alerts_table(alerts_data)
    
    def render_risk_gauge(self, probability):
        """Render risk probability gauge"""
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = probability,
            title = {'text': "Risk Probability (%)"},
            domain = {'x': [0, 1], 'y': [0, 1]},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 25], 'color': "lightgray"},
                    {'range': [25, 50], 'color': "yellow"},
                    {'range': [50, 75], 'color': "orange"},
                    {'range': [75, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    def render_sensor_trends(self):
        """Render sensor data trends"""
        sensor_data, error = self.make_api_request("sensor-data")
        if error:
            st.error("Failed to load sensor data")
            return
        
        # Generate sample trend data
        times = pd.date_range(start=datetime.now()-timedelta(hours=24), 
                             end=datetime.now(), freq='1H')
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=times,
            y=np.random.normal(1.5, 0.3, len(times)),
            mode='lines+markers',
            name='Displacement (mm)',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=times,
            y=np.random.normal(120, 20, len(times)),
            mode='lines+markers',
            name='Strain (μstrain)',
            yaxis='y2',
            line=dict(color='red')
        ))
        
        fig.update_layout(
            title="24-Hour Sensor Trends",
            xaxis_title="Time",
            yaxis=dict(title="Displacement (mm)", side="left"),
            yaxis2=dict(title="Strain (μstrain)", side="right", overlaying="y"),
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_alerts_table(self, alerts_data):
        """Render alerts table"""
        if not alerts_data:
            st.info("No recent alerts")
            return
        
        # Convert alerts to DataFrame
        alerts_df = pd.DataFrame(alerts_data)
        if not alerts_df.empty:
            alerts_df['timestamp'] = pd.to_datetime(alerts_df['timestamp'])
            alerts_df = alerts_df.sort_values('timestamp', ascending=False).head(10)
            
            # Style the table
            styled_df = alerts_df[['timestamp', 'alert_type', 'message', 'severity']].copy()
            styled_df['timestamp'] = styled_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            st.dataframe(styled_df, use_container_width=True)
    
    def render_lstm_page(self):
        """Render LSTM AI predictions page"""
        st.markdown('<div class="main-header">🤖 LSTM AI Predictions</div>', 
                   unsafe_allow_html=True)
        
        # Get LSTM status
        lstm_status, error = self.make_api_request("lstm/status")
        if error:
            st.error(f"Failed to connect to LSTM service: {error}")
            return
        
        # Display LSTM status
        col1, col2, col3 = st.columns(3)
        
        with col1:
            status_color = "#28a745" if lstm_status.get('lstm_available') else "#dc3545"
            st.markdown(f"""
            <div style="background:{status_color};padding:1rem;border-radius:10px;color:white;text-align:center;">
                <h4>LSTM Available</h4>
                <h3>{"✅ Yes" if lstm_status.get('lstm_available') else "❌ No"}</h3>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            status_color = "#28a745" if lstm_status.get('model_loaded') else "#dc3545"
            st.markdown(f"""
            <div style="background:{status_color};padding:1rem;border-radius:10px;color:white;text-align:center;">
                <h4>Model Loaded</h4>
                <h3>{"✅ Yes" if lstm_status.get('model_loaded') else "❌ No"}</h3>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            status_color = "#28a745" if lstm_status.get('model_trained') else "#ffc107"
            st.markdown(f"""
            <div style="background:{status_color};padding:1rem;border-radius:10px;color:white;text-align:center;">
                <h4>Model Trained</h4>
                <h3>{"✅ Yes" if lstm_status.get('model_trained') else "⏳ Training"}</h3>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Training section
        st.markdown("### 🎯 Model Training")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Upload Training Data (CSV)",
                type=['csv'],
                help="Upload CSV file with sensor data for automatic model training"
            )
            
            if uploaded_file:
                if st.button("🚀 Train Model Automatically", type="primary"):
                    self.train_lstm_model(uploaded_file)
        
        with col2:
            if st.button("📊 Generate Sample Data"):
                self.generate_training_data()
        
        # Prediction section
        st.markdown("### 🔮 Real-Time Predictions")
        
        if lstm_status.get('model_trained'):
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🎯 Make Prediction"):
                    self.make_lstm_prediction()
            
            with col2:
                if st.button("📈 Real-Time Monitoring"):
                    self.start_realtime_monitoring()
        else:
            st.warning("⚠️ Model needs to be trained before making predictions. Upload training data above.")
    
    def train_lstm_model(self, uploaded_file):
        """Train LSTM model with uploaded data"""
        with st.spinner("🤖 Training LSTM model automatically..."):
            try:
                # Prepare the file for upload
                files = {"file": uploaded_file.getvalue()}
                data = {"auto_train": True}
                
                result, error = self.make_api_request("lstm/train", method="POST", 
                                                    data=data, files={"file": uploaded_file})
                
                if error:
                    st.error(f"Training failed: {error}")
                else:
                    st.success("🎉 Model trained successfully!")
                    st.json(result)
                    
                    # Auto-refresh status
                    time.sleep(2)
                    st.rerun()
                    
            except Exception as e:
                st.error(f"Training error: {str(e)}")
    
    def generate_training_data(self):
        """Generate sample training data"""
        with st.spinner("📊 Generating sample training data..."):
            try:
                result, error = self.make_api_request("generate-training-data", method="POST")
                
                if error:
                    st.error(f"Failed to generate data: {error}")
                else:
                    st.success("✅ Sample training data generated successfully!")
                    
                    # Show data preview
                    if 'data_preview' in result:
                        st.markdown("### 📋 Data Preview")
                        df = pd.DataFrame(result['data_preview'])
                        st.dataframe(df.head(10))
                        
                        # Provide download link
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="💾 Download Training Data",
                            data=csv,
                            file_name="rockfall_training_data.csv",
                            mime="text/csv"
                        )
            except Exception as e:
                st.error(f"Generation error: {str(e)}")
    
    def make_lstm_prediction(self):
        """Make LSTM prediction"""
        with st.spinner("🔮 Making prediction..."):
            try:
                result, error = self.make_api_request("lstm/predict-realtime")
                
                if error:
                    st.error(f"Prediction failed: {error}")
                else:
                    st.success("✅ Prediction completed!")
                    
                    # Display prediction results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        risk_prob = result.get('risk_probability', 0) * 100
                        st.metric("Risk Probability", f"{risk_prob:.1f}%")
                    
                    with col2:
                        risk_level = result.get('risk_level', 'UNKNOWN')
                        st.metric("Risk Level", risk_level)
                    
                    # Show detailed results
                    st.json(result)
                    
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
    
    def render_lidar_page(self):
        """Render LIDAR 3D visualization page"""
        st.markdown('<div class="main-header">🌍 LIDAR 3D Visualization</div>', 
                   unsafe_allow_html=True)
        
        # Upload section
        st.markdown("### 📤 Upload LIDAR Data")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Upload LIDAR Point Cloud File",
                type=['las', 'laz', 'ply', 'pcd', 'txt', 'csv'],
                help="Supported formats: LAS, LAZ, PLY, PCD, TXT, CSV"
            )
            
            if uploaded_file:
                if st.button("🚀 Process & Analyze", type="primary"):
                    self.process_lidar_file(uploaded_file)
        
        with col2:
            if st.button("📊 Load Sample Data"):
                self.load_sample_lidar_data()
        
        # Visualization section
        st.markdown("### 🎨 3D Point Cloud Visualization")
        
        # Get LIDAR data
        scans_data, error = self.make_api_request("lidar/scans")
        
        if error:
            st.warning(f"No LIDAR data available: {error}")
            st.info("Upload LIDAR files above to start visualization")
            return
        
        if scans_data and len(scans_data) > 0:
            # Scan selector
            scan_options = {f"Scan {scan['id']} - {scan.get('location', 'Unknown')}": scan['id'] 
                           for scan in scans_data}
            
            selected_scan_name = st.selectbox("Select LIDAR Scan", options=list(scan_options.keys()))
            selected_scan_id = scan_options[selected_scan_name]
            
            # Get scan data
            scan_data, error = self.make_api_request(f"lidar/scan/{selected_scan_id}")
            
            if not error and scan_data:
                self.render_3d_point_cloud(scan_data)
                self.render_scan_analysis(scan_data)
            else:
                st.error(f"Failed to load scan data: {error}")
        else:
            st.info("No LIDAR scans available. Upload data to get started.")
    
    def process_lidar_file(self, uploaded_file):
        """Process uploaded LIDAR file"""
        with st.spinner("🔄 Processing LIDAR file..."):
            try:
                files = {"file": uploaded_file.getvalue()}
                data = {
                    "scan_location": "User Upload",
                    "scanner_type": "Unknown",
                    "notes": f"Uploaded: {uploaded_file.name}",
                    "auto_analyze": True
                }
                
                result, error = self.make_api_request("lidar/upload", method="POST", 
                                                    data=data, files={"file": uploaded_file})
                
                if error:
                    st.error(f"Processing failed: {error}")
                else:
                    st.success("🎉 LIDAR file processed successfully!")
                    st.json(result)
                    
                    # Auto-refresh to show new scan
                    time.sleep(2)
                    st.rerun()
                    
            except Exception as e:
                st.error(f"Processing error: {str(e)}")
    
    def render_3d_point_cloud(self, scan_data):
        """Render 3D point cloud visualization"""
        if 'points' not in scan_data or not scan_data['points']:
            st.warning("No point cloud data available for visualization")
            return
        
        # Convert points data to DataFrame
        points = scan_data['points'][:10000]  # Limit for performance
        
        # Create 3D scatter plot
        fig = go.Figure(data=go.Scatter3d(
            x=[p['x'] for p in points],
            y=[p['y'] for p in points],
            z=[p['z'] for p in points],
            mode='markers',
            marker=dict(
                size=2,
                color=[p.get('risk', 0.5) for p in points],
                colorscale='RdYlBu_r',
                colorbar=dict(title="Risk Level"),
                showscale=True
            ),
            text=[f"Risk: {p.get('risk', 0):.2f}" for p in points],
            hovertemplate="X: %{x}<br>Y: %{y}<br>Z: %{z}<br>%{text}<extra></extra>"
        ))
        
        fig.update_layout(
            title="3D Point Cloud Visualization",
            scene=dict(
                xaxis_title='X Coordinate',
                yaxis_title='Y Coordinate',
                zaxis_title='Z Coordinate',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_scan_analysis(self, scan_data):
        """Render scan analysis results"""
        if 'analysis' not in scan_data:
            return
        
        analysis = scan_data['analysis']
        
        st.markdown("### 📊 Scan Analysis Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Points", f"{analysis.get('total_points', 0):,}")
        
        with col2:
            st.metric("High Risk Points", f"{analysis.get('high_risk_points', 0):,}")
        
        with col3:
            st.metric("Average Risk", f"{analysis.get('average_risk', 0):.3f}")
        
        with col4:
            st.metric("Max Displacement", f"{analysis.get('max_displacement', 0):.2f} mm")
        
        # Risk distribution
        if 'risk_distribution' in analysis:
            risk_dist = analysis['risk_distribution']
            
            fig = px.pie(
                values=list(risk_dist.values()),
                names=list(risk_dist.keys()),
                title="Risk Level Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def render_sensors_page(self):
        """Render sensor data page"""
        st.markdown('<div class="main-header">📊 Sensor Data Monitoring</div>', 
                   unsafe_allow_html=True)
        
        # Real-time sensor data
        sensor_data, error = self.make_api_request("sensor-data")
        
        if error:
            st.error(f"Failed to load sensor data: {error}")
            return
        
        # Display latest readings
        if sensor_data:
            st.markdown("### 📡 Latest Sensor Readings")
            
            # Convert to DataFrame
            df = pd.DataFrame(sensor_data)
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp', ascending=False)
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                
                latest_data = df.iloc[0] if not df.empty else {}
                
                with col1:
                    st.metric("Displacement", f"{latest_data.get('displacement', 0):.2f} mm")
                
                with col2:
                    st.metric("Strain", f"{latest_data.get('strain', 0):.0f} μstrain")
                
                with col3:
                    st.metric("Pore Pressure", f"{latest_data.get('pore_pressure', 0):.1f} kPa")
                
                with col4:
                    st.metric("Temperature", f"{latest_data.get('temperature', 0):.1f} °C")
                
                # Data table
                st.markdown("### 📋 Recent Sensor Data")
                st.dataframe(df.head(50), use_container_width=True)
                
                # Download option
                csv = df.to_csv(index=False)
                st.download_button(
                    label="💾 Download Sensor Data",
                    data=csv,
                    file_name=f"sensor_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    def render_alerts_page(self):
        """Render alert management page"""
        st.markdown('<div class="main-header">⚠️ Alert Management</div>', 
                   unsafe_allow_html=True)
        
        # Get alerts data
        alerts_data, error = self.make_api_request("alerts")
        
        if error:
            st.error(f"Failed to load alerts: {error}")
            return
        
        # Alert controls
        st.markdown("### 🎛️ Alert Controls")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🧪 Test Alert System"):
                self.test_alert_system()
        
        with col2:
            if st.button("🔔 Send Test Email"):
                self.send_test_email()
        
        with col3:
            if st.button("📱 Send Test SMS"):
                self.send_test_sms()
        
        # Alerts display
        if alerts_data:
            st.markdown("### 📜 Alert History")
            
            df = pd.DataFrame(alerts_data)
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp', ascending=False)
                
                # Filter controls
                severity_filter = st.multiselect(
                    "Filter by Severity",
                    options=df['severity'].unique(),
                    default=df['severity'].unique()
                )
                
                filtered_df = df[df['severity'].isin(severity_filter)]
                st.dataframe(filtered_df, use_container_width=True)
        else:
            st.info("No alerts available")
    
    def test_alert_system(self):
        """Test the alert system"""
        with st.spinner("🧪 Testing alert system..."):
            result, error = self.make_api_request("test-alerts", method="POST")
            
            if error:
                st.error(f"Alert test failed: {error}")
            else:
                st.success("✅ Alert system test completed!")
                st.json(result)
    
    def render_settings_page(self):
        """Render system settings page"""
        st.markdown('<div class="main-header">⚙️ System Settings</div>', 
                   unsafe_allow_html=True)
        
        # System status
        st.markdown("### 🖥️ System Status")
        
        # Backend health check
        health_data, error = self.make_api_request("health")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            status_color = "#28a745" if not error else "#dc3545"
            st.markdown(f"""
            <div style="background:{status_color};padding:1rem;border-radius:10px;color:white;text-align:center;">
                <h4>Backend API</h4>
                <h3>{"✅ Online" if not error else "❌ Offline"}</h3>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Database status
            db_status = health_data.get('database', 'Unknown') if health_data else 'Unknown'
            db_color = "#28a745" if db_status == 'Connected' else "#dc3545"
            st.markdown(f"""
            <div style="background:{db_color};padding:1rem;border-radius:10px;color:white;text-align:center;">
                <h4>Database</h4>
                <h3>{"✅ " + db_status if db_status != 'Unknown' else "❓ Unknown"}</h3>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            # ML Models status
            models_status = health_data.get('ml_models', 'Unknown') if health_data else 'Unknown'
            models_color = "#28a745" if models_status == 'Loaded' else "#ffc107"
            st.markdown(f"""
            <div style="background:{models_color};padding:1rem;border-radius:10px;color:white;text-align:center;">
                <h4>ML Models</h4>
                <h3>{"✅ " + models_status if models_status != 'Unknown' else "⏳ Loading"}</h3>
            </div>
            """, unsafe_allow_html=True)
        
        # Configuration
        st.markdown("### ⚙️ Configuration")
        
        with st.expander("📧 Email Settings", expanded=False):
            st.text("Email alerts are configured via environment variables")
            st.code("""
            GMAIL_USER=your_email@gmail.com
            GMAIL_APP_PASSWORD=your_app_password
            """)
        
        with st.expander("📱 SMS Settings", expanded=False):
            st.text("SMS alerts via Twilio (optional)")
            st.code("""
            TWILIO_ACCOUNT_SID=your_sid
            TWILIO_AUTH_TOKEN=your_token
            TWILIO_PHONE_NUMBER=your_number
            """)
        
        with st.expander("🔗 API Keys", expanded=False):
            st.text("External service API keys")
            st.code("""
            WEATHER_API_KEY=your_weather_key
            GEOLOGICAL_API_KEY=your_geo_key
            """)
    
    def run(self):
        """Main application runner"""
        # Auto-refresh logic
        if st.session_state.auto_refresh:
            st.session_state.last_update = datetime.now()
        
        # Render sidebar
        self.render_sidebar()
        
        # Render selected page
        page = st.session_state.selected_page
        
        if page == "Dashboard":
            self.render_dashboard()
        elif page == "LSTM":
            self.render_lstm_page()
        elif page == "LIDAR":
            self.render_lidar_page()
        elif page == "Sensors":
            self.render_sensors_page()
        elif page == "Alerts":
            self.render_alerts_page()
        elif page == "Risk":
            self.render_risk_page()
        elif page == "Settings":
            self.render_settings_page()
    
    def render_risk_page(self):
        """Render risk assessment page"""
        st.markdown('<div class="main-header">📈 Risk Assessment</div>', 
                   unsafe_allow_html=True)
        
        # Get risk data
        risk_data, error = self.make_api_request("risk-assessment")
        
        if error:
            st.error(f"Failed to load risk data: {error}")
            return
        
        # Current risk display
        st.markdown("### 🎯 Current Risk Assessment")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            risk_level = risk_data.get('risk_level', 'UNKNOWN')
            probability = risk_data.get('probability', 0) * 100
            
            risk_colors = {
                'LOW': '#28a745',
                'MEDIUM': '#ffc107', 
                'HIGH': '#fd7e14',
                'CRITICAL': '#dc3545'
            }
            
            color = risk_colors.get(risk_level, '#6c757d')
            
            st.markdown(f"""
            <div style="background:{color};padding:2rem;border-radius:15px;color:white;text-align:center;">
                <h2>Risk Level</h2>
                <h1>{risk_level}</h1>
                <h3>{probability:.1f}% Probability</h3>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Risk factors breakdown
            factors = risk_data.get('factors', {})
            
            factor_names = list(factors.keys())
            factor_values = list(factors.values())
            
            fig = px.bar(
                x=factor_values,
                y=factor_names,
                orientation='h',
                title="Risk Factors Contribution",
                color=factor_values,
                color_continuous_scale='RdYlBu_r'
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Historical risk trends
        st.markdown("### 📊 Risk Trends")
        
        # Generate sample historical data
        dates = pd.date_range(start=datetime.now()-timedelta(days=30), 
                             end=datetime.now(), freq='D')
        
        risk_history = np.random.normal(probability/100, 0.1, len(dates))
        risk_history = np.clip(risk_history, 0, 1)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=risk_history * 100,
            mode='lines+markers',
            name='Risk Probability (%)',
            line=dict(color='red', width=2)
        ))
        
        fig.add_hline(y=75, line_dash="dash", line_color="red", 
                      annotation_text="Critical Threshold")
        fig.add_hline(y=50, line_dash="dash", line_color="orange", 
                      annotation_text="High Risk Threshold")
        fig.add_hline(y=25, line_dash="dash", line_color="yellow", 
                      annotation_text="Medium Risk Threshold")
        
        fig.update_layout(
            title="30-Day Risk Probability Trend",
            xaxis_title="Date",
            yaxis_title="Risk Probability (%)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Auto-refresh mechanism for Streamlit
def auto_refresh():
    """Auto-refresh the app every 30 seconds if enabled"""
    if st.session_state.get('auto_refresh', False):
        time.sleep(30)
        st.rerun()

# Main application entry point
if __name__ == "__main__":
    # Initialize and run the dashboard
    dashboard = RockfallDashboard()
    dashboard.run()
    
    # Auto-refresh logic
    if st.session_state.get('auto_refresh', False):
        time.sleep(30)
        st.rerun()