"""
Email Reporting System
Generates and sends risk analysis reports via email
"""

import smtplib
import os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from datetime import datetime, timedelta
import logging
import json
from pathlib import Path
import base64
from jinja2 import Template
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from io import BytesIO
import tempfile

# Try to import reportlab for PDF generation
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("ReportLab not available. Install with: pip install reportlab")

class EmailReportService:
    def __init__(self):
        self.setup_logging()
        
        # Email configuration
        self.smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', '587'))
        self.email_user = os.getenv('EMAIL_USER', '')
        self.email_password = os.getenv('EMAIL_PASSWORD', '')
        
        # Report templates
        self.report_templates_dir = Path("templates/email_reports")
        self.report_templates_dir.mkdir(parents=True, exist_ok=True)
        
    def setup_logging(self):
        """Setup logging for email service"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def generate_risk_charts(self, sensor_data, risk_assessments):
        """Generate charts for the risk report"""
        charts = {}
        
        try:
            # Set style
            plt.style.use('seaborn-v0_8')
            
            # 1. Risk Level Trend Chart
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if risk_assessments:
                dates = [assessment['timestamp'] for assessment in risk_assessments]
                risk_levels = [assessment['probability'] for assessment in risk_assessments]
                
                ax.plot(dates, risk_levels, marker='o', linewidth=2, markersize=6)
                ax.set_title('Risk Level Trend (Last 24 Hours)', fontsize=14, fontweight='bold')
                ax.set_xlabel('Time')
                ax.set_ylabel('Risk Probability (%)')
                ax.grid(True, alpha=0.3)
                
                # Color zones
                ax.axhspan(0, 30, alpha=0.2, color='green', label='Low Risk')
                ax.axhspan(30, 60, alpha=0.2, color='yellow', label='Medium Risk')
                ax.axhspan(60, 80, alpha=0.2, color='orange', label='High Risk')
                ax.axhspan(80, 100, alpha=0.2, color='red', label='Critical Risk')
                
                ax.legend()
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                # Save chart
                chart_buffer = BytesIO()
                plt.savefig(chart_buffer, format='png', dpi=300, bbox_inches='tight')
                chart_buffer.seek(0)
                charts['risk_trend'] = base64.b64encode(chart_buffer.getvalue()).decode()
                plt.close()
            
            # 2. Sensor Readings Chart
            if sensor_data:
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
                
                # Group sensor data by type
                sensor_types = {}
                for reading in sensor_data:
                    sensor_type = reading['sensor_type']
                    if sensor_type not in sensor_types:
                        sensor_types[sensor_type] = {'values': [], 'timestamps': []}
                    sensor_types[sensor_type]['values'].append(reading['value'])
                    sensor_types[sensor_type]['timestamps'].append(reading['timestamp'])
                
                axes = [ax1, ax2, ax3, ax4]
                sensor_type_list = list(sensor_types.keys())[:4]
                
                for i, sensor_type in enumerate(sensor_type_list):
                    if i < len(axes):
                        data = sensor_types[sensor_type]
                        axes[i].plot(data['timestamps'], data['values'], marker='o', linewidth=2)
                        axes[i].set_title(f'{sensor_type.title()} Readings')
                        axes[i].set_xlabel('Time')
                        axes[i].set_ylabel('Value')
                        axes[i].grid(True, alpha=0.3)
                        axes[i].tick_params(axis='x', rotation=45)
                
                plt.tight_layout()
                
                # Save chart
                chart_buffer = BytesIO()
                plt.savefig(chart_buffer, format='png', dpi=300, bbox_inches='tight')
                chart_buffer.seek(0)
                charts['sensor_readings'] = base64.b64encode(chart_buffer.getvalue()).decode()
                plt.close()
            
            # 3. Risk Distribution Pie Chart
            if risk_assessments:
                fig, ax = plt.subplots(figsize=(8, 8))
                
                risk_counts = {'Low': 0, 'Medium': 0, 'High': 0, 'Critical': 0}
                for assessment in risk_assessments:
                    prob = assessment['probability']
                    if prob < 30:
                        risk_counts['Low'] += 1
                    elif prob < 60:
                        risk_counts['Medium'] += 1
                    elif prob < 80:
                        risk_counts['High'] += 1
                    else:
                        risk_counts['Critical'] += 1
                
                labels = []
                sizes = []
                colors_list = []
                
                color_map = {'Low': '#28a745', 'Medium': '#ffc107', 'High': '#fd7e14', 'Critical': '#dc3545'}
                
                for level, count in risk_counts.items():
                    if count > 0:
                        labels.append(f'{level} ({count})')
                        sizes.append(count)
                        colors_list.append(color_map[level])
                
                if sizes:
                    ax.pie(sizes, labels=labels, colors=colors_list, autopct='%1.1f%%', startangle=90)
                    ax.set_title('Risk Level Distribution (Last 24 Hours)', fontsize=14, fontweight='bold')
                    
                    # Save chart
                    chart_buffer = BytesIO()
                    plt.savefig(chart_buffer, format='png', dpi=300, bbox_inches='tight')
                    chart_buffer.seek(0)
                    charts['risk_distribution'] = base64.b64encode(chart_buffer.getvalue()).decode()
                    plt.close()
                    
        except Exception as e:
            self.logger.error(f"Error generating charts: {e}")
        
        return charts
    
    def generate_pdf_report(self, report_data, charts):
        """Generate PDF report using ReportLab"""
        if not REPORTLAB_AVAILABLE:
            return None
            
        try:
            # Create temporary file
            temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
            
            # Create PDF document
            doc = SimpleDocTemplate(temp_pdf.name, pagesize=A4)
            styles = getSampleStyleSheet()
            story = []
            
            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=18,
                spaceAfter=30,
                textColor=colors.darkblue,
                alignment=1  # Center alignment
            )
            
            story.append(Paragraph("Mine Safety Risk Analysis Report", title_style))
            story.append(Spacer(1, 20))
            
            # Report Info
            info_data = [
                ['Report Generated:', report_data['timestamp']],
                ['Site Location:', report_data.get('site_location', 'N/A')],
                ['Current Risk Level:', report_data.get('current_risk_level', 'N/A')],
                ['Risk Probability:', f"{report_data.get('current_probability', 0):.1f}%"]
            ]
            
            info_table = Table(info_data, colWidths=[2*inch, 3*inch])
            info_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(info_table)
            story.append(Spacer(1, 20))
            
            # Risk Assessment Summary
            story.append(Paragraph("Risk Assessment Summary", styles['Heading2']))
            story.append(Spacer(1, 10))
            
            summary_text = f"""
            Based on the latest sensor data and AI analysis, the current risk assessment indicates:
            
            • Risk Level: {report_data.get('current_risk_level', 'Unknown')}
            • Probability: {report_data.get('current_probability', 0):.1f}%
            • Total Sensors Active: {report_data.get('active_sensors', 0)}
            • Alerts Generated: {report_data.get('total_alerts', 0)}
            
            {report_data.get('risk_summary', 'No additional summary available.')}
            """
            
            story.append(Paragraph(summary_text, styles['Normal']))
            story.append(Spacer(1, 20))
            
            # Add charts if available
            if charts:
                story.append(Paragraph("Risk Analysis Charts", styles['Heading2']))
                story.append(Spacer(1, 10))
                
                for chart_name, chart_data in charts.items():
                    try:
                        # Decode base64 chart
                        chart_bytes = base64.b64decode(chart_data)
                        chart_buffer = BytesIO(chart_bytes)
                        
                        # Add chart to PDF
                        chart_img = RLImage(chart_buffer, width=6*inch, height=4*inch)
                        story.append(chart_img)
                        story.append(Spacer(1, 10))
                        
                    except Exception as e:
                        self.logger.error(f"Error adding chart {chart_name}: {e}")
            
            # Recommendations
            if 'recommendations' in report_data:
                story.append(Paragraph("Recommendations", styles['Heading2']))
                story.append(Spacer(1, 10))
                
                for i, rec in enumerate(report_data['recommendations'], 1):
                    story.append(Paragraph(f"{i}. {rec}", styles['Normal']))
                    story.append(Spacer(1, 5))
            
            # Build PDF
            doc.build(story)
            
            return temp_pdf.name
            
        except Exception as e:
            self.logger.error(f"Error generating PDF: {e}")
            return None
    
    def create_html_email_template(self):
        """Create HTML template for email reports"""
        template = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
                .container { max-width: 800px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
                .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
                .header h1 { margin: 0; font-size: 24px; }
                .risk-level { font-size: 18px; font-weight: bold; padding: 10px; border-radius: 5px; margin: 10px 0; }
                .risk-low { background-color: #d4edda; color: #155724; }
                .risk-medium { background-color: #fff3cd; color: #856404; }
                .risk-high { background-color: #f8d7da; color: #721c24; }
                .risk-critical { background-color: #f5c6cb; color: #491217; }
                .stats { display: flex; justify-content: space-between; margin: 20px 0; }
                .stat-item { text-align: center; padding: 10px; background-color: #f8f9fa; border-radius: 5px; }
                .chart-container { margin: 20px 0; text-align: center; }
                .chart-container img { max-width: 100%; height: auto; border-radius: 5px; }
                .recommendations { background-color: #e9ecef; padding: 15px; border-radius: 5px; margin: 20px 0; }
                .footer { text-align: center; margin-top: 30px; padding-top: 20px; border-top: 1px solid #dee2e6; color: #6c757d; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🚨 Mine Safety Risk Analysis Report</h1>
                    <p>Generated on {{ timestamp }}</p>
                </div>
                
                <div class="risk-level risk-{{ risk_class }}">
                    Current Risk Level: {{ current_risk_level }} ({{ current_probability }}%)
                </div>
                
                <div class="stats">
                    <div class="stat-item">
                        <h3>{{ active_sensors }}</h3>
                        <p>Active Sensors</p>
                    </div>
                    <div class="stat-item">
                        <h3>{{ total_alerts }}</h3>
                        <p>Total Alerts</p>
                    </div>
                    <div class="stat-item">
                        <h3>{{ site_location }}</h3>
                        <p>Site Location</p>
                    </div>
                </div>
                
                {% if charts %}
                <h2>Risk Analysis Charts</h2>
                {% for chart_name, chart_data in charts.items() %}
                <div class="chart-container">
                    <img src="data:image/png;base64,{{ chart_data }}" alt="{{ chart_name }}">
                </div>
                {% endfor %}
                {% endif %}
                
                {% if recommendations %}
                <div class="recommendations">
                    <h3>🔍 Recommendations</h3>
                    <ul>
                    {% for rec in recommendations %}
                        <li>{{ rec }}</li>
                    {% endfor %}
                    </ul>
                </div>
                {% endif %}
                
                <div class="footer">
                    <p>This report was automatically generated by the AI-Based Rockfall Prediction System</p>
                    <p>For urgent matters, please contact the emergency response team immediately</p>
                </div>
            </div>
        </body>
        </html>
        """
        return Template(template)
    
    def send_email_report(self, recipient_email, report_data, include_pdf=True):
        """Send email report to recipient"""
        try:
            # Generate charts
            charts = self.generate_risk_charts(
                report_data.get('sensor_data', []),
                report_data.get('risk_assessments', [])
            )
            
            # Create email message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"Mine Safety Risk Report - {report_data.get('current_risk_level', 'Unknown')} Risk Detected"
            msg['From'] = self.email_user
            msg['To'] = recipient_email
            
            # Determine risk class for styling
            risk_level = report_data.get('current_risk_level', '').lower()
            risk_class = {
                'low': 'low',
                'medium': 'medium', 
                'high': 'high',
                'critical': 'critical'
            }.get(risk_level, 'medium')
            
            # Prepare template data
            template_data = {
                **report_data,
                'risk_class': risk_class,
                'charts': charts,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Generate HTML content
            html_template = self.create_html_email_template()
            html_content = html_template.render(**template_data)
            
            # Create HTML part
            html_part = MIMEText(html_content, 'html')
            msg.attach(html_part)
            
            # Generate and attach PDF if requested
            if include_pdf and REPORTLAB_AVAILABLE:
                pdf_path = self.generate_pdf_report(template_data, charts)
                if pdf_path:
                    try:
                        with open(pdf_path, 'rb') as f:
                            pdf_data = f.read()
                        
                        pdf_attachment = MIMEApplication(pdf_data, _subtype='pdf')
                        pdf_attachment.add_header(
                            'Content-Disposition', 
                            'attachment', 
                            filename=f'risk_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
                        )
                        msg.attach(pdf_attachment)
                        
                        # Clean up temporary file
                        os.unlink(pdf_path)
                        
                    except Exception as e:
                        self.logger.error(f"Error attaching PDF: {e}")
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.email_user, self.email_password)
                server.send_message(msg)
            
            self.logger.info(f"Report sent successfully to {recipient_email}")
            return {
                'success': True,
                'message': 'Report sent successfully',
                'recipient': recipient_email
            }
            
        except Exception as e:
            self.logger.error(f"Error sending email report: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def validate_email_config(self):
        """Validate email configuration"""
        missing_config = []
        
        if not self.email_user:
            missing_config.append('EMAIL_USER')
        if not self.email_password:
            missing_config.append('EMAIL_PASSWORD')
            
        return {
            'valid': len(missing_config) == 0,
            'missing_config': missing_config,
            'smtp_server': self.smtp_server,
            'smtp_port': self.smtp_port
        }

# Initialize global email service
email_report_service = EmailReportService()