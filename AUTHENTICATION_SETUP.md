# AI Rockfall Prediction System - Authentication & Report Generation Setup

## 🚀 New Features Added

### 1. User Authentication System
- **Secure Login/Registration**: JWT-based authentication with bcrypt password hashing
- **User Management**: Support for different user roles (User, Admin, Engineer, Supervisor)
- **Protected Routes**: All system features require authentication
- **Session Management**: Persistent login with token storage

### 2. Automated Risk Analysis Reports
- **PDF Report Generation**: Comprehensive risk analysis reports using ReportLab
- **Email Delivery**: Automatic report delivery to registered user email addresses
- **Multiple Report Types**: Daily, Weekly, and Monthly risk analysis reports
- **Report History**: Track all generated reports with status monitoring

### 3. Enhanced User Interface
- **Modern Authentication UI**: Professional login/registration forms with holographic design
- **User Profile Management**: Display user information in navigation bar
- **Reports Dashboard**: Dedicated interface for report generation and history
- **Secure Logout**: Proper session cleanup and security

## 🔧 Setup Instructions

### Backend Configuration

1. **Install Required Python Packages**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Configuration**:
   Copy `.env.example` to `.env` and configure:
   ```bash
   # Authentication
   JWT_SECRET_KEY=your-super-secret-jwt-key-change-in-production
   
   # Email for Reports (Gmail recommended)
   SMTP_EMAIL=your_email@gmail.com
   SMTP_PASSWORD=your_app_password
   
   # Database
   DATABASE_URL=sqlite:///rockfall_system.db
   ```

3. **Gmail App Password Setup** (for report emails):
   - Enable 2-factor authentication on your Gmail account
   - Generate an "App Password" for the application
   - Use this app password in the `SMTP_PASSWORD` field

### Frontend Configuration

1. **Install Node.js Dependencies**:
   ```bash
   cd frontend
   npm install
   ```

2. **No additional configuration required** - authentication is handled automatically

## 🏃 Starting the System

### Option 1: Automated Start (Recommended)
```bash
# Windows
start_complete_system.bat

# Or use Python script
python run_system.py --with-simulator
```

### Option 2: Manual Start
```bash
# Terminal 1: Start Backend
cd backend
python app.py

# Terminal 2: Start Frontend
cd frontend
npm start

# Terminal 3: Start Sensor Simulator (Optional)
python start_simulator.py
```

## 👤 User Guide

### First Time Setup

1. **Access the System**: Open http://localhost:3000 in your browser
2. **Register Account**: Click "Register" and create your account with:
   - Email address (used for reports)
   - Password (securely hashed)
   - Full name and company information
   - Role selection

3. **Login**: Use your credentials to access the system

### Generating Reports

1. **Navigate to Reports**: Click "Reports" in the navigation menu
2. **Choose Report Type**:
   - **Daily**: Current day risk analysis with recent sensor data
   - **Weekly**: 7-day trend analysis with patterns
   - **Monthly**: Comprehensive monthly overview with recommendations
3. **Generate Report**: Click the generate button for your desired report type
4. **Email Delivery**: Report will be automatically generated and emailed to your registered address
5. **View History**: Check the report history table for status and previous reports

### Dashboard Features

- **Risk Assessment**: Real-time risk level monitoring
- **Sensor Data**: Live sensor readings and trends
- **Interactive Maps**: Geographic risk visualization
- **Alert Management**: Critical alert notifications
- **Forecast Models**: Predictive analytics and ML insights

## 🔒 Security Features

### Authentication Security
- **Password Hashing**: Bcrypt with salt for secure password storage
- **JWT Tokens**: Secure token-based authentication with expiration
- **Session Management**: Automatic logout on token expiration
- **Protected Routes**: All API endpoints require valid authentication

### Data Security
- **SQL Injection Protection**: SQLAlchemy ORM with parameterized queries
- **CORS Configuration**: Properly configured cross-origin resource sharing
- **Environment Variables**: Sensitive data stored securely in environment files
- **Input Validation**: Server-side validation for all user inputs

## 📊 Report Features

### Report Content
- **Risk Status Summary**: Current risk level and probability
- **Sensor Data Analysis**: Statistical analysis of recent sensor readings
- **Trend Analysis**: Risk pattern identification over time
- **Recommendations**: Actionable safety recommendations based on risk level
- **Visual Charts**: Graphs and charts for data visualization

### Email Integration
- **HTML Emails**: Professional formatted email reports
- **PDF Attachments**: Detailed reports attached as PDF files
- **Delivery Tracking**: Monitor email delivery status
- **Error Handling**: Automatic retry and error logging

## 🛠 Troubleshooting

### Common Issues

1. **Login Issues**:
   - Check if backend server is running on port 5000
   - Verify database connection in logs
   - Ensure correct email/password combination

2. **Report Generation Failures**:
   - Verify SMTP email configuration in .env file
   - Check Gmail app password is correct
   - Review backend logs for PDF generation errors

3. **Email Delivery Issues**:
   - Confirm Gmail 2FA is enabled
   - Use App Password, not regular Gmail password
   - Check spam/junk folders for delivered reports

4. **Frontend Connection Issues**:
   - Ensure backend is running on port 5000
   - Check proxy configuration in package.json
   - Verify CORS settings in Flask app

### Error Logs
- **Backend Logs**: Check terminal output where `python app.py` is running
- **Frontend Logs**: Check browser developer console (F12)
- **Report Generation**: Check `reports/` directory for generated files

## 📧 Email Configuration Examples

### Gmail Setup
```bash
SMTP_EMAIL=your_email@gmail.com
SMTP_PASSWORD=abcd efgh ijkl mnop  # App password (16 characters with spaces)
```

### Other Email Providers
The system supports any SMTP provider. Update the email configuration in `send_email()` function for other providers like Outlook, Yahoo, etc.

## 🔄 Database Management

### User Management
- Users are stored in SQLite database (`rockfall_system.db`)
- Password hashes are stored securely (never plain text)
- User roles control access levels (future feature)

### Report Tracking
- All report requests are logged in `ReportRequest` table
- Track generation status, email delivery, and timestamps
- History available in Reports dashboard

## 📈 Future Enhancements

### Planned Features
- **Role-based Access Control**: Different permissions for Admin, Engineer, etc.
- **Report Scheduling**: Automatic periodic report generation
- **Advanced Analytics**: More detailed statistical analysis
- **Mobile App**: Native mobile application for field use
- **Real-time Notifications**: WebSocket-based live alerts

## 🆘 Support

### Getting Help
1. Check this documentation first
2. Review error logs in terminal/console
3. Verify environment configuration
4. Test with provided sample data

### System Status
- Backend Health: http://localhost:5000/api/health
- Database Status: Check logs for connection status
- Email Test: Use test scripts in project root

---

**Congratulations!** You now have a fully functional AI Rockfall Prediction System with secure user authentication and automated report generation. The system is ready for production use with proper email configuration and security settings.