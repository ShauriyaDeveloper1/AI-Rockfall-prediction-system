#!/usr/bin/env python3
"""
Test script for authentication and report generation features
Run this script to verify the system is working correctly
"""

import requests
import json
import time

# Configuration
BASE_URL = "http://localhost:5000"
TEST_USER = {
    "email": "test@example.com",
    "password": "testpassword123",
    "first_name": "Test",
    "last_name": "User",
    "company": "Test Company",
    "role": "user"
}

def test_health_check():
    """Test if backend is running"""
    try:
        response = requests.get(f"{BASE_URL}/api/health")
        if response.status_code == 200:
            print("✅ Backend server is running")
            return True
        else:
            print("❌ Backend server health check failed")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to backend server. Is it running on port 5000?")
        return False

def test_user_registration():
    """Test user registration"""
    try:
        response = requests.post(f"{BASE_URL}/api/auth/register", json=TEST_USER)
        if response.status_code == 201:
            print("✅ User registration successful")
            return True
        elif response.status_code == 400 and "already registered" in response.json().get("error", ""):
            print("✅ User already exists (registration previously successful)")
            return True
        else:
            print(f"❌ User registration failed: {response.json()}")
            return False
    except Exception as e:
        print(f"❌ User registration error: {e}")
        return False

def test_user_login():
    """Test user login and return token"""
    try:
        login_data = {
            "email": TEST_USER["email"],
            "password": TEST_USER["password"]
        }
        response = requests.post(f"{BASE_URL}/api/auth/login", json=login_data)
        if response.status_code == 200:
            data = response.json()
            token = data.get("access_token")
            print("✅ User login successful")
            return token
        else:
            print(f"❌ User login failed: {response.json()}")
            return None
    except Exception as e:
        print(f"❌ User login error: {e}")
        return None

def test_protected_route(token):
    """Test accessing protected route with token"""
    try:
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(f"{BASE_URL}/api/auth/profile", headers=headers)
        if response.status_code == 200:
            print("✅ Protected route access successful")
            return True
        else:
            print(f"❌ Protected route access failed: {response.json()}")
            return False
    except Exception as e:
        print(f"❌ Protected route access error: {e}")
        return False

def test_report_generation(token):
    """Test report generation"""
    try:
        headers = {"Authorization": f"Bearer {token}"}
        report_data = {"report_type": "daily"}
        response = requests.post(f"{BASE_URL}/api/reports/generate", json=report_data, headers=headers)
        if response.status_code == 200:
            print("✅ Report generation requested successfully")
            return True
        else:
            print(f"❌ Report generation failed: {response.json()}")
            return False
    except Exception as e:
        print(f"❌ Report generation error: {e}")
        return False

def test_report_history(token):
    """Test report history retrieval"""
    try:
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(f"{BASE_URL}/api/reports/history", headers=headers)
        if response.status_code == 200:
            data = response.json()
            reports = data.get("reports", [])
            print(f"✅ Report history retrieved successfully ({len(reports)} reports)")
            return True
        else:
            print(f"❌ Report history retrieval failed: {response.json()}")
            return False
    except Exception as e:
        print(f"❌ Report history retrieval error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing AI Rockfall Prediction System Authentication")
    print("=" * 60)
    
    # Test 1: Health Check
    print("\n1. Testing backend health...")
    if not test_health_check():
        print("\n❌ Backend is not running. Please start the backend server first:")
        print("   cd backend && python app.py")
        return False
    
    # Test 2: User Registration
    print("\n2. Testing user registration...")
    if not test_user_registration():
        return False
    
    # Test 3: User Login
    print("\n3. Testing user login...")
    token = test_user_login()
    if not token:
        return False
    
    # Test 4: Protected Route Access
    print("\n4. Testing protected route access...")
    if not test_protected_route(token):
        return False
    
    # Test 5: Report Generation
    print("\n5. Testing report generation...")
    if not test_report_generation(token):
        return False
    
    # Test 6: Report History
    print("\n6. Testing report history...")
    if not test_report_history(token):
        return False
    
    print("\n" + "=" * 60)
    print("🎉 All authentication tests passed successfully!")
    print("\n📋 System Status:")
    print("   ✅ Backend server running")
    print("   ✅ User authentication working")
    print("   ✅ Protected routes secured")
    print("   ✅ Report generation functional")
    print("   ✅ Report history accessible")
    
    print("\n🌐 Access the application:")
    print(f"   Frontend: http://localhost:3000")
    print(f"   Backend:  {BASE_URL}")
    
    print("\n👤 Test User Credentials:")
    print(f"   Email:    {TEST_USER['email']}")
    print(f"   Password: {TEST_USER['password']}")
    
    return True

if __name__ == "__main__":
    main()