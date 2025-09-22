#!/usr/bin/env python3
"""
Test Authentication System
"""

import requests
import json

# Configuration
AUTH_BASE_URL = 'http://localhost:5003'

def test_health():
    """Test health endpoint"""
    try:
        response = requests.get(f'{AUTH_BASE_URL}/api/health')
        print(f"Health Check: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_registration():
    """Test user registration"""
    try:
        test_user = {
            'email': 'test@example.com',
            'password': 'Test123!',
            'first_name': 'Test',
            'last_name': 'User',
            'company': 'Test Company'
        }
        
        response = requests.post(f'{AUTH_BASE_URL}/api/auth/register', json=test_user)
        print(f"Registration: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code in [200, 201]
    except Exception as e:
        print(f"Registration test failed: {e}")
        return False

def test_login():
    """Test user login"""
    try:
        login_data = {
            'email': 'admin@rockfall.com',
            'password': 'Admin123!'
        }
        
        response = requests.post(f'{AUTH_BASE_URL}/api/auth/login', json=login_data)
        print(f"Login: {response.status_code}")
        result = response.json()
        print(f"Response: {result}")
        
        if response.status_code == 200 and 'access_token' in result:
            return result['access_token']
        return None
    except Exception as e:
        print(f"Login test failed: {e}")
        return None

def test_protected_route(token):
    """Test protected route with token"""
    try:
        headers = {'Authorization': f'Bearer {token}'}
        response = requests.get(f'{AUTH_BASE_URL}/api/auth/profile', headers=headers)
        print(f"Protected Route: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Protected route test failed: {e}")
        return False

def main():
    print("🔐 Testing Authentication System")
    print("=" * 50)
    
    # Test 1: Health Check
    print("\n1. Testing Health Check...")
    if not test_health():
        print("❌ Health check failed! Make sure auth_backend.py is running.")
        return
    print("✅ Health check passed!")
    
    # Test 2: Registration
    print("\n2. Testing Registration...")
    if test_registration():
        print("✅ Registration test passed!")
    else:
        print("⚠️ Registration test failed (might be normal if user exists)")
    
    # Test 3: Login
    print("\n3. Testing Login...")
    token = test_login()
    if token:
        print("✅ Login test passed!")
        print(f"🔑 Token received: {token[:20]}...")
        
        # Test 4: Protected Route
        print("\n4. Testing Protected Route...")
        if test_protected_route(token):
            print("✅ Protected route test passed!")
        else:
            print("❌ Protected route test failed!")
    else:
        print("❌ Login test failed!")
    
    print("\n" + "=" * 50)
    print("🎉 Authentication tests completed!")

if __name__ == '__main__':
    main()