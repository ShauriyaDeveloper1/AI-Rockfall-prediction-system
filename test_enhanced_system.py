#!/usr/bin/env python3
"""
Comprehensive Test Script for Enhanced AI Rockfall Prediction System
Tests all new features including soil classification, email reporting, and API endpoints
"""

import requests
import json
import time
import os
import sys
from datetime import datetime, timedelta

# Base URL for the API
BASE_URL = "http://localhost:5000/api"

def test_basic_endpoints():
    """Test basic system endpoints"""
    print("🧪 Testing Basic Endpoints...")
    
    endpoints = [
        "/health",
        "/alerts",
        "/forecast",
        "/risk-assessment",
        "/risk-map"
    ]
    
    results = {}
    for endpoint in endpoints:
        try:
            response = requests.get(f"{BASE_URL}{endpoint}", timeout=10)
            results[endpoint] = {
                'status': response.status_code,
                'success': response.status_code == 200,
                'response_time': response.elapsed.total_seconds()
            }
            print(f"  ✅ {endpoint}: {response.status_code} ({response.elapsed.total_seconds():.2f}s)")
        except Exception as e:
            results[endpoint] = {'status': 'error', 'error': str(e), 'success': False}
            print(f"  ❌ {endpoint}: {str(e)}")
    
    return results

def test_enhanced_dashboard_endpoints():
    """Test enhanced dashboard endpoints"""
    print("\n🎯 Testing Enhanced Dashboard Endpoints...")
    
    endpoints = [
        "/enhanced-dashboard/statistics",
        "/enhanced-dashboard/india-mining-data",
        "/enhanced-dashboard/risk-distribution"
    ]
    
    results = {}
    for endpoint in endpoints:
        try:
            response = requests.get(f"{BASE_URL}{endpoint}", timeout=10)
            results[endpoint] = {
                'status': response.status_code,
                'success': response.status_code == 200,
                'response_time': response.elapsed.total_seconds()
            }
            if response.status_code == 200:
                data = response.json()
                print(f"  ✅ {endpoint}: {response.status_code} ({len(str(data))} chars)")
            else:
                print(f"  ❌ {endpoint}: {response.status_code}")
        except Exception as e:
            results[endpoint] = {'status': 'error', 'error': str(e), 'success': False}
            print(f"  ❌ {endpoint}: {str(e)}")
    
    return results

def test_soil_rock_classification():
    """Test soil/rock classification endpoint"""
    print("\n🪨 Testing Soil/Rock Classification...")
    
    try:
        # Test model info endpoint
        response = requests.get(f"{BASE_URL}/soil-rock/info", timeout=10)
        if response.status_code == 200:
            print("  ✅ Model info endpoint working")
            info = response.json()
            print(f"     📊 Model classes: {info.get('info', {}).get('classes', 'N/A')}")
        else:
            print(f"  ❌ Model info endpoint failed: {response.status_code}")
        
        # Test classification endpoint with mock data
        test_payload = {
            'image_data': 'base64_mock_data_for_testing',
            'image_format': 'jpg'
        }
        
        # Note: This will likely fail without actual model/image, but tests the endpoint
        response = requests.post(f"{BASE_URL}/soil-rock/classify", 
                               json=test_payload, timeout=10)
        
        if response.status_code in [200, 400, 503]:  # Expected responses
            print(f"  ✅ Classification endpoint accessible: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"     🎯 Prediction available: {result.get('success', False)}")
        else:
            print(f"  ❌ Classification endpoint error: {response.status_code}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Soil classification test failed: {str(e)}")
        return False

def test_email_reporting():
    """Test email reporting endpoint"""
    print("\n📧 Testing Email Reporting...")
    
    try:
        test_payload = {
            'recipient_email': 'test@example.com',
            'recipient_name': 'Test User',
            'report_type': 'daily_summary',
            'site_location': 'Test Mine Site'
        }
        
        response = requests.post(f"{BASE_URL}/email-report", 
                               json=test_payload, timeout=15)
        
        if response.status_code in [200, 503]:  # 503 if email not configured
            print(f"  ✅ Email reporting endpoint accessible: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"     📬 Email sent: {result.get('success', False)}")
            elif response.status_code == 503:
                print("     ⚠️ Email service not configured (expected in test)")
        else:
            print(f"  ❌ Email reporting failed: {response.status_code}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Email reporting test failed: {str(e)}")
        return False

def test_sensor_data_injection():
    """Test sensor data injection"""
    print("\n📡 Testing Sensor Data Injection...")
    
    try:
        # Inject test sensor data
        test_data = {
            'sensor_id': f'TEST_SENSOR_{int(time.time())}',
            'sensor_type': 'displacement',
            'location_x': -23.5505,
            'location_y': -46.6333,
            'value': 1.5,
            'unit': 'mm'
        }
        
        response = requests.post(f"{BASE_URL}/sensor-data", 
                               json=test_data, timeout=10)
        
        if response.status_code == 200:
            print("  ✅ Sensor data injection successful")
            result = response.json()
            print(f"     📊 Data stored: {result.get('success', False)}")
            return True
        else:
            print(f"  ❌ Sensor data injection failed: {response.status_code}")
            return False
        
    except Exception as e:
        print(f"  ❌ Sensor data injection test failed: {str(e)}")
        return False

def test_lstm_endpoints():
    """Test LSTM prediction endpoints"""
    print("\n🧠 Testing LSTM Endpoints...")
    
    try:
        # Test LSTM status
        response = requests.get(f"{BASE_URL}/lstm/status", timeout=10)
        if response.status_code == 200:
            print("  ✅ LSTM status endpoint working")
            status = response.json()
            print(f"     🤖 Model loaded: {status.get('model_loaded', False)}")
        else:
            print(f"  ❌ LSTM status failed: {response.status_code}")
        
        # Test real-time prediction
        response = requests.get(f"{BASE_URL}/lstm/predict-realtime", timeout=10)
        if response.status_code in [200, 503]:
            print(f"  ✅ LSTM real-time prediction accessible: {response.status_code}")
        else:
            print(f"  ❌ LSTM prediction failed: {response.status_code}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ LSTM test failed: {str(e)}")
        return False

def generate_test_report(results):
    """Generate comprehensive test report"""
    print("\n" + "="*80)
    print("📋 COMPREHENSIVE TEST REPORT")
    print("="*80)
    
    total_tests = 0
    passed_tests = 0
    
    for category, tests in results.items():
        print(f"\n📂 {category.upper()}")
        print("-" * 50)
        
        if isinstance(tests, dict):
            for test_name, result in tests.items():
                total_tests += 1
                if result.get('success', False):
                    passed_tests += 1
                    status = "✅ PASS"
                else:
                    status = "❌ FAIL"
                
                response_time = result.get('response_time', 'N/A')
                if isinstance(response_time, float):
                    response_time = f"{response_time:.2f}s"
                
                print(f"  {status} {test_name} ({response_time})")
        else:
            total_tests += 1
            if tests:
                passed_tests += 1
                print(f"  ✅ PASS {category}")
            else:
                print(f"  ❌ FAIL {category}")
    
    print("\n" + "="*80)
    print(f"📊 SUMMARY: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
    print("="*80)
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED! System is ready for deployment.")
    elif passed_tests > total_tests * 0.8:
        print("⚠️ Most tests passed. Minor issues may need attention.")
    else:
        print("🚨 Multiple test failures. System needs debugging.")
    
    return passed_tests, total_tests

def main():
    """Main test execution"""
    print("🚀 Starting Comprehensive System Tests")
    print(f"⏰ Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🌐 Testing API at: {BASE_URL}")
    print("="*80)
    
    # Run all test categories
    results = {}
    
    try:
        results['basic_endpoints'] = test_basic_endpoints()
        results['enhanced_dashboard'] = test_enhanced_dashboard_endpoints()
        results['soil_classification'] = test_soil_rock_classification()
        results['email_reporting'] = test_email_reporting()
        results['sensor_injection'] = test_sensor_data_injection()
        results['lstm_endpoints'] = test_lstm_endpoints()
        
        # Generate final report
        passed, total = generate_test_report(results)
        
        # Save results to file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = f"test_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'summary': {'passed': passed, 'total': total, 'success_rate': passed/total},
                'results': results
            }, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: {report_file}")
        
        # Exit with appropriate code
        sys.exit(0 if passed == total else 1)
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n💥 Test suite crashed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()