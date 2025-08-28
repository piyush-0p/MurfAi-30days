#!/usr/bin/env python3
"""
Test script for the weather button functionality
Tests both the API endpoint and the button integration
"""

import requests
import json

def test_weather_api():
    """Test the weather API endpoint directly"""
    print("🧪 Testing Weather API Endpoint...")
    
    # Test coordinates (San Francisco)
    test_data = {
        "latitude": 37.7749,
        "longitude": -122.4194
    }
    
    try:
        response = requests.post(
            "http://localhost:8000/api/weather",
            headers={"Content-Type": "application/json"},
            json=test_data,
            timeout=10
        )
        
        if response.status_code == 200:
            weather_data = response.json()
            print("✅ Weather API Response:")
            print(f"   Success: {weather_data.get('success')}")
            print(f"   Location: {weather_data.get('location')}")
            print(f"   Temperature: {weather_data.get('temperature')}°C")
            print(f"   Feels like: {weather_data.get('feels_like')}°C")
            print(f"   Description: {weather_data.get('description')}")
            print(f"   Humidity: {weather_data.get('humidity')}%")
            print(f"   Wind Speed: {weather_data.get('wind_speed')} m/s")
            print(f"   Visibility: {weather_data.get('visibility')} km")
            return True
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Test Error: {e}")
        return False

def test_server_health():
    """Test if the server is running"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        return response.status_code == 200
    except:
        return False

if __name__ == "__main__":
    print("🌤️ Weather Button Functionality Test")
    print("=" * 50)
    
    # Test server health
    if not test_server_health():
        print("❌ Server is not running! Please start the server first:")
        print("   cd /path/to/MurfAIChallenge && python main.py")
        exit(1)
    
    print("✅ Server is running")
    
    # Test weather API
    if test_weather_api():
        print("\n🎉 Weather functionality is working!")
        print("\n📝 How to use the weather button:")
        print("1. Open http://localhost:8000/static/conversation.html")
        print("2. Click the '🌤️ Get Current Weather' button")
        print("3. Allow location access when prompted")
        print("4. See the weather information in the chat")
    else:
        print("\n❌ Weather functionality has issues")
        print("Check the server logs and API key configuration")
    
    print("\n" + "=" * 50)
