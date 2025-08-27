#!/usr/bin/env python3

import asyncio
import sys
import os

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the functions from main.py
from main import detect_skill_request, get_time_info

async def test_time_functionality():
    """Test the time functionality"""
    
    print("🧪 Testing Time & Date Functionality\n")
    
    # Test different time-related phrases
    test_phrases = [
        "what time is it",
        "current time",
        "get current time", 
        "time and date",
        "get time",
        "what's the current date",
        "Get current time and date for any timezone",
        "time in New York",
        "current time in Tokyo"
    ]
    
    print("1. Testing skill detection:")
    for phrase in test_phrases:
        result = await detect_skill_request(phrase)
        if result and result.get('skill') == 'time_zone':
            print(f"✅ '{phrase}' → Detected skill: {result}")
        else:
            print(f"❌ '{phrase}' → No time skill detected (got: {result})")
    
    print("\n2. Testing time function directly:")
    
    # Test with different timezones
    timezones = ["UTC", "America/New_York", "Asia/Tokyo", "Europe/London"]
    
    for tz in timezones:
        result = await get_time_info(tz)
        if result.get('success'):
            print(f"✅ {tz}: {result['formatted']}")
        else:
            print(f"❌ {tz}: {result}")
    
    print("\n🎯 Testing complete!")

if __name__ == "__main__":
    asyncio.run(test_time_functionality())
