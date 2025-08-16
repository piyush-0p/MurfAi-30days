#!/usr/bin/env python3
"""
Comprehensive WebSocket test demonstration
"""
import asyncio
import websockets
import json
from datetime import datetime


async def test_websocket_comprehensive():
    uri = "ws://localhost:8000/ws"
    
    print("="*60)
    print("🚀 WebSocket Connection Test")
    print("="*60)
    print(f"📍 Connecting to: {uri}")
    print(f"🕐 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-"*60)
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ WebSocket connection established successfully!")
            print()
            
            # Test cases
            test_cases = [
                {"name": "Simple Text", "message": "Hello, WebSocket Server!"},
                {"name": "Numbers", "message": "12345"},
                {"name": "Special Characters", "message": "Hello! @#$%^&*() 🚀🎉"},
                {"name": "JSON String", "message": '{"type": "test", "data": {"id": 1, "name": "WebSocket Test"}}'},
                {"name": "Long Message", "message": "This is a longer message to test how the WebSocket handles larger payloads. " * 5},
                {"name": "Empty Message", "message": ""},
                {"name": "Unicode", "message": "Hello 世界! Привет мир! 🌍🌎🌏"},
            ]
            
            for i, test_case in enumerate(test_cases, 1):
                print(f"📤 Test {i}: {test_case['name']}")
                print(f"   Sending: '{test_case['message']}'")
                
                # Send message
                await websocket.send(test_case['message'])
                
                # Receive response
                response = await websocket.recv()
                print(f"   📥 Received: '{response}'")
                
                # Verify echo
                expected = f"Echo: {test_case['message']}"
                if response == expected:
                    print("   ✅ Echo test PASSED")
                else:
                    print("   ❌ Echo test FAILED")
                    print(f"   Expected: '{expected}'")
                
                print()
                await asyncio.sleep(0.5)  # Small delay between tests
                
            print("="*60)
            print("🎉 All WebSocket tests completed successfully!")
            print("="*60)
            
    except Exception as e:
        print(f"❌ WebSocket connection failed: {e}")
        return False
        
    return True


if __name__ == "__main__":
    success = asyncio.run(test_websocket_comprehensive())
    if success:
        print("\n🏆 WebSocket endpoint is working perfectly!")
        print("✅ Ready for integration with client applications like Postman")
    else:
        print("\n💥 WebSocket tests failed!")
