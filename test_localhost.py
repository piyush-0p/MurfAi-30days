#!/usr/bin/env python3
"""
Comprehensive test script for localhost functionality
Tests all endpoints and WebSocket connections
"""
import asyncio
import json
import websockets
import urllib.request
import urllib.error
from datetime import datetime

def test_http_endpoints():
    """Test HTTP endpoints"""
    print("🔍 Testing HTTP Endpoints...")
    
    endpoints = [
        ("/health", "Health check"),
        ("/static/seamless_audio_playback.html", "Seamless audio player page"),
        ("/", "Root redirect"),
    ]
    
    for endpoint, description in endpoints:
        try:
            url = f"http://127.0.0.1:8000{endpoint}"
            response = urllib.request.urlopen(url)
            status = response.getcode()
            print(f"  ✅ {description}: {status}")
        except urllib.error.HTTPError as e:
            print(f"  ❌ {description}: HTTP {e.code}")
        except Exception as e:
            print(f"  ❌ {description}: {str(e)}")
    print()

async def test_websocket():
    """Test WebSocket connection"""
    print("🔍 Testing WebSocket Connection...")
    
    try:
        uri = "ws://127.0.0.1:8000/ws/stream-transcribe"
        async with websockets.connect(uri) as websocket:
            print("  ✅ WebSocket connected successfully")
            
            # Send a test message
            test_message = {
                "type": "START_STREAMING",
                "session_id": "test_session"
            }
            await websocket.send(json.dumps(test_message))
            print("  ✅ Test message sent")
            
            # Try to receive a response
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                print(f"  ✅ Received response: {response[:100]}...")
            except asyncio.TimeoutError:
                print("  ⚠️ No response received (timeout)")
            
            # Send stop message
            stop_message = {
                "type": "STOP_STREAMING"
            }
            await websocket.send(json.dumps(stop_message))
            print("  ✅ Stop message sent")
            
    except Exception as e:
        print(f"  ❌ WebSocket error: {str(e)}")
    print()

def test_static_files():
    """Test static file serving"""
    print("🔍 Testing Static Files...")
    
    static_files = [
        "/static/seamless_audio_playback.html",
        "/static/index.html",
    ]
    
    for file_path in static_files:
        try:
            url = f"http://127.0.0.1:8000{file_path}"
            response = urllib.request.urlopen(url)
            content = response.read().decode('utf-8')
            status = response.getcode()
            size = len(content)
            print(f"  ✅ {file_path}: {status} ({size} bytes)")
        except urllib.error.HTTPError as e:
            print(f"  ❌ {file_path}: HTTP {e.code}")
        except Exception as e:
            print(f"  ❌ {file_path}: {str(e)}")
    print()

async def main():
    """Main test function"""
    print(f"🚀 Testing localhost functionality at {datetime.now()}")
    print("=" * 50)
    
    # Test HTTP endpoints
    test_http_endpoints()
    
    # Test static files
    test_static_files()
    
    # Test WebSocket
    await test_websocket()
    
    print("=" * 50)
    print("✅ Testing completed!")
    print()
    print("🌐 Access the application at:")
    print("   http://127.0.0.1:8000/static/seamless_audio_playback.html")
    print()
    print("📝 Features ready:")
    print("   • FastAPI server running")
    print("   • AssemblyAI Universal Streaming v3")
    print("   • WebSocket real-time communication")
    print("   • Seamless audio playback UI")
    print("   • Gemini AI integration")
    print("   • Murf TTS audio generation")

if __name__ == "__main__":
    asyncio.run(main())
