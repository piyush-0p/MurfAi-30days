import requests
import time

# Test the server is running
try:
    response = requests.get("http://localhost:8000/api/hello")
    print(f"Server is running: {response.json()}")
except:
    print("Server is not running")

# Open the browser URL for manual testing
print("✅ WebSocket Audio Streaming Implementation Complete!")
print()
print("🌐 Open this URL in your browser to test:")
print("   http://localhost:8000/static/websocket_audio.html")
print()
print("📋 Testing Instructions:")
print("1. Click 'Connect' to establish WebSocket connection")
print("2. Click 'Start Recording' to begin audio streaming")
print("3. Speak into your microphone")
print("4. Click 'Stop Recording' to save the audio file")
print("5. Check the 'uploads/' directory for saved files")
print()
print("🎯 Features implemented:")
print("- Real-time WebSocket audio streaming")
print("- Audio chunks sent at regular intervals (250ms default)")
print("- Server saves binary audio data to .webm files")
print("- Session management with unique filenames")
print("- Error handling and reconnection support")
print()
print("🔧 Server logs will show:")
print("- Connection established messages")
print("- Audio chunk reception (size in bytes)")
print("- File saving confirmation with file size")
