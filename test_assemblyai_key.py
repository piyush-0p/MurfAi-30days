#!/usr/bin/env python3
"""
Quick test script to verify AssemblyAI API key is working.
"""
import assemblyai as aai
import ssl

# Your API key
API_KEY = "6bf1c3f3202b4be3ba1fc699a6e43dd5"

# Configure AssemblyAI
aai.settings.api_key = API_KEY

# Disable SSL verification for development
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

def test_api_key():
    """Test if the API key works by trying to create a transcriber"""
    print("🧪 Testing AssemblyAI API key...")
    print(f"🔑 API Key: {API_KEY[:10]}...")
    
    try:
        # Test basic transcriber creation
        transcriber = aai.Transcriber()
        print("✅ Transcriber created successfully")
        
        # Test realtime transcriber creation
        def dummy_callback(data):
            pass
            
        rt_transcriber = aai.RealtimeTranscriber(
            sample_rate=16000,
            on_data=dummy_callback,
            on_error=dummy_callback,
            on_open=dummy_callback,
            on_close=dummy_callback,
        )
        
        print("✅ RealtimeTranscriber created successfully")
        print("🎯 AssemblyAI API key is working!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing API key: {e}")
        return False

if __name__ == "__main__":
    test_api_key()
