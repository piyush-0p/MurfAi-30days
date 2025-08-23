#!/usr/bin/env python3

import asyncio
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import stream_llm_response

async def test_audio_streaming_direct():
    """
    Test the audio streaming functionality by directly calling the LLM function
    """
    print("🧪 Testing direct audio streaming...")
    
    # Test input that should trigger a response
    test_input = "Hello, how are you today?"
    session_id = "test_session_001"
    
    print(f"📝 Input text: '{test_input}'")
    print("🤖 Calling stream_llm_response (without WebSocket - should just log)...")
    
    # Call without WebSocket (log-only mode)
    response = await stream_llm_response(test_input, session_id)
    
    print(f"✅ Response received: {response}")
    print("📸 Check the console output above for base64 audio logging!")
    print("-" * 80)

if __name__ == "__main__":
    asyncio.run(test_audio_streaming_direct())
