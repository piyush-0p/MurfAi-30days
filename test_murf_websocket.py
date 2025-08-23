#!/usr/bin/env python3
"""
Test script for Murf WebSocket TTS integration
"""

import asyncio
import sys
import os
sys.path.append('/Users/apple/Documents/MurfAIChallenge')

# Import the function from main.py
from main import send_to_murf_websocket

async def test_murf_websocket():
    """Test the Murf WebSocket TTS functionality"""
    
    # Test text that would come from LLM
    test_text = "Hello! This is a test of the Murf WebSocket text-to-speech integration with streaming LLM responses."
    
    print(f"🧪 Testing Murf WebSocket TTS with text:")
    print(f"   '{test_text}'")
    print("-" * 80)
    
    try:
        # Call the Murf WebSocket function
        audio_base64 = await send_to_murf_websocket(test_text)
        
        if audio_base64:
            print("\n" + "="*80)
            print("✅ SUCCESS! Murf WebSocket TTS completed!")
            print(f"📊 Generated {len(audio_base64)} characters of base64 encoded audio")
            print("\n🎵 BASE64 ENCODED AUDIO:")
            print("-" * 40)
            print(audio_base64)
            print("-" * 40)
            print(f"\n📸 This is what gets printed to console when LLM streaming triggers Murf TTS!")
            print("\n🎯 Integration working perfectly - ready for LinkedIn screenshot!")
        else:
            print("\n❌ Failed to generate audio from Murf WebSocket")
            
    except Exception as e:
        print(f"\n❌ Error testing Murf WebSocket: {str(e)}")

if __name__ == "__main__":
    print("🚀 Starting Murf WebSocket TTS Test")
    print("="*80)
    asyncio.run(test_murf_websocket())
