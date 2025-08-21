#!/usr/bin/env python3
"""
Test script to demonstrate LLM streaming functionality
This simulates what happens when AssemblyAI sends a final transcript
"""

import asyncio
import httpx
import json
import os
from app.core.config import settings

async def test_llm_streaming():
    """Test the LLM streaming response generation"""
    
    # Sample transcript that would come from AssemblyAI
    test_transcript = "Hello, I'm testing the real-time speech to text functionality with turn detection and LLM responses."
    
    print(f"🧪 Testing LLM streaming with transcript: '{test_transcript}'")
    print("-" * 80)
    
    # Prepare the LLM request
    llm_payload = {
        "contents": [{
            "parts": [{
                "text": f"Please provide a helpful and concise response to this user input: '{test_transcript}'"
            }]
        }]
    }
    
    full_response = ""
    
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            print("🤖 Sending request to Gemini API...")
            
            async with client.stream(
                'POST',
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:streamGenerateContent?key={settings.GEMINI_API_KEY}",
                json=llm_payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                
                if response.status_code != 200:
                    print(f"❌ API Error: {response.status_code}")
                    return
                
                print("✅ Receiving streaming response...")
                print("🔄 LLM Response (streaming):")
                print("-" * 40)
                
                async for line in response.aiter_lines():
                    if line.strip():
                        try:
                            # Parse Server-Sent Events format
                            if line.startswith('data: '):
                                json_data = line[6:]  # Remove 'data: ' prefix
                                data = json.loads(json_data)
                                
                                if 'candidates' in data and data['candidates']:
                                    candidate = data['candidates'][0]
                                    if 'content' in candidate and 'parts' in candidate['content']:
                                        for part in candidate['content']['parts']:
                                            if 'text' in part:
                                                chunk = part['text']
                                                print(chunk, end='', flush=True)
                                                full_response += chunk
                        except json.JSONDecodeError:
                            continue
                
                print("\n" + "-" * 40)
                print(f"📄 Complete LLM Response ({len(full_response)} characters):")
                print(f"'{full_response}'")
                print("\n✅ LLM streaming test completed successfully!")
                
    except Exception as e:
        print(f"❌ Error during LLM streaming test: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_llm_streaming())
