"""
Alternative transcription endpoint using AssemblyAI v3 streaming API with proper SSL handling.
This will replace the problematic RealtimeTranscriber approach.
"""
import asyncio
import json
import websockets
import ssl
import uuid
from typing import Dict, Any
import tempfile
import subprocess
import wave

class AssemblyAIStreamer:
    """
    Custom AssemblyAI streaming client that handles SSL properly.
    """
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.ws = None
        self.session_id = None
        
    async def connect(self, on_transcript=None):
        """Connect to AssemblyAI streaming service"""
        self.on_transcript = on_transcript
        
        # Create SSL context that works with AssemblyAI
        ssl_context = ssl.create_default_context()
        
        # AssemblyAI streaming endpoint
        uri = "wss://api.assemblyai.com/v2/realtime/ws"
        
        try:
            # Connect with proper headers
            extra_headers = {
                "Authorization": self.api_key
            }
            
            self.ws = await websockets.connect(
                uri,
                ssl=ssl_context,
                extra_headers=extra_headers,
                ping_interval=20,
                ping_timeout=10
            )
            
            # Start listening for messages
            asyncio.create_task(self._listen())
            
            print("✅ Connected to AssemblyAI streaming service")
            return True
            
        except Exception as e:
            print(f"❌ Failed to connect to AssemblyAI: {e}")
            return False
    
    async def _listen(self):
        """Listen for messages from AssemblyAI"""
        try:
            async for message in self.ws:
                data = json.loads(message)
                
                if data.get("message_type") == "SessionBegins":
                    self.session_id = data.get("session_id")
                    print(f"🟢 AssemblyAI session began: {self.session_id}")
                    
                elif data.get("message_type") == "PartialTranscript":
                    text = data.get("text", "")
                    if text and self.on_transcript:
                        await self.on_transcript(f"PARTIAL: {text}")
                        
                elif data.get("message_type") == "FinalTranscript":
                    text = data.get("text", "")
                    if text and self.on_transcript:
                        await self.on_transcript(f"FINAL: {text}")
                        
        except Exception as e:
            print(f"❌ Error listening to AssemblyAI: {e}")
    
    async def send_audio(self, audio_data: bytes):
        """Send audio data to AssemblyAI"""
        if not self.ws:
            return False
            
        try:
            # Send binary audio data
            await self.ws.send(audio_data)
            return True
        except Exception as e:
            print(f"❌ Error sending audio: {e}")
            return False
    
    async def close(self):
        """Close the connection"""
        if self.ws:
            await self.ws.close()
            print("🔴 AssemblyAI connection closed")

# Export the streamer for use in main.py
__all__ = ['AssemblyAIStreamer']
