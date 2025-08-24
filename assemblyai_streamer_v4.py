"""
AssemblyAI Universal Streaming v4 - Fixed implementation for real-time transcription.
Based on latest documentation and proper async handling.
"""
import asyncio
import json
import uuid
import time
import websockets
import ssl
from typing import Dict, Any, Callable, Optional
import base64


class AssemblyAIStreamerV4:
    """
    AssemblyAI Universal Streaming implementation using direct WebSocket connection.
    Fixed async handling and proper message formatting.
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.websocket = None
        self.session_id = None
        self.is_connected = False
        self.on_transcript = None
        self.on_turn_end = None
        
        # Turn detection state
        self.last_transcript_time = None
        self.last_final_transcript = ""
        self.turn_detection_delay = 2.0  # 2 seconds of silence to detect turn end
        self.turn_detection_task = None
        
        # WebSocket URL for Universal Streaming
        self.ws_url = "wss://api.assemblyai.com/v2/realtime/ws"
        
    async def connect(self, on_transcript=None, on_turn_end=None):
        """Connect to AssemblyAI Universal Streaming WebSocket"""
        self.on_transcript = on_transcript
        self.on_turn_end = on_turn_end
        
        try:
            print(f"🔄 Connecting to AssemblyAI Universal Streaming...")
            
            # Create SSL context
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            # Connect to WebSocket with proper authentication format
            # AssemblyAI expects the token as a query parameter
            ws_url_with_auth = f"{self.ws_url}?token={self.api_key}"
            
            self.websocket = await websockets.connect(
                ws_url_with_auth,
                ssl=ssl_context
            )
            
            print(f"✅ Connected to AssemblyAI WebSocket")
            
            # Send initial configuration message with auth
            config_message = {
                "audio_data": None,  # No audio data in config message
                "sample_rate": 16000
            }
            
            await self.websocket.send(json.dumps(config_message))
            print(f"📤 Sent configuration to AssemblyAI")
            
            self.is_connected = True
            self.session_id = str(uuid.uuid4())[:8]
            
            # Start listening for messages
            asyncio.create_task(self._listen_for_messages())
            
            print(f"✅ AssemblyAI Universal Streaming session {self.session_id} ready")
            return True
            
        except Exception as e:
            print(f"❌ Failed to connect to AssemblyAI: {e}")
            self.is_connected = False
            return False
    
    async def _listen_for_messages(self):
        """Listen for messages from AssemblyAI WebSocket"""
        try:
            while self.is_connected and self.websocket:
                try:
                    message = await self.websocket.recv()
                    await self._handle_message(message)
                except websockets.exceptions.ConnectionClosed:
                    print(f"🔴 AssemblyAI WebSocket connection closed")
                    self.is_connected = False
                    break
                except Exception as e:
                    print(f"❌ Error receiving message: {e}")
                    
        except Exception as e:
            print(f"❌ Error in message listener: {e}")
        finally:
            self.is_connected = False
    
    async def _handle_message(self, message):
        """Handle incoming messages from AssemblyAI"""
        try:
            data = json.loads(message)
            
            # Handle transcript messages
            if "text" in data:
                transcript_text = data["text"]
                is_final = data.get("message_type") == "FinalTranscript"
                confidence = data.get("confidence", 0.0)
                
                print(f"🎯 Transcript: '{transcript_text}' (final: {is_final}, confidence: {confidence:.2f})")
                
                # Update turn detection state
                if transcript_text.strip():
                    self.last_transcript_time = time.time()
                    
                    # Store final transcript for turn detection
                    if is_final:
                        self.last_final_transcript = transcript_text.strip()
                
                # Call transcript callback
                if self.on_transcript:
                    await self.on_transcript({
                        'text': transcript_text,
                        'is_final': is_final,
                        'confidence': confidence,
                        'session_id': self.session_id
                    })
                
                # Handle turn detection for final transcripts
                if is_final and transcript_text.strip():
                    await self._schedule_turn_detection()
            
            # Handle session information
            elif data.get("message_type") == "SessionInformation":
                print(f"📋 AssemblyAI Session Info: {data}")
                
            else:
                print(f"📨 Other AssemblyAI message: {data}")
                
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse AssemblyAI message: {e}")
        except Exception as e:
            print(f"❌ Error handling AssemblyAI message: {e}")
    
    async def _schedule_turn_detection(self):
        """Schedule turn end detection after silence period"""
        try:
            # Cancel existing turn detection
            if self.turn_detection_task:
                self.turn_detection_task.cancel()
            
            # Schedule new turn detection
            self.turn_detection_task = asyncio.create_task(
                self._detect_turn_end()
            )
            
        except Exception as e:
            print(f"❌ Error scheduling turn detection: {e}")
    
    async def _detect_turn_end(self):
        """Detect when user has finished speaking (turn end)"""
        try:
            # Wait for the detection delay
            await asyncio.sleep(self.turn_detection_delay)
            
            # Check if we still have the same last transcript time
            current_time = time.time()
            if (self.last_transcript_time and 
                current_time - self.last_transcript_time >= self.turn_detection_delay):
                
                print(f"🔚 Turn end detected after {self.turn_detection_delay}s silence")
                
                # Call turn end callback
                if self.on_turn_end:
                    await self.on_turn_end(self.last_final_transcript)
                    
        except asyncio.CancelledError:
            # Turn detection was cancelled (new speech detected)
            print(f"⏹️ Turn detection cancelled (new speech)")
            pass
        except Exception as e:
            print(f"❌ Error in turn detection: {e}")
    
    async def send_audio(self, audio_data: bytes):
        """Send audio data to AssemblyAI"""
        if not self.is_connected or not self.websocket:
            print(f"❌ Cannot send audio - not connected to AssemblyAI")
            return False
        
        try:
            # Encode audio data as base64
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            
            # Send audio message
            message = {
                "audio_data": audio_base64
            }
            
            await self.websocket.send(json.dumps(message))
            return True
            
        except Exception as e:
            print(f"❌ Error sending audio to AssemblyAI: {e}")
            return False
    
    async def close(self):
        """Close the AssemblyAI connection"""
        try:
            self.is_connected = False
            
            # Cancel turn detection
            if self.turn_detection_task:
                self.turn_detection_task.cancel()
            
            # Close WebSocket
            if self.websocket:
                await self.websocket.close()
                print(f"✅ AssemblyAI connection closed")
                
        except Exception as e:
            print(f"❌ Error closing AssemblyAI connection: {e}")
    
    def __del__(self):
        """Cleanup on destruction"""
        if self.is_connected:
            try:
                asyncio.create_task(self.close())
            except:
                pass
