"""
Simple AssemblyAI transcription using HTTP API with chunked processing.
This is a more reliable approach than WebSocket for now.
"""
import asyncio
import httpx
import json
import uuid
import time
from typing import Dict, Any, Callable, Optional
import base64
import io


class AssemblyAIHttpStreamer:
    """
    AssemblyAI transcription using HTTP API with simulated streaming.
    More reliable than WebSocket but still provides chunk-by-chunk processing.
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session_id = None
        self.is_connected = False
        self.on_transcript = None
        self.on_turn_end = None
        
        # Audio buffer for accumulating chunks
        self.audio_buffer = io.BytesIO()
        self.buffer_size = 0
        self.max_buffer_size = 1024 * 1024  # 1MB buffer
        
        # Turn detection state
        self.last_transcript_time = None
        self.last_final_transcript = ""
        self.turn_detection_delay = 2.0
        self.turn_detection_task = None
        
        # Processing state
        self.processing_task = None
        self.should_process = False
        
        # HTTP client
        self.client = None
        
    async def connect(self, on_transcript=None, on_turn_end=None):
        """Initialize the HTTP streaming session"""
        self.on_transcript = on_transcript
        self.on_turn_end = on_turn_end
        
        try:
            print(f"🔄 Initializing AssemblyAI HTTP streaming...")
            
            # Create HTTP client
            self.client = httpx.AsyncClient(
                timeout=30.0,
                headers={
                    "authorization": self.api_key,
                    "content-type": "application/json"
                }
            )
            
            self.is_connected = True
            self.session_id = str(uuid.uuid4())[:8]
            self.should_process = True
            
            # Start processing task
            self.processing_task = asyncio.create_task(self._process_audio_buffer())
            
            print(f"✅ AssemblyAI HTTP streaming session {self.session_id} ready")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize AssemblyAI HTTP streaming: {e}")
            self.is_connected = False
            return False
    
    async def _process_audio_buffer(self):
        """Process accumulated audio buffer periodically"""
        try:
            while self.should_process:
                # Wait for some audio to accumulate
                await asyncio.sleep(3.0)  # Process every 3 seconds
                
                if self.buffer_size > 0:
                    await self._process_current_buffer()
                    
        except asyncio.CancelledError:
            print(f"🔚 Audio processing task cancelled")
        except Exception as e:
            print(f"❌ Error in audio processing: {e}")
    
    async def _process_current_buffer(self):
        """Process the current audio buffer"""
        try:
            if self.buffer_size == 0:
                return
                
            print(f"🎯 Processing audio buffer: {self.buffer_size} bytes")
            
            # Get current buffer content
            buffer_content = self.audio_buffer.getvalue()
            
            # Reset buffer for new audio
            self.audio_buffer = io.BytesIO()
            self.buffer_size = 0
            
            # Convert to base64 for AssemblyAI upload
            audio_base64 = base64.b64encode(buffer_content).decode('utf-8')
            
            # Upload audio data
            upload_response = await self.client.post(
                "https://api.assemblyai.com/v2/upload",
                data=buffer_content,
                headers={"content-type": "application/octet-stream"}
            )
            
            if upload_response.status_code != 200:
                print(f"❌ Failed to upload audio: {upload_response.text}")
                return
                
            upload_url = upload_response.json()["upload_url"]
            print(f"✅ Audio uploaded: {upload_url}")
            
            # Submit transcription request
            transcript_request = {
                "audio_url": upload_url,
                "language_code": "en",
                "punctuate": True,
                "format_text": True
            }
            
            transcript_response = await self.client.post(
                "https://api.assemblyai.com/v2/transcript",
                json=transcript_request
            )
            
            if transcript_response.status_code != 200:
                print(f"❌ Failed to submit transcription: {transcript_response.text}")
                return
                
            transcript_data = transcript_response.json()
            transcript_id = transcript_data["id"]
            
            print(f"🎯 Transcription submitted: {transcript_id}")
            
            # Poll for results
            await self._poll_transcription_results(transcript_id)
            
        except Exception as e:
            print(f"❌ Error processing audio buffer: {e}")
    
    async def _poll_transcription_results(self, transcript_id: str):
        """Poll for transcription results"""
        try:
            max_attempts = 30  # 30 seconds timeout
            attempt = 0
            
            while attempt < max_attempts:
                # Get transcription status
                status_response = await self.client.get(
                    f"https://api.assemblyai.com/v2/transcript/{transcript_id}"
                )
                
                if status_response.status_code != 200:
                    print(f"❌ Failed to get transcription status: {status_response.text}")
                    return
                    
                result_data = status_response.json()
                status = result_data.get("status")
                
                if status == "completed":
                    # Process the transcription result
                    transcript_text = result_data.get("text", "")
                    confidence = result_data.get("confidence", 0.0)
                    
                    if transcript_text.strip():
                        print(f"🎯 Transcription result: '{transcript_text}' (confidence: {confidence:.2f})")
                        
                        # Update turn detection
                        self.last_transcript_time = time.time()
                        self.last_final_transcript = transcript_text.strip()
                        
                        # Call transcript callback
                        if self.on_transcript:
                            await self.on_transcript({
                                'text': transcript_text,
                                'is_final': True,
                                'confidence': confidence,
                                'session_id': self.session_id
                            })
                        
                        # Schedule turn detection
                        await self._schedule_turn_detection()
                    else:
                        print(f"🔇 Empty transcription result")
                    
                    return
                    
                elif status == "error":
                    error_msg = result_data.get("error", "Unknown error")
                    print(f"❌ Transcription error: {error_msg}")
                    return
                    
                else:
                    # Still processing
                    print(f"⏳ Transcription in progress: {status}")
                    await asyncio.sleep(1.0)
                    attempt += 1
            
            print(f"⏰ Transcription timeout after {max_attempts} seconds")
            
        except Exception as e:
            print(f"❌ Error polling transcription results: {e}")
    
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
            print(f"⏹️ Turn detection cancelled (new speech)")
            pass
        except Exception as e:
            print(f"❌ Error in turn detection: {e}")
    
    async def send_audio(self, audio_data: bytes):
        """Add audio data to the buffer for processing"""
        if not self.is_connected:
            print(f"❌ Cannot send audio - not connected to AssemblyAI")
            return False
        
        try:
            # Add audio data to buffer
            self.audio_buffer.write(audio_data)
            self.buffer_size += len(audio_data)
            
            # If buffer is getting large, trigger immediate processing
            if self.buffer_size >= self.max_buffer_size:
                asyncio.create_task(self._process_current_buffer())
            
            return True
            
        except Exception as e:
            print(f"❌ Error adding audio to buffer: {e}")
            return False
    
    async def close(self):
        """Close the HTTP streaming session"""
        try:
            self.is_connected = False
            self.should_process = False
            
            # Cancel tasks
            if self.processing_task:
                self.processing_task.cancel()
            if self.turn_detection_task:
                self.turn_detection_task.cancel()
            
            # Close HTTP client
            if self.client:
                await self.client.aclose()
                
            print(f"✅ AssemblyAI HTTP streaming connection closed")
                
        except Exception as e:
            print(f"❌ Error closing AssemblyAI HTTP streaming: {e}")
    
    def __del__(self):
        """Cleanup on destruction"""
        if self.is_connected:
            try:
                asyncio.create_task(self.close())
            except:
                pass
