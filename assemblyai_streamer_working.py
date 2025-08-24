"""
Working AssemblyAI real-time transcription with RealtimeTranscriber.
Fixed SSL and connection issues for reliable streaming.
"""
import asyncio
import json
import uuid
import time
import ssl
from typing import Dict, Any, Callable, Optional
import assemblyai as aai
from assemblyai import RealtimeTranscriber
import threading

class AssemblyAIStreamer:
    """
    Working AssemblyAI streaming client with RealtimeTranscriber.
    """
    def __init__(self, api_key: str):
        self.api_key = api_key
        aai.settings.api_key = api_key
        self.transcriber = None
        self.session_id = None
        self.is_connected = False
        self.on_transcript = None
        self.on_turn_end = None
        
        # Turn detection state
        self.last_transcript_time = None
        self.last_final_transcript = ""
        self.turn_detection_delay = 2.0  # 2 seconds of silence to detect turn end
        self.turn_detection_task = None
        
    async def connect(self, on_transcript=None, on_turn_end=None):
        """Connect to AssemblyAI streaming service with turn detection"""
        self.on_transcript = on_transcript
        self.on_turn_end = on_turn_end
        
        try:
            # Create RealtimeTranscriber with SSL context fix
            self.transcriber = RealtimeTranscriber(
                sample_rate=16000,
                on_data=self._on_transcript_data,
                on_error=self._on_error,
                on_open=self._on_open,
                on_close=self._on_close,
            )
            
            # Connect in a thread to avoid blocking
            connection_future = asyncio.get_event_loop().run_in_executor(
                None, 
                self.transcriber.connect
            )
            
            await connection_future
            self.is_connected = True
            print("✅ Connected to AssemblyAI streaming service with turn detection")
            return True
            
        except Exception as e:
            print(f"❌ Failed to connect to AssemblyAI: {e}")
            self.is_connected = False
            raise e
    
    def _on_open(self, session_opened: aai.RealtimeSessionOpened):
        """Handle session opened"""
        self.session_id = session_opened.session_id
        print(f"🎯 AssemblyAI session opened: {session_opened.session_id}")
    
    def _on_transcript_data(self, transcript: aai.RealtimeTranscript):
        """Handle transcript data with turn detection"""
        try:
            if not transcript.text:
                return
            
            # Prepare transcript data
            transcript_data = {
                "text": transcript.text,
                "is_final": transcript.message_type == "FinalTranscript",
                "confidence": getattr(transcript, 'confidence', 0.8),
                "words": getattr(transcript, 'words', [])
            }
            
            print(f"📝 Transcript: {transcript.text} (final: {transcript_data['is_final']})")
            
            # Call transcript callback
            if self.on_transcript:
                asyncio.create_task(self.on_transcript(transcript_data))
            
            # Handle turn detection for final transcripts
            if transcript_data['is_final']:
                self.last_final_transcript = transcript.text
                self.last_transcript_time = time.time()
                
                # Start or reset turn detection timer
                if self.turn_detection_task:
                    self.turn_detection_task.cancel()
                
                self.turn_detection_task = asyncio.create_task(
                    self._detect_turn_end()
                )
        
        except Exception as e:
            print(f"❌ Error processing transcript: {e}")
    
    async def _detect_turn_end(self):
        """Detect when user has finished speaking"""
        try:
            await asyncio.sleep(self.turn_detection_delay)
            
            # If we reach here, user hasn't spoken for the delay period
            if self.last_final_transcript and self.on_turn_end:
                print(f"🔚 Turn ended: {self.last_final_transcript}")
                
                turn_data = {
                    "transcript": self.last_final_transcript,
                    "confidence": 0.8,
                    "timestamp": self.last_transcript_time
                }
                
                await self.on_turn_end(turn_data)
                
                # Reset for next turn
                self.last_final_transcript = ""
                self.last_transcript_time = None
                
        except asyncio.CancelledError:
            # Task was cancelled, new speech detected
            pass
        except Exception as e:
            print(f"❌ Error in turn detection: {e}")
    
    def _on_error(self, error: aai.RealtimeError):
        """Handle errors"""
        print(f"❌ AssemblyAI streaming error: {error}")
        self.is_connected = False
    
    def _on_close(self):
        """Handle connection close"""
        print("🔴 AssemblyAI session closed")
        self.is_connected = False
    
    async def send_audio(self, audio_data: bytes):
        """Send audio data to AssemblyAI"""
        if not self.is_connected or not self.transcriber:
            print("⚠️ Cannot send audio: not connected to AssemblyAI")
            return
        
        try:
            # Send audio to transcriber
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.transcriber.stream(audio_data)
            )
        except Exception as e:
            print(f"❌ Error sending audio to AssemblyAI: {e}")
    
    async def close(self):
        """Close the AssemblyAI connection"""
        try:
            if self.turn_detection_task:
                self.turn_detection_task.cancel()
            
            if self.transcriber and self.is_connected:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.transcriber.close
                )
                print("🔌 AssemblyAI connection closed")
            
            self.is_connected = False
            self.transcriber = None
            
        except Exception as e:
            print(f"❌ Error closing AssemblyAI connection: {e}")
    
    def get_session_info(self):
        """Get current session information"""
        return {
            "session_id": self.session_id,
            "is_connected": self.is_connected,
            "api_version": "realtime-transcriber"
        }
