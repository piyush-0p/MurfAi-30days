"""
Enhanced AssemblyAI real-time transcription with Universal Streaming v3 API.
Uses the new streaming.v3 client with built-in turn detection.
"""
import asyncio
import json
import uuid
import time
import logging
from typing import Dict, Any, Callable, Optional, Type
import assemblyai as aai
from assemblyai.streaming.v3 import (
    BeginEvent,
    StreamingClient,
    StreamingClientOptions,
    StreamingError,
    StreamingEvents,
    StreamingParameters,
    StreamingSessionParameters,
    TerminationEvent,
    TurnEvent,
)
import threading

class AssemblyAIStreamer:
    """
    Enhanced AssemblyAI streaming client with turn detection.
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
            # Create RealtimeTranscriber
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
            return False
    
    def _on_open(self, session_opened: aai.RealtimeSessionOpened):
        """Handle session opened"""
        self.session_id = session_opened.session_id
        print(f"🟢 AssemblyAI session opened: {self.session_id}")
    
    def _on_transcript_data(self, transcript: aai.RealtimeTranscript):
        """Handle transcript data with turn detection"""
        try:
            current_time = time.time()
            
            if transcript.message_type == aai.RealtimeTranscriptType.partial_transcript:
                # Handle partial transcript
                text = transcript.text
                if text and self.on_transcript:
                    asyncio.create_task(self.on_transcript(f"PARTIAL: {text}"))
                
                # Update last activity time for any speech
                if text.strip():
                    self.last_transcript_time = current_time
                    
            elif transcript.message_type == aai.RealtimeTranscriptType.final_transcript:
                # Handle final transcript
                text = transcript.text
                if text and self.on_transcript:
                    asyncio.create_task(self.on_transcript(f"FINAL: {text}"))
                
                # Store final transcript for turn detection
                if text.strip():
                    self.last_final_transcript = text
                    self.last_transcript_time = current_time
                    
                    # Start or restart turn detection timer
                    self._start_turn_detection_timer()
                    
        except Exception as e:
            print(f"❌ Error handling transcript: {e}")
    
    def _start_turn_detection_timer(self):
        """Start turn detection timer"""
        # Cancel existing timer
        if self.turn_detection_task:
            self.turn_detection_task.cancel()
        
        # Start new timer
        self.turn_detection_task = asyncio.create_task(self._check_turn_end())
    
    async def _check_turn_end(self):
        """Check if turn has ended after delay"""
        try:
            # Wait for the turn detection delay
            await asyncio.sleep(self.turn_detection_delay)
            
            # Check if we've had speech recently
            current_time = time.time()
            if (self.last_transcript_time and 
                current_time - self.last_transcript_time >= self.turn_detection_delay):
                
                # Turn has ended
                if self.last_final_transcript and self.on_turn_end:
                    print(f"🔇 Turn ended after {self.turn_detection_delay}s silence")
                    await self.on_turn_end(self.last_final_transcript)
                    
                    # Reset state
                    self.last_final_transcript = ""
                    
        except asyncio.CancelledError:
            # Timer was cancelled, which is normal
            pass
        except Exception as e:
            print(f"❌ Error in turn detection: {e}")
    
    def _on_error(self, error: RealtimeError):
        """Handle streaming errors"""
        print(f"❌ AssemblyAI streaming error: {error}")
    
    def _on_close(self):
        """Handle session close"""
        print("🔴 AssemblyAI session closed")
        self.is_connected = False
    
    async def send_audio(self, audio_data: bytes):
        """Send audio data to AssemblyAI"""
        if not self.is_connected or not self.transcriber:
            return False
            
        try:
            # Send audio data in executor to avoid blocking
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.transcriber.stream(audio_data)
            )
            return True
            
        except Exception as e:
            print(f"❌ Error sending audio: {e}")
            return False
    
    async def close(self):
        """Close the connection"""
        if self.turn_detection_task:
            self.turn_detection_task.cancel()
            
        if self.transcriber and self.is_connected:
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.transcriber.close
                )
                print("🔴 AssemblyAI connection closed")
            except Exception as e:
                print(f"❌ Error closing connection: {e}")
        
        self.is_connected = False

# Export the streamer for use in main.py
__all__ = ['AssemblyAIStreamer']
