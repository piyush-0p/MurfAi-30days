"""
Enhanced AssemblyAI real-time transcription with Universal Streaming v3 API.
Uses the new streaming.v3 client with built-in turn detection.
"""
import asyncio
import json
import uuid
import time
import logging
from typing import Dict, Any, Callable, Optional, Type, TYPE_CHECKING
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
    Enhanced AssemblyAI streaming client using Universal Streaming v3 API.
    """
    def __init__(self, api_key: str):
        self.api_key = api_key
        aai.settings.api_key = api_key
        self.client = None
        self.session_id = None
        self.is_connected = False
        self.on_transcript = None
        self.on_turn_end = None
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Audio buffer for streaming
        self.audio_buffer = asyncio.Queue()
        
    async def connect(self, on_transcript=None, on_turn_end=None):
        """Connect to AssemblyAI Universal Streaming service"""
        self.on_transcript = on_transcript
        self.on_turn_end = on_turn_end
        
        try:
            # Create StreamingClient with v3 API
            self.client = StreamingClient(
                StreamingClientOptions(
                    api_key=self.api_key,
                    api_host="streaming.assemblyai.com",
                )
            )
            
            # Register event handlers
            self.client.on(StreamingEvents.Begin, self._on_begin)
            self.client.on(StreamingEvents.Turn, self._on_turn)
            self.client.on(StreamingEvents.Termination, self._on_terminated)
            self.client.on(StreamingEvents.Error, self._on_error)
            
            # Connect to streaming service
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.connect(
                    StreamingParameters(
                        sample_rate=16000,
                        format_turns=True,  # Enable turn formatting
                        encoding="pcm_s16le",
                        min_end_of_turn_silence_when_confident=560,  # 560ms for better turn detection
                    )
                )
            )
            
            self.is_connected = True
            self.session_id = str(uuid.uuid4())
            print("✅ Connected to AssemblyAI Universal Streaming service with turn detection")
            
        except Exception as e:
            print(f"❌ Failed to connect to AssemblyAI Universal Streaming: {e}")
            self.is_connected = False
            raise e
    
    def _on_begin(self, client: Type["StreamingClient"], event: BeginEvent):
        """Handle session begin event"""
        self.session_id = event.id
        print(f"🎯 Universal Streaming session started: {event.id}")
    
    def _on_turn(self, client: Type["StreamingClient"], event: TurnEvent):
        """Handle turn event with transcript and turn detection"""
        try:
            # Process transcript data
            if self.on_transcript and event.transcript:
                transcript_data = {
                    "text": event.transcript,
                    "is_final": event.end_of_turn,
                    "turn_order": event.turn_order,
                    "confidence": getattr(event, 'end_of_turn_confidence', 0.8),
                    "words": getattr(event, 'words', [])
                }
                
                # Call transcript callback
                if asyncio.iscoroutinefunction(self.on_transcript):
                    # Schedule coroutine for execution
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            asyncio.create_task(self.on_transcript(transcript_data))
                        else:
                            loop.run_until_complete(self.on_transcript(transcript_data))
                    except RuntimeError:
                        # If no event loop, call synchronously
                        print("⚠️ No event loop available for async transcript callback")
                else:
                    self.on_transcript(transcript_data)
            
            # Handle end of turn
            if event.end_of_turn and self.on_turn_end:
                turn_data = {
                    "transcript": event.transcript,
                    "turn_order": event.turn_order,
                    "is_formatted": event.turn_is_formatted,
                    "confidence": getattr(event, 'end_of_turn_confidence', 0.8)
                }
                
                print(f"🔚 Turn ended: {event.transcript} (formatted: {event.turn_is_formatted})")
                
                # Handle async turn end callback
                if asyncio.iscoroutinefunction(self.on_turn_end):
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            asyncio.create_task(self.on_turn_end(turn_data))
                        else:
                            loop.run_until_complete(self.on_turn_end(turn_data))
                    except RuntimeError:
                        # If no event loop, call synchronously  
                        print("⚠️ No event loop available for async turn end callback")
                else:
                    self.on_turn_end(turn_data)
        
        except Exception as e:
            print(f"❌ Error processing turn event: {e}")
    
    def _on_terminated(self, client: Type["StreamingClient"], event: TerminationEvent):
        """Handle session termination"""
        print(f"🔴 Universal Streaming session terminated: {event.audio_duration_seconds}s processed")
        self.is_connected = False
    
    def _on_error(self, client: Type["StreamingClient"], error: StreamingError):
        """Handle streaming errors"""
        print(f"❌ Universal Streaming error: {error}")
        self.is_connected = False
    
    async def send_audio(self, audio_data: bytes):
        """Send audio data to AssemblyAI Universal Streaming"""
        if not self.is_connected or not self.client:
            print("⚠️ Cannot send audio: not connected to Universal Streaming")
            return
        
        try:
            # Send audio to streaming client
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.send_audio(audio_data)
            )
        except Exception as e:
            print(f"❌ Error sending audio to Universal Streaming: {e}")
    
    async def close(self):
        """Close the Universal Streaming connection"""
        try:
            if self.client and self.is_connected:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.client.disconnect(terminate=True)
                )
                print("🔌 Universal Streaming connection closed")
            
            self.is_connected = False
            self.client = None
            
        except Exception as e:
            print(f"❌ Error closing Universal Streaming connection: {e}")
    
    def get_session_info(self):
        """Get current session information"""
        return {
            "session_id": self.session_id,
            "is_connected": self.is_connected,
            "api_version": "universal-streaming-v3"
        }
