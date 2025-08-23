from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form, Path, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import Dict, List, Optional
import os
import httpx
import json
import uuid
from datetime import datetime
import aiofiles
import assemblyai as aai
# Remove v3 imports - we'll use the working RealtimeTranscriber with SSL fix
from assemblyai_streamer import AssemblyAIStreamer
import asyncio
import threading
import queue
import tempfile
import subprocess
import wave
import websockets
import base64
import io
import ssl

app = FastAPI(title="MurfAI Challenge API")

# In-memory chat history storage
chat_sessions: Dict[str, List[Dict[str, str]]] = {}

# Create uploads directory if it doesn't exist
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# AssemblyAI Configuration
# You'll need to set your AssemblyAI API key here
# For development, you can set it directly or use environment variable
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY", "6bf1c3f3202b4be3ba1fc699a6e43dd5")

# Initialize AssemblyAI with API key
if ASSEMBLYAI_API_KEY != "your_assemblyai_api_key_here":
    aai.settings.api_key = ASSEMBLYAI_API_KEY
    print("✅ AssemblyAI API key configured successfully")
else:
    print("⚠️  Warning: AssemblyAI API key not set. Please set ASSEMBLYAI_API_KEY environment variable or update main.py")

# Gemini API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyAGHYYINDcdMGZgE6VXCSGlKhKEIdcDjFg")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:streamGenerateContent"

if GEMINI_API_KEY != "your_gemini_api_key_here":
    print("✅ Gemini API key configured successfully")
else:
    print("⚠️  Warning: Gemini API key not set. Please set GEMINI_API_KEY environment variable or update main.py")

# LLM Streaming Response Function
async def stream_llm_response(user_input: str, session_id: str = None, websocket: WebSocket = None) -> str:
    """
    Stream LLM response using Gemini API and accumulate the full response.
    If websocket is provided, will stream audio chunks to the client.
    Returns the complete accumulated response.
    """
    if not user_input.strip():
        return ""
    
    try:
        print(f"🤖 Starting LLM streaming for input: '{user_input[:50]}...'")
        
        # Prepare the request payload for streaming
        payload = {
            "contents": [{
                "parts": [{
                    "text": f"You are a helpful AI assistant. Please respond to: {user_input}"
                }]
            }],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 500,
                "topP": 0.8,
                "topK": 40
            }
        }
        
        headers = {
            "Content-Type": "application/json",
        }
        
        # Add API key to URL
        url = f"{GEMINI_API_URL}?key={GEMINI_API_KEY}&alt=sse"
        
        accumulated_response = ""
        chunk_count = 0
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            async with client.stream(
                "POST", 
                url, 
                headers=headers, 
                json=payload
            ) as response:
                
                if response.status_code != 200:
                    error_text = await response.atext()
                    print(f"❌ LLM API error {response.status_code}: {error_text}")
                    return f"Error: Failed to get LLM response ({response.status_code})"
                
                print("🔄 Streaming LLM response...")
                
                async for line in response.aiter_lines():
                    if line.strip():
                        # Parse Server-Sent Events format
                        if line.startswith("data: "):
                            chunk_count += 1
                            data_str = line[6:]  # Remove "data: " prefix
                            
                            if data_str.strip() == "[DONE]":
                                break
                                
                            try:
                                chunk_data = json.loads(data_str)
                                
                                # Extract text from Gemini streaming response
                                if "candidates" in chunk_data:
                                    for candidate in chunk_data["candidates"]:
                                        if "content" in candidate:
                                            if "parts" in candidate["content"]:
                                                for part in candidate["content"]["parts"]:
                                                    if "text" in part:
                                                        chunk_text = part["text"]
                                                        accumulated_response += chunk_text
                                                        print(f"🔄 Chunk {chunk_count}: {chunk_text}")
                                                        
                            except json.JSONDecodeError as e:
                                print(f"⚠️ Failed to parse streaming chunk: {e}")
                                continue
        
        print(f"✅ LLM streaming complete! Total chunks: {chunk_count}")
        print(f"🎯 Complete LLM response: {accumulated_response}")
        
        # Send the complete LLM response to Murf WebSocket for TTS and stream to client
        if accumulated_response.strip():
            if websocket and websocket.client_state.name == "CONNECTED":
                # Stream audio chunks to client
                await send_to_murf_websocket(accumulated_response.strip(), websocket_client=websocket)
            else:
                # Log-only mode (no streaming to client)
                murf_audio_base64 = await send_to_murf_websocket(accumulated_response.strip())
                if murf_audio_base64:
                    print(f"🔊 Murf WebSocket TTS generated {len(murf_audio_base64)} characters of base64 audio:")
                    print(f"📄 Base64 Audio: {murf_audio_base64[:100]}..." if len(murf_audio_base64) > 100 else f"📄 Base64 Audio: {murf_audio_base64}")
        
        return accumulated_response.strip()
        
    except Exception as e:
        error_msg = f"❌ Error streaming LLM response: {str(e)}"
        print(error_msg)
        return f"Error: {str(e)}"

async def stream_audio_chunks_to_client(audio_base64: str, websocket: WebSocket, chunk_size: int = 1024):
    """
    Stream base64 audio data to client in chunks and wait for acknowledgments.
    
    Args:
        audio_base64: The complete base64 encoded audio data
        websocket: WebSocket connection to the client
        chunk_size: Size of each chunk to send (default 1024 characters)
    """
    try:
        print(f"🔊 Starting audio stream to client: {len(audio_base64)} characters in chunks of {chunk_size}")
        
        # Send audio start notification
        await websocket.send_json({
            "type": "AUDIO_START",
            "total_size": len(audio_base64),
            "chunk_size": chunk_size,
            "total_chunks": (len(audio_base64) + chunk_size - 1) // chunk_size
        })
        
        # Send audio data in chunks
        chunk_count = 0
        for i in range(0, len(audio_base64), chunk_size):
            chunk = audio_base64[i:i + chunk_size]
            chunk_count += 1
            
            # Send chunk to client
            await websocket.send_json({
                "type": "AUDIO_CHUNK",
                "chunk_index": chunk_count,
                "data": chunk,
                "is_final": i + chunk_size >= len(audio_base64)
            })
            
            print(f"📤 Sent audio chunk {chunk_count}: {len(chunk)} characters")
            
            # Wait for client acknowledgment (with timeout)
            try:
                # Use a timeout to avoid hanging if client doesn't respond
                ack_received = False
                timeout_count = 0
                max_timeout_ms = 100  # 100ms timeout per chunk
                
                while not ack_received and timeout_count < max_timeout_ms:
                    try:
                        # Check if there's an incoming message (non-blocking)
                        ack_msg = await asyncio.wait_for(
                            websocket.receive(),
                            timeout=0.001  # 1ms check
                        )
                        
                        if ack_msg.get("type") == "websocket.receive" and ack_msg.get("text"):
                            ack_data = json.loads(ack_msg["text"])
                            if (ack_data.get("type") == "AUDIO_CHUNK_ACK" and 
                                ack_data.get("chunk_index") == chunk_count):
                                ack_received = True
                                print(f"✅ Client acknowledged audio chunk {chunk_count}")
                                break
                                
                    except asyncio.TimeoutError:
                        timeout_count += 1
                        continue
                    except Exception as ack_error:
                        print(f"⚠️ Error waiting for chunk {chunk_count} acknowledgment: {ack_error}")
                        break
                
                if not ack_received:
                    print(f"⚠️ No acknowledgment received for chunk {chunk_count} (continuing anyway)")
            
            except Exception as chunk_error:
                print(f"❌ Error processing chunk {chunk_count}: {chunk_error}")
        
        # Send completion notification
        await websocket.send_json({
            "type": "AUDIO_COMPLETE",
            "total_chunks_sent": chunk_count
        })
        
        print(f"🎉 Audio streaming complete! Sent {chunk_count} chunks to client")
        
    except Exception as e:
        print(f"❌ Error streaming audio to client: {e}")

async def send_to_murf_websocket(text: str, context_id: str = "stream_tts_context_001", websocket_client: WebSocket = None) -> Optional[str]:
    """
    Send text to Murf API for TTS conversion and return base64 encoded audio.
    If websocket_client is provided, streams audio chunks to the client.
    Since Murf WebSocket endpoint returned HTTP 405, we'll use the HTTP API 
    and encode the result as base64 for demonstration.
    
    Args:
        text: The text to convert to speech
        context_id: Context ID for tracking (static to avoid context limits)
        websocket_client: Optional WebSocket to stream chunks to client
    
    Returns:
        Base64 encoded audio data or None if failed
    """
    # Murf API key - using the hardcoded one from existing endpoints
    murf_api_key = "ap2_1633e776-b13b-4a5d-9826-1001621abe70"
    
    try:
        print(f"🎤 Sending text to Murf API: '{text[:50]}...'")
        
        # Use HTTP API since WebSocket returned HTTP 405
        murf_api_url = "https://api.murf.ai/v1/speech/generate"
        
        headers = {
            "api-key": murf_api_key,
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        payload = {
            "voiceId": "en-US-cooper",
            "text": text,
            "format": "mp3",
            "sampleRate": 44100,
            "speed": 1.0,
            "pitch": 0
        }
        
        print("📤 Making HTTP request to Murf API...")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                murf_api_url,
                headers=headers,
                json=payload
            )
            
            print(f"📥 Murf API responded with status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                audio_url = result.get("audioFile", result.get("url", result.get("audioUrl")))
                
                if audio_url:
                    print(f"� Got Murf audio URL: {audio_url}")
                    
                    # Download the audio file and convert to base64
                    audio_response = await client.get(audio_url)
                    if audio_response.status_code == 200:
                        audio_bytes = audio_response.content
                        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                        
                        print(f"🎉 Murf TTS success! Generated {len(audio_base64)} characters of base64 audio")
                        print(f"📊 Original audio size: {len(audio_bytes)} bytes")
                        
                        # Stream to client if WebSocket provided
                        if websocket_client and websocket_client.client_state.name == "CONNECTED":
                            await stream_audio_chunks_to_client(audio_base64, websocket_client)
                        
                        return audio_base64
                    else:
                        print(f"❌ Failed to download audio from URL: {audio_response.status_code}")
                        return None
                else:
                    print("⚠️ No audio URL found in Murf response")
                    return None
            else:
                error_text = response.text
                print(f"❌ Murf API error {response.status_code}: {error_text}")
                return None
                
    except Exception as e:
        print(f"❌ Error with Murf API: {str(e)}")
        return None

# Audio conversion helper functions
def convert_webm_to_pcm(webm_data: bytes) -> bytes:
    """
    Convert WebM audio data to 16kHz, 16-bit, mono PCM format for AssemblyAI.
    Returns raw PCM bytes.
    """
    try:
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as temp_input:
            temp_input.write(webm_data)
            temp_input_path = temp_input.name
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_output:
            temp_output_path = temp_output.name
        
        # Use ffmpeg to convert WebM to 16kHz, 16-bit, mono WAV
        cmd = [
            'ffmpeg', '-y',
            '-i', temp_input_path,
            '-ar', '16000',  # 16kHz sample rate
            '-ac', '1',      # mono
            '-sample_fmt', 's16',  # 16-bit
            '-f', 'wav',
            temp_output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ FFmpeg error: {result.stderr}")
            return b''
        
        # Read the converted WAV file and extract PCM data
        with wave.open(temp_output_path, 'rb') as wav_file:
            pcm_data = wav_file.readframes(wav_file.getnframes())
        
        # Clean up temp files
        try:
            os.unlink(temp_input_path)
            os.unlink(temp_output_path)
        except:
            pass
        
        return pcm_data
        
    except Exception as e:
        print(f"❌ Audio conversion error: {e}")
        return b''


def accumulate_webm_chunks(chunks: List[bytes]) -> bytes:
    """
    Combine multiple WebM chunks into a single WebM file for processing.
    """
    return b''.join(chunks)

# Pydantic models for request/response
class TTSRequest(BaseModel):
    text: str
    voice_id: str = "en-US-cooper"  # Default voice (valid Cooper voice)
    api_key: str  # Murf API key

class TTSResponse(BaseModel):
    success: bool
    audio_url: str = None
    error: str = None
    fallback: bool = False

class AudioUploadResponse(BaseModel):
    success: bool
    filename: str = None
    content_type: str = None
    size: int = None
    upload_path: str = None
    error: str = None

class TranscriptionResponse(BaseModel):
    success: bool
    transcript: str = None
    confidence: float = None
    duration: float = None
    word_count: int = None
    error: str = None

class EchoResponse(BaseModel):
    success: bool
    audio_url: str = None
    transcript: str = None
    voice_id: str = None
    confidence: float = None
    word_count: int = None
    error: str = None

class LLMQueryRequest(BaseModel):
    text: str

class LLMQueryResponse(BaseModel):
    success: bool
    response: str = None
    audio_url: str = None
    transcript: str = None
    voice_id: str = None
    error: str = None
    model: str = None
    usage: dict = None


app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve the main HTML page"""
    with open("static/index.html", "r") as file:
        html_content = file.read()
    return HTMLResponse(content=html_content)

@app.get("/api/hello")
async def hello_world():
    """Simple API endpoint"""
    return {"message": "Hello from FastAPI!", "status": "success"}

@app.get("/api/data")
async def get_data():
    """Example API endpoint that returns some data"""
    return {
        "data": [
            {"id": 1, "name": "Item 1", "value": 100},
            {"id": 2, "name": "Item 2", "value": 200},
            {"id": 3, "name": "Item 3", "value": 300}
        ],
        "total": 3
    }


@app.post("/api/tts/generate", response_model=TTSResponse)
async def generate_tts(request: TTSRequest):
    """
    Generate audio from text using Murf's TTS API, with robust error handling and fallback.
    """
    try:
        murf_api_url = "https://api.murf.ai/v1/speech/generate"
        headers = {
            "api-key": request.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        payload = {
            "voiceId": request.voice_id,
            "text": request.text,
            "format": "mp3",
            "sampleRate": 44100,
            "speed": 1.0,
            "pitch": 0
        }
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    murf_api_url,
                    headers=headers,
                    json=payload
                )
                if response.status_code == 200:
                    result = response.json()
                    audio_url = result.get("audioFile", result.get("url", result.get("audioUrl")))
                    if audio_url:
                        return TTSResponse(success=True, audio_url=audio_url)
                    else:
                        raise Exception("Audio URL not found in Murf API response")
                else:
                    raise Exception(f"Murf API error (status {response.status_code}): {response.text}")
        except Exception as e:
            # Fallback: serve static fallback audio
            fallback_url = "/static/fallback.mp3"
            return TTSResponse(success=True, audio_url=fallback_url, error=f"TTS fallback: {str(e)}", fallback=True)
    except Exception as e:
        return TTSResponse(success=False, error=f"Unexpected error: {str(e)}", fallback=True)

@app.post("/api/upload/audio", response_model=AudioUploadResponse)
async def upload_audio(audio_file: UploadFile = File(...)):
    """
    Upload audio file from Echo Bot recording
    
    Args:
        audio_file: The uploaded audio file (WebM format from MediaRecorder)
        
    Returns:
        AudioUploadResponse with file details or error message
    """
    try:
        # Validate file type
        if not audio_file.content_type or not audio_file.content_type.startswith('audio/'):
            return AudioUploadResponse(
                success=False,
                error=f"Invalid file type. Expected audio file, got: {audio_file.content_type}"
            )
        
        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_extension = ".webm"  # Default for MediaRecorder
        
        # Try to get proper extension from content type
        if audio_file.content_type == "audio/webm":
            file_extension = ".webm"
        elif audio_file.content_type == "audio/wav":
            file_extension = ".wav"
        elif audio_file.content_type == "audio/mp3":
            file_extension = ".mp3"
        elif audio_file.content_type == "audio/ogg":
            file_extension = ".ogg"
            
        unique_filename = f"recording_{timestamp}_{uuid.uuid4().hex[:8]}{file_extension}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)
        
        # Read file content
        content = await audio_file.read()
        file_size = len(content)
        
        # Validate file size (limit to 10MB)
        max_size = 10 * 1024 * 1024  # 10MB
        if file_size > max_size:
            return AudioUploadResponse(
                success=False,
                error=f"File too large. Maximum size: {max_size // (1024*1024)}MB, got: {file_size // (1024*1024)}MB"
            )
        
        # Save file asynchronously
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(content)
        
        # Verify file was saved
        if not os.path.exists(file_path):
            return AudioUploadResponse(
                success=False,
                error="Failed to save file to server"
            )
        
        return AudioUploadResponse(
            success=True,
            filename=unique_filename,
            content_type=audio_file.content_type,
            size=file_size,
            upload_path=f"/uploads/{unique_filename}"
        )
        
    except Exception as e:
        # Clean up file if it was partially created
        if 'file_path' in locals() and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass
                
        return AudioUploadResponse(
            success=False,
            error=f"Upload failed: {str(e)}"
        )

@app.post("/api/transcribe/file", response_model=TranscriptionResponse)
async def transcribe_audio_file(audio_file: UploadFile = File(...)):
    """
    Transcribe audio file using AssemblyAI
    
    Args:
        audio_file: The uploaded audio file to transcribe
        
    Returns:
        TranscriptionResponse with transcript text or error message
    """
    try:
        # Validate AssemblyAI API key
        if ASSEMBLYAI_API_KEY == "your_assemblyai_api_key_here":
            return TranscriptionResponse(
                success=False,
                error="AssemblyAI API key not configured. Please set ASSEMBLYAI_API_KEY environment variable."
            )
        
        # Validate file type
        if not audio_file.content_type or not audio_file.content_type.startswith('audio/'):
            return TranscriptionResponse(
                success=False,
                error=f"Invalid file type. Expected audio file, got: {audio_file.content_type}"
            )
        
        # Read audio file content
        audio_data = await audio_file.read()
        file_size = len(audio_data)
        
        # Validate file size (limit to 25MB for AssemblyAI)
        max_size = 25 * 1024 * 1024  # 25MB
        if file_size > max_size:
            return TranscriptionResponse(
                success=False,
                error=f"File too large. Maximum size: {max_size // (1024*1024)}MB, got: {file_size // (1024*1024)}MB"
            )
        
        # Initialize AssemblyAI transcriber
        transcriber = aai.Transcriber()
        
        # Transcribe audio from binary data
        transcript = transcriber.transcribe(audio_data)
        
        # Check if transcription was successful
        if transcript.status == aai.TranscriptStatus.error:
            return TranscriptionResponse(
                success=False,
                error=f"Transcription failed: {transcript.error}"
            )
        
        # Count words in transcript
        word_count = len(transcript.text.split()) if transcript.text else 0
        
        # Get confidence score (average of word confidences if available)
        confidence = None
        if hasattr(transcript, 'confidence') and transcript.confidence:
            confidence = transcript.confidence
        elif hasattr(transcript, 'words') and transcript.words:
            # Calculate average confidence from individual words
            confidences = [word.confidence for word in transcript.words if word.confidence is not None]
            if confidences:
                confidence = sum(confidences) / len(confidences)
        
        return TranscriptionResponse(
            success=True,
            transcript=transcript.text or "",
            confidence=confidence,
            duration=transcript.audio_duration if hasattr(transcript, 'audio_duration') else None,
            word_count=word_count
        )
        
    except Exception as e:
        return TranscriptionResponse(
            success=False,
            error=f"Transcription failed: {str(e)}"
        )

@app.post("/api/tts/echo", response_model=EchoResponse)
async def echo_with_murf_voice(
    audio_file: UploadFile = File(...),
    voice_id: str = Form("en-US-cooper"),
    murf_api_key: str = Form(...)
):
    """
    Echo Bot v2: Transcribe audio and replay with Murf voice
    
    Args:
        audio_file: The uploaded audio file to transcribe and echo
        voice_id: Murf voice ID to use for speech generation
        murf_api_key: Murf API key for TTS generation
        
    Returns:
        EchoResponse with Murf audio URL and transcript details
    """
    try:
        # Validate required parameters
        if not murf_api_key or murf_api_key.strip() == "":
            return EchoResponse(
                success=False,
                error="Murf API key is required for echo functionality"
            )
        
        # Validate AssemblyAI API key
        if ASSEMBLYAI_API_KEY == "your_assemblyai_api_key_here":
            return EchoResponse(
                success=False,
                error="AssemblyAI API key not configured. Please set ASSEMBLYAI_API_KEY environment variable."
            )
        
        # Validate file type
        if not audio_file.content_type or not audio_file.content_type.startswith('audio/'):
            return EchoResponse(
                success=False,
                error=f"Invalid file type. Expected audio file, got: {audio_file.content_type}"
            )
        
        # Read audio file content
        audio_data = await audio_file.read()
        file_size = len(audio_data)
        
        # Validate file size (limit to 25MB for AssemblyAI)
        max_size = 25 * 1024 * 1024  # 25MB
        if file_size > max_size:
            return EchoResponse(
                success=False,
                error=f"File too large. Maximum size: {max_size // (1024*1024)}MB, got: {file_size // (1024*1024)}MB"
            )
        
        # Step 1: Transcribe audio using AssemblyAI
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(audio_data)
        
        # Check if transcription was successful
        if transcript.status == aai.TranscriptStatus.error:
            return EchoResponse(
                success=False,
                error=f"Transcription failed: {transcript.error}"
            )
        
        # Extract transcript text
        transcript_text = transcript.text or ""
        if not transcript_text.strip():
            return EchoResponse(
                success=False,
                error="No speech detected in audio file"
            )
        
        # Calculate transcript metrics
        word_count = len(transcript_text.split()) if transcript_text else 0
        confidence = None
        if hasattr(transcript, 'confidence') and transcript.confidence:
            confidence = transcript.confidence
        elif hasattr(transcript, 'words') and transcript.words:
            confidences = [word.confidence for word in transcript.words if word.confidence is not None]
            if confidences:
                confidence = sum(confidences) / len(confidences)
        
        # Step 2: Generate speech using Murf API
        murf_api_url = "https://api.murf.ai/v1/speech/generate"
        
        headers = {
            "api-key": murf_api_key,
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        payload = {
            "voiceId": voice_id,
            "text": transcript_text,
            "format": "mp3",
            "sampleRate": 44100,
            "speed": 1.0,
            "pitch": 0
        }
        
        # Make request to Murf API
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                murf_api_url,
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                audio_url = result.get("audioFile", result.get("url", result.get("audioUrl")))
                
                if audio_url:
                    return EchoResponse(
                        success=True,
                        audio_url=audio_url,
                        transcript=transcript_text,
                        voice_id=voice_id,
                        confidence=confidence,
                        word_count=word_count
                    )
                else:
                    return EchoResponse(
                        success=False,
                        error="Audio URL not found in Murf API response",
                        transcript=transcript_text,
                        voice_id=voice_id,
                        confidence=confidence,
                        word_count=word_count
                    )
            else:
                error_detail = response.text
                return EchoResponse(
                    success=False,
                    error=f"Murf API error (status {response.status_code}): {error_detail}",
                    transcript=transcript_text,
                    voice_id=voice_id,
                    confidence=confidence,
                    word_count=word_count
                )
                
    except httpx.TimeoutException:
        return EchoResponse(
            success=False,
            error="Request to Murf API timed out"
        )
    except Exception as e:
        return EchoResponse(
            success=False,
            error=f"Echo processing failed: {str(e)}"
        )

@app.post("/llm/query", response_model=LLMQueryResponse)
async def query_llm(
    audio_file: UploadFile = File(...),
    voice_id: str = Form("en-US-cooper"),
    murf_api_key: str = Form(...)
):
    """
    Voice-to-Voice AI Pipeline: Audio → Transcription → LLM → TTS → Audio Response
    
    Args:
        audio_file: The uploaded audio file to transcribe and process
        voice_id: Murf voice ID to use for speech generation
        murf_api_key: Murf API key for TTS generation
        
    Returns:
        LLMQueryResponse with LLM response text and Murf audio URL
    """
    try:
        # Validate required parameters
        if not murf_api_key or murf_api_key.strip() == "":
            return LLMQueryResponse(
                success=False,
                error="Murf API key is required for voice response generation"
            )
        
        # Validate Gemini API key
        if GEMINI_API_KEY == "your_gemini_api_key_here":
            return LLMQueryResponse(
                success=False,
                error="Gemini API key not configured. Please set GEMINI_API_KEY environment variable."
            )
        
        # Validate AssemblyAI API key
        if ASSEMBLYAI_API_KEY == "your_assemblyai_api_key_here":
            return LLMQueryResponse(
                success=False,
                error="AssemblyAI API key not configured. Please set ASSEMBLYAI_API_KEY environment variable."
            )
        
        # Validate file type
        if not audio_file.content_type or not audio_file.content_type.startswith('audio/'):
            return LLMQueryResponse(
                success=False,
                error=f"Invalid file type. Expected audio file, got: {audio_file.content_type}"
            )
        
        # Read audio file content
        audio_data = await audio_file.read()
        file_size = len(audio_data)
        
        # Validate file size (limit to 25MB for AssemblyAI)
        max_size = 25 * 1024 * 1024  # 25MB
        if file_size > max_size:
            return LLMQueryResponse(
                success=False,
                error=f"File too large. Maximum size: {max_size // (1024*1024)}MB, got: {file_size // (1024*1024)}MB"
            )
        
        # Step 1: Transcribe audio using AssemblyAI
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(audio_data)
        
        # Check if transcription was successful
        if transcript.status == aai.TranscriptStatus.error:
            return LLMQueryResponse(
                success=False,
                error=f"Transcription failed: {transcript.error}"
            )
        
        # Extract transcript text
        transcript_text = transcript.text or ""
        if not transcript_text.strip():
            return LLMQueryResponse(
                success=False,
                error="No speech detected in audio file"
            )
        
        # Step 2: Send transcript to Gemini LLM
        headers = {
            "Content-Type": "application/json"
        }
        
        # Prepare payload for Gemini API
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": f"Please provide a concise and helpful response to the following question or statement (maximum 500 words): {transcript_text}"
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.7,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 500,  # Reduced to keep responses shorter
            },
            "safetySettings": [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                }
            ]
        }
        
        # Construct the full URL with API key for Gemini
        gemini_url = f"{GEMINI_API_URL}?key={GEMINI_API_KEY}"
        
        # Make request to Gemini API
        llm_response_text = ""
        usage_info = {}
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                gemini_url,
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract the response text from Gemini's response structure
                if "candidates" in result and len(result["candidates"]) > 0:
                    candidate = result["candidates"][0]
                    if "content" in candidate and "parts" in candidate["content"]:
                        llm_response_text = candidate["content"]["parts"][0].get("text", "")
                        
                        # Extract usage information if available
                        if "usageMetadata" in result:
                            usage_info = result["usageMetadata"]
                    else:
                        return LLMQueryResponse(
                            success=False,
                            error="Invalid response structure from Gemini API",
                            transcript=transcript_text
                        )
                else:
                    return LLMQueryResponse(
                        success=False,
                        error="No response candidates returned from Gemini API",
                        transcript=transcript_text
                    )
            else:
                error_detail = response.text
                return LLMQueryResponse(
                    success=False,
                    error=f"Gemini API error (status {response.status_code}): {error_detail}",
                    transcript=transcript_text
                )
        
        # Step 3: Convert LLM response to speech using Murf API
        # Note: Murf API has a 3000 character limit for text
        MAX_MURF_CHARS = 2900  # Leave some buffer
        
        # Truncate the response if it's too long
        if len(llm_response_text) > MAX_MURF_CHARS:
            # Try to truncate at a sentence boundary
            truncated_text = llm_response_text[:MAX_MURF_CHARS]
            last_period = truncated_text.rfind('.')
            last_exclamation = truncated_text.rfind('!')
            last_question = truncated_text.rfind('?')
            
            # Find the last sentence ending
            last_sentence_end = max(last_period, last_exclamation, last_question)
            
            if last_sentence_end > MAX_MURF_CHARS * 0.7:  # If we can keep at least 70% of text
                tts_text = truncated_text[:last_sentence_end + 1]
            else:
                # Just truncate at word boundary
                truncated_text = llm_response_text[:MAX_MURF_CHARS]
                last_space = truncated_text.rfind(' ')
                tts_text = truncated_text[:last_space] + "..."
            
            print(f"🔍 Truncated text from {len(llm_response_text)} to {len(tts_text)} characters")
        else:
            tts_text = llm_response_text
        
        murf_api_url = "https://api.murf.ai/v1/speech/generate"
        
        murf_headers = {
            "api-key": murf_api_key,
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        murf_payload = {
            "voiceId": voice_id,
            "text": tts_text,
            "format": "mp3",
            "sampleRate": 44100,
            "speed": 1.0,
            "pitch": 0
        }
        
        # Make request to Murf API
        async with httpx.AsyncClient(timeout=30.0) as client:
            murf_response = await client.post(
                murf_api_url,
                headers=murf_headers,
                json=murf_payload
            )
            
            print(f"🔍 Murf API Status: {murf_response.status_code}")
            print(f"🔍 Murf API Response: {murf_response.text}")
            
            if murf_response.status_code == 200:
                murf_result = murf_response.json()
                print(f"🔍 Murf Result Keys: {list(murf_result.keys())}")
                print(f"🔍 Full Murf Response: {murf_result}")
                
                # Try different possible keys for the audio URL
                audio_url = None
                possible_keys = ["audioFile", "url", "audioUrl", "audio_url", "file", "download_url", "stream_url"]
                
                for key in possible_keys:
                    if key in murf_result and murf_result[key]:
                        audio_url = murf_result[key]
                        print(f"🔍 Found audio URL with key '{key}': {audio_url}")
                        break
                
                if not audio_url:
                    # If no direct URL, check for nested objects
                    if "data" in murf_result and isinstance(murf_result["data"], dict):
                        for key in possible_keys:
                            if key in murf_result["data"] and murf_result["data"][key]:
                                audio_url = murf_result["data"][key]
                                print(f"🔍 Found audio URL in data.{key}: {audio_url}")
                                break
                
                print(f"🔍 Final Audio URL: {audio_url}")
                
                if audio_url:
                    return LLMQueryResponse(
                        success=True,
                        response=llm_response_text,
                        audio_url=audio_url,
                        transcript=transcript_text,
                        voice_id=voice_id,
                        model="gemini-1.5-flash-latest",
                        usage=usage_info
                    )
                else:
                    return LLMQueryResponse(
                        success=False,
                        error="Audio URL not found in Murf API response",
                        response=llm_response_text,
                        transcript=transcript_text,
                        voice_id=voice_id,
                        model="gemini-1.5-flash-latest",
                        usage=usage_info
                    )
            else:
                # Parse Murf API error
                try:
                    error_detail = murf_response.json()
                    error_message = error_detail.get("errorMessage", murf_response.text)
                except:
                    error_message = murf_response.text
                    
                print(f"🔍 Murf API Error: {error_message}")
                    
                return LLMQueryResponse(
                    success=False,
                    error=f"Murf API error (status {murf_response.status_code}): {error_message}",
                    response=llm_response_text,
                    transcript=transcript_text,
                    voice_id=voice_id,
                    model="gemini-1.5-flash-latest",
                    usage=usage_info
                )
                
    except httpx.TimeoutException:
        return LLMQueryResponse(
            success=False,
            error="Request timeout - please try again"
        )
    except Exception as e:
        return LLMQueryResponse(
            success=False,
            error=f"Voice-to-voice processing failed: {str(e)}"
        )

@app.post("/agent/chat/{session_id}", response_model=LLMQueryResponse)
async def agent_chat(
    session_id: str = Path(..., description="Session ID for chat history"),
    audio_file: UploadFile = File(...),
    voice_id: str = Form("en-US-cooper"),
    murf_api_key: str = Form(...)
):
    """
    Conversational Voice-to-Voice AI Pipeline with chat history.
    
    Args:
        session_id: Unique session identifier for chat history
        audio_file: User's audio message
        voice_id: Murf voice ID for response synthesis
        murf_api_key: Murf API key for TTS generation
        
    Returns:
        LLMQueryResponse with audio_url, transcript, and chat history
    """
    try:
        # Validate required parameters
        if not murf_api_key or murf_api_key.strip() == "":
            return LLMQueryResponse(
                success=False,
                error="Murf API key is required for voice response generation"
            )
        
        # Validate Gemini API key
        if GEMINI_API_KEY == "your_gemini_api_key_here":
            return LLMQueryResponse(
                success=False,
                error="Gemini API key not configured. Please set GEMINI_API_KEY environment variable."
            )
        
        # Validate AssemblyAI API key
        if ASSEMBLYAI_API_KEY == "your_assemblyai_api_key_here":
            return LLMQueryResponse(
                success=False,
                error="AssemblyAI API key not configured. Please set ASSEMBLYAI_API_KEY environment variable."
            )
        
        # Validate file type
        if not audio_file.content_type or not audio_file.content_type.startswith('audio/'):
            return LLMQueryResponse(
                success=False,
                error=f"Invalid file type. Expected audio file, got: {audio_file.content_type}"
            )
        
        # Read audio file content
        audio_data = await audio_file.read()
        file_size = len(audio_data)
        
        # Validate file size (limit to 25MB for AssemblyAI)
        max_size = 25 * 1024 * 1024  # 25MB
        if file_size > max_size:
            return LLMQueryResponse(
                success=False,
                error=f"File too large. Maximum size: {max_size // (1024*1024)}MB, got: {file_size // (1024*1024)}MB"
            )
        
        # Step 1: Transcribe audio using AssemblyAI
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(audio_data)
        
        # Check if transcription was successful
        if transcript.status == aai.TranscriptStatus.error:
            return LLMQueryResponse(
                success=False,
                error=f"Transcription failed: {transcript.error}"
            )
        
        # Extract transcript text
        transcript_text = transcript.text or ""
        if not transcript_text.strip():
            return LLMQueryResponse(
                success=False,
                error="No speech detected in audio file"
            )
        
        # Step 2: Fetch and update chat history for session
        history = chat_sessions.get(session_id, [])
        
        # Add new user message to history
        history.append({"role": "user", "content": transcript_text})
        
        # Prepare conversation history for Gemini API
        gemini_contents = []
        for msg in history:
            # Convert role: "assistant" -> "model" for Gemini API
            role = "model" if msg["role"] == "assistant" else "user"
            gemini_contents.append({
                "parts": [{"text": msg["content"]}],
                "role": role
            })
        
        # Prepare payload for Gemini API
        payload = {
            "contents": gemini_contents,
            "generationConfig": {
                "temperature": 0.7,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 500,  # Keep responses shorter for TTS
            },
            "safetySettings": [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                }
            ]
        }
        
        # Construct the full URL with API key for Gemini
        gemini_url = f"{GEMINI_API_URL}?key={GEMINI_API_KEY}"
        
        # Make request to Gemini API
        llm_response_text = ""
        usage_info = {}
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                gemini_url,
                headers={"Content-Type": "application/json"},
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract the response text from Gemini's response structure
                if "candidates" in result and len(result["candidates"]) > 0:
                    candidate = result["candidates"][0]
                    if "content" in candidate and "parts" in candidate["content"]:
                        llm_response_text = candidate["content"]["parts"][0].get("text", "")
                        
                        # Extract usage information if available
                        if "usageMetadata" in result:
                            usage_info = result["usageMetadata"]
                    else:
                        return LLMQueryResponse(
                            success=False,
                            error="Invalid response structure from Gemini API",
                            transcript=transcript_text
                        )
                else:
                    return LLMQueryResponse(
                        success=False,
                        error="No response candidates returned from Gemini API",
                        transcript=transcript_text
                    )
            else:
                error_detail = response.text
                return LLMQueryResponse(
                    success=False,
                    error=f"Gemini API error (status {response.status_code}): {error_detail}",
                    transcript=transcript_text
                )
        
        # Add assistant response to chat history
        history.append({"role": "assistant", "content": llm_response_text})
        
        # Store updated history (keep last 20 messages to manage memory)
        chat_sessions[session_id] = history[-20:]
        
        # Step 3: Convert LLM response to speech using Murf API
        # Note: Murf API has a 3000 character limit for text
        MAX_MURF_CHARS = 2900  # Leave some buffer
        
        # Truncate the response if it's too long
        if len(llm_response_text) > MAX_MURF_CHARS:
            # Try to truncate at a sentence boundary
            truncated_text = llm_response_text[:MAX_MURF_CHARS]
            last_period = truncated_text.rfind('.')
            last_exclamation = truncated_text.rfind('!')
            last_question = truncated_text.rfind('?')
            
            # Find the last sentence ending
            last_sentence_end = max(last_period, last_exclamation, last_question)
            
            if last_sentence_end > MAX_MURF_CHARS * 0.7:  # If we can keep at least 70% of text
                tts_text = truncated_text[:last_sentence_end + 1]
            else:
                # Just truncate at word boundary
                truncated_text = llm_response_text[:MAX_MURF_CHARS]
                last_space = truncated_text.rfind(' ')
                tts_text = truncated_text[:last_space] + "..."
            
            print(f"🔍 Truncated text from {len(llm_response_text)} to {len(tts_text)} characters")
        else:
            tts_text = llm_response_text
        
        murf_api_url = "https://api.murf.ai/v1/speech/generate"
        
        murf_headers = {
            "api-key": murf_api_key,
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        murf_payload = {
            "voiceId": voice_id,
            "text": tts_text,
            "format": "mp3",
            "sampleRate": 44100,
            "speed": 1.0,
            "pitch": 0
        }
        
        # Make request to Murf API
        async with httpx.AsyncClient(timeout=30.0) as client:
            murf_response = await client.post(
                murf_api_url,
                headers=murf_headers,
                json=murf_payload
            )
            
            print(f"🔍 Murf API Status: {murf_response.status_code}")
            print(f"🔍 Murf API Response: {murf_response.text}")
            
            if murf_response.status_code == 200:
                murf_result = murf_response.json()
                print(f"🔍 Murf Result Keys: {list(murf_result.keys())}")
                print(f"🔍 Full Murf Response: {murf_result}")
                
                # Try different possible keys for the audio URL
                audio_url = None
                possible_keys = ["audioFile", "url", "audioUrl", "audio_url", "file", "download_url", "stream_url"]
                
                for key in possible_keys:
                    if key in murf_result and murf_result[key]:
                        audio_url = murf_result[key]
                        print(f"🔍 Found audio URL with key '{key}': {audio_url}")
                        break
                
                if not audio_url:
                    # If no direct URL, check for nested objects
                    if "data" in murf_result and isinstance(murf_result["data"], dict):
                        for key in possible_keys:
                            if key in murf_result["data"] and murf_result["data"][key]:
                                audio_url = murf_result["data"][key]
                                print(f"🔍 Found audio URL in data.{key}: {audio_url}")
                                break
                
                print(f"🔍 Final Audio URL: {audio_url}")
                
                if audio_url:
                    return LLMQueryResponse(
                        success=True,
                        response=llm_response_text,
                        audio_url=audio_url,
                        transcript=transcript_text,
                        voice_id=voice_id,
                        model="gemini-2.0-flash",
                        usage=usage_info
                    )
                else:
                    return LLMQueryResponse(
                        success=False,
                        error="Audio URL not found in Murf API response",
                        response=llm_response_text,
                        transcript=transcript_text,
                        voice_id=voice_id,
                        model="gemini-2.0-flash",
                        usage=usage_info
                    )
            else:
                # Parse Murf API error
                try:
                    error_detail = murf_response.json()
                    error_message = error_detail.get("errorMessage", murf_response.text)
                except:
                    error_message = murf_response.text
                    
                print(f"🔍 Murf API Error: {error_message}")
                    
                return LLMQueryResponse(
                    success=False,
                    error=f"Murf API error (status {murf_response.status_code}): {error_message}",
                    response=llm_response_text,
                    transcript=transcript_text,
                    voice_id=voice_id,
                    model="gemini-2.0-flash",
                    usage=usage_info
                )
                
    except httpx.TimeoutException:
        return LLMQueryResponse(
            success=False,
            error="Request timeout - please try again"
        )
    except Exception as e:
        return LLMQueryResponse(
            success=False,
            error=f"Agent chat processing failed: {str(e)}"
        )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time audio streaming.
    Clients can connect and stream audio data, server will save it to files.
    """
    await websocket.accept()
    print("🔌 WebSocket connection established")
    
    # Initialize audio recording session
    session_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    audio_filename = f"uploads/websocket_recording_{timestamp}_{session_id}.webm"
    audio_chunks = []
    
    print(f"📁 Audio will be saved to: {audio_filename}")
    
    try:
        await websocket.send_text(f"Connected! Session ID: {session_id}. Ready to receive audio data.")
        
        while True:
            # Receive message (could be binary or text)
            message = await websocket.receive()
            
            if message["type"] == "websocket.receive":
                # Handle binary data (audio chunks)
                if message.get("bytes") is not None:
                    audio_data = message["bytes"]
                    print(f"🎤 Received audio chunk: {len(audio_data)} bytes")
                    audio_chunks.append(audio_data)
                    await websocket.send_text(f"Received {len(audio_data)} bytes")
                # Handle text data (control messages)
                elif message.get("text") is not None:
                    text_message = message["text"]
                    print(f"📨 Received text message: {text_message}")
                    if text_message == "STOP_RECORDING":
                        if audio_chunks:
                            print(f"💾 Saving {len(audio_chunks)} audio chunks to {audio_filename}")
                            os.makedirs("uploads", exist_ok=True)
                            with open(audio_filename, 'wb') as audio_file:
                                for chunk in audio_chunks:
                                    audio_file.write(chunk)
                            file_size = os.path.getsize(audio_filename)
                            success_msg = f"✅ Recording saved! File: {audio_filename}, Size: {file_size} bytes, Chunks: {len(audio_chunks)}"
                            print(success_msg)
                            await websocket.send_text(success_msg)
                            audio_chunks = []
                            session_id = str(uuid.uuid4())[:8]
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            audio_filename = f"uploads/websocket_recording_{timestamp}_{session_id}.webm"
                        else:
                            await websocket.send_text("No audio data received to save.")
                    elif text_message == "START_RECORDING":
                        audio_chunks = []
                        session_id = str(uuid.uuid4())[:8]
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        audio_filename = f"uploads/websocket_recording_{timestamp}_{session_id}.webm"
                        await websocket.send_text(f"🎤 Started new recording session: {session_id}")
                    else:
                        response = f"Echo: {text_message}"
                        await websocket.send_text(response)
            
    except WebSocketDisconnect:
        print("🔌 WebSocket connection closed")
    except Exception as e:
        print(f"❌ WebSocket error: {str(e)}")
    finally:
        # Save any remaining audio data before closing
        if audio_chunks:
            print(f"💾 Saving final {len(audio_chunks)} audio chunks before disconnect")
            try:
                os.makedirs("uploads", exist_ok=True)
                with open(audio_filename, 'wb') as audio_file:
                    for chunk in audio_chunks:
                        audio_file.write(chunk)
                print(f"✅ Final recording saved: {audio_filename}")
            except Exception as e:
                print(f"❌ Error saving final recording: {str(e)}")


@app.websocket("/ws/transcribe")
async def websocket_transcribe_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for batch audio transcription using AssemblyAI standard API.
    This approach is more reliable than real-time streaming.
    """
    await websocket.accept()
    print("🔌 WebSocket transcription connection established")
    
    # Initialize session
    session_id = str(uuid.uuid4())[:8]
    
    print(f"🎯 Starting transcription session: {session_id}")
    
    # Accumulate WebM chunks for batch processing
    webm_chunks = []
    is_recording = False
    
    async def transcribe_batch():
        """Transcribe accumulated chunks using standard AssemblyAI API"""
        nonlocal webm_chunks
        
        if not webm_chunks:
            return
        
        try:
            print(f"� Transcribing batch of {len(webm_chunks)} chunks")
            
            # Combine all chunks
            combined_webm = accumulate_webm_chunks(webm_chunks)
            
            if len(combined_webm) < 1000:  # Skip very small chunks
                print("⚠️ Skipping small audio chunk")
                return
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as temp_file:
                temp_file.write(combined_webm)
                temp_file_path = temp_file.name
            
            try:
                # Use AssemblyAI standard API for transcription
                transcriber = aai.Transcriber()
                transcript = transcriber.transcribe(temp_file_path)
                
                if transcript.status == aai.TranscriptStatus.error:
                    error_msg = f"❌ Transcription error: {transcript.error}"
                    print(error_msg)
                    try:
                        await websocket.send_text(f"ERROR: {transcript.error}")
                        print(f"📤 Sent error to client: {transcript.error}")
                    except Exception as send_error:
                        print(f"❌ Failed to send error to client: {send_error}")
                else:
                    result_text = transcript.text or ""
                    if result_text.strip():
                        print(f"✅ Transcription: {result_text}")
                        if websocket.client_state.name == "CONNECTED":
                            try:
                                await websocket.send_text(f"TRANSCRIPT: {result_text}")
                                print(f"📤 Sent to client: TRANSCRIPT: {result_text}")
                                await asyncio.sleep(0.1)  # Small delay to ensure message delivery
                            except Exception as send_error:
                                print(f"❌ Failed to send to client: {send_error}")
                        else:
                            print(f"⚠️ WebSocket not connected, cannot send transcript")
                    else:
                        print("⚠️ No speech detected in audio")
                        if websocket.client_state.name == "CONNECTED":
                            try:
                                await websocket.send_text("INFO: No speech detected")
                            except Exception as send_error:
                                print(f"❌ Failed to send to client: {send_error}")
                        
            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
                    
            # Clear processed chunks
            webm_chunks = []
            
        except Exception as e:
            error_msg = f"❌ Batch transcription error: {e}"
            print(error_msg)
            await websocket.send_text(f"ERROR: {str(e)}")
    
    try:
        try:
            await websocket.send_text(f"🎯 Session {session_id} ready. Send START_TRANSCRIPTION to begin.")
            print(f"📤 Sent welcome message to client")
        except Exception as send_error:
            print(f"❌ Failed to send welcome message: {send_error}")
        
        while True:
            try:
                message = await websocket.receive()
                
                if message["type"] == "websocket.receive":
                    if message.get("bytes") is not None:
                        # Handle binary audio data (WebM chunks)
                        if is_recording:
                            audio_data = message["bytes"]
                            print(f"🎤 Received WebM chunk: {len(audio_data)} bytes")
                            webm_chunks.append(audio_data)
                            
                            # Process in batches of 20 chunks for better results
                            if len(webm_chunks) >= 20:
                                await transcribe_batch()
                        
                    elif message.get("text") is not None:
                        text_message = message["text"]
                        print(f"📨 Control message: {text_message}")
                        
                        if text_message == "START_TRANSCRIPTION":
                            is_recording = True
                            webm_chunks = []  # Reset chunks
                            try:
                                await websocket.send_text("🎤 Recording started - speak now")
                                print(f"📤 Sent to client: Recording started")
                            except Exception as send_error:
                                print(f"❌ Failed to send start message: {send_error}")
                            print(f"🎯 Transcription session {session_id} started")
                            
                        elif text_message == "STOP_TRANSCRIPTION":
                            is_recording = False
                            # Process any remaining chunks
                            if webm_chunks:
                                await transcribe_batch()
                            try:
                                await websocket.send_text("⏹️ Recording stopped")
                                print(f"📤 Sent to client: Recording stopped")
                            except Exception as send_error:
                                print(f"❌ Failed to send stop message: {send_error}")
                            print(f"⏹️ Transcription session {session_id} stopped")
                            
                        else:
                            try:
                                await websocket.send_text(f"Echo: {text_message}")
                            except Exception as send_error:
                                print(f"❌ Failed to send echo: {send_error}")
                
                elif message["type"] == "websocket.disconnect":
                    print(f"🔌 Client disconnected normally")
                    break
                    
            except RuntimeError as e:
                if "disconnect message has been received" in str(e):
                    print(f"🔌 Client disconnected")
                    break
                else:
                    raise e
                            
    except WebSocketDisconnect:
        print("🔌 WebSocket transcription connection closed by client")
    except Exception as e:
        print(f"❌ WebSocket transcription error: {str(e)}")
        # Don't try to send error message if connection is already closed
        
    finally:
        print(f"🔚 Transcription session {session_id} ended")


@app.websocket("/ws/stream-transcribe")
async def websocket_stream_transcribe_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time streaming transcription with turn detection using AssemblyAI.
    Uses the custom AssemblyAIStreamer for real-time processing with turn detection.
    """
    await websocket.accept()
    print("🔌 Real-time streaming WebSocket connection established")
    
    # Initialize session
    session_id = str(uuid.uuid4())[:8]
    print(f"🎯 Starting real-time streaming session: {session_id}")
    
    # Initialize AssemblyAI streamer
    streamer = AssemblyAIStreamer(ASSEMBLYAI_API_KEY)
    is_recording = False
    current_turn_transcript = ""
    
    async def on_transcript(transcript_data):
        """Handle transcript messages from AssemblyAI"""
        nonlocal current_turn_transcript
        
        try:
            if transcript_data.startswith("PARTIAL:"):
                partial_text = transcript_data[8:].strip()  # Remove "PARTIAL: "
                if partial_text and websocket.client_state.name == "CONNECTED":
                    await websocket.send_text(f"PARTIAL: {partial_text}")
                    print(f"📤 Sent partial: {partial_text}")
                    
            elif transcript_data.startswith("FINAL:"):
                final_text = transcript_data[6:].strip()  # Remove "FINAL: "
                if final_text:
                    current_turn_transcript += " " + final_text if current_turn_transcript else final_text
                    if websocket.client_state.name == "CONNECTED":
                        await websocket.send_text(f"FINAL: {final_text}")
                        print(f"📤 Sent final: {final_text}")
                        
        except Exception as e:
            print(f"❌ Error sending transcript: {e}")
    
    async def on_turn_end(turn_transcript):
        """Handle turn end events from AssemblyAI and trigger LLM streaming response"""
        try:
            # Use the accumulated transcript or the turn transcript
            final_turn_text = turn_transcript or current_turn_transcript
            
            if final_turn_text.strip():
                print(f"🔇 Turn ended with transcript: {final_turn_text}")
                if websocket.client_state.name == "CONNECTED":
                    # Send turn end notification with the complete transcript
                    await websocket.send_text(f"TURN_END: {final_turn_text.strip()}")
                    print(f"📤 Sent turn end: {final_turn_text.strip()}")
                
                # 🤖 NEW: Trigger LLM streaming response
                print(f"🚀 Triggering LLM streaming response for: '{final_turn_text.strip()}'")
                
                # Start LLM streaming in the background (non-blocking)
                asyncio.create_task(
                    process_llm_streaming_response(final_turn_text.strip(), session_id, websocket)
                )
                
            else:
                print(f"🔇 Turn ended (no speech detected)")
                if websocket.client_state.name == "CONNECTED":
                    await websocket.send_text("TURN_END_SILENCE")
                    print(f"📤 Sent turn end (silence)")
            
            # Reset for next turn
            current_turn_transcript = ""
            
        except Exception as e:
            print(f"❌ Error handling turn end: {e}")
    
    async def process_llm_streaming_response(user_input: str, session_id: str, websocket: WebSocket):
        """Process LLM streaming response and accumulate the result, streaming audio to client"""
        try:
            print(f"🤖 [Session {session_id}] Processing LLM response for: '{user_input}'")
            
            # Stream the LLM response and accumulate (passing WebSocket for audio streaming)
            llm_response = await stream_llm_response(user_input, session_id, websocket)
            
            if llm_response:
                print(f"✅ [Session {session_id}] LLM Response Complete:")
                print(f"   User: {user_input}")
                print(f"   Assistant: {llm_response}")
                print("-" * 80)
            else:
                print(f"⚠️ [Session {session_id}] No LLM response generated")
                
        except Exception as e:
            print(f"❌ [Session {session_id}] Error processing LLM response: {e}")
    
    try:
        # Connect to AssemblyAI streaming service
        connected = await streamer.connect(on_transcript=on_transcript, on_turn_end=on_turn_end)
        
        if not connected:
            await websocket.send_text("ERROR: Failed to connect to AssemblyAI streaming service")
            return
        
        # Send welcome message
        try:
            await websocket.send_text(f"🎯 Real-time streaming session {session_id} ready. Send START_STREAMING to begin.")
            print(f"📤 Sent welcome message to client")
        except Exception as send_error:
            print(f"❌ Failed to send welcome message: {send_error}")
        
        while True:
            try:
                message = await websocket.receive()
                
                if message["type"] == "websocket.receive":
                    if message.get("bytes") is not None:
                        # Handle binary audio data
                        if is_recording:
                            audio_data = message["bytes"]
                            print(f"🎤 Streaming audio chunk: {len(audio_data)} bytes")
                            
                            # Convert WebM to PCM if needed and send to AssemblyAI
                            pcm_data = convert_webm_to_pcm(audio_data)
                            if pcm_data:
                                await streamer.send_audio(pcm_data)
                            
                    elif message.get("text") is not None:
                        text_message = message["text"]
                        print(f"📨 Control message: {text_message}")
                        
                        if text_message == "START_STREAMING":
                            is_recording = True
                            current_turn_transcript = ""  # Reset transcript
                            try:
                                await websocket.send_text("🎤 Real-time streaming started - speak now")
                                print(f"📤 Sent to client: Streaming started")
                            except Exception as send_error:
                                print(f"❌ Failed to send start message: {send_error}")
                            print(f"🎯 Real-time streaming session {session_id} started")
                            
                        elif text_message == "STOP_STREAMING":
                            is_recording = False
                            try:
                                await websocket.send_text("⏹️ Real-time streaming stopped")
                                print(f"📤 Sent to client: Streaming stopped")
                            except Exception as send_error:
                                print(f"❌ Failed to send stop message: {send_error}")
                            print(f"⏹️ Real-time streaming session {session_id} stopped")
                            
                        else:
                            try:
                                await websocket.send_text(f"Echo: {text_message}")
                            except Exception as send_error:
                                print(f"❌ Failed to send echo: {send_error}")
                
                elif message["type"] == "websocket.disconnect":
                    print(f"🔌 Client disconnected normally")
                    break
                    
            except RuntimeError as e:
                if "disconnect message has been received" in str(e):
                    print(f"🔌 Client disconnected")
                    break
                else:
                    raise e
                            
    except WebSocketDisconnect:
        print("🔌 Real-time streaming WebSocket connection closed by client")
    except Exception as e:
        print(f"❌ Real-time streaming WebSocket error: {str(e)}")
        
    finally:
        # Clean up AssemblyAI connection
        if streamer:
            await streamer.close()
        print(f"🔚 Real-time streaming session {session_id} ended")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
