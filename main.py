from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form, Path
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import Dict, List
import os
import httpx
import json
import uuid
from datetime import datetime
import aiofiles
import assemblyai as aai

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
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

if GEMINI_API_KEY != "your_gemini_api_key_here":
    print("✅ Gemini API key configured successfully")
else:
    print("⚠️  Warning: Gemini API key not set. Please set GEMINI_API_KEY environment variable or update main.py")

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
