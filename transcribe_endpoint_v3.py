@app.websocket("/ws/transcribe")
async def websocket_transcribe_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time audio streaming with AssemblyAI v3 transcription.
    Accepts WebM audio chunks and converts them to PCM for AssemblyAI.
    """
    await websocket.accept()
    print("🔌 WebSocket transcription connection established")
    
    # Initialize session
    session_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"🎯 Starting transcription session: {session_id}")
    
    # AssemblyAI v3 streaming client
    streaming_client = None
    webm_chunks = []  # Accumulate WebM chunks for conversion
    is_transcribing = False
    
    async def send_to_client(message):
        """Helper to send messages to WebSocket client safely"""
        try:
            await websocket.send_text(message)
        except Exception as e:
            print(f"❌ Error sending to client: {e}")
    
    # AssemblyAI v3 event handlers
    def on_begin(client, event: BeginEvent):
        print(f"🟢 AssemblyAI session started: {event.id}")
        asyncio.create_task(send_to_client(f"TRANSCRIPTION_STARTED:{event.id}"))
        
    def on_turn(client, event: TurnEvent):
        print(f"📝 Turn: {event.transcript} (end_of_turn: {event.end_of_turn})")
        
        # Send to client
        transcription_data = {
            "transcript": event.transcript,
            "end_of_turn": event.end_of_turn,
            "turn_is_formatted": event.turn_is_formatted,
            "confidence": getattr(event, 'end_of_turn_confidence', 0.0)
        }
        asyncio.create_task(send_to_client(f"TRANSCRIPT:{json.dumps(transcription_data)}"))
        
    def on_terminated(client, event: TerminationEvent):
        print(f"🔴 AssemblyAI session terminated: {event.audio_duration_seconds} seconds processed")
        asyncio.create_task(send_to_client(f"TRANSCRIPTION_ENDED:{event.audio_duration_seconds}"))
        
    def on_error(client, error: StreamingError):
        print(f"❌ AssemblyAI streaming error: {error}")
        asyncio.create_task(send_to_client(f"ERROR:{error}"))

    async def convert_webm_to_pcm(webm_data: bytes) -> bytes:
        """Convert WebM audio data to PCM format using ffmpeg"""
        try:
            with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as webm_file:
                webm_file.write(webm_data)
                webm_file.flush()
                
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as pcm_file:
                    # Convert WebM to PCM using ffmpeg
                    result = subprocess.run([
                        'ffmpeg', '-i', webm_file.name,
                        '-ar', '16000',  # 16kHz sample rate
                        '-ac', '1',      # mono
                        '-f', 's16le',   # PCM 16-bit little-endian
                        pcm_file.name,
                        '-y'  # overwrite output
                    ], capture_output=True, text=True)
                    
                    if result.returncode != 0:
                        print(f"❌ FFmpeg conversion failed: {result.stderr}")
                        return b''
                    
                    # Read PCM data
                    with open(pcm_file.name, 'rb') as f:
                        pcm_data = f.read()
                    
                    # Clean up temp files
                    os.unlink(webm_file.name)
                    os.unlink(pcm_file.name)
                    
                    return pcm_data
                    
        except Exception as e:
            print(f"❌ Error converting WebM to PCM: {e}")
            return b''
    
    async def process_accumulated_audio():
        """Process accumulated WebM chunks periodically"""
        nonlocal webm_chunks, streaming_client
        
        if not webm_chunks or not streaming_client or not is_transcribing:
            return
            
        try:
            # Combine all WebM chunks
            combined_webm = b''.join(webm_chunks)
            print(f"🎵 Processing {len(combined_webm)} bytes of WebM audio")
            
            # Convert to PCM
            pcm_data = await convert_webm_to_pcm(combined_webm)
            
            if pcm_data:
                print(f"🎵 Converted to {len(pcm_data)} bytes of PCM")
                # Send PCM data to AssemblyAI
                streaming_client.send_audio_data(pcm_data)
            
            # Clear processed chunks
            webm_chunks = []
            
        except Exception as e:
            print(f"❌ Error processing audio: {e}")
    
    try:
        # Initialize AssemblyAI v3 streaming client
        streaming_client = StreamingClient(
            StreamingClientOptions(
                api_key=os.getenv("ASSEMBLYAI_API_KEY"),
                api_host="streaming.assemblyai.com"
            )
        )
        
        # Set up event handlers
        streaming_client.on(StreamingEvents.Begin, on_begin)
        streaming_client.on(StreamingEvents.Turn, on_turn)
        streaming_client.on(StreamingEvents.Termination, on_terminated)
        streaming_client.on(StreamingEvents.Error, on_error)
        
        await websocket.send_text(f"Connected! Session: {session_id}. Ready for audio transcription.")
        
        # Start periodic audio processing
        async def periodic_process():
            while is_transcribing:
                await asyncio.sleep(1.0)  # Process every second
                await process_accumulated_audio()
        
        process_task = None
        
        try:
            while True:
                message = await websocket.receive()
                
                if message["type"] == "websocket.receive":
                    if "bytes" in message:
                        # Audio data received
                        audio_data = message["bytes"]
                        print(f"🎤 Received audio chunk for transcription: {len(audio_data)} bytes")
                        
                        # Accumulate WebM chunks
                        if is_transcribing:
                            webm_chunks.append(audio_data)
                            
                    elif "text" in message:
                        # Control message
                        text_message = message["text"]
                        print(f"📨 Control message: {text_message}")
                        
                        if text_message == "START_TRANSCRIPTION":
                            print(f"🎯 Transcription session {session_id} started")
                            is_transcribing = True
                            
                            # Connect to AssemblyAI
                            streaming_client.connect(
                                StreamingParameters(
                                    sample_rate=16000,
                                    format_turns=True,
                                    encoding="pcm_s16le"
                                )
                            )
                            print("🚀 Connected to AssemblyAI v3 streaming service")
                            
                            # Start periodic processing
                            process_task = asyncio.create_task(periodic_process())
                            
                            await websocket.send_text(f"TRANSCRIPTION_READY:{session_id}")
                            
                        elif text_message == "STOP_TRANSCRIPTION":
                            print(f"⏹️ Transcription session {session_id} stopped")
                            is_transcribing = False
                            
                            if process_task:
                                process_task.cancel()
                                
                            if streaming_client:
                                try:
                                    streaming_client.disconnect(terminate=True)
                                    print("🔴 AssemblyAI session closed")
                                except Exception as e:
                                    print(f"❌ Error disconnecting: {e}")
                                    
                            await websocket.send_text(f"TRANSCRIPTION_STOPPED:{session_id}")
                            
                        else:
                            await websocket.send_text(f"Unknown control message: {text_message}")
                            
        except asyncio.CancelledError:
            pass
        finally:
            if process_task:
                try:
                    process_task.cancel()
                    await process_task
                except asyncio.CancelledError:
                    pass
                            
    except WebSocketDisconnect:
        print("🔌 WebSocket transcription connection closed")
    except Exception as e:
        print(f"❌ AssemblyAI transcription setup error: {str(e)}")
        try:
            await websocket.send_text(f"Transcription Setup Error: {str(e)}")
        except:
            pass
        
    finally:
        # Clean up AssemblyAI connection
        if streaming_client:
            try:
                streaming_client.disconnect(terminate=True)
                print("🔴 AssemblyAI v3 client closed")
            except Exception as e:
                print(f"❌ Error closing streaming client: {e}")
        
        print(f"🔚 Transcription session {session_id} ended")
