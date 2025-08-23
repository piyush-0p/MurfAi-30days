#!/usr/bin/env python3

import asyncio
import json
import websockets
import time

async def test_audio_streaming():
    """
    Test the audio streaming functionality by connecting to the WebSocket
    and simulating a voice interaction.
    """
    uri = "ws://localhost:8000/ws/stream-transcribe"
    
    try:
        print("🔌 Connecting to WebSocket...")
        async with websockets.connect(uri) as websocket:
            print("✅ WebSocket connected!")
            
            # Wait for welcome message
            welcome = await websocket.recv()
            print(f"📥 Received: {welcome}")
            
            # Start streaming
            print("🚀 Starting streaming...")
            await websocket.send("START_STREAMING")
            
            # Wait for streaming confirmation
            streaming_msg = await websocket.recv()
            print(f"📥 Received: {streaming_msg}")
            
            # Simulate some speech detection (this would be turn end)
            print("🎤 Simulating speech detection...")
            
            # Keep connection alive and listen for audio streaming
            print("👂 Listening for audio streaming messages...")
            audio_chunks_received = 0
            total_audio_size = 0
            
            # Listen for messages for 30 seconds max
            timeout_count = 0
            max_timeout = 300  # 30 seconds (100ms intervals)
            
            while timeout_count < max_timeout:
                try:
                    # Check for messages with short timeout
                    message = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                    
                    try:
                        # Try to parse as JSON (audio streaming)
                        data = json.loads(message)
                        
                        if data.get('type') == 'AUDIO_START':
                            print(f"🎵 Audio streaming started:")
                            print(f"   Total size: {data['total_size']} chars")
                            print(f"   Total chunks: {data['total_chunks']}")
                            print(f"   Chunk size: {data['chunk_size']}")
                            
                        elif data.get('type') == 'AUDIO_CHUNK':
                            audio_chunks_received += 1
                            chunk_size = len(data['data'])
                            total_audio_size += chunk_size
                            
                            print(f"📦 Received audio chunk {data['chunk_index']}: {chunk_size} chars")
                            print(f"   Preview: {data['data'][:50]}...")
                            print(f"   Is final: {data.get('is_final', False)}")
                            
                            # Send acknowledgment
                            ack = {
                                'type': 'AUDIO_CHUNK_ACK',
                                'chunk_index': data['chunk_index']
                            }
                            await websocket.send(json.dumps(ack))
                            print(f"✅ Sent acknowledgment for chunk {data['chunk_index']}")
                            
                        elif data.get('type') == 'AUDIO_COMPLETE':
                            print(f"🎉 Audio streaming complete!")
                            print(f"   Total chunks received: {audio_chunks_received}")
                            print(f"   Total audio data: {total_audio_size} characters")
                            print(f"📸 SCREENSHOT READY - Audio streaming acknowledgments logged!")
                            break
                            
                    except json.JSONDecodeError:
                        # Regular text message
                        print(f"📥 Text message: {message}")
                        
                        if "TURN_END:" in message:
                            print("🔇 Turn detected - LLM should start processing...")
                    
                    timeout_count = 0  # Reset timeout on message received
                    
                except asyncio.TimeoutError:
                    timeout_count += 1
                    if timeout_count % 50 == 0:  # Every 5 seconds
                        print(f"⏳ Waiting... ({timeout_count/10:.1f}s)")
            
            if timeout_count >= max_timeout:
                print("⏰ Timeout reached - ending test")
            
            print("🏁 Test complete!")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("🧪 Starting audio streaming test...")
    asyncio.run(test_audio_streaming())
