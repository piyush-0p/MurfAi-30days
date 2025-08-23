#!/usr/bin/env python3

import asyncio
import json
import websockets

async def test_websocket_audio_streaming():
    """
    Test WebSocket audio streaming by simulating a turn end event
    """
    uri = "ws://localhost:8000/ws/stream-transcribe"
    
    try:
        print("🔌 Connecting to WebSocket...")
        async with websockets.connect(uri) as websocket:
            print("✅ WebSocket connected!")
            
            # Wait for welcome message
            welcome = await websocket.recv()
            print(f"📥 Received: {welcome}")
            
            # We need to manually trigger a turn end to test audio streaming
            # Since we can't easily simulate the full speech pipeline, let's use 
            # the assemblyai_streamer to manually send a turn end event
            
            print("🎤 Manually triggering LLM response...")
            
            # Listen for messages and audio streaming
            print("👂 Listening for audio streaming messages...")
            audio_chunks_received = 0
            total_audio_size = 0
            
            timeout_count = 0
            max_timeout = 300  # 30 seconds
            
            while timeout_count < max_timeout:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                    
                    try:
                        # Try to parse as JSON (audio streaming)
                        data = json.loads(message)
                        
                        if data.get('type') == 'AUDIO_START':
                            print(f"🎵 Audio streaming started!")
                            print(f"   📊 Total size: {data['total_size']} chars")
                            print(f"   📦 Total chunks: {data['total_chunks']}")
                            print(f"   🔢 Chunk size: {data['chunk_size']}")
                            
                        elif data.get('type') == 'AUDIO_CHUNK':
                            audio_chunks_received += 1
                            chunk_size = len(data['data'])
                            total_audio_size += chunk_size
                            
                            print(f"📦 Received audio chunk {data['chunk_index']}/{data.get('total_chunks', '?')}: {chunk_size} chars")
                            
                            # Send acknowledgment 
                            ack = {
                                'type': 'AUDIO_CHUNK_ACK',
                                'chunk_index': data['chunk_index']
                            }
                            await websocket.send(json.dumps(ack))
                            print(f"✅ Client acknowledged audio chunk {data['chunk_index']}")
                            
                        elif data.get('type') == 'AUDIO_COMPLETE':
                            print(f"🎉 Audio streaming complete!")
                            print(f"📊 Final stats:")
                            print(f"   🔢 Total chunks received: {audio_chunks_received}")
                            print(f"   📝 Total audio data: {total_audio_size} characters")
                            print(f"")
                            print(f"📸 SUCCESS! Audio data streamed and acknowledged by client!")
                            print(f"📸 Ready for LinkedIn screenshot!")
                            return
                            
                    except json.JSONDecodeError:
                        # Regular text message
                        print(f"📥 Text: {message}")
                    
                    timeout_count = 0  # Reset on message received
                    
                except asyncio.TimeoutError:
                    timeout_count += 1
                    if timeout_count == 50:  # After 5 seconds of no activity
                        print("🚀 No automatic activity detected. Manually triggering test...")
                        # For testing, we'll simulate what would happen after a turn end
                        # by calling the streaming function directly
                        
                        # Import and call the function to simulate real scenario
                        import sys
                        import os
                        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
                        
                        from main import stream_llm_response
                        
                        print("🤖 Manually calling stream_llm_response with WebSocket...")
                        test_input = "Tell me a short joke"
                        
                        # This should stream audio chunks to our WebSocket client
                        response = await stream_llm_response(test_input, "manual_test", websocket)
                        print(f"🎯 LLM Response: {response}")
            
            print("⏰ Test completed")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("🧪 Starting WebSocket audio streaming test...")
    asyncio.run(test_websocket_audio_streaming())
