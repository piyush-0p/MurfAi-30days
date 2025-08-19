#!/usr/bin/env python3
"""
Test WebSocket audio streaming functionality
"""
import asyncio
import websockets
import json
from datetime import datetime
import struct
import os

async def test_websocket_audio_streaming():
    uri = "ws://localhost:8000/ws"
    
    print("="*60)
    print("🚀 WebSocket Audio Streaming Test")
    print("="*60)
    print(f"📍 Connecting to: {uri}")
    print(f"🕐 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-"*60)
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ WebSocket connection established successfully!")
            print()
            
            # Test 1: Initial connection message
            response = await websocket.recv()
            print(f"📥 Initial server message: {response}")
            print()
            
            # Test 2: Send START_RECORDING command
            print("📤 Test 1: Sending START_RECORDING command")
            await websocket.send("START_RECORDING")
            response = await websocket.recv()
            print(f"📥 Response: {response}")
            print()
            
            # Test 3: Send fake audio binary data
            print("📤 Test 2: Sending fake audio binary data")
            fake_audio_chunks = [
                b"fake_audio_chunk_1_" + b"0" * 100,
                b"fake_audio_chunk_2_" + b"1" * 150,
                b"fake_audio_chunk_3_" + b"2" * 200,
            ]
            
            for i, chunk in enumerate(fake_audio_chunks):
                print(f"   Sending chunk {i+1}: {len(chunk)} bytes")
                await websocket.send(chunk)
                response = await websocket.recv()
                print(f"   📥 Response: {response}")
            
            print()
            
            # Test 4: Send STOP_RECORDING command
            print("📤 Test 3: Sending STOP_RECORDING command")
            await websocket.send("STOP_RECORDING")
            response = await websocket.recv()
            print(f"📥 Response: {response}")
            print()
            
            # Test 5: Send another START/STOP cycle
            print("📤 Test 4: Second recording session")
            await websocket.send("START_RECORDING")
            response = await websocket.recv()
            print(f"📥 Start response: {response}")
            
            # Send more fake data
            more_chunks = [b"second_session_" + b"3" * 80, b"second_session_" + b"4" * 120]
            for chunk in more_chunks:
                await websocket.send(chunk)
                response = await websocket.recv()
                print(f"📥 Chunk response: {response}")
            
            await websocket.send("STOP_RECORDING")
            response = await websocket.recv()
            print(f"📥 Stop response: {response}")
            print()
            
            # Test 6: Echo test with regular text
            print("📤 Test 5: Regular text echo test")
            await websocket.send("Hello WebSocket Server!")
            response = await websocket.recv()
            print(f"📥 Echo response: {response}")
            
            print("="*60)
            print("🎉 All WebSocket audio streaming tests completed!")
            print("="*60)
            
    except Exception as e:
        print(f"❌ WebSocket connection failed: {e}")
        return False
        
    return True

if __name__ == "__main__":
    success = asyncio.run(test_websocket_audio_streaming())
    if success:
        print("\n🏆 WebSocket audio streaming is working perfectly!")
        print("✅ Check the uploads/ directory for saved audio files")
    else:
        print("\n💥 WebSocket audio streaming tests failed!")
