#!/usr/bin/env python3
"""
Simple WebSocket audio streaming test
"""
import asyncio
import websockets
import json
from datetime import datetime

async def test_simple():
    uri = "ws://localhost:8000/ws"
    
    print("Connecting to WebSocket...")
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected!")
            
            # Get initial message
            response = await websocket.recv()
            print(f"Server: {response}")
            
            # Test START_RECORDING
            print("Sending START_RECORDING...")
            await websocket.send("START_RECORDING")
            response = await websocket.recv()
            print(f"Server: {response}")
            
            # Send some fake audio data
            print("Sending fake audio data...")
            fake_audio = b"fake_audio_data_" + b"1" * 100
            await websocket.send(fake_audio)
            response = await websocket.recv()
            print(f"Server: {response}")
            
            # Send more fake audio data
            fake_audio2 = b"fake_audio_data_" + b"2" * 150
            await websocket.send(fake_audio2)
            response = await websocket.recv()
            print(f"Server: {response}")
            
            # Test STOP_RECORDING
            print("Sending STOP_RECORDING...")
            await websocket.send("STOP_RECORDING")
            response = await websocket.recv()
            print(f"Server: {response}")
            
            print("✅ Test completed successfully!")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_simple())
