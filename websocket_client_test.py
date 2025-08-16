#!/usr/bin/env python3
"""
Simple WebSocket client to test the /ws endpoint
"""
import asyncio
import websockets
import json


async def test_websocket():
    uri = "ws://localhost:8000/ws"
    
    try:
        async with websockets.connect(uri) as websocket:
            print("🔌 Connected to WebSocket server!")
            
            # Send some test messages
            test_messages = [
                "Hello WebSocket!",
                "This is a test message",
                "How are you doing?",
                "Testing echo functionality",
                json.dumps({"type": "test", "message": "JSON test"})
            ]
            
            for message in test_messages:
                print(f"📤 Sending: {message}")
                await websocket.send(message)
                
                response = await websocket.recv()
                print(f"📥 Received: {response}")
                print("-" * 50)
                
                # Wait a bit between messages
                await asyncio.sleep(1)
                
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    print("Starting WebSocket client test...")
    asyncio.run(test_websocket())
    print("✅ WebSocket test completed!")
