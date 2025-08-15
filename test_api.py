#!/usr/bin/env python3
"""
Test script for the refactored MurfAI Challenge API
"""

import asyncio
import httpx
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_api_endpoints():
    """Test the main API endpoints"""
    base_url = "http://localhost:8000"
    
    async with httpx.AsyncClient() as client:
        try:
            # Test health endpoint
            logger.info("Testing health endpoint...")
            response = await client.get(f"{base_url}/api/health")
            print(f"Health check: {response.status_code} - {response.json()}")
            
            # Test hello endpoint
            logger.info("Testing hello endpoint...")
            response = await client.get(f"{base_url}/api/hello")
            print(f"Hello: {response.status_code} - {response.json()}")
            
            # Test data endpoint
            logger.info("Testing data endpoint...")
            response = await client.get(f"{base_url}/api/data")
            print(f"Data: {response.status_code} - {response.json()}")
            
        except httpx.RequestError as e:
            logger.error(f"Request error: {e}")
            print("Make sure the server is running: python main_refactored.py")

if __name__ == "__main__":
    print("Testing refactored API endpoints...")
    print("Make sure to start the server first: python main_refactored.py")
    print("-" * 50)
    asyncio.run(test_api_endpoints())
