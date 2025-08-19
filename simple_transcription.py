"""
Simple working transcription solution using AssemblyAI standard API.
This bypasses the real-time streaming issues by using batch processing.
"""
import asyncio
import tempfile
import subprocess
import assemblyai as aai
import os

# Configure AssemblyAI
API_KEY = "6bf1c3f3202b4be3ba1fc699a6e43dd5"
aai.settings.api_key = API_KEY

async def transcribe_webm_chunks(webm_chunks, session_id="test"):
    """
    Transcribe accumulated WebM chunks using AssemblyAI standard API.
    This is more reliable than real-time streaming.
    """
    if not webm_chunks:
        return ""
    
    try:
        # Combine all chunks
        combined_webm = b''.join(webm_chunks)
        print(f"🔄 Processing {len(webm_chunks)} chunks ({len(combined_webm)} bytes)")
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as temp_file:
            temp_file.write(combined_webm)
            temp_file_path = temp_file.name
        
        try:
            # Transcribe using AssemblyAI standard API
            transcriber = aai.Transcriber()
            
            print(f"🚀 Sending to AssemblyAI for transcription...")
            transcript = transcriber.transcribe(temp_file_path)
            
            if transcript.status == aai.TranscriptStatus.error:
                print(f"❌ Transcription error: {transcript.error}")
                return ""
            
            result_text = transcript.text or ""
            print(f"✅ Transcription result: {result_text}")
            
            return result_text
            
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_file_path)
            except:
                pass
                
    except Exception as e:
        print(f"❌ Transcription error: {e}")
        return ""

# Test the function
if __name__ == "__main__":
    print("🧪 Testing transcription function...")
    
    # Create dummy WebM data for testing
    test_chunks = [b"dummy_webm_data"]
    
    async def test():
        result = await transcribe_webm_chunks(test_chunks)
        print(f"Test result: {result}")
    
    asyncio.run(test())
