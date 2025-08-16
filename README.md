# MurfAI Challenge - AI Voice Assistant

A comprehensive full-stack AI voice assistant application that combines speech-to-text, text-to-speech, and large language models to create an interactive conversational experience.

## 🎯 Project Overview

This application is an advanced AI voice assistant that allows users to:
- Record voice messages and get them transcribed
- Generate natural-sounding speech from text using Murf AI
- Have conversations with AI using Google's Gemini model
- Upload and process audio files
- Maintain chat sessions with conversation history

## 🏗️ Project Structure

```
MurfAIChallenge/
├── main.py                 # FastAPI backend server with all API endpoints
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
├── generated_audio.mp3    # Sample generated audio file
├── uploads/               # Directory for uploaded audio files
│   └── *.webm            # Recorded audio files
└── static/               # Frontend files
    ├── index.html        # Main web application
    ├── index_new.html    # Alternative UI version
    ├── index_old.html    # Legacy UI version
    ├── app.js           # JavaScript functionality
    └── fallback.mp3     # Fallback audio file
```

## ✨ Features

### 🎤 Voice Recording & Transcription
- **Real-time voice recording** using Web Audio API
- **Audio file upload** support (WebM, MP3, WAV formats)
- **Speech-to-text transcription** using AssemblyAI
- **Confidence scoring** and word count analysis
- **Audio file management** with automatic uploads directory

### 🔊 Text-to-Speech (TTS)
- **Murf AI integration** for high-quality voice synthesis
- **Multiple voice options** with customizable voice IDs
- **Audio generation** from text input
- **Fallback audio** support for error scenarios
- **Echo functionality** - transcribe audio and convert back to speech

### 🤖 AI Conversation
- **Google Gemini 2.0 Flash** integration for intelligent responses
- **Conversational memory** with session-based chat history
- **Multi-turn conversations** with context awareness
- **Structured responses** with usage statistics
- **Error handling** and graceful fallbacks

### 🌐 Web Interface
- **Modern responsive design** with gradient backgrounds
- **Real-time audio visualization** during recording
- **Interactive chat interface** with message history
- **File upload with drag-and-drop** support
- **Loading states and error handling**
- **Keyboard shortcuts** for enhanced UX

## 🛠️ Technologies Used

### Backend
- **FastAPI** - Modern, fast web framework for building APIs
- **Uvicorn** - ASGI server for running the application
- **AssemblyAI** - Speech-to-text transcription service
- **Google Gemini API** - Large language model for AI responses
- **Murf AI API** - Text-to-speech generation
- **aiofiles** - Asynchronous file operations
- **httpx** - Async HTTP client for API calls

### Frontend
- **HTML5** - Modern web markup with audio/video support
- **CSS3** - Advanced styling with gradients, animations, and responsive design
- **Vanilla JavaScript** - Pure JS with modern ES6+ features
- **Web Audio API** - For real-time audio recording
- **Fetch API** - For asynchronous HTTP requests

### File Handling
- **python-multipart** - For handling file uploads
- **Pydantic** - Data validation and serialization
- **UUID** - Unique filename generation
- **datetime** - Timestamp management

## 🚀 Setup and Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager
- Modern web browser with microphone access

### 1. Environment Setup
```bash
# Navigate to project directory
cd /Users/apple/Documents/MurfAIChallenge

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate     # On Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Variables
Set up the following environment variables for full functionality:

```bash
# Required API Keys
export ASSEMBLYAI_API_KEY="your_assemblyai_api_key_here"
export GEMINI_API_KEY="your_gemini_api_key_here"

# Optional: Add to your shell profile (~/.bashrc, ~/.zshrc, etc.)
echo 'export ASSEMBLYAI_API_KEY="your_assemblyai_api_key_here"' >> ~/.zshrc
echo 'export GEMINI_API_KEY="your_gemini_api_key_here"' >> ~/.zshrc
```

#### How to get API Keys:

**AssemblyAI API Key:**
1. Sign up at [AssemblyAI](https://www.assemblyai.com/)
2. Go to your dashboard
3. Copy your API key from the account settings

**Google Gemini API Key:**
1. Visit [Google AI Studio](https://makersuite.google.com/)
2. Sign in with your Google account
3. Create a new API key
4. Copy the generated key

**Murf AI API Key:**
- The Murf API key is passed as a parameter in TTS requests
- Sign up at [Murf AI](https://murf.ai/) to get your API key

### 3. Run the Application
```bash
# Start the FastAPI server
python main.py

# The server will start on http://localhost:8000
```

### 4. Access the Application
- **Main Application**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Alternative API Docs**: http://localhost:8000/redoc

## 📡 API Endpoints

### Core Endpoints
- `GET /` - Serves the main web application
- `GET /api/hello` - Health check endpoint
- `GET /api/data` - Sample data endpoint

### Audio Processing
- `POST /api/upload/audio` - Upload audio files for processing
- `POST /api/transcribe/file` - Transcribe uploaded audio files
- `POST /api/tts/echo` - Transcribe audio and convert back to speech

### AI Services
- `POST /api/tts/generate` - Generate speech from text using Murf AI
- `POST /llm/query` - Single query to Gemini AI
- `POST /agent/chat/{session_id}` - Conversational AI with session memory

## 🎮 Usage Guide

### 1. Voice Recording
1. Click the "Start Recording" button
2. Speak into your microphone
3. Click "Stop Recording" to end
4. The audio will be automatically transcribed

### 2. File Upload
1. Use the file upload area or drag-and-drop
2. Select audio files (WebM, MP3, WAV)
3. Files are automatically processed and transcribed

### 3. Text-to-Speech
1. Enter text in the input field
2. Provide your Murf AI API key
3. Select a voice ID (optional)
4. Click generate to create audio

### 4. AI Conversation
1. Type your message or use voice input
2. The AI will respond with text
3. Optionally convert responses to speech
4. Chat history is maintained per session

## 🔧 Development

### Running in Development Mode
The server runs with auto-reload enabled by default:
```bash
python main.py
# Server automatically restarts on code changes
```

### File Structure Details
- **main.py**: Contains all FastAPI routes and business logic
- **static/**: Frontend assets served directly by FastAPI
- **uploads/**: Automatically created directory for audio uploads
- **requirements.txt**: All Python dependencies with versions

### Adding New Features
1. Add new routes in `main.py`
2. Update frontend in `static/index.html` and `static/app.js`
3. Test using the interactive API docs at `/docs`

## 🐛 Troubleshooting

### Common Issues

**"AssemblyAI API key not set" Warning:**
- Set the `ASSEMBLYAI_API_KEY` environment variable
- Or update the key directly in `main.py` (not recommended for production)

**"Gemini API key not set" Warning:**
- Set the `GEMINI_API_KEY` environment variable
- Or update the key directly in `main.py` (not recommended for production)

**Microphone Access Denied:**
- Check browser permissions for microphone access
- Ensure you're accessing via HTTPS or localhost

**File Upload Errors:**
- Check file format (WebM, MP3, WAV supported)
- Ensure file size is reasonable (< 25MB recommended)
- Verify uploads directory permissions

### Logs and Debugging
- Check terminal output for detailed error messages
- Use browser developer tools for frontend debugging
- API documentation at `/docs` for testing endpoints

## 📄 License

This project is part of the MurfAI Challenge and is intended for educational and demonstration purposes.

## 🤝 Contributing

This is a challenge project, but feel free to:
1. Fork the repository
2. Create feature branches
3. Submit pull requests
4. Report issues and suggestions

## 📞 Support

For questions or issues:
1. Check the troubleshooting section above
2. Review API documentation at `/docs`
3. Check browser console for frontend errors
4. Verify all API keys are properly configured
