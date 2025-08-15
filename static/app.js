// Main JavaScript file for MurfAI Challenge Frontend

// API base URL
const API_BASE = '';

// DOM elements
const responseDiv = document.getElementById('response');
const dataContainer = document.getElementById('dataContainer');

/**
 * Display response in the response div
 * @param {*} data - Data to display
 * @param {string} type - Type of response (success, error, loading)
 */
function displayResponse(data, type = 'success') {
    responseDiv.className = type === 'error' ? 'error' : '';
    
    if (typeof data === 'object') {
        responseDiv.textContent = JSON.stringify(data, null, 2);
    } else {
        responseDiv.textContent = data;
    }
}

/**
 * Clear the response div
 */
function clearResponse() {
    responseDiv.textContent = '';
    responseDiv.className = '';
}

/**
 * Test the hello API endpoint
 */
async function testHelloAPI() {
    try {
        displayResponse('Loading...', 'loading');
        
        const response = await fetch(`${API_BASE}/api/hello`);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        displayResponse(data, 'success');
        
    } catch (error) {
        console.error('Error fetching hello API:', error);
        displayResponse(`Error: ${error.message}`, 'error');
    }
}

/**
 * Fetch data from the API and display it
 */
async function fetchData() {
    try {
        // Update data container with loading message
        dataContainer.innerHTML = '<p class="loading">Loading data...</p>';
        
        // Also show loading in response div
        displayResponse('Fetching data...', 'loading');
        
        const response = await fetch(`${API_BASE}/api/data`);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        
        // Display response in response div
        displayResponse(result, 'success');
        
        // Display data in a nice grid format
        displayDataGrid(result.data);
        
    } catch (error) {
        console.error('Error fetching data:', error);
        displayResponse(`Error: ${error.message}`, 'error');
        dataContainer.innerHTML = `<p class="error">Failed to load data: ${error.message}</p>`;
    }
}

/**
 * Display data in a grid format
 * @param {Array} data - Array of data items
 */
function displayDataGrid(data) {
    if (!data || data.length === 0) {
        dataContainer.innerHTML = '<p>No data available</p>';
        return;
    }
    
    const gridHTML = data.map(item => `
        <div class="data-item">
            <h3>${item.name}</h3>
            <p><strong>ID:</strong> ${item.id}</p>
            <p><strong>Value:</strong> ${item.value}</p>
        </div>
    `).join('');
    
    dataContainer.innerHTML = `
        <div class="data-grid">
            ${gridHTML}
        </div>
        <p style="margin-top: 15px;"><strong>Total items:</strong> ${data.length}</p>
    `;
}

/**
 * Initialize the application
 */
function initApp() {
    console.log('MurfAI Challenge Frontend initialized');
    
    // You can add any initialization logic here
    // For example, automatically fetch some data on page load
    
    // Add event listeners for keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        // Ctrl/Cmd + Enter to test hello API
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            testHelloAPI();
        }
        
        // Ctrl/Cmd + D to fetch data
        if ((e.ctrlKey || e.metaKey) && e.key === 'd') {
            e.preventDefault();
            fetchData();
        }
    });
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', initApp);

// Optional: Add some utility functions
const utils = {
    /**
     * Format timestamp to readable string
     * @param {Date} date 
     */
    formatTime: (date = new Date()) => {
        return date.toLocaleTimeString();
    },
    
    /**
     * Add timestamp to console logs
     * @param {string} message 
     */
    log: (message) => {
        console.log(`[${utils.formatTime()}] ${message}`);
    }
};

/**
 * Generate TTS audio from text input
 */
async function generateTTS() {
    try {
        const textInput = document.getElementById('ttsText');
        const voiceSelect = document.getElementById('voiceSelect');
        const apiKeyInput = document.getElementById('apiKey');
        const generateBtn = document.getElementById('generateTTSBtn');
        const ttsResponseDiv = document.getElementById('ttsResponse');
        const audioPlayerDiv = document.getElementById('audioPlayer');
        const audioElement = document.getElementById('generatedAudio');
        
        // Validate inputs
        const text = textInput.value.trim();
        const voiceId = voiceSelect.value;
        const apiKey = apiKeyInput.value.trim();
        
        if (!text) {
            displayTTSResponse('Please enter some text to convert to speech.', 'error');
            return;
        }
        
        if (!apiKey) {
            displayTTSResponse('Please enter your Murf API key.', 'error');
            return;
        }
        
        // Update UI for loading state
        generateBtn.disabled = true;
        generateBtn.textContent = '⏳ Generating...';
        audioPlayerDiv.style.display = 'none';
        displayTTSResponse('Generating speech, please wait...', 'loading');
        
        // Prepare request payload
        const payload = {
            text: text,
            voice_id: voiceId,
            api_key: apiKey
        };
        
        // Make API request
        const response = await fetch(`${API_BASE}/api/tts/generate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(payload)
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        
        if (result.success && result.audio_url) {
            // Success - display and play audio
            displayTTSResponse({
                message: 'Speech generated successfully!',
                voice: voiceId,
                text_length: text.length,
                audio_url: result.audio_url
            }, 'success');
            
            // Set up audio player
            audioElement.src = result.audio_url;
            audioPlayerDiv.style.display = 'block';
            
            // Auto-play the audio
            try {
                await audioElement.play();
                utils.log('Audio playback started automatically');
            } catch (playError) {
                console.warn('Auto-play failed, user interaction required:', playError);
                displayTTSResponse('Audio generated! Click the play button below to listen.', 'success');
            }
            
        } else {
            // API returned error
            displayTTSResponse(`TTS Generation failed: ${result.error || 'Unknown error'}`, 'error');
        }
        
    } catch (error) {
        console.error('Error generating TTS:', error);
        displayTTSResponse(`Error: ${error.message}`, 'error');
    } finally {
        // Reset button state
        const generateBtn = document.getElementById('generateTTSBtn');
        generateBtn.disabled = false;
        generateBtn.textContent = '🎵 Generate Speech';
    }
}

/**
 * Display TTS response in the TTS response div
 * @param {*} data - Data to display
 * @param {string} type - Type of response (success, error, loading)
 */
function displayTTSResponse(data, type = 'success') {
    const ttsResponseDiv = document.getElementById('ttsResponse');
    
    if (!ttsResponseDiv) return;
    
    ttsResponseDiv.className = type === 'error' ? 'error' : '';
    
    if (typeof data === 'object') {
        ttsResponseDiv.innerHTML = `
            <div style="margin-top: 15px; padding: 15px; background-color: #d4edda; border-radius: 5px; border-left: 4px solid #28a745;">
                <strong>✅ ${data.message}</strong><br>
                <small>Voice: ${data.voice} | Text length: ${data.text_length} characters</small><br>
                <small>Audio URL: <a href="${data.audio_url}" target="_blank" style="color: #007bff;">Open in new tab</a></small>
            </div>
        `;
    } else {
        const bgColor = type === 'error' ? '#f8d7da' : type === 'loading' ? '#e2e3e5' : '#d4edda';
        const borderColor = type === 'error' ? '#dc3545' : type === 'loading' ? '#6c757d' : '#28a745';
        const icon = type === 'error' ? '❌' : type === 'loading' ? '⏳' : '✅';
        
        ttsResponseDiv.innerHTML = `
            <div style="margin-top: 15px; padding: 15px; background-color: ${bgColor}; border-radius: 5px; border-left: 4px solid ${borderColor};">
                ${icon} ${data}
            </div>
        `;
    }
}

/**
 * Clear TTS response and audio player
 */
function clearTTS() {
    const ttsResponseDiv = document.getElementById('ttsResponse');
    const audioPlayerDiv = document.getElementById('audioPlayer');
    const audioElement = document.getElementById('generatedAudio');
    
    if (ttsResponseDiv) ttsResponseDiv.innerHTML = '';
    if (audioPlayerDiv) audioPlayerDiv.style.display = 'none';
    if (audioElement) {
        audioElement.pause();
        audioElement.src = '';
    }
}

// Echo Bot functionality
let mediaRecorder;
let recordedChunks = [];
let recordingTimer;
let recordingStartTime;

/**
 * Start voice recording using MediaRecorder API
 */
async function startRecording() {
    try {
        const startBtn = document.getElementById('startRecordBtn');
        const stopBtn = document.getElementById('stopRecordBtn');
        const status = document.getElementById('recordingStatus');
        const timer = document.getElementById('recordingTimer');
        const response = document.getElementById('echoBotResponse');
        
        // Clear previous recordings
        clearRecording();
        
        // Request microphone access
        displayEchoBotResponse('Requesting microphone access...', 'loading');
        
        const stream = await navigator.mediaDevices.getUserMedia({ 
            audio: {
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true
            } 
        });
        
        // Create MediaRecorder
        recordedChunks = [];
        mediaRecorder = new MediaRecorder(stream);
        
        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                recordedChunks.push(event.data);
            }
        };
        
        mediaRecorder.onstop = () => {
            // Stop all tracks to turn off microphone light
            stream.getTracks().forEach(track => track.stop());
            processRecording();
        };
        
        mediaRecorder.onerror = (event) => {
            console.error('MediaRecorder error:', event.error);
            displayEchoBotResponse(`Recording error: ${event.error}`, 'error');
            resetRecordingUI();
        };
        
        // Start recording
        mediaRecorder.start();
        recordingStartTime = Date.now();
        
        // Update UI
        startBtn.disabled = true;
        stopBtn.disabled = false;
        startBtn.classList.add('recording');
        status.className = 'recording-status status-recording';
        status.textContent = '🔴 Recording...';
        timer.style.display = 'block';
        
        // Start timer
        recordingTimer = setInterval(updateTimer, 100);
        
        displayEchoBotResponse('Recording started! Speak into your microphone.', 'success');
        
    } catch (error) {
        console.error('Error starting recording:', error);
        let errorMessage = 'Failed to start recording. ';
        
        if (error.name === 'NotAllowedError') {
            errorMessage += 'Please allow microphone access and try again.';
        } else if (error.name === 'NotFoundError') {
            errorMessage += 'No microphone found. Please check your audio devices.';
        } else {
            errorMessage += error.message;
        }
        
        displayEchoBotResponse(errorMessage, 'error');
        resetRecordingUI();
    }
}

/**
 * Stop voice recording
 */
function stopRecording() {
    try {
        if (mediaRecorder && mediaRecorder.state === 'recording') {
            mediaRecorder.stop();
            
            // Clear timer
            if (recordingTimer) {
                clearInterval(recordingTimer);
                recordingTimer = null;
            }
            
            // Update UI
            const startBtn = document.getElementById('startRecordBtn');
            const stopBtn = document.getElementById('stopRecordBtn');
            const status = document.getElementById('recordingStatus');
            
            stopBtn.disabled = true;
            startBtn.classList.remove('recording');
            status.className = 'recording-status status-recorded';
            status.textContent = '⏹️ Processing recording...';
            
            displayEchoBotResponse('Recording stopped. Processing audio...', 'loading');
        }
    } catch (error) {
        console.error('Error stopping recording:', error);
        displayEchoBotResponse(`Error stopping recording: ${error.message}`, 'error');
        resetRecordingUI();
    }
}

/**
 * Process the recorded audio and make it playable
 */
function processRecording() {
    try {
        if (recordedChunks.length === 0) {
            displayEchoBotResponse('No audio data recorded. Please try again.', 'error');
            resetRecordingUI();
            return;
        }
        
        // Create blob from recorded chunks
        const blob = new Blob(recordedChunks, { type: 'audio/webm' });
        const audioUrl = URL.createObjectURL(blob);
        
        // Set up audio player
        const audioElement = document.getElementById('echoAudio');
        const audioPlayerDiv = document.getElementById('echoAudioPlayer');
        
        audioElement.src = audioUrl;
        audioPlayerDiv.style.display = 'block';
        
        // Update status
        const status = document.getElementById('recordingStatus');
        status.className = 'recording-status status-recorded';
        status.textContent = '✅ Recording ready to play!';
        
        // Calculate recording duration
        const duration = recordingStartTime ? (Date.now() - recordingStartTime) / 1000 : 0;
        
        displayEchoBotResponse({
            message: 'Recording completed successfully!',
            duration: duration.toFixed(1),
            size: `${(blob.size / 1024).toFixed(1)} KB`,
            type: blob.type
        }, 'success');
        
        // Auto-play the recording
        setTimeout(() => {
            audioElement.play().catch(error => {
                console.warn('Auto-play failed:', error);
                displayEchoBotResponse('Recording ready! Click the play button to listen.', 'success');
            });
        }, 500);
        
        // Echo with Murf voice instead of just transcribing
        echoWithMurfVoice(blob, duration);
        
        // Reset UI for next recording
        resetRecordingUI();
        
    } catch (error) {
        console.error('Error processing recording:', error);
        displayEchoBotResponse(`Error processing recording: ${error.message}`, 'error');
        resetRecordingUI();
    }
}

/**
 * Clear the current recording and reset UI
 */
function clearRecording() {
    const audioElement = document.getElementById('echoAudio');
    const audioPlayerDiv = document.getElementById('echoAudioPlayer');
    const response = document.getElementById('echoBotResponse');
    
    // Stop and clear audio
    if (audioElement) {
        audioElement.pause();
        if (audioElement.src) {
            URL.revokeObjectURL(audioElement.src);
            audioElement.src = '';
        }
    }
    
    // Hide audio player
    if (audioPlayerDiv) {
        audioPlayerDiv.style.display = 'none';
    }
    
    // Clear response
    if (response) {
        response.innerHTML = '';
    }
    
    // Clear recorded data
    recordedChunks = [];
    
    // Reset UI
    resetRecordingUI();
    
    const status = document.getElementById('recordingStatus');
    if (status) {
        status.className = 'recording-status status-idle';
        status.textContent = '🟢 Ready to record';
    }
}

/**
 * Reset recording UI to initial state
 */
function resetRecordingUI() {
    const startBtn = document.getElementById('startRecordBtn');
    const stopBtn = document.getElementById('stopRecordBtn');
    const timer = document.getElementById('recordingTimer');
    
    if (startBtn) {
        startBtn.disabled = false;
        startBtn.classList.remove('recording');
    }
    
    if (stopBtn) {
        stopBtn.disabled = true;
    }
    
    if (timer) {
        timer.style.display = 'none';
        timer.textContent = '00:00';
    }
    
    // Clear timer if running
    if (recordingTimer) {
        clearInterval(recordingTimer);
        recordingTimer = null;
    }
}

/**
 * Update the recording timer display
 */
function updateTimer() {
    if (!recordingStartTime) return;
    
    const elapsed = (Date.now() - recordingStartTime) / 1000;
    const minutes = Math.floor(elapsed / 60);
    const seconds = Math.floor(elapsed % 60);
    
    const timer = document.getElementById('recordingTimer');
    if (timer) {
        timer.textContent = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
    }
}

/**
 * Display Echo Bot response messages
 * @param {*} data - Data to display
 * @param {string} type - Type of response (success, error, loading, upload-success, transcription-success)
 */
function displayEchoBotResponse(data, type = 'success') {
    const responseDiv = document.getElementById('echoBotResponse');
    
    if (!responseDiv) return;
    
    if (typeof data === 'object') {
        if (type === 'echo-success') {
            responseDiv.innerHTML = `
                <div style="margin-top: 15px; padding: 15px; background-color: #e7f3ff; border-radius: 5px; border-left: 4px solid #007bff;">
                    <strong>🎉 ${data.message}</strong><br>
                    <div style="margin: 10px 0; padding: 10px; background-color: #f8f9fa; border-radius: 3px; border-left: 3px solid #6c757d;">
                        <strong>📝 Your words:</strong><br>
                        <em style="font-size: 16px; line-height: 1.4;">"${data.transcript || 'No speech detected'}"</em>
                    </div>
                    ${data.audio_url ? `
                    <div style="margin: 10px 0; padding: 10px; background-color: #e7f3ff; border-radius: 3px; border-left: 3px solid #007bff;">
                        <strong>🎙️ Murf Voice:</strong> ${data.voice_id}<br>
                        <strong>🔊 Status:</strong> Generated and ready to play!
                    </div>` : `
                    <div style="margin: 10px 0; padding: 10px; background-color: #fff3cd; border-radius: 3px; border-left: 3px solid #856404;">
                        <strong>⏳ Status:</strong> Transcription complete, Murf integration pending
                    </div>`}
                    <small><strong>Stats:</strong> ${data.word_count} words | Duration: ${data.duration}s${data.confidence ? ` | Confidence: ${(data.confidence * 100).toFixed(1)}%` : ''}</small>
                </div>
            `;
        } else if (type === 'transcription-success') {
            responseDiv.innerHTML = `
                <div style="margin-top: 15px; padding: 15px; background-color: #e7f3ff; border-radius: 5px; border-left: 4px solid #007bff;">
                    <strong>🎯 ${data.message}</strong><br>
                    <div style="margin: 10px 0; padding: 10px; background-color: #f8f9fa; border-radius: 3px; border-left: 3px solid #6c757d;">
                        <strong>📝 Transcript:</strong><br>
                        <em style="font-size: 16px; line-height: 1.4;">"${data.transcript || 'No speech detected'}"</em>
                    </div>
                    <small><strong>Stats:</strong> ${data.word_count} words | Duration: ${data.duration}s${data.confidence ? ` | Confidence: ${(data.confidence * 100).toFixed(1)}%` : ''}</small>
                </div>
            `;
        } else if (type === 'upload-success') {
            responseDiv.innerHTML = `
                <div style="margin-top: 15px; padding: 15px; background-color: #cce5ff; border-radius: 5px; border-left: 4px solid #007bff;">
                    <strong>📤 ${data.message}</strong><br>
                    <small><strong>Local:</strong> Duration: ${data.duration}s | Size: ${data.size}</small><br>
                    <small><strong>Server:</strong> ${data.filename} | ${data.content_type}</small><br>
                    <small><strong>Path:</strong> ${data.server_path}</small>
                </div>
            `;
        } else {
            responseDiv.innerHTML = `
                <div style="margin-top: 15px; padding: 15px; background-color: #d4edda; border-radius: 5px; border-left: 4px solid #28a745;">
                    <strong>✅ ${data.message}</strong><br>
                    <small>Duration: ${data.duration}s | Size: ${data.size} | Format: ${data.type}</small>
                </div>
            `;
        }
    } else {
        const bgColor = type === 'error' ? '#f8d7da' : type === 'loading' ? '#e2e3e5' : '#d4edda';
        const borderColor = type === 'error' ? '#dc3545' : type === 'loading' ? '#6c757d' : '#28a745';
        const icon = type === 'error' ? '❌' : type === 'loading' ? '⏳' : '✅';
        
        responseDiv.innerHTML = `
            <div style="margin-top: 15px; padding: 15px; background-color: ${bgColor}; border-radius: 5px; border-left: 4px solid ${borderColor};">
                ${icon} ${data}
            </div>
        `;
    }
}

/**
 * Echo recorded audio using Murf voice (transcribe + TTS)
 * @param {Blob} audioBlob - Audio blob to process
 * @param {number} duration - Recording duration in seconds
 */
async function echoWithMurfVoice(audioBlob, duration) {
    try {
        // Get Murf configuration from Echo Bot form
        const murfApiKey = document.getElementById('echoMurfApiKey')?.value?.trim();
        const voiceId = document.getElementById('echoVoiceSelect')?.value || 'en-US-cooper';
        
        // Validate Murf API key
        if (!murfApiKey) {
            displayEchoBotResponse('Please enter your Murf API key to use Echo Bot v2!', 'error');
            return;
        }
        
        // Show processing status
        displayEchoBotResponse('🎯 Processing your voice... (Transcribing + Generating Murf voice)', 'loading');
        
        // Create FormData for file upload
        const formData = new FormData();
        formData.append('audio_file', audioBlob, 'recording.webm');
        formData.append('voice_id', voiceId);
        formData.append('murf_api_key', murfApiKey);
        
        // Send to echo endpoint
        const response = await fetch(`${API_BASE}/api/tts/echo`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`Echo request failed with status ${response.status}`);
        }
        
        const result = await response.json();
        
        if (result.success) {
            // Success - show echo results
            displayEchoBotResponse({
                message: result.audio_url ? 'Echo generated successfully with Murf voice!' : 'Transcription completed (Murf integration pending)',
                transcript: result.transcript,
                voice_id: result.voice_id,
                word_count: result.word_count,
                confidence: result.confidence,
                duration: duration.toFixed(1),
                audio_url: result.audio_url
            }, 'echo-success');
            
            // If we have Murf audio URL, set up audio player
            if (result.audio_url) {
                const audioElement = document.getElementById('echoAudio');
                const audioPlayerDiv = document.getElementById('echoAudioPlayer');
                
                audioElement.src = result.audio_url;
                audioPlayerDiv.style.display = 'block';
                
                // Auto-play the Murf audio
                try {
                    await audioElement.play();
                    utils.log('Murf echo audio playback started automatically');
                } catch (playError) {
                    console.warn('Auto-play failed, user interaction required:', playError);
                }
            }
            
            utils.log(`Echo processed: "${result.transcript}" with voice ${result.voice_id}`);
        } else {
            // Server returned error
            displayEchoBotResponse(`Echo failed: ${result.error}`, 'error');
        }
        
    } catch (error) {
        console.error('Error processing echo:', error);
        displayEchoBotResponse(`Echo error: ${error.message}`, 'error');
    }
}

// ========================
// Voice-to-Voice AI Chat Functions
// ========================

let aiChatRecorder = null;
let aiChatRecordingTimer = null;
let aiChatStartTime = null;

/**
 * Start recording for AI chat
 */
function startAIChatRecording() {
    const apiKey = document.getElementById('aiChatMurfApiKey').value;
    if (!apiKey.trim()) {
        alert('Please enter your Murf API key first!');
        return;
    }
    
    navigator.mediaDevices.getUserMedia({ audio: true })
        .then(stream => {
            aiChatRecorder = new MediaRecorder(stream);
            aiChatRecorder.chunks = [];
            
            aiChatRecorder.ondataavailable = (e) => {
                aiChatRecorder.chunks.push(e.data);
            };
            
            aiChatRecorder.onstop = async () => {
                const audioBlob = new Blob(aiChatRecorder.chunks, { type: 'audio/wav' });
                await processAIChatAudio(audioBlob);
                stream.getTracks().forEach(track => track.stop());
            };
            
            aiChatRecorder.start();
            
            // Update UI
            document.getElementById('aiChatStartRecordBtn').disabled = true;
            document.getElementById('aiChatStopRecordBtn').disabled = false;
            document.getElementById('aiChatRecordingStatus').textContent = '🔴 Recording your question...';
            document.getElementById('aiChatRecordingStatus').className = 'recording-status status-recording';
            
            // Start timer
            aiChatStartTime = Date.now();
            document.getElementById('aiChatRecordingTimer').style.display = 'block';
            aiChatRecordingTimer = setInterval(updateAIChatTimer, 100);
        })
        .catch(err => {
            console.error('Error accessing microphone:', err);
            alert('Error accessing microphone. Please check permissions.');
        });
}

/**
 * Stop recording for AI chat
 */
function stopAIChatRecording() {
    if (aiChatRecorder && aiChatRecorder.state === 'recording') {
        aiChatRecorder.stop();
        
        // Update UI
        document.getElementById('aiChatStartRecordBtn').disabled = false;
        document.getElementById('aiChatStopRecordBtn').disabled = true;
        document.getElementById('aiChatRecordingStatus').textContent = '🤖 Processing your question with AI...';
        document.getElementById('aiChatRecordingStatus').className = 'recording-status status-processing';
        
        // Stop timer
        clearInterval(aiChatRecordingTimer);
        document.getElementById('aiChatRecordingTimer').style.display = 'none';
    }
}

/**
 * Update the recording timer for AI chat
 */
function updateAIChatTimer() {
    const elapsed = Date.now() - aiChatStartTime;
    const seconds = Math.floor(elapsed / 1000);
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    document.getElementById('aiChatRecordingTimer').textContent = 
        `${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`;
}

/**
 * Process the recorded audio for AI chat
 * @param {Blob} audioBlob - The recorded audio blob
 */
async function processAIChatAudio(audioBlob) {
    const apiKey = document.getElementById('aiChatMurfApiKey').value;
    const voiceId = document.getElementById('aiChatVoiceSelect').value;
    
    try {
        const formData = new FormData();
        formData.append('audio_file', audioBlob, 'recording.wav');
        formData.append('murf_api_key', apiKey);
        formData.append('voice_id', voiceId);
        
        document.getElementById('aiChatResponse').innerHTML = '<p>🧠 AI is thinking about your question...</p>';
        
        const response = await fetch('/llm/query', {
            method: 'POST',
            body: formData
        });
        
        if (response.ok) {
            const result = await response.json();
            
            // Display response info
            document.getElementById('aiChatResponse').innerHTML = `
                <div style="padding: 15px; background-color: #f8f9fa; border-radius: 5px; margin: 10px 0;">
                    <h4>📝 Your Question (Transcript):</h4>
                    <p style="font-style: italic; background-color: #e9ecef; padding: 10px; border-radius: 3px;">"${result.transcript}"</p>
                    <h4>🤖 AI Response:</h4>
                    <p style="background-color: #d1ecf1; padding: 10px; border-radius: 3px;">${result.response}</p>
                    <p><small><strong>Voice:</strong> ${result.voice_id}</small></p>
                </div>
            `;
            
            // Check if audio URL is valid before trying to play
            if (result.audio_url && result.audio_url !== null && result.audio_url !== 'null') {
                // Play AI response audio
                const audioPlayer = document.getElementById('aiChatAudio');
                audioPlayer.src = result.audio_url;
                document.getElementById('aiChatAudioPlayer').style.display = 'block';
            } else {
                // Show message that audio generation failed
                document.getElementById('aiChatResponse').innerHTML += `
                    <div style="padding: 10px; background-color: #fff3cd; color: #856404; border-radius: 3px; margin-top: 10px;">
                        ⚠️ Audio generation failed, but text response is available above.
                    </div>
                `;
                console.warn('Audio URL is null or invalid:', result.audio_url);
            }
            
            // Update status
            document.getElementById('aiChatRecordingStatus').textContent = '✅ AI response ready!';
            document.getElementById('aiChatRecordingStatus').className = 'recording-status status-complete';
            
            utils.log(`AI Chat processed: "${result.transcript}" -> "${result.response}" with voice ${result.voice_id}`);
            
        } else {
            const error = await response.json();
            document.getElementById('aiChatResponse').innerHTML = 
                `<p style="color: red;">❌ Error: ${error.detail}</p>`;
            
            document.getElementById('aiChatRecordingStatus').textContent = '❌ Processing failed';
            document.getElementById('aiChatRecordingStatus').className = 'recording-status status-error';
        }
        
    } catch (error) {
        console.error('Error processing AI chat:', error);
        document.getElementById('aiChatResponse').innerHTML = 
            '<p style="color: red;">❌ Network error occurred</p>';
        
        document.getElementById('aiChatRecordingStatus').textContent = '❌ Network error';
        document.getElementById('aiChatRecordingStatus').className = 'recording-status status-error';
    }
}

/**
 * Clear the AI chat interface
 */
function clearAIChat() {
    document.getElementById('aiChatResponse').innerHTML = '';
    document.getElementById('aiChatAudioPlayer').style.display = 'none';
    document.getElementById('aiChatRecordingStatus').textContent = '🟢 Ready to record your question';
    document.getElementById('aiChatRecordingStatus').className = 'recording-status status-idle';
    utils.log('AI Chat cleared');
}

// ========================
// Conversational Agent Functions
// ========================

let agentRecorder = null;
let agentRecordingTimer = null;
let agentStartTime = null;
let currentSessionId = null;
let isAutoRecordingEnabled = false;

/**
 * Initialize session ID from URL or generate new one
 */
function initializeSession() {
    const urlParams = new URLSearchParams(window.location.search);
    currentSessionId = urlParams.get('session');
    
    if (!currentSessionId) {
        // Generate new session ID
        currentSessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
        
        // Update URL without refreshing page
        const newUrl = new URL(window.location);
        newUrl.searchParams.set('session', currentSessionId);
        window.history.replaceState({}, '', newUrl);
    }
    
    // Display session ID
    const sessionIdElement = document.getElementById('sessionId');
    if (sessionIdElement) {
        sessionIdElement.textContent = currentSessionId;
    }
    
    utils.log(`Session initialized: ${currentSessionId}`);
}

/**
 * Start recording for conversational agent
 */
function startAgentRecording() {
    const apiKey = document.getElementById('agentMurfApiKey').value;
    if (!apiKey.trim()) {
        alert('Please enter your Murf API key first!');
        return;
    }
    
    navigator.mediaDevices.getUserMedia({ audio: true })
        .then(stream => {
            agentRecorder = new MediaRecorder(stream);
            agentRecorder.chunks = [];
            
            agentRecorder.ondataavailable = (e) => {
                agentRecorder.chunks.push(e.data);
            };
            
            agentRecorder.onstop = async () => {
                const audioBlob = new Blob(agentRecorder.chunks, { type: 'audio/wav' });
                await processAgentAudio(audioBlob);
                stream.getTracks().forEach(track => track.stop());
            };
            
            agentRecorder.start();
            
            // Update UI
            document.getElementById('agentStartRecordBtn').disabled = true;
            document.getElementById('agentStopRecordBtn').disabled = false;
            document.getElementById('agentRecordingStatus').textContent = '🔴 Recording your message...';
            document.getElementById('agentRecordingStatus').className = 'recording-status status-recording';
            
            // Start timer
            agentStartTime = Date.now();
            document.getElementById('agentRecordingTimer').style.display = 'block';
            agentRecordingTimer = setInterval(updateAgentTimer, 100);
        })
        .catch(err => {
            console.error('Error accessing microphone:', err);
            alert('Error accessing microphone. Please check permissions.');
        });
}

/**
 * Stop recording for conversational agent
 */
function stopAgentRecording() {
    if (agentRecorder && agentRecorder.state === 'recording') {
        agentRecorder.stop();
        
        // Update UI
        document.getElementById('agentStartRecordBtn').disabled = false;
        document.getElementById('agentStopRecordBtn').disabled = true;
        document.getElementById('agentRecordingStatus').textContent = '🤖 Processing with AI (with chat history)...';
        document.getElementById('agentRecordingStatus').className = 'recording-status status-processing';
        
        // Stop timer
        clearInterval(agentRecordingTimer);
        document.getElementById('agentRecordingTimer').style.display = 'none';
    }
}

/**
 * Update the recording timer for conversational agent
 */
function updateAgentTimer() {
    const elapsed = Date.now() - agentStartTime;
    const seconds = Math.floor(elapsed / 1000);
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    document.getElementById('agentRecordingTimer').textContent = 
        `${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`;
}

/**
 * Process the recorded audio for conversational agent
 * @param {Blob} audioBlob - The recorded audio blob
 */
async function processAgentAudio(audioBlob) {
    const apiKey = document.getElementById('agentMurfApiKey').value;
    const voiceId = document.getElementById('agentVoiceSelect').value;
    
    try {
        const formData = new FormData();
        formData.append('audio_file', audioBlob, 'recording.wav');
        formData.append('murf_api_key', apiKey);
        formData.append('voice_id', voiceId);
        
        // Show processing status
        updateChatHistory('system', '🧠 AI is processing your message with conversation context...');
        
        const response = await fetch(`/agent/chat/${currentSessionId}`, {
            method: 'POST',
            body: formData
        });
        
        if (response.ok) {
            const result = await response.json();
            
            // Add user message to chat history
            addMessageToChat('user', result.transcript);
            
            // Add AI response to chat history  
            addMessageToChat('assistant', result.response);
            
            // Remove processing message
            removeChatMessage('system');
            
            // Check if audio URL is valid before trying to play
            if (result.audio_url && result.audio_url !== null && result.audio_url !== 'null') {
                // Play AI response audio
                const audioPlayer = document.getElementById('agentAudio');
                console.log('🔊 Setting audio source:', result.audio_url);
                audioPlayer.src = result.audio_url;
                
                // Enable auto-recording after audio finishes
                isAutoRecordingEnabled = true;
                
                // Auto-play the audio
                try {
                    console.log('🎵 Attempting to play audio...');
                    
                    // Test if the audio element can load the URL first
                    audioPlayer.addEventListener('loadstart', () => {
                        console.log('🔄 Audio started loading...');
                    });
                    
                    audioPlayer.addEventListener('canplay', () => {
                        console.log('✅ Audio can play - ready for playback');
                    });
                    
                    audioPlayer.addEventListener('error', (e) => {
                        console.error('❌ Audio error:', e);
                        addMessageToChat('system', `❌ Audio playback error. Manual link: <a href="${result.audio_url}" target="_blank">Play Audio</a>`);
                    });
                    
                    const playPromise = audioPlayer.play();
                    await playPromise;
                    console.log('✅ Audio playback started successfully!');
                } catch (playError) {
                    console.warn('⚠️ Auto-play failed:', playError);
                    // Add a temporary visible audio element for manual playback
                    addMessageToChat('system', `🔊 AI response audio ready! <audio controls src="${result.audio_url}" style="width: 100%; margin: 10px 0;"></audio>`);
                    // Still enable auto-recording
                    setTimeout(() => {
                        if (isAutoRecordingEnabled) {
                            autoStartNextRecording();
                        }
                    }, 1000);
                }
            } else {
                // Show message that audio generation failed but continue conversation
                addMessageToChat('system', '⚠️ Audio generation failed, but you can continue the conversation.');
                console.warn('Audio URL is null or invalid:', result.audio_url);
                
                // Still enable auto-recording for next message
                setTimeout(() => {
                    autoStartNextRecording();
                }, 2000);
            }
            
            // Update status
            resetRecordButton();
            document.getElementById('agentRecordingStatus').textContent = '✅ AI response ready!';
            document.getElementById('agentRecordingStatus').className = 'status-indicator status-complete';
            
            utils.log(`Agent processed: "${result.transcript}" -> "${result.response}" with voice ${result.voice_id}`);
            
        } else {
            const error = await response.json();
            removeChatMessage('system');
            addMessageToChat('system', `❌ Error: ${error.detail || 'Unknown error'}`);
            
            resetRecordButton();
            document.getElementById('agentRecordingStatus').textContent = '❌ Processing failed';
            document.getElementById('agentRecordingStatus').className = 'status-indicator status-error';
        }
        
    } catch (error) {
        console.error('Error processing agent audio:', error);
        removeChatMessage('system');
        addMessageToChat('system', '❌ Network error occurred');
        
        resetRecordButton();
        document.getElementById('agentRecordingStatus').textContent = '❌ Network error';
        document.getElementById('agentRecordingStatus').className = 'status-indicator status-error';
    }
}

/**
 * Add a message to the chat history display
 */
function addMessageToChat(role, content) {
    const chatHistory = document.getElementById('chatHistory');
    if (!chatHistory) return;
    
    // Clear "no conversation" message if it exists
    const emptyState = chatHistory.querySelector('.empty-state');
    if (emptyState) {
        chatHistory.innerHTML = '';
    }
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message ${role}`;
    messageDiv.innerHTML = `
        <div>${content}</div>
        <div class="timestamp">${new Date().toLocaleTimeString()}</div>
    `;
    
    chatHistory.appendChild(messageDiv);
    
    // Scroll to bottom
    chatHistory.scrollTop = chatHistory.scrollHeight;
}

/**
 * Update an existing system message (for processing status)
 */
function updateChatHistory(role, content) {
    // Remove any existing system messages first
    removeChatMessage('system');
    addMessageToChat(role, content);
}

/**
 * Remove messages of a specific role from chat
 */
function removeChatMessage(role) {
    const chatHistory = document.getElementById('chatHistory');
    if (!chatHistory) return;
    
    const messages = chatHistory.querySelectorAll(`.chat-message.${role}`);
    messages.forEach(msg => msg.remove());
}

/**
 * Clear all chat history
 */
function clearChatHistory() {
    const chatHistory = document.getElementById('chatHistory');
    if (chatHistory) {
        chatHistory.innerHTML = '<div class="empty-state">No conversation yet. Click the microphone button to start chatting!</div>';
    }
    
    // Hide audio player (it's hidden by default in new UI)
    const audioPlayer = document.getElementById('agentAudio');
    if (audioPlayer) {
        audioPlayer.pause();
        audioPlayer.src = '';
    }
    
    // Reset status and button
    resetRecordButton();
    
    // Disable auto-recording
    isAutoRecordingEnabled = false;
    
    utils.log('Chat history cleared');
}

/**
 * Called when agent audio finishes playing
 */
function onAgentAudioEnded() {
    if (isAutoRecordingEnabled) {
        setTimeout(() => {
            autoStartNextRecording();
        }, 1000); // 1 second delay before auto-recording
    }
}

/**
 * Automatically start recording the next message
 */
function autoStartNextRecording() {
    if (!isAutoRecordingEnabled) return;
    
    // Update status to show auto-recording countdown
    const status = document.getElementById('agentRecordingStatus');
    status.textContent = '🎤 Auto-recording in 3 seconds... (or click to start manually)';
    status.className = 'status-indicator auto-recording';
    
    setTimeout(() => {
        if (isAutoRecordingEnabled && (!agentRecorder || agentRecorder.state === 'inactive')) {
            toggleRecording(); // Use the new unified function
        }
    }, 3000);
}

/**
 * Toggle recording function for the new unified button
 */
function toggleRecording() {
    const button = document.getElementById('mainRecordButton');
    const status = document.getElementById('agentRecordingStatus');
    const timer = document.getElementById('agentRecordingTimer');
    
    if (!agentRecorder || agentRecorder.state === 'inactive') {
        // Start recording
        const apiKey = document.getElementById('agentMurfApiKey').value;
        if (!apiKey.trim()) {
            alert('Please enter your Murf API key first!');
            return;
        }
        
        navigator.mediaDevices.getUserMedia({ audio: true })
            .then(stream => {
                agentRecorder = new MediaRecorder(stream);
                agentRecorder.chunks = [];
                
                agentRecorder.ondataavailable = (e) => {
                    agentRecorder.chunks.push(e.data);
                };
                
                agentRecorder.onstop = async () => {
                    const audioBlob = new Blob(agentRecorder.chunks, { type: 'audio/wav' });
                    await processAgentAudio(audioBlob);
                    stream.getTracks().forEach(track => track.stop());
                };
                
                agentRecorder.start();
                
                // Update UI for recording state
                button.className = 'record-button recording';
                button.innerHTML = '⏹️ Stop';
                status.textContent = '🔴 Recording your message...';
                status.className = 'status-indicator status-recording';
                
                // Start timer
                agentStartTime = Date.now();
                timer.classList.add('visible');
                agentRecordingTimer = setInterval(updateAgentTimer, 100);
            })
            .catch(err => {
                console.error('Error accessing microphone:', err);
                alert('Error accessing microphone. Please check permissions.');
            });
    } else if (agentRecorder.state === 'recording') {
        // Stop recording
        agentRecorder.stop();
        
        // Update UI for processing state
        button.className = 'record-button processing';
        button.innerHTML = '⟳ Processing';
        button.disabled = true;
        status.textContent = '🤖 Processing with AI (with chat history)...';
        status.className = 'status-indicator status-processing';
        
        // Stop timer
        clearInterval(agentRecordingTimer);
        timer.classList.remove('visible');
    }
}

/**
 * Reset the main record button to idle state
 */
function resetRecordButton() {
    const button = document.getElementById('mainRecordButton');
    const status = document.getElementById('agentRecordingStatus');
    
    button.className = 'record-button idle';
    button.innerHTML = '🎤 Start';
    button.disabled = false;
    status.textContent = '🟢 Ready to start conversation';
    status.className = 'status-indicator status-idle';
}

/**
 * Update agent timer for new UI
 */
function updateAgentTimer() {
    const elapsed = Date.now() - agentStartTime;
    const seconds = Math.floor(elapsed / 1000);
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    document.getElementById('agentRecordingTimer').textContent = 
        `${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`;
}

// Initialize session when page loads
document.addEventListener('DOMContentLoaded', () => {
    initializeSession();
});

// Export for potential use in other scripts
window.MurfAIApp = {
    testHelloAPI,
    fetchData,
    clearResponse,
    generateTTS,
    clearTTS,
    startRecording,
    stopRecording,
    clearRecording,
    echoWithMurfVoice,
    startAIChatRecording,
    stopAIChatRecording,
    clearAIChat,
    startAgentRecording,
    stopAgentRecording,
    clearChatHistory,
    toggleRecording,
    resetRecordButton,
    utils
};
