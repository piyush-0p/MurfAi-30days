// API Configuration Module
const ApiConfig = {
    // API key storage keys
    STORAGE_KEYS: {
        ASSEMBLYAI: 'assemblyai_api_key',
        GEMINI: 'gemini_api_key',
        WEATHER: 'weather_api_key',
        TAVILY: 'tavily_api_key',
        MURF: 'murf_api_key'
    },

    // Initialize sidebar functionality
    init() {
        this.setupEventListeners();
        
        // Create the sidebar if it doesn't exist
        if (!document.getElementById('apiConfigSidebar')) {
            this.createSidebar();
        }
    },

    // Set up event listeners
    setupEventListeners() {
        // After sidebar is created
        document.addEventListener('sidebar-created', () => {
            document.getElementById('openApiConfigBtn')?.addEventListener('click', this.openSidebar);
            document.getElementById('closeApiConfigBtn')?.addEventListener('click', this.closeSidebar);
            document.getElementById('saveApiConfigBtn')?.addEventListener('click', this.saveApiKeys);
            
            // Load saved keys when sidebar opens
            document.getElementById('apiConfigSidebar')?.addEventListener('sidebar-opened', this.loadSavedKeys);
            
            // Close sidebar when clicking outside
            document.addEventListener('click', (e) => {
                const sidebar = document.getElementById('apiConfigSidebar');
                const openBtn = document.getElementById('openApiConfigBtn');
                
                if (sidebar?.classList.contains('active') && 
                    !sidebar.contains(e.target) && 
                    e.target !== openBtn) {
                    this.closeSidebar();
                }
            });
        });
    },

    // Create the sidebar HTML
    createSidebar() {
        const sidebar = document.createElement('div');
        sidebar.id = 'apiConfigSidebar';
        sidebar.className = 'api-config-sidebar';
        
        sidebar.innerHTML = `
            <div class="api-config-content">
                <div class="api-config-header">
                    <h2>🔑 API Configuration</h2>
                    <button id="closeApiConfigBtn" class="close-btn" aria-label="Close">×</button>
                </div>
                
                <div class="api-config-body">
                    <p class="api-config-description">
                        💡 <strong>Configure your API keys</strong><br>
                        Enter your personal API keys below to use your own accounts. 
                        All keys are stored securely on your device and never shared.
                    </p>
                    
                    <div class="api-config-section">
                        <label for="assemblyaiKey">
                            <span class="api-name">🎤 AssemblyAI</span>
                            <span class="api-purpose">Real-time Speech Recognition</span>
                        </label>
                        <div class="api-input-group">
                            <input type="password" id="assemblyaiKey" placeholder="Enter your AssemblyAI API Key" class="api-input">
                            <button class="toggle-visibility-btn" data-for="assemblyaiKey" aria-label="Toggle visibility">👁️</button>
                        </div>
                    </div>
                    
                    <div class="api-config-section">
                        <label for="geminiKey">
                            <span class="api-name">🧠 Google Gemini</span>
                            <span class="api-purpose">Advanced AI Language Model</span>
                        </label>
                        <div class="api-input-group">
                            <input type="password" id="geminiKey" placeholder="Enter your Google Gemini API Key" class="api-input">
                            <button class="toggle-visibility-btn" data-for="geminiKey" aria-label="Toggle visibility">👁️</button>
                        </div>
                    </div>
                    
                    <div class="api-config-section">
                        <label for="murfKey">
                            <span class="api-name">🎙️ MurfAI</span>
                            <span class="api-purpose">High-Quality Text-to-Speech</span>
                        </label>
                        <div class="api-input-group">
                            <input type="password" id="murfKey" placeholder="Enter your MurfAI API Key" class="api-input">
                            <button class="toggle-visibility-btn" data-for="murfKey" aria-label="Toggle visibility">👁️</button>
                        </div>
                    </div>
                    
                    <div class="api-config-section">
                        <label for="weatherKey">
                            <span class="api-name">⛅ OpenWeatherMap</span>
                            <span class="api-purpose">Real-time Weather Data</span>
                        </label>
                        <div class="api-input-group">
                            <input type="password" id="weatherKey" placeholder="Enter your OpenWeatherMap API Key" class="api-input">
                            <button class="toggle-visibility-btn" data-for="weatherKey" aria-label="Toggle visibility">👁️</button>
                        </div>
                    </div>
                    
                    <div class="api-config-section">
                        <label for="tavilyKey">
                            <span class="api-name">🔍 Tavily</span>
                            <span class="api-purpose">Intelligent Web Search</span>
                        </label>
                        <div class="api-input-group">
                            <input type="password" id="tavilyKey" placeholder="Enter your Tavily API Key" class="api-input">
                            <button class="toggle-visibility-btn" data-for="tavilyKey" aria-label="Toggle visibility">👁️</button>
                        </div>
                    </div>
                    
                    <button id="saveApiConfigBtn" class="api-save-btn">
                        💾 Save Configuration
                    </button>
                    
                    <div id="apiConfigStatus" class="api-config-status"></div>
                </div>
            </div>
        `;
        
        document.body.appendChild(sidebar);
        
        // Add enhanced toggle visibility listeners with improved UX
        const toggleBtns = sidebar.querySelectorAll('.toggle-visibility-btn');
        toggleBtns.forEach(btn => {
            btn.addEventListener('click', () => {
                const inputId = btn.getAttribute('data-for');
                const input = document.getElementById(inputId);
                if (input) {
                    if (input.type === 'password') {
                        input.type = 'text';
                        btn.textContent = '🔒';
                        btn.setAttribute('aria-label', 'Hide password');
                    } else {
                        input.type = 'password';
                        btn.textContent = '👁️';
                        btn.setAttribute('aria-label', 'Show password');
                    }
                }
            });
        });
        
        // Add input focus/blur effects
        const inputs = sidebar.querySelectorAll('.api-input');
        inputs.forEach(input => {
            const section = input.closest('.api-config-section');
            
            input.addEventListener('focus', () => {
                section.classList.add('focused');
            });
            
            input.addEventListener('blur', () => {
                section.classList.remove('focused');
                if (input.value.trim()) {
                    input.classList.add('filled');
                } else {
                    input.classList.remove('filled');
                }
            });
            
            input.addEventListener('input', () => {
                if (input.value.trim()) {
                    input.classList.add('filled');
                } else {
                    input.classList.remove('filled');
                }
            });
        });
        
        // Add open button to header with enhanced styling
        const headerSection = document.querySelector('.app-title');
        if (headerSection) {
            const configBtn = document.createElement('button');
            configBtn.id = 'openApiConfigBtn';
            configBtn.className = 'api-config-open-btn';
            configBtn.innerHTML = '⚙️ API Settings';
            configBtn.title = 'Configure API Keys';
            
            // Insert after the title
            headerSection.parentNode.insertBefore(configBtn, headerSection.nextSibling);
        }
        
        // Dispatch event when sidebar is created
        document.dispatchEvent(new CustomEvent('sidebar-created'));
    },

    // Open the sidebar
    openSidebar() {
        const sidebar = document.getElementById('apiConfigSidebar');
        if (sidebar) {
            sidebar.classList.add('active');
            sidebar.dispatchEvent(new CustomEvent('sidebar-opened'));
        }
    },

    // Close the sidebar
    closeSidebar() {
        const sidebar = document.getElementById('apiConfigSidebar');
        if (sidebar) {
            sidebar.classList.remove('active');
        }
    },

    // Load saved API keys
    loadSavedKeys() {
        const keys = ApiConfig.STORAGE_KEYS;
        document.getElementById('assemblyaiKey').value = localStorage.getItem(keys.ASSEMBLYAI) || '';
        document.getElementById('geminiKey').value = localStorage.getItem(keys.GEMINI) || '';
        document.getElementById('weatherKey').value = localStorage.getItem(keys.WEATHER) || '';
        document.getElementById('tavilyKey').value = localStorage.getItem(keys.TAVILY) || '';
        document.getElementById('murfKey').value = localStorage.getItem(keys.MURF) || '';
        
        // Update UI for filled inputs
        const inputs = document.querySelectorAll('.api-input');
        inputs.forEach(input => {
            if (input.value.trim()) {
                input.classList.add('filled');
            }
        });
    },

    // Save API keys to localStorage and update server
    saveApiKeys() {
        const keys = ApiConfig.STORAGE_KEYS;
        const statusEl = document.getElementById('apiConfigStatus');
        const saveBtn = document.getElementById('saveApiConfigBtn');
        const keysData = {
            assemblyai: document.getElementById('assemblyaiKey').value,
            gemini: document.getElementById('geminiKey').value,
            weather: document.getElementById('weatherKey').value,
            tavily: document.getElementById('tavilyKey').value,
            murf: document.getElementById('murfKey').value
        };
        
        // Add loading state
        saveBtn.classList.add('loading');
        saveBtn.textContent = 'Saving...';
        
        // Save to localStorage
        localStorage.setItem(keys.ASSEMBLYAI, keysData.assemblyai);
        localStorage.setItem(keys.GEMINI, keysData.gemini);
        localStorage.setItem(keys.WEATHER, keysData.weather);
        localStorage.setItem(keys.TAVILY, keysData.tavily);
        localStorage.setItem(keys.MURF, keysData.murf);
        
        // Send to server via WebSocket if connected
        if (window.ws && window.ws.readyState === WebSocket.OPEN) {
            window.ws.send(JSON.stringify({
                type: 'API_CONFIG_UPDATE',
                keys: keysData
            }));
            
            // Show saving message
            statusEl.textContent = '🔄 Updating API keys...';
            statusEl.className = 'api-config-status api-status-info';
            
            // Remove loading state after a delay
            setTimeout(() => {
                saveBtn.classList.remove('loading');
                saveBtn.textContent = '💾 Save Configuration';
            }, 1000);
        } else {
            // Show error if not connected
            statusEl.textContent = '❌ Error: Not connected to server';
            statusEl.className = 'api-config-status api-status-error';
            
            saveBtn.classList.remove('loading');
            saveBtn.textContent = '💾 Save Configuration';
        }
    },

    // Handle config update response from server
    handleConfigUpdateResponse(success, message) {
        const statusEl = document.getElementById('apiConfigStatus');
        if (success) {
            statusEl.textContent = `✅ ${message || 'API keys updated successfully!'}`;
            statusEl.className = 'api-config-status api-status-success';
            
            // Add a celebration effect
            const sidebar = document.getElementById('apiConfigSidebar');
            sidebar.style.transform = 'scale(1.02)';
            setTimeout(() => {
                sidebar.style.transform = '';
            }, 200);
        } else {
            statusEl.textContent = `❌ ${message || 'Error updating API keys'}`;
            statusEl.className = 'api-config-status api-status-error';
        }
        
        // Clear status message after 5 seconds
        setTimeout(() => {
            statusEl.textContent = '';
            statusEl.className = 'api-config-status';
        }, 5000);
    }
};

// Initialize API config when document is ready
document.addEventListener('DOMContentLoaded', () => {
    ApiConfig.init();
});
