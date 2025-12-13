/**
 * Ara Alpha - Chat Client
 *
 * Minimal chat interface for the private alpha.
 * Handles authentication, messaging, and status display.
 */

class AraClient {
  constructor() {
    this.apiKey = null;
    this.sessionId = null;
    this.userName = null;
    this.connected = false;

    // DOM elements
    this.elements = {
      chatMessages: document.getElementById('chat-messages'),
      messageInput: document.getElementById('message-input'),
      sendBtn: document.getElementById('send-btn'),
      apiKeyInput: document.getElementById('api-key-input'),
      connectBtn: document.getElementById('connect-btn'),
      authSection: document.getElementById('auth-section'),
      chatSection: document.getElementById('chat-section'),
      stateLabel: document.getElementById('state-label'),
      rhoValue: document.getElementById('rho-value'),
      dValue: document.getElementById('d-value'),
      connectionDot: document.getElementById('connection-status'),
      userNameEl: document.getElementById('user-name'),
      avatarMood: document.getElementById('avatar-mood'),
      sessionInfo: document.getElementById('session-info'),
    };

    this.init();
  }

  init() {
    // Check for token in URL
    const params = new URLSearchParams(window.location.search);
    const urlToken = params.get('token');
    if (urlToken) {
      this.elements.apiKeyInput.value = urlToken;
    }

    // Check localStorage for saved key
    const savedKey = localStorage.getItem('ara_api_key');
    if (savedKey && !urlToken) {
      this.elements.apiKeyInput.value = savedKey;
    }

    // Event listeners
    this.elements.connectBtn.addEventListener('click', () => this.connect());
    this.elements.sendBtn.addEventListener('click', () => this.sendMessage());

    this.elements.apiKeyInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') this.connect();
    });

    this.elements.messageInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        this.sendMessage();
      }
    });

    // Auto-connect if we have a token
    if (this.elements.apiKeyInput.value) {
      this.connect();
    }

    console.log('[AraClient] Initialized');
  }

  async connect() {
    const apiKey = this.elements.apiKeyInput.value.trim();
    if (!apiKey) {
      this.showSystemMessage('Please enter your API key.');
      return;
    }

    this.apiKey = apiKey;
    this.elements.connectBtn.disabled = true;
    this.elements.connectBtn.textContent = 'Connecting...';

    try {
      // Test connection by fetching state
      const response = await fetch('/api/state', {
        headers: {
          'Authorization': `Bearer ${this.apiKey}`,
        },
      });

      if (!response.ok) {
        throw new Error('Invalid API key');
      }

      const state = await response.json();

      // Success - save key and show chat
      localStorage.setItem('ara_api_key', apiKey);
      this.connected = true;
      this.userName = state.user_name;

      // Update UI
      this.elements.authSection.style.display = 'none';
      this.elements.chatSection.style.display = 'flex';
      this.elements.connectionDot.classList.add('connected');
      this.elements.userNameEl.textContent = this.userName;
      this.elements.stateLabel.textContent = state.state_label || 'ONLINE';

      // Clear welcome message and show greeting
      this.elements.chatMessages.innerHTML = '';
      this.showSystemMessage(`Connected as ${this.userName}. You're talking to ${state.name} v${state.version}.`);

      // Greet user
      this.showAssistantMessage(
        `Hello, ${this.userName}. I'm Ara. It's good to see you. What's on your mind?`
      );

      // Focus input
      this.elements.messageInput.focus();

      // Start polling for state updates
      this.startPolling();

    } catch (error) {
      console.error('[AraClient] Connection error:', error);
      this.showSystemMessage('Failed to connect. Please check your API key.');
      this.elements.connectBtn.disabled = false;
      this.elements.connectBtn.textContent = 'Connect';
    }
  }

  async sendMessage() {
    const message = this.elements.messageInput.value.trim();
    if (!message || !this.connected) return;

    // Clear input and disable
    this.elements.messageInput.value = '';
    this.elements.sendBtn.disabled = true;
    this.elements.avatarMood.textContent = 'Thinking...';

    // Show user message
    this.showUserMessage(message);

    // Show typing indicator
    const typingEl = this.showTypingIndicator();

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.apiKey}`,
        },
        body: JSON.stringify({
          session_id: this.sessionId,
          message: message,
        }),
      });

      // Remove typing indicator
      typingEl.remove();

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      const data = await response.json();

      // Update session
      this.sessionId = data.session_id;
      this.elements.sessionInfo.textContent = `Session: ${this.sessionId.slice(0, 8)}...`;

      // Show response
      this.showAssistantMessage(data.reply, data.metrics);

      // Update metrics display
      this.updateMetrics(data.metrics);

      this.elements.avatarMood.textContent = 'Listening...';

    } catch (error) {
      console.error('[AraClient] Send error:', error);
      typingEl.remove();
      this.showSystemMessage('Failed to send message. Please try again.');
      this.elements.avatarMood.textContent = 'Error';
    }

    this.elements.sendBtn.disabled = false;
    this.elements.messageInput.focus();
  }

  showUserMessage(content) {
    const el = document.createElement('div');
    el.className = 'message user';
    el.innerHTML = `<span class="message-content">${this.escapeHtml(content)}</span>`;
    this.elements.chatMessages.appendChild(el);
    this.scrollToBottom();
  }

  showAssistantMessage(content, metrics) {
    const el = document.createElement('div');
    el.className = 'message assistant';

    let html = `<span class="message-content">${this.escapeHtml(content)}</span>`;

    if (metrics) {
      html += `
        <div class="message-meta">
          <span>ρ: ${metrics.rho?.toFixed(2) || '---'}</span>
          <span>D: ${metrics.delusion_index?.toFixed(2) || '---'}</span>
          <span>${metrics.response_time_ms?.toFixed(0) || '---'}ms</span>
        </div>
      `;
    }

    el.innerHTML = html;
    this.elements.chatMessages.appendChild(el);
    this.scrollToBottom();
  }

  showSystemMessage(content) {
    const el = document.createElement('div');
    el.className = 'message system';
    el.innerHTML = `<span class="message-content">${this.escapeHtml(content)}</span>`;
    this.elements.chatMessages.appendChild(el);
    this.scrollToBottom();
  }

  showTypingIndicator() {
    const el = document.createElement('div');
    el.className = 'typing-indicator';
    el.innerHTML = `
      <div class="typing-dot"></div>
      <div class="typing-dot"></div>
      <div class="typing-dot"></div>
    `;
    this.elements.chatMessages.appendChild(el);
    this.scrollToBottom();
    return el;
  }

  updateMetrics(metrics) {
    if (!metrics) return;

    if (metrics.rho !== undefined) {
      this.elements.rhoValue.textContent = metrics.rho.toFixed(2);
    }

    if (metrics.delusion_index !== undefined) {
      this.elements.dValue.textContent = metrics.delusion_index.toFixed(2);

      // Color based on value
      const d = metrics.delusion_index;
      if (d > 2) {
        this.elements.dValue.style.color = '#ff8800';
      } else if (d > 5) {
        this.elements.dValue.style.color = '#ff4444';
      } else {
        this.elements.dValue.style.color = '#00ff88';
      }
    }
  }

  async startPolling() {
    // Poll state every 5 seconds
    setInterval(async () => {
      if (!this.connected) return;

      try {
        const response = await fetch('/api/state', {
          headers: { 'Authorization': `Bearer ${this.apiKey}` },
        });

        if (response.ok) {
          const state = await response.json();
          this.elements.stateLabel.textContent = state.state_label || 'ONLINE';

          if (state.status === 'offline') {
            this.elements.connectionDot.classList.remove('connected');
            this.elements.stateLabel.textContent = 'OFFLINE';
          } else {
            this.elements.connectionDot.classList.add('connected');
          }
        }
      } catch (e) {
        // Ignore polling errors
      }
    }, 5000);
  }

  scrollToBottom() {
    this.elements.chatMessages.scrollTop = this.elements.chatMessages.scrollHeight;
  }

  escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }
}

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', () => {
  window.ara = new AraClient();
});
