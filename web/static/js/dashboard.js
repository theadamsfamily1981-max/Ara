// T-FAN Dashboard JavaScript

const API_BASE = window.location.origin;
const API_TOKEN = 'tfan-secure-token-change-me'; // TODO: Get from env/login

class TFANDashboard {
    constructor() {
        this.ws = null;
        this.connectWebSocket();
        this.setupNavigation();
        this.setupMetrics();
        this.setupPareto();
        this.setupTraining();
        this.setupConfigs();
        this.loadConfigs();
    }

    // ========================================================================
    // WebSocket Connection
    // ========================================================================

    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/metrics`;

        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
            console.log('✓ WebSocket connected');
            this.updateConnectionStatus(true);
            // Send ping every 30 seconds
            setInterval(() => {
                if (this.ws.readyState === WebSocket.OPEN) {
                    this.ws.send('ping');
                }
            }, 30000);
        };

        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleWebSocketMessage(data);
        };

        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.updateConnectionStatus(false);
        };

        this.ws.onclose = () => {
            console.log('WebSocket disconnected');
            this.updateConnectionStatus(false);
            // Reconnect after 5 seconds
            setTimeout(() => this.connectWebSocket(), 5000);
        };
    }

    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'metrics':
                this.updateMetrics(data.data);
                break;
            case 'weights_updated':
                this.showStatus('pareto-status', 'Weights updated successfully', 'success');
                break;
            case 'training_started':
                this.onTrainingStarted(data.data);
                break;
            case 'training_stopped':
                this.onTrainingStopped();
                break;
            case 'config_selected':
                this.showStatus('pareto-status', `Config selected: ${data.data.config}`, 'success');
                break;
        }
    }

    updateConnectionStatus(connected) {
        const dot = document.getElementById('connection-status');
        const text = document.getElementById('connection-text');

        if (connected) {
            dot.classList.remove('disconnected');
            dot.classList.add('connected');
            text.textContent = 'Connected';
        } else {
            dot.classList.remove('connected');
            dot.classList.add('disconnected');
            text.textContent = 'Disconnected';
        }
    }

    // ========================================================================
    // Navigation
    // ========================================================================

    setupNavigation() {
        const navItems = document.querySelectorAll('.nav-item');
        const sections = document.querySelectorAll('.section');

        navItems.forEach(item => {
            item.addEventListener('click', (e) => {
                e.preventDefault();
                const sectionId = item.dataset.section;

                // Update nav active state
                navItems.forEach(nav => nav.classList.remove('active'));
                item.classList.add('active');

                // Show/hide sections
                sections.forEach(section => {
                    if (section.id === `${sectionId}-section`) {
                        section.classList.add('active');
                    } else {
                        section.classList.remove('active');
                    }
                });
            });
        });
    }

    // ========================================================================
    // Metrics
    // ========================================================================

    setupMetrics() {
        // Initial load
        this.fetchMetrics();
    }

    async fetchMetrics() {
        try {
            const response = await fetch(`${API_BASE}/api/metrics`);
            const data = await response.json();
            this.updateMetrics(data);
        } catch (error) {
            console.error('Error fetching metrics:', error);
        }
    }

    updateMetrics(data) {
        // Update metric cards
        document.getElementById('metric-accuracy').textContent = data.accuracy.toFixed(3);
        document.getElementById('metric-latency').textContent = data.latency_ms.toFixed(1);
        document.getElementById('metric-hypervolume').textContent = Math.round(data.hypervolume);
        document.getElementById('metric-epr-cv').textContent = data.epr_cv.toFixed(3);

        // Update training status
        const badge = document.getElementById('training-badge');
        const step = document.getElementById('training-step');

        if (data.training_active) {
            badge.textContent = '● Training';
            badge.classList.add('active');
            step.textContent = `Step: ${data.step}`;
        } else {
            badge.textContent = '● Idle';
            badge.classList.remove('active');
            step.textContent = 'Step: 0';
        }
    }

    // ========================================================================
    // Pareto
    // ========================================================================

    setupPareto() {
        // Setup weight sliders
        const weights = ['accuracy', 'latency', 'epr', 'topo', 'energy'];

        weights.forEach(weight => {
            const slider = document.getElementById(`weight-${weight}`);
            const value = document.getElementById(`value-${weight}`);

            slider.addEventListener('input', (e) => {
                value.textContent = parseFloat(e.target.value).toFixed(1);
            });
        });

        // Apply weights button
        document.getElementById('apply-weights-btn').addEventListener('click', () => {
            this.applyParetoWeights();
        });

        // Run Pareto button
        document.getElementById('run-pareto-btn').addEventListener('click', () => {
            this.runParetoOptimization();
        });

        // Load initial Pareto front
        this.loadParetoFront();
    }

    async applyParetoWeights() {
        const weights = {
            neg_accuracy: parseFloat(document.getElementById('weight-accuracy').value),
            latency: parseFloat(document.getElementById('weight-latency').value),
            epr_cv: parseFloat(document.getElementById('weight-epr').value),
            topo_gap: parseFloat(document.getElementById('weight-topo').value),
            energy: parseFloat(document.getElementById('weight-energy').value)
        };

        try {
            const response = await fetch(`${API_BASE}/api/pareto/weights`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${API_TOKEN}`
                },
                body: JSON.stringify(weights)
            });

            if (response.ok) {
                this.showStatus('pareto-status', 'Weights applied successfully!', 'success');
            } else {
                throw new Error('Failed to apply weights');
            }
        } catch (error) {
            this.showStatus('pareto-status', 'Error applying weights', 'error');
            console.error(error);
        }
    }

    async loadParetoFront() {
        try {
            const response = await fetch(`${API_BASE}/api/pareto/front`);
            const data = await response.json();

            if (data.n_pareto_points > 0) {
                this.plotParetoFront(data);
            }
        } catch (error) {
            console.error('Error loading Pareto front:', error);
        }
    }

    plotParetoFront(data) {
        const configs = data.configurations;

        // Extract objectives
        const accuracy = configs.map(c => -c.objectives[0]); // Negate back
        const latency = configs.map(c => c.objectives[1]);
        const epr_cv = configs.map(c => c.objectives[2]);

        // Create Plotly trace
        const trace = {
            x: accuracy,
            y: latency,
            mode: 'markers',
            type: 'scatter',
            marker: {
                size: 12,
                color: epr_cv,
                colorscale: 'Viridis',
                showscale: true,
                colorbar: {
                    title: 'EPR CV'
                }
            },
            text: configs.map((c, i) =>
                `Config ${i}<br>` +
                `Accuracy: ${accuracy[i].toFixed(3)}<br>` +
                `Latency: ${latency[i].toFixed(1)}ms<br>` +
                `EPR CV: ${epr_cv[i].toFixed(3)}`
            ),
            hoverinfo: 'text'
        };

        const layout = {
            title: 'Pareto Front: Accuracy vs Latency',
            xaxis: { title: 'Accuracy' },
            yaxis: { title: 'Latency (ms)' },
            paper_bgcolor: '#1e293b',
            plot_bgcolor: '#1e293b',
            font: { color: '#e2e8f0' },
            hovermode: 'closest'
        };

        Plotly.newPlot('pareto-plot', [trace], layout, {responsive: true});
    }

    async runParetoOptimization() {
        const iterations = parseInt(document.getElementById('pareto-iterations').value);
        const initial = parseInt(document.getElementById('pareto-initial').value);

        this.showStatus('pareto-status', 'Starting Pareto optimization...', 'info');

        try {
            const response = await fetch(`${API_BASE}/api/pareto/run?n_iterations=${iterations}&n_initial=${initial}`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${API_TOKEN}`
                }
            });

            const data = await response.json();

            if (data.status === 'started') {
                this.showStatus('pareto-status',
                    `Optimization started (task ID: ${data.task_id}). This may take several minutes...`,
                    'info'
                );

                // Poll for completion
                this.pollParetoTask(data.task_id);
            }
        } catch (error) {
            this.showStatus('pareto-status', 'Error starting optimization', 'error');
            console.error(error);
        }
    }

    async pollParetoTask(taskId) {
        const checkStatus = async () => {
            try {
                const response = await fetch(`${API_BASE}/api/pareto/status/${taskId}`);
                const data = await response.json();

                if (data.status === 'completed') {
                    this.showStatus('pareto-status',
                        `✓ Optimization complete! Found ${data.n_points} Pareto points, HV: ${data.hypervolume.toFixed(0)}`,
                        'success'
                    );
                    this.loadParetoFront(); // Reload visualization
                } else if (data.status === 'failed') {
                    this.showStatus('pareto-status', `✗ Optimization failed: ${data.error}`, 'error');
                } else {
                    // Still running, check again in 5 seconds
                    setTimeout(checkStatus, 5000);
                }
            } catch (error) {
                console.error('Error checking task status:', error);
            }
        };

        checkStatus();
    }

    // ========================================================================
    // Training
    // ========================================================================

    setupTraining() {
        document.getElementById('start-training-btn').addEventListener('click', () => {
            this.startTraining();
        });

        document.getElementById('stop-training-btn').addEventListener('click', () => {
            this.stopTraining();
        });

        // Poll training status
        setInterval(() => this.updateTrainingStatus(), 5000);
    }

    async startTraining() {
        const config = document.getElementById('training-config').value;
        const maxSteps = parseInt(document.getElementById('training-steps').value);

        const request = {
            config_path: config,
            max_steps: maxSteps,
            logdir: 'runs/web_training'
        };

        try {
            const response = await fetch(`${API_BASE}/api/training/start`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${API_TOKEN}`
                },
                body: JSON.stringify(request)
            });

            if (response.ok) {
                const data = await response.json();
                this.addLog(`Training started (PID: ${data.pid})`);
                document.getElementById('start-training-btn').disabled = true;
                document.getElementById('stop-training-btn').disabled = false;
            } else {
                const error = await response.json();
                this.addLog(`Error: ${error.detail}`);
            }
        } catch (error) {
            this.addLog(`Error starting training: ${error}`);
            console.error(error);
        }
    }

    async stopTraining() {
        try {
            const response = await fetch(`${API_BASE}/api/training/stop`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${API_TOKEN}`
                }
            });

            if (response.ok) {
                this.addLog('Training stopped');
                document.getElementById('start-training-btn').disabled = false;
                document.getElementById('stop-training-btn').disabled = true;
            }
        } catch (error) {
            this.addLog(`Error stopping training: ${error}`);
            console.error(error);
        }
    }

    async updateTrainingStatus() {
        try {
            const response = await fetch(`${API_BASE}/api/training/status`);
            const data = await response.json();

            const startBtn = document.getElementById('start-training-btn');
            const stopBtn = document.getElementById('stop-training-btn');

            if (data.active) {
                startBtn.disabled = true;
                stopBtn.disabled = false;
            } else {
                startBtn.disabled = false;
                stopBtn.disabled = true;
            }
        } catch (error) {
            console.error('Error updating training status:', error);
        }
    }

    onTrainingStarted(data) {
        this.addLog(`[System] Training started with config: ${data.config}`);
    }

    onTrainingStopped() {
        this.addLog('[System] Training stopped');
    }

    addLog(message) {
        const logOutput = document.getElementById('training-log-output');
        const timestamp = new Date().toLocaleTimeString();
        const logEntry = document.createElement('div');
        logEntry.textContent = `[${timestamp}] ${message}`;
        logOutput.appendChild(logEntry);
        logOutput.scrollTop = logOutput.scrollHeight;
    }

    // ========================================================================
    // Configs
    // ========================================================================

    setupConfigs() {
        // Config management will be handled here
    }

    async loadConfigs() {
        try {
            const response = await fetch(`${API_BASE}/api/configs`);
            const data = await response.json();

            const container = document.getElementById('configs-list-container');
            container.innerHTML = '';

            data.configs.forEach(config => {
                const item = document.createElement('div');
                item.className = 'config-item';
                item.innerHTML = `
                    <div>
                        <strong>${config.name}</strong><br>
                        <small>${config.path}</small>
                    </div>
                    <button class="btn btn-primary" onclick="dashboard.selectConfig('${config.path}')">
                        Select
                    </button>
                `;
                container.appendChild(item);
            });
        } catch (error) {
            console.error('Error loading configs:', error);
        }
    }

    async selectConfig(configPath) {
        try {
            const response = await fetch(`${API_BASE}/api/configs/select?config_path=${encodeURIComponent(configPath)}`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${API_TOKEN}`
                }
            });

            if (response.ok) {
                alert(`Config selected: ${configPath}`);
            }
        } catch (error) {
            alert('Error selecting config');
            console.error(error);
        }
    }

    // ========================================================================
    // Utilities
    // ========================================================================

    showStatus(elementId, message, type) {
        const element = document.getElementById(elementId);
        element.textContent = message;
        element.className = `status-message ${type}`;
        element.style.display = 'block';

        // Auto-hide after 5 seconds
        setTimeout(() => {
            element.style.display = 'none';
        }, 5000);
    }
}

// Initialize dashboard when DOM is ready
let dashboard;
document.addEventListener('DOMContentLoaded', () => {
    dashboard = new TFANDashboard();
});
