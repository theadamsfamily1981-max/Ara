/**
 * T-FAN Topology Screensaver - WebGL Version
 *
 * Four visualization modes:
 * 1. Barcode Nebula - 3D persistence barcodes
 * 2. Landscape Waterfall - Flowing persistence landscapes
 * 3. Poincaré Orbits - Hyperbolic embeddings
 * 4. Pareto Galaxy - Multi-objective stars
 */

class TFANScreensaver {
    constructor() {
        this.canvas = document.getElementById('screensaver');
        this.modes = ['barcode', 'landscape', 'poincare', 'pareto'];
        this.currentMode = 1; // Start with landscape
        this.paused = false;
        this.time = 0;

        // Metrics from WebSocket/API
        this.metrics = {
            epr_cv: 0.10,
            accuracy: 0.0,
            latency_ms: 0.0,
            hypervolume: 0.0,
            topo_gap: 0.015,
            topo_cos: 0.93
        };

        this.initThree();
        this.initControls();
        this.initWebSocket();
        this.createVisualization();
        this.animate();
    }

    initThree() {
        // Scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x020306);
        this.scene.fog = new THREE.Fog(0x020306, 10, 50);

        // Camera
        const aspect = window.innerWidth / window.innerHeight;
        this.camera = new THREE.PerspectiveCamera(60, aspect, 0.1, 100);
        this.camera.position.set(0, 5, 10);
        this.camera.lookAt(0, 0, 0);

        // Renderer
        this.renderer = new THREE.WebGLRenderer({
            canvas: this.canvas,
            antialias: true,
            alpha: false
        });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

        // Orbit controls
        this.controls = new THREE.OrbitControls(this.camera, this.canvas);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.autoRotate = true;
        this.controls.autoRotateSpeed = 0.5;

        // Lights
        const ambientLight = new THREE.AmbientLight(0x404040, 1.0);
        this.scene.add(ambientLight);

        const pointLight1 = new THREE.PointLight(0x667eea, 2.0);
        pointLight1.position.set(10, 10, 10);
        this.scene.add(pointLight1);

        const pointLight2 = new THREE.PointLight(0x764ba2, 1.5);
        pointLight2.position.set(-10, 5, -10);
        this.scene.add(pointLight2);

        // Handle window resize
        window.addEventListener('resize', () => this.onResize());
    }

    initControls() {
        // Keyboard controls
        document.addEventListener('keydown', (e) => {
            const key = e.key.toLowerCase();

            switch(key) {
                case 'm':
                case 'tab':
                    e.preventDefault();
                    this.cycleMode();
                    break;
                case 'p':
                case ' ':
                    e.preventDefault();
                    this.togglePause();
                    break;
                case 'h':
                    e.preventDefault();
                    this.toggleHelp();
                    break;
                case 'f':
                    e.preventDefault();
                    this.toggleFullscreen();
                    break;
                case 'q':
                case 'escape':
                    if (document.fullscreenElement) {
                        document.exitFullscreen();
                    }
                    break;
            }
        });
    }

    initWebSocket() {
        // Try to connect to metrics WebSocket
        // Falls back to HTTP polling if WebSocket unavailable
        const wsUrl = this.getWebSocketURL();

        if (wsUrl) {
            this.connectWebSocket(wsUrl);
        } else {
            // Fallback to HTTP polling
            this.startHTTPPolling();
        }
    }

    getWebSocketURL() {
        // Try to determine WebSocket URL from page URL or default
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const host = window.location.hostname || 'localhost';
        const port = window.location.port || '8000';

        // Check if running from API server
        if (window.location.port === '8000') {
            return `${protocol}//${host}:${port}/ws/metrics`;
        }

        // Try default API port
        return `ws://localhost:8000/ws/metrics`;
    }

    connectWebSocket(url) {
        try {
            this.ws = new WebSocket(url);

            this.ws.onopen = () => {
                console.log('✓ WebSocket connected');
                this.updateWSStatus(true);
            };

            this.ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    if (data.type === 'metrics_update') {
                        this.updateMetrics(data.data);
                    }
                } catch (e) {
                    console.error('Failed to parse WebSocket message:', e);
                }
            };

            this.ws.onerror = (error) => {
                console.warn('WebSocket error, falling back to HTTP polling');
                this.updateWSStatus(false);
                this.startHTTPPolling();
            };

            this.ws.onclose = () => {
                this.updateWSStatus(false);
                // Try to reconnect after 5 seconds
                setTimeout(() => this.connectWebSocket(url), 5000);
            };
        } catch (e) {
            console.warn('WebSocket unavailable, using HTTP polling');
            this.startHTTPPolling();
        }
    }

    startHTTPPolling() {
        // Poll metrics from HTTP endpoint
        setInterval(async () => {
            try {
                const response = await fetch('http://localhost:8000/api/metrics');
                if (response.ok) {
                    const data = await response.json();
                    this.updateMetrics(data);
                    this.updateWSStatus(true);
                }
            } catch (e) {
                this.updateWSStatus(false);
            }
        }, 2000);
    }

    updateMetrics(data) {
        this.metrics = {
            epr_cv: data.epr_cv || 0.10,
            accuracy: data.accuracy || 0.0,
            latency_ms: data.latency_ms || 0.0,
            hypervolume: data.hypervolume || 0.0,
            topo_gap: data.topo_gap || 0.015,
            topo_cos: data.topo_cos || 0.93
        };

        // Update HUD
        document.getElementById('metric-epr').textContent = `EPR-CV: ${this.metrics.epr_cv.toFixed(3)}`;
        document.getElementById('metric-acc').textContent = `Accuracy: ${this.metrics.accuracy.toFixed(3)}`;
        document.getElementById('metric-lat').textContent = `Latency: ${this.metrics.latency_ms.toFixed(1)}ms`;
        document.getElementById('metric-hv').textContent = `HV: ${Math.round(this.metrics.hypervolume)}`;
    }

    updateWSStatus(connected) {
        const status = document.getElementById('ws-status');
        if (connected) {
            status.textContent = '🟢 Connected';
            status.className = 'connected';
        } else {
            status.textContent = '⚫ Disconnected';
            status.className = 'disconnected';
        }
    }

    cycleMode() {
        this.currentMode = (this.currentMode + 1) % this.modes.length;
        this.createVisualization();

        const modeName = this.modes[this.currentMode].toUpperCase();
        document.getElementById('mode-display').textContent = `Mode: ${modeName}`;
    }

    togglePause() {
        this.paused = !this.paused;
    }

    toggleHelp() {
        const help = document.getElementById('help-overlay');
        help.classList.toggle('hidden');
    }

    toggleFullscreen() {
        if (!document.fullscreenElement) {
            document.documentElement.requestFullscreen();
        } else {
            document.exitFullscreen();
        }
    }

    createVisualization() {
        // Clear existing objects
        while(this.scene.children.length > 0) {
            this.scene.remove(this.scene.children[0]);
        }

        // Re-add lights
        const ambientLight = new THREE.AmbientLight(0x404040, 1.0);
        this.scene.add(ambientLight);

        const pointLight1 = new THREE.PointLight(0x667eea, 2.0);
        pointLight1.position.set(10, 10, 10);
        this.scene.add(pointLight1);

        const pointLight2 = new THREE.PointLight(0x764ba2, 1.5);
        pointLight2.position.set(-10, 5, -10);
        this.scene.add(pointLight2);

        // Create mode-specific visualization
        const mode = this.modes[this.currentMode];

        switch(mode) {
            case 'barcode':
                this.createBarcodeNebula();
                break;
            case 'landscape':
                this.createLandscapeWaterfall();
                break;
            case 'poincare':
                this.createPoincareOrbits();
                break;
            case 'pareto':
                this.createParetoGalaxy();
                break;
        }
    }

    createBarcodeNebula() {
        // Generate persistence diagram (fake data for now)
        const n = 200;
        const bars = [];

        for (let i = 0; i < n; i++) {
            const birth = Math.random() * 3.0;
            const persistence = Math.random() * 2.0;
            const death = birth + persistence;
            const y = (i / n) * 10 - 5;

            bars.push({ birth, death, y, persistence });
        }

        // Sort by persistence (longest first)
        bars.sort((a, b) => b.persistence - a.persistence);

        // Create 3D barcode using instanced meshes
        const geometry = new THREE.BoxGeometry(0.05, 0.05, 1);
        const material = new THREE.MeshPhongMaterial({
            color: 0x667eea,
            emissive: 0x667eea,
            emissiveIntensity: 0.3
        });

        bars.slice(0, 150).forEach(bar => {
            const length = bar.death - bar.birth;
            const mesh = new THREE.Mesh(geometry.clone(), material.clone());

            mesh.scale.z = length;
            mesh.position.set(
                bar.birth + length / 2 - 1.5,
                bar.y,
                Math.sin(bar.y * 0.3) * 2
            );

            // Color by persistence
            const hue = 0.6 + (bar.persistence / 3.0) * 0.2;
            mesh.material.color.setHSL(hue, 0.8, 0.6);
            mesh.material.emissive.setHSL(hue, 0.8, 0.3);

            this.scene.add(mesh);
            mesh.userData.isBarcode = true;
            mesh.userData.initialY = bar.y;
        });
    }

    createLandscapeWaterfall() {
        // Create flowing persistence landscape layers
        const layers = 6;
        const resolution = 100;

        for (let layer = 0; layer < layers; layer++) {
            const vertices = [];
            const colors = [];

            for (let i = 0; i < resolution; i++) {
                const x = (i / resolution) * 10 - 5;
                const t = this.time * 0.1;

                // Landscape function (sum of triangular peaks)
                let y = 0;
                const peaks = 5;
                for (let p = 0; p < peaks; p++) {
                    const center = ((p + t * 0.1) % 10) - 5;
                    const width = 1.0 + Math.sin(t + p) * 0.3;
                    const height = (layers - layer) / layers * (1.0 + Math.sin(t * 0.5 + p) * 0.2);
                    y += Math.max(0, height - Math.abs(x - center) / width);
                }

                const z = layer * 1.5 - 4;
                vertices.push(x, y, z);

                // Color gradient
                const hue = 0.5 + layer / layers * 0.3;
                const color = new THREE.Color().setHSL(hue, 0.8, 0.5 + y * 0.1);
                colors.push(color.r, color.g, color.b);
            }

            const geometry = new THREE.BufferGeometry();
            geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
            geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

            const material = new THREE.LineBasicMaterial({
                vertexColors: true,
                linewidth: 2
            });

            const line = new THREE.Line(geometry, material);
            line.userData.isLandscape = true;
            line.userData.layer = layer;
            this.scene.add(line);
        }
    }

    createPoincareOrbits() {
        // Create Poincaré disk boundary
        const circleGeometry = new THREE.RingGeometry(4.9, 5.0, 128);
        const circleMaterial = new THREE.MeshBasicMaterial({
            color: 0x667eea,
            side: THREE.DoubleSide,
            transparent: true,
            opacity: 0.5
        });
        const circle = new THREE.Mesh(circleGeometry, circleMaterial);
        circle.rotation.x = Math.PI / 2;
        this.scene.add(circle);

        // Create hyperbolic points
        const n = 600;
        const geometry = new THREE.BufferGeometry();
        const positions = [];
        const colors = [];
        const sizes = [];

        for (let i = 0; i < n; i++) {
            // Hierarchical structure: radial ~ level
            const level = Math.floor(Math.random() * 6);
            const r = Math.pow(level / 6, 0.9) * 4.5;
            const theta = Math.random() * Math.PI * 2;

            const x = r * Math.cos(theta);
            const z = r * Math.sin(theta);
            const y = (Math.random() - 0.5) * 0.5;

            positions.push(x, y, z);

            // Color by hierarchy level
            const hue = 0.5 + level / 12;
            const color = new THREE.Color().setHSL(hue, 0.8, 0.6);
            colors.push(color.r, color.g, color.b);

            sizes.push(2.0 + level * 0.5);
        }

        geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
        geometry.setAttribute('size', new THREE.Float32BufferAttribute(sizes, 1));

        const material = new THREE.PointsMaterial({
            size: 0.1,
            vertexColors: true,
            transparent: true,
            opacity: 0.8,
            sizeAttenuation: true
        });

        const points = new THREE.Points(geometry, material);
        points.userData.isPoincare = true;
        this.scene.add(points);
    }

    createParetoGalaxy() {
        // Generate fake Pareto front (non-dominated configs)
        const m = 220;
        const configs = [];

        // Generate random 5D points
        for (let i = 0; i < m; i++) {
            configs.push({
                obj: [
                    Math.random(),
                    Math.random(),
                    Math.random(),
                    Math.random(),
                    Math.random()
                ]
            });
        }

        // Non-dominated filter (simple)
        const nonDominated = configs.filter(a => {
            return !configs.some(b => {
                if (a === b) return false;
                return b.obj.every((val, i) => val <= a.obj[i]) &&
                       b.obj.some((val, i) => val < a.obj[i]);
            });
        });

        // Project to 2D and create stars
        const geometry = new THREE.BufferGeometry();
        const positions = [];
        const colors = [];
        const sizes = [];

        nonDominated.forEach(config => {
            // Simple 2D projection
            const weights = [1.0, 0.2, 0.6, 0.5, 0.1];
            let x = 0, z = 0;
            config.obj.forEach((val, i) => {
                x += val * weights[i];
                z += val * weights[(i + 1) % 5];
            });

            x = (x - 0.5) * 10;
            z = (z - 0.5) * 10;
            const y = (Math.random() - 0.5) * 2;

            positions.push(x, y, z);

            // Color by first objective (proxy for accuracy)
            const hue = 0.5 + config.obj[0] * 0.3;
            const color = new THREE.Color().setHSL(hue, 0.9, 0.6);
            colors.push(color.r, color.g, color.b);

            sizes.push(3.0 + Math.random() * 2.0);
        });

        geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

        // Create star sprites
        const material = new THREE.PointsMaterial({
            size: 0.2,
            vertexColors: true,
            transparent: true,
            opacity: 0.9,
            sizeAttenuation: true,
            blending: THREE.AdditiveBlending
        });

        const points = new THREE.Points(geometry, material);
        points.userData.isPareto = true;
        this.scene.add(points);
    }

    animate() {
        requestAnimationFrame(() => this.animate());

        if (!this.paused) {
            this.time += 0.016;
            this.updateScene();
        }

        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }

    updateScene() {
        // Tension driven by EPR-CV
        const tension = 0.4 + this.metrics.epr_cv * 2.0;

        // Update mode-specific animations
        const mode = this.modes[this.currentMode];

        switch(mode) {
            case 'barcode':
                this.updateBarcodeNebula(tension);
                break;
            case 'landscape':
                this.updateLandscapeWaterfall(tension);
                break;
            case 'poincare':
                this.updatePoincareOrbits(tension);
                break;
            case 'pareto':
                this.updateParetoGalaxy(tension);
                break;
        }
    }

    updateBarcodeNebula(tension) {
        this.scene.children.forEach(child => {
            if (child.userData.isBarcode) {
                // Gentle floating motion
                child.position.y = child.userData.initialY +
                    Math.sin(this.time * tension * 0.5 + child.position.x) * 0.2;

                // Pulse emissive
                const pulse = 0.3 + Math.sin(this.time * 2 + child.position.y) * 0.2;
                child.material.emissiveIntensity = pulse;
            }
        });
    }

    updateLandscapeWaterfall(tension) {
        // Recreate landscape to show flow
        if (Math.floor(this.time * 30) % 2 === 0) {
            this.scene.children.filter(c => c.userData.isLandscape).forEach(child => {
                this.scene.remove(child);
            });
            this.createLandscapeWaterfall();
        }
    }

    updatePoincareOrbits(tension) {
        this.scene.children.forEach(child => {
            if (child.userData.isPoincare) {
                // Rotate points around disk
                child.rotation.y += 0.001 * tension;
            }
        });
    }

    updateParetoGalaxy(tension) {
        this.scene.children.forEach(child => {
            if (child.userData.isPareto) {
                // Gentle rotation
                child.rotation.y += 0.002 * tension;

                // Twinkle effect
                child.material.opacity = 0.7 + Math.sin(this.time * 3) * 0.2;
            }
        });
    }

    onResize() {
        const width = window.innerWidth;
        const height = window.innerHeight;

        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();

        this.renderer.setSize(width, height);
    }
}

// Start screensaver when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    const screensaver = new TFANScreensaver();
});
