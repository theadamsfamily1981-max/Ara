/**
 * ARA Cognitive Cockpit - Main Controller
 *
 * Renders the sci-fi HUD with real-time telemetry visualization:
 * - Triple-ring Cognitive Core gauge (ρ, D, Π)
 * - Body Heat hologram
 * - Avalanche scope oscilloscope
 * - Mental modes switcher
 * - Sanity timeline
 * - Status ticker
 */

class CognitiveCockpit {
  constructor(options = {}) {
    this.dataPath = options.dataPath || '/cognitive_state.json';
    this.pollInterval = options.pollInterval || 1000;
    this.onModeChange = options.onModeChange || null;

    this.state = null;
    this.connected = false;
    this.lastUpdate = 0;

    // Canvas contexts
    this.coreCtx = null;
    this.scopeCtx = null;
    this.timelineCtx = null;

    // Animation
    this.animationFrame = null;
    this.ringAngles = { rho: 0, delusion: 0, precision: 0 };

    this.init();
  }

  init() {
    // Get canvas contexts
    const coreCanvas = document.getElementById('core-canvas');
    const scopeCanvas = document.getElementById('scope-canvas');
    const timelineCanvas = document.getElementById('timeline-canvas');

    if (coreCanvas) this.coreCtx = coreCanvas.getContext('2d');
    if (scopeCanvas) this.scopeCtx = scopeCanvas.getContext('2d');
    if (timelineCanvas) this.timelineCtx = timelineCanvas.getContext('2d');

    // Set up mode buttons
    this.setupModeButtons();

    // Start polling
    this.startPolling();

    // Start animation loop
    this.animate();

    console.log('[CognitiveCockpit] Initialized');
  }

  setupModeButtons() {
    const buttons = document.querySelectorAll('.mode-btn');
    buttons.forEach(btn => {
      btn.addEventListener('click', () => {
        const mode = btn.dataset.mode;
        this.setMentalMode(mode);
      });
    });
  }

  async startPolling() {
    const poll = async () => {
      try {
        const response = await fetch(this.dataPath + '?t=' + Date.now());
        if (response.ok) {
          this.state = await response.json();
          this.connected = true;
          this.lastUpdate = Date.now();
          this.updateUI();
        } else {
          this.connected = false;
        }
      } catch (e) {
        this.connected = false;
        console.warn('[CognitiveCockpit] Fetch error:', e.message);
      }

      this.updateConnectionStatus();
      setTimeout(poll, this.pollInterval);
    };

    poll();
  }

  updateConnectionStatus() {
    const dot = document.querySelector('.connection-dot');
    const text = document.querySelector('.connection-text');

    if (dot) {
      dot.classList.toggle('connected', this.connected);
    }
    if (text) {
      text.textContent = this.connected ? 'LINKED' : 'OFFLINE';
    }
  }

  updateUI() {
    if (!this.state) return;

    this.updateStateLabel();
    this.updateThermalPanel();
    this.updateMentalModePanel();
    this.updateTicker();
  }

  updateStateLabel() {
    const label = document.getElementById('state-label');
    const sublabel = document.getElementById('state-sublabel');

    if (label && this.state.state_label) {
      label.textContent = 'STATE: ' + this.state.state_label;

      // Color based on criticality state
      const critState = this.state.criticality?.state || 'edge';
      label.className = 'state-label ' + critState;
    }

    if (sublabel && this.state.criticality) {
      const rho = this.state.criticality.rho.toFixed(3);
      sublabel.textContent = `ρ=${rho}`;
    }
  }

  updateThermalPanel() {
    const thermal = this.state.thermal;
    if (!thermal) return;

    // Update thermal zones on SVG
    thermal.zones.forEach(zone => {
      const element = document.getElementById(`thermal-zone-${zone.zone_id}`);
      if (element) {
        element.classList.remove('cool', 'nominal', 'warming', 'critical');
        element.classList.add(zone.status);
      }
    });

    // Update badges
    const statusBadge = document.getElementById('thermal-status-value');
    if (statusBadge) {
      statusBadge.textContent = thermal.overall_status.toUpperCase();
      statusBadge.className = 'badge-value ' + thermal.overall_status;
    }

    const reflexBadge = document.getElementById('reflex-state-value');
    if (reflexBadge) {
      reflexBadge.textContent = thermal.reflex_state.toUpperCase();
    }

    const fanBadge = document.getElementById('fan-mode-value');
    if (fanBadge) {
      fanBadge.textContent = thermal.fan_mode.toUpperCase();
    }

    // Hottest temp
    const tempValue = document.getElementById('hottest-temp');
    if (tempValue && thermal.hottest) {
      tempValue.textContent = thermal.hottest.temperature_c.toFixed(1) + '°C';
    }
  }

  updateMentalModePanel() {
    const mode = this.state.mental_mode;
    if (!mode) return;

    // Update active button
    const buttons = document.querySelectorAll('.mode-btn');
    buttons.forEach(btn => {
      btn.classList.toggle('active', btn.dataset.mode === mode.mode);
    });

    // Update drive meters
    this.updateDriveMeter('goal', mode.extrinsic_weight);
    this.updateDriveMeter('curiosity', mode.intrinsic_weight);
    this.updateDriveMeter('energy', mode.energy_budget);
  }

  updateDriveMeter(name, value) {
    const fill = document.getElementById(`drive-${name}`);
    if (fill) {
      fill.style.width = (value * 100) + '%';
    }
  }

  updateTicker() {
    const ticker = this.state.ticker;
    if (!ticker) return;

    const icon = document.getElementById('ticker-icon');
    const message = document.getElementById('ticker-message');
    const timestamp = document.getElementById('ticker-timestamp');

    if (icon) {
      icon.className = 'ticker-icon ' + ticker.severity;
      icon.textContent = ticker.severity === 'critical' ? '⚠' :
                         ticker.severity === 'warning' ? '⚡' : '◈';
    }

    if (message) {
      message.textContent = ticker.message;
    }

    if (timestamp && this.state.timestamp) {
      const date = new Date(this.state.timestamp * 1000);
      timestamp.textContent = date.toLocaleTimeString();
    }
  }

  async setMentalMode(mode) {
    console.log('[CognitiveCockpit] Setting mode:', mode);

    if (this.onModeChange) {
      this.onModeChange(mode);
    }

    // For now, just update locally (in real impl, this would POST to daemon)
    if (this.state && this.state.mental_mode) {
      this.state.mental_mode.mode = mode;

      const presets = {
        worker: { extrinsic_weight: 0.8, intrinsic_weight: 0.2, energy_budget: 0.6 },
        scientist: { extrinsic_weight: 0.4, intrinsic_weight: 0.7, energy_budget: 0.5 },
        chill: { extrinsic_weight: 0.2, intrinsic_weight: 0.2, energy_budget: 0.3 },
      };

      if (presets[mode]) {
        Object.assign(this.state.mental_mode, presets[mode]);
      }

      this.updateMentalModePanel();
    }
  }

  // =========================================================================
  // Canvas Rendering
  // =========================================================================

  animate() {
    this.renderCoreGauge();
    this.renderAvalancheScope();
    this.renderSanityTimeline();

    this.animationFrame = requestAnimationFrame(() => this.animate());
  }

  renderCoreGauge() {
    const ctx = this.coreCtx;
    if (!ctx) return;

    const canvas = ctx.canvas;
    const dpr = window.devicePixelRatio || 1;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;

    // Handle DPR
    if (canvas.width !== width * dpr || canvas.height !== height * dpr) {
      canvas.width = width * dpr;
      canvas.height = height * dpr;
      ctx.scale(dpr, dpr);
    }

    ctx.clearRect(0, 0, width, height);

    const cx = width / 2;
    const cy = height / 2;
    const maxRadius = Math.min(width, height) / 2 - 10;

    // Ring parameters
    const rings = [
      { radius: maxRadius * 0.5, width: 12, color: '#00ffff', value: this.state?.criticality?.rho || 0.85, max: 1.5, key: 'rho' },
      { radius: maxRadius * 0.7, width: 10, color: '#ff00ff', value: this.normalizeDelusion(this.state?.delusion?.D), max: 1, key: 'delusion' },
      { radius: maxRadius * 0.88, width: 8, color: '#ff8800', value: this.normalizePrecision(this.state?.precision?.ratio), max: 1, key: 'precision' },
    ];

    rings.forEach(ring => {
      // Animate towards target
      const target = (ring.value / ring.max) * Math.PI * 1.8;
      this.ringAngles[ring.key] += (target - this.ringAngles[ring.key]) * 0.08;

      this.drawArcRing(ctx, cx, cy, ring.radius, ring.width, ring.color, this.ringAngles[ring.key]);
    });

    // Center decoration
    this.drawCenterHex(ctx, cx, cy, 25);
  }

  normalizeDelusion(D) {
    if (!D) return 0.5;
    // Map D to 0-1 where 1 = balanced
    const logD = Math.log10(Math.max(D, 0.001));
    // -1 to 1 -> 0 to 1
    return Math.max(0, Math.min(1, 0.5 + logD * 0.25));
  }

  normalizePrecision(ratio) {
    if (!ratio) return 0.5;
    // Similar normalization
    return Math.max(0, Math.min(1, ratio / 2));
  }

  drawArcRing(ctx, cx, cy, radius, lineWidth, color, angle) {
    const startAngle = -Math.PI * 0.9;

    // Background track
    ctx.beginPath();
    ctx.arc(cx, cy, radius, startAngle, startAngle + Math.PI * 1.8);
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
    ctx.lineWidth = lineWidth;
    ctx.lineCap = 'round';
    ctx.stroke();

    // Active arc
    ctx.beginPath();
    ctx.arc(cx, cy, radius, startAngle, startAngle + angle);
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.lineCap = 'round';

    // Glow effect
    ctx.shadowColor = color;
    ctx.shadowBlur = 15;
    ctx.stroke();
    ctx.shadowBlur = 0;

    // End dot
    const endX = cx + Math.cos(startAngle + angle) * radius;
    const endY = cy + Math.sin(startAngle + angle) * radius;

    ctx.beginPath();
    ctx.arc(endX, endY, lineWidth / 2 + 2, 0, Math.PI * 2);
    ctx.fillStyle = '#ffffff';
    ctx.shadowColor = color;
    ctx.shadowBlur = 10;
    ctx.fill();
    ctx.shadowBlur = 0;
  }

  drawCenterHex(ctx, cx, cy, size) {
    ctx.beginPath();
    for (let i = 0; i < 6; i++) {
      const angle = (Math.PI / 3) * i - Math.PI / 2;
      const x = cx + size * Math.cos(angle);
      const y = cy + size * Math.sin(angle);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.strokeStyle = 'rgba(0, 255, 255, 0.5)';
    ctx.lineWidth = 1;
    ctx.stroke();
  }

  renderAvalancheScope() {
    const ctx = this.scopeCtx;
    if (!ctx) return;

    const canvas = ctx.canvas;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;

    // Handle DPR
    const dpr = window.devicePixelRatio || 1;
    if (canvas.width !== width * dpr || canvas.height !== height * dpr) {
      canvas.width = width * dpr;
      canvas.height = height * dpr;
      ctx.scale(dpr, dpr);
    }

    ctx.clearRect(0, 0, width, height);

    const avalanches = this.state?.avalanches;
    if (!avalanches || !avalanches.scatter) return;

    const scatter = avalanches.scatter;
    if (scatter.length === 0) return;

    // Log-log plot bounds
    const minLogS = 0, maxLogS = 4;  // log10(size)
    const minLogF = -4, maxLogF = 0; // log10(freq)

    const padX = 30, padY = 20;
    const plotW = width - padX * 2;
    const plotH = height - padY * 2;

    // Draw scatter points
    ctx.fillStyle = '#00ffff';
    scatter.forEach(pt => {
      const x = padX + ((pt.log_size - minLogS) / (maxLogS - minLogS)) * plotW;
      const y = padY + ((maxLogF - pt.log_freq) / (maxLogF - minLogF)) * plotH;

      ctx.beginPath();
      ctx.arc(x, y, 3, 0, Math.PI * 2);
      ctx.fill();
    });

    // Draw fitted power law line
    const tau = avalanches.fitted_tau || 1.5;
    ctx.beginPath();
    ctx.strokeStyle = '#ff00ff';
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);

    for (let logS = minLogS; logS <= maxLogS; logS += 0.1) {
      // P(S) ~ S^(-tau), so log P = -tau * log S + const
      const logF = -tau * logS + 2;  // arbitrary constant for visualization
      const x = padX + ((logS - minLogS) / (maxLogS - minLogS)) * plotW;
      const y = padY + ((maxLogF - Math.max(minLogF, Math.min(maxLogF, logF))) / (maxLogF - minLogF)) * plotH;

      if (logS === minLogS) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.setLineDash([]);

    // Tau label
    ctx.fillStyle = '#ff00ff';
    ctx.font = '10px monospace';
    ctx.fillText(`τ = ${tau.toFixed(2)}`, width - 60, 20);

    // Update cascade status
    const statusEl = document.getElementById('cascade-status');
    if (statusEl) {
      statusEl.textContent = 'CASCADES: ' + (avalanches.cascade_state || 'STABLE').toUpperCase();
      statusEl.className = 'cascade-status ' + (avalanches.cascade_state || 'stable');
    }

    // Update stats
    const tauEl = document.getElementById('tau-value');
    if (tauEl) tauEl.textContent = tau.toFixed(2);

    const r2El = document.getElementById('r2-value');
    if (r2El) r2El.textContent = (avalanches.fit_r_squared || 0).toFixed(2);
  }

  renderSanityTimeline() {
    const ctx = this.timelineCtx;
    if (!ctx) return;

    const canvas = ctx.canvas;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;

    // Handle DPR
    const dpr = window.devicePixelRatio || 1;
    if (canvas.width !== width * dpr || canvas.height !== height * dpr) {
      canvas.width = width * dpr;
      canvas.height = height * dpr;
      ctx.scale(dpr, dpr);
    }

    ctx.clearRect(0, 0, width, height);

    const timeline = this.state?.sanity_timeline;
    if (!timeline || timeline.length === 0) return;

    // Plot bounds: log10(D) from -2 to 2
    const minLogD = -2, maxLogD = 2;
    const padX = 10, padY = 10;
    const plotW = width - padX * 2;
    const plotH = height - padY * 2;

    // Draw center line (balanced)
    ctx.beginPath();
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    const centerY = padY + plotH / 2;
    ctx.moveTo(padX, centerY);
    ctx.lineTo(width - padX, centerY);
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw timeline
    ctx.beginPath();
    ctx.strokeStyle = '#ff00ff';
    ctx.lineWidth = 2;

    timeline.forEach((pt, i) => {
      const x = padX + (i / (timeline.length - 1 || 1)) * plotW;
      const normalizedLogD = (pt.log_D - minLogD) / (maxLogD - minLogD);
      const y = padY + (1 - normalizedLogD) * plotH;

      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });

    ctx.shadowColor = '#ff00ff';
    ctx.shadowBlur = 5;
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Draw event markers
    timeline.forEach((pt, i) => {
      const x = padX + (i / (timeline.length - 1 || 1)) * plotW;
      const normalizedLogD = (pt.log_D - minLogD) / (maxLogD - minLogD);
      const y = padY + (1 - normalizedLogD) * plotH;

      if (pt.guardrail) {
        ctx.fillStyle = '#ffff00';
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.fill();
      }

      if (pt.hallucination) {
        ctx.fillStyle = '#ff0044';
        ctx.beginPath();
        ctx.arc(x, y, 5, 0, Math.PI * 2);
        ctx.fill();
      }
    });

    // Update indicators
    const guardrailIndicator = document.getElementById('guardrail-indicator');
    const halluIndicator = document.getElementById('hallucination-indicator');

    if (guardrailIndicator) {
      const recent = timeline.slice(-5).some(p => p.guardrail);
      guardrailIndicator.classList.toggle('active', recent);
    }

    if (halluIndicator) {
      const recent = timeline.slice(-5).some(p => p.hallucination);
      halluIndicator.classList.toggle('active', recent);
    }
  }

  destroy() {
    if (this.animationFrame) {
      cancelAnimationFrame(this.animationFrame);
    }
  }
}

// =========================================================================
// Initialize on DOM Ready
// =========================================================================

document.addEventListener('DOMContentLoaded', () => {
  window.cockpit = new CognitiveCockpit({
    dataPath: '/cognitive_state.json',
    pollInterval: 1000,
    onModeChange: (mode) => {
      console.log('Mode changed to:', mode);
      // In production: POST to daemon socket or D-Bus
    }
  });
});
