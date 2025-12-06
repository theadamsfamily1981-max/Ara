"""
Spike Pipeline Orrery - Animated visualization of SNN data flow.

Shows spike propagation through neural network layers as an astronomical
orrery - layers as orbital rings, neurons as planets, spikes as moons
orbiting and jumping between layers.

The metaphor: Your neural network is a solar system. Spikes are moons
launched from planet to planet. Watch the dance of computation.
"""

from __future__ import annotations

import math
import json
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from .base import (
    GiftGenerator,
    GiftArtifact,
    ArtifactType,
    register_generator,
)


@dataclass
class NeuralLayer:
    """A layer in the SNN to visualize."""
    id: str
    name: str
    neuron_count: int
    layer_type: str = "hidden"       # input, hidden, output
    firing_rate: float = 0.5         # 0-1, average firing rate
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SpikeConnection:
    """Connection between layers showing spike flow."""
    source_layer: str
    target_layer: str
    weight: float = 0.5              # 0-1, connection strength
    spike_rate: float = 0.5          # 0-1, spikes per time unit
    delay: float = 1.0               # Time delay in ms


@register_generator
class OrreryGenerator(GiftGenerator):
    """
    Generates animated HTML orrery visualization of SNN spike flow.

    Neural layers become orbital rings:
    - Inner ring = input layer
    - Outer rings = successive hidden layers
    - Outermost = output layer

    Neurons are planets on each ring.
    Spikes are animated particles traveling between layers.
    """

    name = "orrery"
    description = "SNN spike flow as astronomical orrery"
    artifact_type = ArtifactType.HTML

    def __init__(self, theme: str = "ara", width: int = 800, height: int = 800):
        super().__init__(theme)
        self.width = width
        self.height = height

    def generate(self, data: Dict[str, Any]) -> GiftArtifact:
        """
        Generate orrery HTML from SNN data.

        Expected data format:
        {
            "layers": [
                {"id": "input", "name": "Input", "neuron_count": 128,
                 "layer_type": "input", "firing_rate": 0.3},
                {"id": "hidden1", "name": "Hidden 1", "neuron_count": 256,
                 "layer_type": "hidden", "firing_rate": 0.5},
                ...
            ],
            "connections": [
                {"source_layer": "input", "target_layer": "hidden1",
                 "weight": 0.8, "spike_rate": 0.6},
                ...
            ],
            "title": "SNN Spike Pipeline",  # optional
        }
        """
        layers = self._parse_layers(data.get("layers", []))
        connections = self._parse_connections(data.get("connections", []))
        title = data.get("title", "Spike Pipeline Orrery")

        html = self._render_html(layers, connections, title)

        return GiftArtifact(
            id=self._make_id("orrery"),
            name=title,
            artifact_type=ArtifactType.HTML,
            content=html,
            description=(
                f"Animated orrery of {len(layers)} neural layers. "
                f"Watch spikes flow through the network as orbiting particles."
            ),
            generator=self.name,
            source_data={
                "layer_count": len(layers),
                "connection_count": len(connections),
            },
            width=self.width,
            height=self.height,
            theme=self.theme,
        )

    def _parse_layers(self, raw: List[Dict]) -> List[NeuralLayer]:
        """Parse raw layer dicts into NeuralLayer objects."""
        layers = []
        for i, l in enumerate(raw):
            layers.append(NeuralLayer(
                id=l.get("id", f"layer_{i}"),
                name=l.get("name", f"Layer {i}"),
                neuron_count=l.get("neuron_count", 64),
                layer_type=l.get("layer_type", "hidden"),
                firing_rate=l.get("firing_rate", 0.5),
                properties=l.get("properties", {}),
            ))
        return layers

    def _parse_connections(self, raw: List[Dict]) -> List[SpikeConnection]:
        """Parse raw connection dicts into SpikeConnection objects."""
        connections = []
        for c in raw:
            connections.append(SpikeConnection(
                source_layer=c.get("source_layer", ""),
                target_layer=c.get("target_layer", ""),
                weight=c.get("weight", 0.5),
                spike_rate=c.get("spike_rate", 0.5),
                delay=c.get("delay", 1.0),
            ))
        return connections

    def _render_html(
        self,
        layers: List[NeuralLayer],
        connections: List[SpikeConnection],
        title: str,
    ) -> str:
        """Render the complete HTML with embedded SVG and animation."""
        colors = self.colors

        # Prepare data for JavaScript
        layer_data = [
            {
                "id": l.id,
                "name": l.name,
                "neuronCount": l.neuron_count,
                "type": l.layer_type,
                "firingRate": l.firing_rate,
            }
            for l in layers
        ]

        connection_data = [
            {
                "source": c.source_layer,
                "target": c.target_layer,
                "weight": c.weight,
                "spikeRate": c.spike_rate,
                "delay": c.delay,
            }
            for c in connections
        ]

        return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            background: {colors["bg"]};
            color: {colors["fg"]};
            font-family: 'Monaco', 'Menlo', monospace;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 20px;
            min-height: 100vh;
        }}
        h1 {{
            font-size: 18px;
            margin-bottom: 10px;
            color: {colors["accent"]};
        }}
        .container {{
            position: relative;
        }}
        #orrery {{
            background: {colors["bg"]};
        }}
        .stats {{
            position: absolute;
            top: 10px;
            right: 10px;
            background: {colors["grid"]};
            padding: 10px;
            border-radius: 4px;
            font-size: 11px;
        }}
        .stats div {{
            margin: 4px 0;
        }}
        .stat-label {{
            color: {colors["muted"]};
        }}
        .stat-value {{
            color: {colors["accent"]};
        }}
        .legend {{
            display: flex;
            gap: 20px;
            margin-top: 10px;
            font-size: 11px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        .legend-dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
        }}
        .controls {{
            margin-top: 15px;
            display: flex;
            gap: 10px;
        }}
        button {{
            background: {colors["grid"]};
            color: {colors["fg"]};
            border: 1px solid {colors["muted"]};
            padding: 6px 12px;
            border-radius: 4px;
            cursor: pointer;
            font-family: inherit;
            font-size: 11px;
        }}
        button:hover {{
            background: {colors["muted"]};
        }}
        button.active {{
            background: {colors["accent"]};
            color: {colors["bg"]};
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <div class="container">
        <svg id="orrery" width="{self.width}" height="{self.height}"></svg>
        <div class="stats">
            <div><span class="stat-label">Layers:</span> <span class="stat-value" id="stat-layers">0</span></div>
            <div><span class="stat-label">Neurons:</span> <span class="stat-value" id="stat-neurons">0</span></div>
            <div><span class="stat-label">Active Spikes:</span> <span class="stat-value" id="stat-spikes">0</span></div>
            <div><span class="stat-label">Avg Firing:</span> <span class="stat-value" id="stat-firing">0%</span></div>
        </div>
    </div>
    <div class="legend">
        <div class="legend-item">
            <div class="legend-dot" style="background: {colors["accent"]}"></div>
            <span>Input</span>
        </div>
        <div class="legend-item">
            <div class="legend-dot" style="background: {colors["accent3"]}"></div>
            <span>Hidden</span>
        </div>
        <div class="legend-item">
            <div class="legend-dot" style="background: {colors["accent2"]}"></div>
            <span>Output</span>
        </div>
        <div class="legend-item">
            <div class="legend-dot" style="background: {colors["warning"]}"></div>
            <span>Spike</span>
        </div>
    </div>
    <div class="controls">
        <button id="btn-pause">Pause</button>
        <button id="btn-speed-slow">0.5x</button>
        <button id="btn-speed-normal" class="active">1x</button>
        <button id="btn-speed-fast">2x</button>
    </div>

    <script>
        // Data
        const layers = {json.dumps(layer_data)};
        const connections = {json.dumps(connection_data)};

        // Colors
        const colors = {{
            bg: "{colors["bg"]}",
            fg: "{colors["fg"]}",
            accent: "{colors["accent"]}",
            accent2: "{colors["accent2"]}",
            accent3: "{colors["accent3"]}",
            muted: "{colors["muted"]}",
            warning: "{colors["warning"]}",
            grid: "{colors["grid"]}",
        }};

        // Canvas setup
        const svg = document.getElementById('orrery');
        const width = {self.width};
        const height = {self.height};
        const centerX = width / 2;
        const centerY = height / 2;
        const maxRadius = Math.min(width, height) * 0.42;

        // Animation state
        let paused = false;
        let speed = 1;
        let spikes = [];
        let time = 0;

        // Layer positioning
        const layerPositions = {{}};
        layers.forEach((layer, i) => {{
            const radiusRatio = (i + 1) / (layers.length + 1);
            layerPositions[layer.id] = {{
                radius: maxRadius * radiusRatio,
                neurons: []
            }};

            // Position neurons around the ring
            const neuronCount = Math.min(layer.neuronCount, 32); // Cap for display
            for (let j = 0; j < neuronCount; j++) {{
                const angle = (2 * Math.PI * j / neuronCount) - Math.PI / 2;
                layerPositions[layer.id].neurons.push({{
                    x: centerX + layerPositions[layer.id].radius * Math.cos(angle),
                    y: centerY + layerPositions[layer.id].radius * Math.sin(angle),
                    angle: angle
                }});
            }}
        }});

        // Create SVG elements
        function createSVGElement(tag, attrs) {{
            const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
            for (const [key, value] of Object.entries(attrs)) {{
                el.setAttribute(key, value);
            }}
            return el;
        }}

        // Draw static elements
        function drawStatic() {{
            // Background gradient
            const defs = createSVGElement('defs', {{}});
            defs.innerHTML = `
                <radialGradient id="bgGradient">
                    <stop offset="0%" stop-color="${{colors.grid}}" stop-opacity="0.3"/>
                    <stop offset="100%" stop-color="${{colors.bg}}" stop-opacity="0"/>
                </radialGradient>
                <filter id="glow">
                    <feGaussianBlur stdDeviation="2" result="blur"/>
                    <feMerge>
                        <feMergeNode in="blur"/>
                        <feMergeNode in="SourceGraphic"/>
                    </feMerge>
                </filter>
            `;
            svg.appendChild(defs);

            // Center glow
            const centerGlow = createSVGElement('circle', {{
                cx: centerX, cy: centerY, r: maxRadius * 0.8,
                fill: 'url(#bgGradient)'
            }});
            svg.appendChild(centerGlow);

            // Orbital rings
            layers.forEach((layer, i) => {{
                const pos = layerPositions[layer.id];
                const ring = createSVGElement('circle', {{
                    cx: centerX, cy: centerY, r: pos.radius,
                    fill: 'none', stroke: colors.grid,
                    'stroke-width': 1, 'stroke-dasharray': '4,6',
                    opacity: 0.5
                }});
                svg.appendChild(ring);
            }});

            // Neurons (planets)
            layers.forEach(layer => {{
                const pos = layerPositions[layer.id];
                const color = layer.type === 'input' ? colors.accent :
                             layer.type === 'output' ? colors.accent2 : colors.accent3;

                pos.neurons.forEach((neuron, i) => {{
                    const size = 3 + 2 * layer.firingRate;
                    const planet = createSVGElement('circle', {{
                        cx: neuron.x, cy: neuron.y, r: size,
                        fill: color, opacity: 0.6 + 0.4 * layer.firingRate,
                        filter: 'url(#glow)'
                    }});
                    svg.appendChild(planet);
                }});

                // Layer label
                const labelAngle = -Math.PI / 4;
                const labelX = centerX + pos.radius * Math.cos(labelAngle);
                const labelY = centerY + pos.radius * Math.sin(labelAngle);
                const label = createSVGElement('text', {{
                    x: labelX + 10, y: labelY,
                    fill: colors.muted, 'font-size': 10,
                    'font-family': 'monospace'
                }});
                label.textContent = layer.name;
                svg.appendChild(label);
            }});
        }}

        // Create spike group
        const spikeGroup = createSVGElement('g', {{ id: 'spikes' }});

        // Spawn a spike
        function spawnSpike(connection) {{
            const sourcePos = layerPositions[connection.source];
            const targetPos = layerPositions[connection.target];
            if (!sourcePos || !targetPos) return;

            // Pick random neurons
            const sourceNeuron = sourcePos.neurons[
                Math.floor(Math.random() * sourcePos.neurons.length)
            ];
            const targetNeuron = targetPos.neurons[
                Math.floor(Math.random() * targetPos.neurons.length)
            ];

            if (!sourceNeuron || !targetNeuron) return;

            spikes.push({{
                x: sourceNeuron.x,
                y: sourceNeuron.y,
                targetX: targetNeuron.x,
                targetY: targetNeuron.y,
                progress: 0,
                speed: 0.5 + Math.random() * 0.5,
                weight: connection.weight,
                el: null
            }});
        }}

        // Update spikes
        function updateSpikes(dt) {{
            // Maybe spawn new spikes
            connections.forEach(conn => {{
                if (Math.random() < conn.spikeRate * dt * speed * 0.5) {{
                    spawnSpike(conn);
                }}
            }});

            // Update existing spikes
            spikes = spikes.filter(spike => {{
                spike.progress += dt * spike.speed * speed * 0.01;

                if (spike.progress >= 1) {{
                    if (spike.el) spike.el.remove();
                    return false;
                }}

                // Update position
                const t = spike.progress;
                // Curved path using quadratic easing
                const eased = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;

                spike.currentX = spike.x + (spike.targetX - spike.x) * eased;
                spike.currentY = spike.y + (spike.targetY - spike.y) * eased;

                // Create element if needed
                if (!spike.el) {{
                    spike.el = createSVGElement('circle', {{
                        r: 2 + 2 * spike.weight,
                        fill: colors.warning,
                        filter: 'url(#glow)'
                    }});
                    spikeGroup.appendChild(spike.el);
                }}

                spike.el.setAttribute('cx', spike.currentX);
                spike.el.setAttribute('cy', spike.currentY);
                spike.el.setAttribute('opacity', 0.8 - 0.5 * t);

                return true;
            }});
        }}

        // Update stats
        function updateStats() {{
            document.getElementById('stat-layers').textContent = layers.length;
            document.getElementById('stat-neurons').textContent =
                layers.reduce((sum, l) => sum + l.neuronCount, 0);
            document.getElementById('stat-spikes').textContent = spikes.length;
            const avgFiring = layers.reduce((sum, l) => sum + l.firingRate, 0) / layers.length;
            document.getElementById('stat-firing').textContent =
                (avgFiring * 100).toFixed(0) + '%';
        }}

        // Animation loop
        let lastTime = performance.now();
        function animate(currentTime) {{
            if (!paused) {{
                const dt = (currentTime - lastTime) / 16.67; // Normalize to ~60fps
                updateSpikes(dt);
                time += dt;
            }}
            lastTime = currentTime;
            updateStats();
            requestAnimationFrame(animate);
        }}

        // Initialize
        drawStatic();
        svg.appendChild(spikeGroup);

        // Controls
        document.getElementById('btn-pause').addEventListener('click', (e) => {{
            paused = !paused;
            e.target.textContent = paused ? 'Play' : 'Pause';
            e.target.classList.toggle('active', paused);
        }});

        document.querySelectorAll('[id^="btn-speed"]').forEach(btn => {{
            btn.addEventListener('click', (e) => {{
                document.querySelectorAll('[id^="btn-speed"]').forEach(b =>
                    b.classList.remove('active'));
                e.target.classList.add('active');
                if (e.target.id === 'btn-speed-slow') speed = 0.5;
                else if (e.target.id === 'btn-speed-normal') speed = 1;
                else if (e.target.id === 'btn-speed-fast') speed = 2;
            }});
        }});

        // Start animation
        requestAnimationFrame(animate);
    </script>
</body>
</html>
'''


# =============================================================================
# Convenience Function
# =============================================================================

def generate_orrery(
    data: Dict[str, Any],
    theme: str = "ara",
    width: int = 800,
    height: int = 800,
) -> GiftArtifact:
    """
    Generate an SNN orrery visualization.

    Args:
        data: SNN topology data (layers, connections)
        theme: Color theme
        width: Canvas width
        height: Canvas height

    Returns:
        GiftArtifact containing the HTML
    """
    gen = OrreryGenerator(theme=theme, width=width, height=height)
    return gen.generate(data)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'NeuralLayer',
    'SpikeConnection',
    'OrreryGenerator',
    'generate_orrery',
]
