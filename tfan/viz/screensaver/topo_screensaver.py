#!/usr/bin/env python3
"""
T-FAN Topology Screensaver

Living visualization of T-FAN's mathematical foundations:
- Barcode Nebula: Animated persistence barcodes from streaming point clouds
- Landscape Waterfall: Stacked persistence landscapes flowing in time
- Poincaré Orbits: Hyperbolic embeddings with geodesic drift
- Pareto Galaxy: Non-dominated configs as stars in objective space

Usage:
    python topo_screensaver.py --mode landscape --fullscreen
    python topo_screensaver.py --metrics http://localhost:8000/api/metrics
"""

import argparse
import math
import time
import threading
import json
import random
import numpy as np
from vispy import app, scene, color
from vispy.visuals import Line, Markers, Text
from ripser import ripser
from persim import PersistenceLandscape
import requests

# ---------- Config ----------
PALETTE = color.get_colormap('viridis')
BG = (0.02, 0.03, 0.06, 1.0)
FPS = 60

# Telemetry source (optional Prometheus-exported metrics)
DEFAULT_METRICS = dict(
    epr_cv=0.10,
    ttw_p95_ms=3.0,
    topo_cos=0.93,
    wass_gap=0.015,
    pgu_p95_ms=150.0,
    accuracy=0.0,
    latency_ms=0.0,
    hypervolume=0.0
)

# ---------- Data Generators ----------
def swiss_roll(n=1200, t=0.0):
    """Animated point cloud with slow drift → varied topology."""
    theta = np.linspace(1.5*np.pi, 4.5*np.pi, n)
    z = np.linspace(-2, 2, n)
    r = theta + 0.1*np.sin(2*t + 0.3*theta)
    x = r*np.cos(theta) + 0.05*np.cos(t*0.7)*np.random.randn(n)
    y = r*np.sin(theta) + 0.05*np.sin(t*0.9)*np.random.randn(n)
    P = np.stack([x, y, z], axis=1)
    P = P[:, :2]  # 2D for faster PD
    P += 0.6*np.stack([np.sin(t*0.2), np.cos(t*0.17)])  # slow drift
    return P

def circle_clusters(n=1200, k=3, t=0.0):
    """k drifting circles → controllable H1 cycles."""
    pts = []
    for i in range(k):
        r = 1.0 + 0.2*np.sin(t*0.3 + i)
        th = np.random.rand(n//k)*2*np.pi
        cx = 2.5*np.cos(i*2*np.pi/k + t*0.1)
        cy = 2.5*np.sin(i*2*np.pi/k + t*0.1)
        x = cx + r*np.cos(th) + 0.06*np.random.randn(len(th))
        y = cy + r*np.sin(th) + 0.06*np.random.randn(len(th))
        pts.append(np.stack([x, y], 1))
    return np.concatenate(pts, 0)

# ---------- Topology ----------
def compute_pd(P, maxdim=1):
    """Ripser for Vietoris–Rips persistence diagram."""
    res = ripser(P, maxdim=maxdim)
    return res['dgms']  # list: [H0, H1, ...] each shape (m,2)

def landscape_from_pd(dgm, k=5, resolution=400, xrange=(0, 3.0)):
    """Build persistence landscape from single homology degree."""
    if dgm.size == 0:
        xs = np.linspace(xrange[0], xrange[1], resolution)
        return xs, np.zeros((k, resolution))
    pl = PersistenceLandscape(dgms=[dgm], homology_degree=0)
    xs = np.linspace(xrange[0], xrange[1], resolution)
    # Stack first k layers; missing layers → zeros
    layers = []
    for i in range(1, k+1):
        yi = pl(xs, i)
        if yi is None:
            yi = np.zeros_like(xs)
        layers.append(yi)
    return xs, np.vstack(layers)

# ---------- Hyperbolic (optional stub – renders a disk & points) ----------
def poincare_sample(n=400, t=0.0):
    """Fake a hierarchy: radial ~ level, angle ~ branch."""
    levels = np.random.choice(6, size=n, p=np.linspace(1, 3, 6)/np.linspace(1, 3, 6).sum())
    r = 0.85*(levels/levels.max())**0.9 + 0.03*np.random.rand(n)
    a = np.random.rand(n)*2*np.pi + 0.1*np.sin(0.2*t + levels)
    x, y = r*np.cos(a), r*np.sin(a)
    return np.stack([x, y], 1)

# ---------- Pareto front (static demo) ----------
def fake_pareto(m=140):
    """Generate fake Pareto front: [neg_acc, latency_ms, epr_cv, topo_gap, energy]."""
    pts = np.random.rand(m, 5)
    # Non-dominated filter (simple)
    dom = np.zeros(m, dtype=bool)
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            if np.all(pts[j] <= pts[i]) and np.any(pts[j] < pts[i]):
                dom[i] = True
                break
    nd = pts[~dom]
    return nd

# ---------- Telemetry polling ----------
class Telemetry:
    """Poll metrics from HTTP endpoint or use defaults."""
    def __init__(self, url=None):
        self.url = url
        self.values = DEFAULT_METRICS.copy()
        self._stop = False
        if url:
            threading.Thread(target=self._poll, daemon=True).start()

    def _poll(self):
        while not self._stop:
            try:
                r = requests.get(self.url, timeout=1.5)
                r.raise_for_status()
                data = r.json()
                # Map API response to expected keys
                self.values.update({
                    'epr_cv': data.get('epr_cv', self.values['epr_cv']),
                    'accuracy': data.get('accuracy', self.values['accuracy']),
                    'latency_ms': data.get('latency_ms', self.values['latency_ms']),
                    'hypervolume': data.get('hypervolume', self.values['hypervolume']),
                    'topo_gap': data.get('topo_gap', self.values.get('topo_gap', 0.015)),
                })
            except Exception as e:
                pass
            time.sleep(2.0)

    def stop(self):
        self._stop = True

# ---------- VisApp ----------
class TopoScreensaver(app.Canvas):
    """Main screensaver canvas with 4 visualization modes."""
    MODES = ['barcode', 'landscape', 'poincare', 'pareto']

    def __init__(self, mode, telemetry: Telemetry, fullscreen=False):
        app.Canvas.__init__(
            self, title='T-FAN Topology Screensaver',
            keys='interactive', fullscreen=fullscreen,
            size=(1600, 900), bgcolor=BG
        )
        self.telemetry = telemetry
        self.mode_idx = self.MODES.index(mode) if mode in self.MODES else 1
        self.t0 = time.time()
        self.last = self.t0

        # Scene
        self.unfreeze()
        self.view = scene.SceneCanvas(keys=None, bgcolor=BG).central_widget.add_view()
        self.view.camera = scene.cameras.PanZoomCamera(aspect=1)
        self.view.camera.rect = (-3, -2, 6, 4)

        # Visuals
        self.lines = []
        self.points = Markers(parent=self.view.scene)
        self.text = Text('', color='white', pos=(20, 20), font_size=12, parent=self.scene)

        self.timer = app.Timer(1.0/FPS, connect=self._on_timer, start=True)
        self.paused = False
        self.freeze()

    # ---------- Drawing helpers ----------
    def clear_lines(self):
        for ln in self.lines:
            ln.parent = None
        self.lines = []

    def draw_barcode(self, dgm, y0=-1.0, y_step=0.03):
        """Draw persistence barcode diagram."""
        self.clear_lines()
        if dgm.size == 0:
            return
        # Normalize births/deaths into [0,3]
        b = np.nan_to_num(dgm[:, 0], nan=0.0, posinf=0.0)
        d = np.nan_to_num(dgm[:, 1], nan=3.0, posinf=3.0)
        order = np.argsort(-(d-b))  # longest first
        y = y0
        for idx in order[:200]:  # cap for perf
            x0, x1 = b[idx], d[idx]
            y += y_step
            pts = np.array([[x0, y], [x1, y]], dtype=np.float32)
            persistence = (d[idx] - b[idx]) / 3.0
            ln = scene.Line(
                pts, color=PALETTE.map(persistence),
                width=2, parent=self.view.scene
            )
            self.lines.append(ln)
        self.view.camera.rect = (0, -0.2, 3.0, 1.0)

    def draw_landscape(self, xs, layers):
        """Draw persistence landscape layers."""
        self.clear_lines()
        if layers.size == 0:
            return
        # Stack layers with vertical offsets; color by layer index
        k = layers.shape[0]
        for i in range(k):
            y = layers[i] + 0.15*i
            pts = np.stack([xs, y], 1).astype(np.float32)
            col = PALETTE.map((i+1)/(k+1))
            ln = scene.Line(pts, color=col, width=2, parent=self.view.scene)
            self.lines.append(ln)
        self.view.camera.rect = (xs.min(), -0.1, xs.ptp(), 0.2*k + 0.4)

    def draw_poincare(self, X):
        """Draw Poincaré disk with points."""
        self.clear_lines()
        # Draw unit circle
        th = np.linspace(0, 2*np.pi, 256)
        circ = np.stack([np.cos(th), np.sin(th)], 1).astype(np.float32)
        self.lines.append(
            scene.Line(circ, color=(0.5, 0.5, 0.7, 0.6),
                      width=2, parent=self.view.scene)
        )
        # Nodes
        self.points.set_data(
            X.astype(np.float32),
            face_color=(0.3, 0.8, 1.0, 0.9),
            size=2.5, edge_width=0
        )
        self.view.camera.rect = (-1.1, -1.1, 2.2, 2.2)

    def draw_pareto(self, ND):
        """Draw Pareto front in 2D projection."""
        self.clear_lines()
        # Project 5D → 2D (simple fixed)
        w = np.array([+1.0, +0.2, +0.6, +0.5, +0.1])  # custom lens
        x = ND @ (w[:5]/np.linalg.norm(w))
        y = ND @ (np.roll(w, 1)[:5]/np.linalg.norm(np.roll(w, 1)))
        pts = np.stack([x, y], 1)
        c = (ND[:, 0] - ND[:, 0].min()) / (ND[:, 0].ptp() + 1e-6)  # color by accuracy proxy
        cols = PALETTE.map(c)
        self.points.set_data(
            pts.astype(np.float32),
            face_color=cols, size=3.0, edge_width=0
        )
        self.view.camera.rect = (
            pts[:, 0].min()-0.1, pts[:, 1].min()-0.1,
            pts[:, 0].ptp()+0.2, pts[:, 1].ptp()+0.2
        )

    # ---------- Frame update ----------
    def _on_timer(self, ev):
        if self.paused:
            return
        t = time.time() - self.t0
        # Telemetry blending (affects speed/brightness)
        tel = self.telemetry.values
        tension = np.clip(0.4 + 2.0*tel.get('epr_cv', 0.1), 0.4, 1.2)  # 0.4–1.2
        tint = np.clip(0.5 + 1.5*(1.0 - tel.get('topo_cos', 0.9)), 0.5, 1.0)  # 0.5–1.0

        # Mode
        mode = self.MODES[self.mode_idx]
        if mode == 'barcode':
            P = circle_clusters(n=1200, k=3, t=t*tension)
            dgms = compute_pd(P, maxdim=1)
            dgm1 = dgms[1] if len(dgms) > 1 else np.empty((0, 2))
            self.draw_barcode(dgm1, y0=-0.05, y_step=0.01 + 0.01*tint)
        elif mode == 'landscape':
            P = swiss_roll(n=1000, t=t*tension)
            dgms = compute_pd(P, maxdim=0)  # landscape over H0 for speed
            dgm0 = dgms[0] if len(dgms) > 0 else np.empty((0, 2))
            xs, layers = landscape_from_pd(dgm0, k=6, resolution=400, xrange=(0, 3.0))
            self.draw_landscape(xs, layers)
        elif mode == 'poincare':
            X = poincare_sample(n=600, t=t*tension)
            self.draw_poincare(X)
        else:  # pareto
            ND = fake_pareto(m=220)
            self.draw_pareto(ND)

        self.text.text = (
            f"Mode: {mode.upper()}  |  "
            f"EPR-CV={tel.get('epr_cv', 0):.3f}  "
            f"Accuracy={tel.get('accuracy', 0):.3f}  "
            f"Latency={tel.get('latency_ms', 0):.1f}ms  "
            f"HV={tel.get('hypervolume', 0):.0f}"
        )

        self.update()

    # ---------- Events ----------
    def on_key_press(self, ev):
        k = ev.key.name.lower() if ev.key else ''
        if k in ('m', 'tab'):
            self.mode_idx = (self.mode_idx + 1) % len(self.MODES)
        elif k in ('p', 'space'):
            self.paused = not self.paused
        elif k in ('q', 'escape'):
            self.telemetry.stop()
            self.close()

def main():
    ap = argparse.ArgumentParser(description='T-FAN Topology Screensaver')
    ap.add_argument(
        '--mode', default='landscape',
        choices=['barcode', 'landscape', 'poincare', 'pareto'],
        help='Visualization mode (cycle with M)'
    )
    ap.add_argument(
        '--fullscreen', action='store_true',
        help='Run in fullscreen mode'
    )
    ap.add_argument(
        '--metrics',
        help='HTTP endpoint for live metrics (e.g., http://localhost:8000/api/metrics)'
    )
    args = ap.parse_args()

    tel = Telemetry(url=args.metrics)
    canvas = TopoScreensaver(args.mode, tel, fullscreen=args.fullscreen)
    canvas.show()
    app.run()

if __name__ == "__main__":
    main()
