# T-FAN Quick Start Demo 🚀

**3 ways to interact with your T-FAN infrastructure in under 5 minutes!**

---

## Option 1: Web Dashboard (Easiest) 🌐

### Start the Server
```bash
cd /home/user/Quanta-meis-nib-cis
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### Access the Dashboard
Open your browser to: **http://localhost:8000**

### What You'll See
- **Metrics Tab** - Real-time training metrics (accuracy, latency, hypervolume)
- **Pareto Tab** - Interactive weight sliders + Plotly visualization
- **Training Tab** - Start/stop training controls
- **Configs Tab** - Browse and select configurations

### Try It Out
1. Click "Metrics" - see current status
2. Click "Pareto" - adjust weight sliders
3. Click "Apply Weights" - update optimization priorities in real-time
4. Watch the WebSocket status indicator (top right) - should show "Connected 🟢"

---

## Option 2: REST API (Programmatic) 🔌

### Start the Server (if not already running)
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### Test with curl

**Get Current Metrics:**
```bash
curl http://localhost:8000/api/metrics | jq
```

**Get Pareto Weights:**
```bash
curl http://localhost:8000/api/pareto/weights | jq
```

**Update Pareto Weights (Live Tuning):**
```bash
curl -X POST http://localhost:8000/api/pareto/weights \
  -H "Authorization: Bearer tfan-secure-token-change-me" \
  -H "Content-Type: application/json" \
  -d '{
    "neg_accuracy": 15.0,
    "latency": 0.5,
    "epr_cv": 2.0,
    "topo_gap": 1.0,
    "energy": 0.3
  }' | jq
```

**List Available Configs:**
```bash
curl http://localhost:8000/api/configs | jq
```

### Test with Python

```python
import requests

API_BASE = "http://localhost:8000"
TOKEN = "tfan-secure-token-change-me"
headers = {"Authorization": f"Bearer {TOKEN}"}

# Get metrics
response = requests.get(f"{API_BASE}/api/metrics")
print("Metrics:", response.json())

# Update weights
weights = {
    "neg_accuracy": 12.0,
    "latency": 0.8,
    "epr_cv": 2.0,
    "topo_gap": 1.0,
    "energy": 0.5
}
response = requests.post(
    f"{API_BASE}/api/pareto/weights",
    json=weights,
    headers=headers
)
print("Updated:", response.json())

# Get Pareto front
response = requests.get(f"{API_BASE}/api/pareto/front")
print("Front:", response.json())
```

---

## Option 3: GNOME Desktop App (Native Linux) 🖥️

### Install (One-Time Setup)
```bash
cd /home/user/Quanta-meis-nib-cis/gnome-tfan
./install.sh
```

### Enable Extension
```bash
gnome-extensions enable tfan@quanta-meis-nib-cis
```

### Launch Dashboard
```bash
tfan-gnome
```

### What You'll See
- **System Tray Icon** - Click for quick metrics
- **GTK4 Dashboard** - Modern native app with:
  - Overview view - Metric cards with gradients
  - Training view - Start/stop controls
  - Pareto view - Weight sliders
  - Configs view - Configuration manager
  - System view - Status and diagnostics

### Features
- Live metric updates every 2 seconds
- GitHub auto-loader (paste repo URL)
- One-click training
- Gradient cards (purple → blue)
- Glassmorphism effects

---

## Option 4: Command Line (Scripts) 💻

### Visualize Pareto Front
```bash
cd /home/user/Quanta-meis-nib-cis

# Basic visualization
python dashboards/pareto_app.py --results artifacts/pareto/pareto_front.json

# With best config highlighted
python dashboards/pareto_app.py --results artifacts/pareto/pareto_front.json --show-best

# A/B comparison
python dashboards/pareto_app.py \
  --results run2/pareto_front.json \
  --compare run1/pareto_front.json
```

### Load Best Config in Python
```python
from tfan.runtime import ModelSelector

# Auto-loads configs/auto/best.yaml
selector = ModelSelector()
config = selector.get_config()

print(f"Model: {config['model']['name']}")
print(f"Heads: {config['model']['n_heads']}")
print(f"Accuracy: {config['pareto_metrics']['accuracy']}")

# With CLI overrides
selector = ModelSelector(overrides={"n_heads": 16})
config = selector.get_config()
```

### Run Promotion Script
```bash
python scripts/promote_auto_best.py
```

This will:
1. Load `configs/auto/best.yaml`
2. Run smoke evaluation on 3 datasets
3. Verify all gates (accuracy, latency, EPR CV, topo gap, HV)
4. Generate detailed report
5. Exit 0 if all gates pass, exit 1 if any fail

---

## Real-Time WebSocket Demo 🔴

### Python WebSocket Client
```python
import asyncio
import websockets
import json

async def stream_metrics():
    uri = "ws://localhost:8000/ws/metrics"
    async with websockets.connect(uri) as websocket:
        print("Connected to metrics stream...")

        async for message in websocket:
            data = json.loads(message)
            print(f"[{data['type']}] {data['data']}")

# Run it
asyncio.run(stream_metrics())
```

### JavaScript WebSocket Client
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/metrics');

ws.onopen = () => {
    console.log('Connected to T-FAN metrics stream');
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log(`[${data.type}]`, data.data);

    if (data.type === 'metrics_update') {
        console.log(`Step: ${data.data.step}`);
        console.log(`Accuracy: ${data.data.accuracy}`);
        console.log(`Latency: ${data.data.latency_ms}ms`);
    }
};

ws.onerror = (error) => {
    console.error('WebSocket error:', error);
};
```

---

## Updating Metrics (for Testing)

### Emit Demo Metrics
```bash
python scripts/emit_demo_metrics.py
```

This writes sample metrics to `~/.cache/tfan/metrics.json`:
```json
{
  "training_active": true,
  "step": 1234,
  "accuracy": 0.923,
  "latency_ms": 145.2,
  "hypervolume": 47500,
  "epr_cv": 0.12,
  "topo_gap": 0.015,
  "timestamp": "2025-11-17T12:34:56"
}
```

The API and dashboards will automatically pick up these changes.

---

## Running a Full Pareto Optimization

### Method 1: Python Script
```python
from tfan.pareto_v2 import ParetoRunner, ParetoRunnerConfig

config = ParetoRunnerConfig(
    n_initial_points=10,
    n_iterations=50,
    output_dir='artifacts/pareto',
    seed=42
)

runner = ParetoRunner(config)
runner.run()

# Export best config
best_config = runner.get_best_config({
    "neg_accuracy": 10.0,
    "latency": 1.0,
    "epr_cv": 2.0,
    "topo_gap": 1.0,
    "energy": 0.5
})

runner.export_config(best_config, "configs/auto/best.yaml")
```

### Method 2: Via API
```bash
curl -X POST "http://localhost:8000/api/pareto/run?n_iterations=50&n_initial=10" \
  -H "Authorization: Bearer tfan-secure-token-change-me"
```

### Method 3: GitHub Actions (CI)
Trigger workflow manually or wait for weekly cron:
```bash
# Via gh CLI (if available)
gh workflow run pareto_optimization.yml \
  --field n_iterations=50 \
  --field n_initial=10
```

Or go to GitHub Actions tab in browser and click "Run workflow".

---

## Verifying Everything Works

### Run All Checks
```bash
# 1. Check API loads
python -c "from api.main import app; print(f'✓ {app.title} - {len(app.routes)} routes')"

# 2. Check services
python -c "
from api.services.metrics_service import MetricsService
from api.services.pareto_service import ParetoService
from api.services.training_service import TrainingService
print('✓ All services initialized')
"

# 3. Check pareto_v2 module
python -c "from tfan.pareto_v2 import ParetoRunner; print('✓ ParetoRunner ready')"

# 4. Check runtime module
python -c "from tfan.runtime import ModelSelector; print('✓ ModelSelector ready')"

# 5. Check configs
ls -lh configs/auto/best.yaml && echo "✓ Best config exists"

# 6. Check metrics
ls -lh ~/.cache/tfan/metrics.json && echo "✓ Metrics cache exists"

# 7. Check web files
ls -lh web/templates/dashboard.html web/static/css/dashboard.css web/static/js/dashboard.js
echo "✓ Web dashboard files exist"
```

### Run Tests
```bash
# Run Pareto tests
pytest tests/pareto/ -v

# Run runtime tests
pytest tests/runtime/ -v

# Run all tests
pytest tests/ -v
```

---

## Production Deployment

### With Systemd Service
Create `/etc/systemd/system/tfan-api.service`:
```ini
[Unit]
Description=T-FAN REST API
After=network.target

[Service]
Type=simple
User=tfan
WorkingDirectory=/opt/tfan
Environment="TFAN_API_TOKEN=your-secure-token-here"
ExecStart=/opt/tfan/venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable tfan-api
sudo systemctl start tfan-api
sudo systemctl status tfan-api
```

### With Docker
```bash
docker build -t tfan-api .
docker run -d -p 8000:8000 \
  -v ~/.cache/tfan:/root/.cache/tfan \
  -v ./configs:/app/configs \
  tfan-api
```

### With Nginx Reverse Proxy
```nginx
server {
    listen 80;
    server_name tfan.example.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## Troubleshooting

**API won't start:**
```bash
# Check dependencies
pip install -r requirements-api.txt

# Check port not in use
lsof -i :8000

# Run with debug logging
uvicorn api.main:app --reload --log-level debug
```

**WebSocket won't connect:**
```bash
# Install websockets support
pip install uvicorn[standard]

# Check firewall
sudo ufw allow 8000/tcp
```

**Metrics not updating:**
```bash
# Create cache directory
mkdir -p ~/.cache/tfan

# Emit test metrics
python scripts/emit_demo_metrics.py

# Check file permissions
ls -l ~/.cache/tfan/metrics.json
```

**GNOME extension not loading:**
```bash
# Check extension installed
gnome-extensions list | grep tfan

# View logs
journalctl -f -o cat /usr/bin/gnome-shell

# Restart GNOME Shell
Alt+F2, type 'r', press Enter
```

---

## 🎉 You're Ready!

All infrastructure is deployed and ready to use. Choose your preferred interface:
- **Web Dashboard** - Beautiful browser UI
- **REST API** - Programmatic access
- **GNOME App** - Native Linux desktop
- **CLI Scripts** - Terminal workflows

**Next Steps:**
1. Start the API server
2. Open http://localhost:8000
3. Explore the 4 dashboard views
4. Try live weight tuning
5. Run a Pareto optimization
6. Deploy to production

---

**Built with FastAPI, WebSockets, and engineering excellence** 🔥
