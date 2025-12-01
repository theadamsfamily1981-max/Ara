# T-FAN REST API 🚀

**Modern FastAPI server for T-FAN with live weight tuning, training control, and real-time metrics.**

## Features

✨ **REST API**
- Live Pareto weight tuning
- Training start/stop control
- Config management
- Real-time metrics endpoint

🔌 **WebSocket**
- Live metric streaming
- Real-time status updates
- Broadcast notifications

🎨 **Web Dashboard**
- Beautiful modern UI
- Interactive Pareto visualization (Plotly.js)
- Live metric cards
- Weight tuning sliders
- Training controls

🔒 **Security**
- Token-based authentication
- CORS support
- Secure WebSocket connections

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements-api.txt

# Install T-FAN package
pip install -e .
```

### Run Server

```bash
# Development mode
uvicorn api.main:app --reload

# Production mode
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Access Dashboard

Open your browser to:
```
http://localhost:8000
```

## API Endpoints

### Metrics

**GET /api/metrics**
```bash
curl http://localhost:8000/api/metrics
```

Response:
```json
{
  "training_active": true,
  "step": 1234,
  "accuracy": 0.923,
  "latency_ms": 145.2,
  "hypervolume": 47500,
  "epr_cv": 0.12,
  "topo_gap": 0.015,
  "timestamp": "2025-01-15T10:30:00"
}
```

### Pareto Optimization

**GET /api/pareto/weights**
```bash
curl http://localhost:8000/api/pareto/weights
```

**POST /api/pareto/weights**
```bash
curl -X POST http://localhost:8000/api/pareto/weights \
  -H "Authorization: Bearer tfan-secure-token-change-me" \
  -H "Content-Type: application/json" \
  -d '{
    "neg_accuracy": 10.0,
    "latency": 1.0,
    "epr_cv": 2.0,
    "topo_gap": 1.0,
    "energy": 0.5
  }'
```

**GET /api/pareto/front**
```bash
curl http://localhost:8000/api/pareto/front
```

**POST /api/pareto/run**
```bash
curl -X POST "http://localhost:8000/api/pareto/run?n_iterations=100&n_initial=10" \
  -H "Authorization: Bearer tfan-secure-token-change-me"
```

### Training Control

**POST /api/training/start**
```bash
curl -X POST http://localhost:8000/api/training/start \
  -H "Authorization: Bearer tfan-secure-token-change-me" \
  -H "Content-Type: application/json" \
  -d '{
    "config_path": "configs/auto/best.yaml",
    "max_steps": 20000,
    "logdir": "runs/api_training"
  }'
```

**POST /api/training/stop**
```bash
curl -X POST http://localhost:8000/api/training/stop \
  -H "Authorization: Bearer tfan-secure-token-change-me"
```

**GET /api/training/status**
```bash
curl http://localhost:8000/api/training/status
```

### Configuration

**GET /api/configs**
```bash
curl http://localhost:8000/api/configs
```

**POST /api/configs/select**
```bash
curl -X POST "http://localhost:8000/api/configs/select?config_path=configs/7b/quanta_focus.yaml" \
  -H "Authorization: Bearer tfan-secure-token-change-me"
```

### WebSocket

**Connect to metrics stream:**
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/metrics');

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log(data);
};
```

## Web Dashboard Usage

### 1. Metrics View 📊
- Live metric cards (accuracy, latency, hypervolume, EPR CV)
- Training status badge
- Real-time updates every 2 seconds

### 2. Pareto View 🎯
- **Weight Sliders:** Adjust optimization priorities
- **Apply Weights:** Update decision weights live
- **Visualization:** Interactive Plotly scatter plot
- **Run Optimization:** Start new Pareto sweep

### 3. Training View 🚀
- **Config Selector:** Choose training configuration
- **Max Steps:** Set training duration
- **Start/Stop:** Control training session
- **Live Logs:** Real-time training output

### 4. Configs View ⚙️
- List all available configs
- Select active configuration
- One-click config switching

## Architecture

```
api/
├── main.py                  # FastAPI app + endpoints
├── models/
│   └── schemas.py          # Pydantic models
└── services/
    ├── metrics_service.py  # Metrics management
    ├── pareto_service.py   # Pareto optimization
    └── training_service.py # Training control

web/
├── templates/
│   └── dashboard.html      # Main dashboard
└── static/
    ├── css/
    │   └── dashboard.css   # Styling
    └── js/
        └── dashboard.js    # Interactive logic
```

## Security

### API Token

Change the default token in `api/main.py`:
```python
API_TOKEN = "your-secure-token-here"
```

Or use environment variable:
```bash
export TFAN_API_TOKEN="your-secure-token"
```

### CORS

Update allowed origins in `api/main.py`:
```python
allow_origins=["https://yourdomain.com"]
```

## Integration with T-FAN

The API integrates seamlessly with T-FAN infrastructure:

1. **Reads metrics** from `~/.cache/tfan/metrics.json`
2. **Uses Pareto v2** module for optimization
3. **Launches training** via `training/train.py`
4. **Manages configs** in `configs/` directory

## Live Weight Tuning Example

```python
import requests

# Update weights to prioritize accuracy over latency
weights = {
    "neg_accuracy": 15.0,  # Increased
    "latency": 0.5,        # Decreased
    "epr_cv": 2.0,
    "topo_gap": 1.0,
    "energy": 0.5
}

response = requests.post(
    'http://localhost:8000/api/pareto/weights',
    json=weights,
    headers={'Authorization': 'Bearer tfan-secure-token-change-me'}
)

print(response.json())
```

## Production Deployment

### With Gunicorn

```bash
pip install gunicorn
gunicorn api.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

### With Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install -r requirements-api.txt
RUN pip install -e .

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### With Nginx (Reverse Proxy)

```nginx
server {
    listen 80;
    server_name tfan.yourdomain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```

## Troubleshooting

**WebSocket connection fails:**
```bash
# Check if uvicorn supports WebSockets
pip install uvicorn[standard]
```

**Metrics not updating:**
```bash
# Ensure metrics file exists
mkdir -p ~/.cache/tfan
echo '{"training_active": false, "step": 0, "accuracy": 0.0}' > ~/.cache/tfan/metrics.json
```

**Training won't start:**
```bash
# Check config file exists
ls configs/auto/best.yaml

# Check training script
python training/train.py --help
```

## Development

### Run in development mode:
```bash
uvicorn api.main:app --reload --log-level debug
```

### Run tests:
```bash
pytest tests/api/
```

### API Documentation:
Visit `http://localhost:8000/docs` for interactive Swagger UI

---

**Built with FastAPI, WebSockets, and love** ❤️

*Part of the T-FAN Neural Optimizer project*
