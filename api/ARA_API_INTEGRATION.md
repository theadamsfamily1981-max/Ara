# Ara API Integration Guide 🤖🔌

Complete guide for connecting Ara avatar system to T-FAN via REST API and WebSocket.

## API Endpoints

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/ara/command` | POST | Process voice command |
| `/api/ara/status` | GET | Get current system state |
| `/api/ara/status/report` | GET | Get spoken status report |
| `/api/ara/history` | GET | Get command history |
| `/api/ara/avatar` | POST | Update avatar config |
| `/api/ara/mode/{mode}` | POST | Set workspace mode |
| `/api/ara/personality` | GET | Get personality config |
| `/api/ara/commands` | GET | List available commands |
| `/ws/ara` | WS | Bidirectional event stream |

---

## REST API Usage

### 1. Process Voice Command

**Endpoint:** `POST /api/ara/command`

Ara sends natural language commands, API returns action and response text.

**Request:**
```json
{
  "command": "show me GPU stats",
  "context": {
    "current_view": "dashboard"
  }
}
```

**Response:**
```json
{
  "success": true,
  "action": "cockpit_view",
  "params": {
    "view": "gpu"
  },
  "response": "Switching to GPU metrics",
  "original_command": "show me GPU stats"
}
```

**Python Example:**
```python
import httpx

async def send_command(command: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/api/ara/command",
            json={"command": command}
        )
        result = response.json()

        # Ara speaks the response
        ara.speak(result['response'])

        # Execute the action
        if result['success']:
            execute_action(result['action'], result['params'])

        return result
```

### 2. Get System State

**Endpoint:** `GET /api/ara/status`

Returns complete system state for Ara to understand context.

**Response:**
```json
{
  "workspace_mode": "work",
  "current_view": "dashboard",
  "training_active": true,
  "topology_visible": false,
  "fullscreen": false,
  "cockpit_active": true,
  "mode": "work",
  "avatar": {
    "profile": "Professional",
    "style": "Realistic",
    "mood": "Focused"
  },
  "personality": {
    "mode": "professional",
    "speech_style": "formal",
    "proactivity": "high",
    "humor_level": "low",
    "detail_level": "technical"
  },
  "timestamp": "2025-11-18T10:30:00"
}
```

**Python Example:**
```python
async def get_system_state():
    async with httpx.AsyncClient() as client:
        response = await client.get("http://localhost:8000/api/ara/status")
        state = response.json()

        # Adjust Ara's personality
        ara.set_personality(state['personality'])

        return state
```

### 3. Get Status Report

**Endpoint:** `GET /api/ara/status/report`

Returns natural language status report for Ara to speak.

**Response:**
```json
{
  "report_text": "System status: Training active at 94.5% accuracy. Currently in work mode, viewing dashboard.",
  "mode": "work",
  "system_state": {...}
}
```

**Python Example:**
```python
async def speak_status():
    async with httpx.AsyncClient() as client:
        response = await client.get("http://localhost:8000/api/ara/status/report")
        report = response.json()

        # Ara speaks the report
        ara.speak(report['report_text'])
```

### 4. Update Avatar Config

**Endpoint:** `POST /api/ara/avatar`

Update Ara's visual appearance from cockpit HUD or voice command.

**Request:**
```json
{
  "profile": "Casual",
  "style": "Realistic",
  "mood": "Friendly"
}
```

**Response:**
```json
{
  "status": "success",
  "avatar": {
    "profile": "Casual",
    "style": "Realistic",
    "mood": "Friendly"
  }
}
```

### 5. Set Workspace Mode

**Endpoint:** `POST /api/ara/mode/{mode}`

Switch between work and relax modes.

**Example:** `POST /api/ara/mode/relax`

**Response:**
```json
{
  "status": "success",
  "mode": "relax",
  "personality": {
    "mode": "conversational",
    "speech_style": "casual",
    "proactivity": "moderate",
    "humor_level": "moderate",
    "detail_level": "simplified"
  }
}
```

### 6. Get Available Commands

**Endpoint:** `GET /api/ara/commands`

List all supported voice commands for help/documentation.

**Response:**
```json
{
  "commands": [
    "show dashboard",
    "show pareto",
    "show topology",
    "topology landscape",
    "work mode",
    "start training",
    ...
  ],
  "categories": {
    "navigation": ["show dashboard", "show pareto", ...],
    "topology": ["topology barcode", "topology landscape", ...],
    "control": ["start training", "stop training", "work mode", ...],
    "avatar": ["avatar professional", "avatar casual", ...]
  }
}
```

---

## WebSocket Communication

### Connect to WebSocket

**Endpoint:** `ws://localhost:8000/api/ara/ws`

Bidirectional stream for real-time communication.

**Python Example:**
```python
import asyncio
import websockets
import json

async def ara_websocket_client():
    uri = "ws://localhost:8000/api/ara/ws"

    async with websockets.connect(uri) as websocket:
        # Receive initial state
        initial = await websocket.recv()
        state = json.loads(initial)
        print(f"Initial state: {state}")

        # Start listening for events
        async for message in websocket:
            event = json.loads(message)
            await handle_event(event)

async def handle_event(event):
    event_type = event.get('type')
    data = event.get('data', {})

    if event_type == 'command_processed':
        print(f"Command processed: {data['command']} → {data['action']}")

    elif event_type == 'mode_changed':
        print(f"Mode changed to: {data['mode']}")
        ara.set_personality(data['personality'])

    elif event_type == 'avatar_updated':
        print(f"Avatar updated: {data}")
        ara.update_appearance(data)

    elif event_type == 'state_updated':
        print(f"State updated: {data}")
```

### Send Commands via WebSocket

```python
async def send_ws_command(websocket, command: str):
    await websocket.send(json.dumps({
        'type': 'command',
        'data': {
            'command': command
        }
    }))

    # Wait for response
    response = await websocket.recv()
    result = json.loads(response)

    if result['type'] == 'command_response':
        ara.speak(result['data']['response'])
```

### Message Types

**Client → Server:**
- `command` - Process voice command
- `state_update` - Update system state
- `ping` - Keep-alive
- `get_state` - Request current state

**Server → Client:**
- `initial_state` - Sent on connection
- `command_response` - Response to command
- `command_processed` - Broadcast when command processed
- `mode_changed` - Workspace mode changed
- `avatar_updated` - Avatar config changed
- `state_updated` - System state changed
- `pong` - Keep-alive response

---

## Voice Command Mapping

### Navigation Commands

| Command | Action | Parameters |
|---------|--------|------------|
| "show dashboard" | switch_view | {view: "dashboard"} |
| "show pareto" | switch_view | {view: "pareto"} |
| "show training" | switch_view | {view: "training"} |
| "show topology" | show_topology | {fullscreen: false} |
| "show gpu" | cockpit_view | {view: "gpu"} |
| "show cpu" | cockpit_view | {view: "cpu"} |
| "show network" | cockpit_view | {view: "network"} |
| "show storage" | cockpit_view | {view: "storage"} |
| "show overview" | cockpit_view | {view: "overview"} |

### Topology Commands

| Command | Action | Parameters |
|---------|--------|------------|
| "topology barcode" | set_topology_mode | {mode: "barcode"} |
| "topology landscape" | set_topology_mode | {mode: "landscape"} |
| "topology poincare" | set_topology_mode | {mode: "poincare"} |
| "topology pareto" | set_topology_mode | {mode: "pareto"} |
| "engage topology" | show_topology | {fullscreen: true} |
| "hide topology" | hide_topology | {} |

### Control Commands

| Command | Action | Parameters |
|---------|--------|------------|
| "work mode" | set_workspace_mode | {mode: "work"} |
| "relax mode" | set_workspace_mode | {mode: "relax"} |
| "start training" | start_training | {config: "default"} |
| "stop training" | stop_training | {} |
| "fullscreen" | toggle_fullscreen | {} |
| "minimize" | minimize_window | {} |
| "restore" | restore_window | {} |

### Avatar Commands

| Command | Action | Parameters |
|---------|--------|------------|
| "avatar professional" | set_avatar_profile | {profile: "Professional"} |
| "avatar casual" | set_avatar_profile | {profile: "Casual"} |
| "avatar scientist" | set_avatar_profile | {profile: "Scientist"} |

### Query Commands

| Command | Action | Parameters |
|---------|--------|------------|
| "status report" | get_status_report | {} |
| "metrics report" | get_metrics_report | {} |

---

## Configuration Files

### Ara Config (ara_config.yaml)

Place in Ara repo to configure T-FAN connection:

```yaml
# T-FAN API connection
tfan_api:
  base_url: "http://localhost:8000"
  ws_url: "ws://localhost:8000/api/ara/ws"
  timeout: 30

# Personality defaults
personality:
  work:
    speech_style: "formal"
    proactivity: "high"
    humor_level: "low"
    detail_level: "technical"
  relax:
    speech_style: "casual"
    proactivity: "moderate"
    humor_level: "moderate"
    detail_level: "simplified"

# Avatar defaults
avatar:
  default_profile: "Default"
  default_style: "Realistic"
  default_mood: "Neutral"

# Event handling
events:
  announce_training_milestones: true
  announce_alerts: true
  alert_thresholds:
    epr_cv: 0.15
    latency_ms: 200
    temperature: 80
```

### Environment Variables

```bash
# T-FAN API
export TFAN_API_URL=http://localhost:8000
export TFAN_WS_URL=ws://localhost:8000/api/ara/ws
export TFAN_API_TOKEN=tfan-secure-token-change-me

# Ara settings
export ARA_DEFAULT_MODE=work
export ARA_ANNOUNCE_MILESTONES=true
```

---

## Integration Examples

### Complete Ara Voice Handler

```python
import asyncio
import httpx
import websockets
import json

class AraVoiceHandler:
    def __init__(self, api_url="http://localhost:8000"):
        self.api_url = api_url
        self.ws_url = api_url.replace("http", "ws") + "/api/ara/ws"
        self.websocket = None
        self.state = {}

    async def connect(self):
        """Connect to T-FAN WebSocket."""
        self.websocket = await websockets.connect(self.ws_url)

        # Get initial state
        initial = await self.websocket.recv()
        self.state = json.loads(initial).get('data', {})

        # Start event listener
        asyncio.create_task(self._listen_events())

    async def _listen_events(self):
        """Listen for events from T-FAN."""
        async for message in self.websocket:
            event = json.loads(message)
            await self._handle_event(event)

    async def _handle_event(self, event):
        """Handle incoming event."""
        event_type = event.get('type')
        data = event.get('data', {})

        if event_type == 'mode_changed':
            # Update personality
            self.state['mode'] = data['mode']
            self.state['personality'] = data['personality']

        elif event_type == 'state_updated':
            self.state.update(data)

    async def process_command(self, command: str) -> str:
        """
        Process voice command and return response text.

        Args:
            command: Natural language command from user

        Returns:
            str: Response text for TTS
        """
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.api_url}/api/ara/command",
                json={"command": command}
            )
            result = response.json()

            if result['success']:
                # Execute local actions if needed
                await self._execute_action(result['action'], result['params'])

            return result['response']

    async def _execute_action(self, action: str, params: dict):
        """Execute action locally (e.g., update D-Bus)."""
        # This is where you'd call D-Bus or other local systems
        pass

    async def get_status_report(self) -> str:
        """Get spoken status report."""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.api_url}/api/ara/status/report")
            report = response.json()
            return report['report_text']


# Usage in Ara's main loop
async def main():
    handler = AraVoiceHandler()
    await handler.connect()

    # Example: process voice command
    user_said = "show me GPU stats"
    response = await handler.process_command(user_said)
    print(f"Ara says: {response}")

    # Example: get status report
    report = await handler.get_status_report()
    print(f"Status: {report}")

asyncio.run(main())
```

### Proactive Event Handling

```python
class AraProactiveHandler:
    """Handle events and announce milestones proactively."""

    def __init__(self, tts_engine):
        self.tts = tts_engine
        self.announced_milestones = set()

    async def handle_metrics_update(self, metrics):
        """Check metrics and announce milestones."""
        accuracy = metrics.get('accuracy', 0)

        # Announce accuracy milestones
        if accuracy >= 0.95 and '95%' not in self.announced_milestones:
            await self.tts.speak("Great news! We've hit 95% accuracy.")
            self.announced_milestones.add('95%')

        if accuracy >= 0.99 and '99%' not in self.announced_milestones:
            await self.tts.speak("Excellent! 99% accuracy achieved.")
            self.announced_milestones.add('99%')

        # Check for alerts
        if metrics.get('epr_cv', 0) > 0.15:
            await self.tts.speak("Heads up - topology is getting unstable.")

        if metrics.get('latency_ms', 0) > 200:
            await self.tts.speak("Latency is above 200 milliseconds.")
```

---

## Error Handling

### HTTP Errors

```python
async def safe_command(command: str):
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(
                "http://localhost:8000/api/ara/command",
                json={"command": command}
            )
            response.raise_for_status()
            return response.json()

    except httpx.ConnectError:
        return {
            'success': False,
            'response': "I can't connect to T-FAN right now."
        }

    except httpx.TimeoutException:
        return {
            'success': False,
            'response': "T-FAN is not responding."
        }

    except httpx.HTTPStatusError as e:
        return {
            'success': False,
            'response': f"Error: {e.response.status_code}"
        }
```

### WebSocket Reconnection

```python
async def robust_ws_connection():
    while True:
        try:
            async with websockets.connect(
                "ws://localhost:8000/api/ara/ws",
                ping_interval=30,
                ping_timeout=10
            ) as websocket:
                async for message in websocket:
                    await handle_message(message)

        except websockets.ConnectionClosed:
            print("WebSocket closed, reconnecting...")
            await asyncio.sleep(2)

        except Exception as e:
            print(f"WebSocket error: {e}, reconnecting...")
            await asyncio.sleep(5)
```

---

## Testing

### Test Command Processing

```bash
# Test command endpoint
curl -X POST http://localhost:8000/api/ara/command \
  -H "Content-Type: application/json" \
  -d '{"command": "show gpu stats"}'

# Test status endpoint
curl http://localhost:8000/api/ara/status

# Test status report
curl http://localhost:8000/api/ara/status/report

# Test available commands
curl http://localhost:8000/api/ara/commands
```

### Test WebSocket

```python
# test_ara_ws.py
import asyncio
import websockets
import json

async def test():
    uri = "ws://localhost:8000/api/ara/ws"
    async with websockets.connect(uri) as ws:
        # Get initial state
        msg = await ws.recv()
        print(f"Initial: {msg}")

        # Send command
        await ws.send(json.dumps({
            'type': 'command',
            'data': {'command': 'show topology'}
        }))

        # Get response
        response = await ws.recv()
        print(f"Response: {response}")

        # Send ping
        await ws.send(json.dumps({'type': 'ping', 'data': {}}))
        pong = await ws.recv()
        print(f"Pong: {pong}")

asyncio.run(test())
```

---

## Security Notes

1. **API Token**: Change the default token in production
   ```python
   API_TOKEN = os.getenv("TFAN_API_TOKEN", "your-secure-token")
   ```

2. **CORS**: Restrict origins in production
   ```python
   allow_origins=["http://localhost:3000", "http://your-domain.com"]
   ```

3. **WebSocket Auth**: Add token verification to WebSocket
   ```python
   @router.websocket("/ws")
   async def websocket(ws: WebSocket, token: str = Query(...)):
       if token != API_TOKEN:
           await ws.close(code=4001)
           return
   ```

---

## Next Steps

1. **Implement in Ara repo**: Create `tfan_client.py` using this guide
2. **Add to voice pipeline**: Route voice → command → API → response → TTS
3. **Subscribe to events**: Listen for training milestones, alerts
4. **Sync state**: Keep Ara's state in sync with T-FAN
5. **Handle offline**: Graceful degradation when T-FAN unavailable

---

**Part of the T-FAN + Ara integration project** 🚀✨
