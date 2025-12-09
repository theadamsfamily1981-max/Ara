# Ara Hive - Postgres-Backed Distributed Job Scheduler

A bee colony-inspired job scheduler using PostgreSQL for shared state.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      POSTGRES (Master)                      │
│  ┌─────────┐  ┌─────────┐  ┌─────────────────────────────┐  │
│  │  tasks  │  │  nodes  │  │          sites              │  │
│  │ (queue) │  │ (boxes) │  │  (task_type + node + q_hat) │  │
│  └─────────┘  └─────────┘  └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
        │               │                    │
        ▼               ▼                    ▼
   ┌─────────┐    ┌─────────┐         ┌─────────┐
   │ Worker1 │    │ Worker2 │   ...   │ WorkerN │
   │ bee_    │    │ bee_    │         │ bee_    │
   │ agent.py│    │ agent.py│         │ agent.py│
   └─────────┘    └─────────┘         └─────────┘
```

## Quick Start

### 1. Setup Master (Postgres)

```bash
# Install Postgres
sudo apt install postgresql postgresql-contrib

# Create database
sudo -u postgres createdb ara_hive

# Apply schema
sudo -u postgres psql ara_hive -f schema.sql
```

### 2. Setup Worker (Any Machine)

```bash
git clone <repo> ara-hive
cd ara-hive

# Create virtualenv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with your master's IP:
# DB_DSN=dbname=ara_hive user=postgres host=192.168.1.100
```

### 3. Register Node + Start Worker

```bash
source venv/bin/activate
export $(cat .env | xargs)

# Register this machine (creates node + sites)
python register_node.py worker --task-types dummy_cpu

# Start bee agent
python bee_agent.py --task-type dummy_cpu --node-id 1
```

### 4. Seed Tasks

```bash
# Seed 20 dummy tasks
python seed_tasks.py dummy_cpu 20

# Watch them drain
python seed_tasks.py --status
```

### 5. Run Evaporation (on master)

```bash
# One-shot (for cron)
python evaporate.py

# Or as daemon
python evaporate.py --daemon --interval 60
```

## Files

| File | Purpose |
|------|---------|
| `schema.sql` | PostgreSQL schema (tasks, nodes, sites, views) |
| `bee_agent.py` | Worker that claims and executes tasks |
| `register_node.py` | Register machine + send heartbeats |
| `evaporate.py` | Pheromone decay + stale node detection |
| `seed_tasks.py` | Inject test tasks into queue |

## Key Concepts

### Tasks
Jobs in the queue. Workers claim them with `FOR UPDATE SKIP LOCKED`.

### Nodes
Physical machines. Send heartbeats to prove they're alive.

### Sites
`(task_type, node_id)` pairs with pheromone values:
- `q_hat`: Smoothed reward (EMA of 1/duration)
- `intensity`: Waggle dance strength = 0.1 + q_hat

### Waggle Dance Selection
Workers pick sites proportional to intensity. Better sites (lower latency) accumulate more intensity and attract more jobs.

### Evaporation
Sites that aren't updated fade: `intensity *= 0.95`. This prevents stale pheromone from dominating.

## Views

```sql
-- See the waggle board
SELECT * FROM waggle_board;

-- Queue status by task type
SELECT * FROM queue_status;

-- Node health (healthy/stale/dead)
SELECT * FROM node_health;
```

## Customizing Task Execution

Edit `execute_task()` in `bee_agent.py`:

```python
def execute_task(task_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    if task_type == "image_gen":
        return run_stable_diffusion(payload["prompt"])
    elif task_type == "transcode":
        return ffmpeg_transcode(payload["input"], payload["output"])
    else:
        # Default: sleep
        time.sleep(payload.get("work_ms", 500) / 1000.0)
        return {"status": "ok"}
```

## Scaling

1. **More workers**: Just run `bee_agent.py` on more machines
2. **More task types**: Create sites for new types via `register_node.py --task-types type1 type2`
3. **GPU tasks**: Register with `python register_node.py gpu --gpu --task-types image_gen`
