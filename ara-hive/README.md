# Ara Hive - Phase 1

Bee colony-inspired job scheduler using PostgreSQL.

```
ara-hive/
  db/
    schema.sql           # Tables: nodes, sites, tasks
    seed_dummy_tasks.sql # Test data
  src/
    register_node.py     # Node heartbeat
    bee_agent.py         # Worker bee
    evaporate_sites.py   # Pheromone decay
  requirements.txt
```

## Quick Start

### 1. Master: Setup Postgres

```bash
sudo -u postgres createuser ara -P        # password: ara
sudo -u postgres createdb ara_hive -O ara
psql -U ara -d ara_hive -f db/schema.sql
```

### 2. Each Node: Setup Python

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Each Node: Register + Start Worker

```bash
export ARA_HIVE_DSN="dbname=ara_hive user=ara password=ara host=<MASTER_IP>"
python src/register_node.py --role worker
python src/bee_agent.py
```

### 4. Master: Seed Test Tasks

```bash
psql -U ara -d ara_hive -f db/seed_dummy_tasks.sql
```

### 5. Master: Evaporation Cron

```cron
* * * * * cd /path/to/ara-hive && ARA_HIVE_DSN="..." venv/bin/python src/evaporate_sites.py
```

## Waggle Board Queries

```sql
-- Top sites by pheromone
SELECT s.id, s.task_type, n.hostname, s.q_hat, s.intensity, s.congestion
FROM sites s JOIN nodes n ON n.id = s.node_id
ORDER BY s.intensity DESC LIMIT 10;

-- Task throughput by minute
SELECT task_type, date_trunc('minute', finished_at) AS minute, count(*) AS done
FROM tasks WHERE status = 'done'
GROUP BY task_type, minute ORDER BY minute DESC;

-- Node health
SELECT id, hostname, role, cpu_load_pct, mem_used_pct, last_heartbeat
FROM nodes ORDER BY role, hostname;
```

## Next: Swap `time.sleep` for Real Work

Edit `src/bee_agent.py` and replace the sleep with your actual job:

```python
# In bee_agent.py, after claiming task:
t0 = time.time()
# time.sleep(work_ms / 1000.0)  # <-- remove this
result = your_real_job(payload)   # <-- add this
duration = time.time() - t0
```

The hive mechanics stay identical.
