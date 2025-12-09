-- =============================================================================
-- Ara Hive Schema - Phase 1 (Postgres)
-- =============================================================================
-- Usage:
--   sudo -u postgres createuser ara -P
--   sudo -u postgres createdb ara_hive -O ara
--   psql -U ara -d ara_hive -f db/schema.sql
-- =============================================================================

-- 1.1 Enum for task status
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'task_status') THEN
    CREATE TYPE task_status AS ENUM ('pending', 'in_progress', 'done', 'failed');
  END IF;
END$$;

-- 1.2 Nodes: each machine in the hive
CREATE TABLE IF NOT EXISTS nodes (
  id             SERIAL PRIMARY KEY,
  hostname       TEXT UNIQUE NOT NULL,
  role           TEXT NOT NULL,              -- 'worker', 'gpu', 'pi', 'master'
  last_heartbeat TIMESTAMPTZ DEFAULT now(),
  cpu_load_pct   NUMERIC(5,2) DEFAULT 0,     -- 0–100
  mem_used_pct   NUMERIC(5,2) DEFAULT 0,
  gpu_load_pct   NUMERIC(5,2) DEFAULT 0,
  meta           JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_nodes_role ON nodes(role);

-- 1.3 Sites: (task_type, node) pair + bee metrics
CREATE TABLE IF NOT EXISTS sites (
  id          SERIAL PRIMARY KEY,
  task_type   TEXT NOT NULL,              -- e.g., 'dummy_cpu'
  node_id     INT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
  q_hat       NUMERIC(10,4) DEFAULT 0,    -- estimated reward (jobs/sec)
  intensity   NUMERIC(10,4) DEFAULT 1,    -- pheromone strength
  congestion  INT DEFAULT 0,              -- active tasks on this site
  last_update TIMESTAMPTZ DEFAULT now(),
  UNIQUE(task_type, node_id)
);

CREATE INDEX IF NOT EXISTS idx_sites_task_intensity
  ON sites(task_type, intensity DESC);

-- 1.4 Tasks: simple queue for Phase 1
CREATE TABLE IF NOT EXISTS tasks (
  id           BIGSERIAL PRIMARY KEY,
  task_type    TEXT NOT NULL,              -- 'dummy_cpu'
  status       task_status NOT NULL DEFAULT 'pending',
  site_id      INT REFERENCES sites(id),
  payload      JSONB DEFAULT '{}'::jsonb,
  created_at   TIMESTAMPTZ DEFAULT now(),
  updated_at   TIMESTAMPTZ DEFAULT now(),
  started_at   TIMESTAMPTZ,
  finished_at  TIMESTAMPTZ,
  attempt      INT DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_tasks_type_status
  ON tasks(task_type, status);
