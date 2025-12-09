-- =============================================================================
-- Ara Hive Schema - Postgres
-- =============================================================================
-- Apply with: sudo -u postgres psql ara_hive -f schema.sql
-- =============================================================================

-- Bite 0: Task Queue
-- =============================================================================

CREATE TYPE task_status AS ENUM ('pending', 'in_progress', 'done', 'failed');

CREATE TABLE tasks (
  id           BIGSERIAL PRIMARY KEY,
  task_type    TEXT NOT NULL,
  status       task_status NOT NULL DEFAULT 'pending',
  payload      JSONB DEFAULT '{}'::jsonb,
  result       JSONB DEFAULT NULL,
  error        TEXT DEFAULT NULL,
  created_at   TIMESTAMPTZ DEFAULT now(),
  updated_at   TIMESTAMPTZ DEFAULT now(),
  started_at   TIMESTAMPTZ,
  finished_at  TIMESTAMPTZ,
  attempt      INT DEFAULT 0,
  node_id      INT DEFAULT NULL,  -- Which node processed this (Bite 1+)
  site_id      INT DEFAULT NULL   -- Which site was used (Bite 2+)
);

CREATE INDEX idx_tasks_type_status ON tasks(task_type, status);
CREATE INDEX idx_tasks_created ON tasks(created_at);

-- =============================================================================
-- Bite 1: Nodes (Machines)
-- =============================================================================

CREATE TABLE nodes (
  id             SERIAL PRIMARY KEY,
  hostname       TEXT UNIQUE NOT NULL,
  role           TEXT NOT NULL,              -- 'worker','gpu','pi','master'
  last_heartbeat TIMESTAMPTZ DEFAULT now(),
  cpu_load_pct   NUMERIC(5,2) DEFAULT 0,
  mem_used_pct   NUMERIC(5,2) DEFAULT 0,
  gpu_load_pct   NUMERIC(5,2) DEFAULT 0,
  gpu_mem_pct    NUMERIC(5,2) DEFAULT 0,
  active         BOOLEAN DEFAULT true,
  meta           JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX idx_nodes_role ON nodes(role);
CREATE INDEX idx_nodes_active ON nodes(active) WHERE active = true;

-- =============================================================================
-- Bite 2: Sites (task_type + node) with Waggle Fields
-- =============================================================================

CREATE TABLE sites (
  id          SERIAL PRIMARY KEY,
  task_type   TEXT NOT NULL,
  node_id     INT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
  q_hat       NUMERIC(10,4) DEFAULT 0,       -- Smoothed reward estimate
  intensity   NUMERIC(10,4) DEFAULT 1,       -- Waggle dance intensity
  congestion  INT DEFAULT 0,                 -- Current jobs on this site
  job_count   BIGINT DEFAULT 0,              -- Total jobs completed
  total_time  NUMERIC(12,4) DEFAULT 0,       -- Total execution time
  last_update TIMESTAMPTZ DEFAULT now(),
  UNIQUE(task_type, node_id)
);

CREATE INDEX idx_sites_task_intensity ON sites(task_type, intensity DESC);
CREATE INDEX idx_sites_node ON sites(node_id);

-- =============================================================================
-- Helper Functions
-- =============================================================================

-- Function to auto-update updated_at
CREATE OR REPLACE FUNCTION update_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER tasks_update_timestamp
    BEFORE UPDATE ON tasks
    FOR EACH ROW EXECUTE FUNCTION update_timestamp();

-- =============================================================================
-- Convenience Views
-- =============================================================================

-- Waggle board: see all sites with node info
CREATE VIEW waggle_board AS
SELECT
    s.id AS site_id,
    n.hostname,
    n.role,
    s.task_type,
    s.q_hat,
    s.intensity,
    s.congestion,
    s.job_count,
    CASE WHEN s.job_count > 0
         THEN round(s.total_time / s.job_count, 3)
         ELSE 0 END AS avg_time,
    s.last_update,
    n.cpu_load_pct,
    n.mem_used_pct,
    n.gpu_load_pct
FROM sites s
JOIN nodes n ON n.id = s.node_id
WHERE n.active = true
ORDER BY s.task_type, s.intensity DESC;

-- Queue status: pending/running/done counts
CREATE VIEW queue_status AS
SELECT
    task_type,
    COUNT(*) FILTER (WHERE status = 'pending') AS pending,
    COUNT(*) FILTER (WHERE status = 'in_progress') AS running,
    COUNT(*) FILTER (WHERE status = 'done') AS done,
    COUNT(*) FILTER (WHERE status = 'failed') AS failed,
    COUNT(*) AS total
FROM tasks
GROUP BY task_type;

-- Node health: show nodes that haven't heartbeated recently
CREATE VIEW node_health AS
SELECT
    id,
    hostname,
    role,
    active,
    cpu_load_pct,
    mem_used_pct,
    gpu_load_pct,
    last_heartbeat,
    now() - last_heartbeat AS since_heartbeat,
    CASE
        WHEN now() - last_heartbeat > interval '5 minutes' THEN 'dead'
        WHEN now() - last_heartbeat > interval '1 minute' THEN 'stale'
        ELSE 'healthy'
    END AS health
FROM nodes
ORDER BY last_heartbeat DESC;
