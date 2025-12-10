-- =============================================================================
-- Ara Hive Schema - Video Jobs Extension
-- =============================================================================
-- Usage: psql -U ara -d ara_hive -f db/schema_video.sql
-- =============================================================================

-- 2.1 Video job status enum
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'video_job_status') THEN
    CREATE TYPE video_job_status AS ENUM ('queued', 'running', 'completed', 'failed');
  END IF;
END$$;

-- 2.2 Top-level video jobs (logical job, fans out to blocks)
CREATE TABLE IF NOT EXISTS video_jobs (
  id            BIGSERIAL PRIMARY KEY,
  user_id       TEXT,                          -- who submitted
  request_id    TEXT,                          -- client-side correlation id
  status        video_job_status NOT NULL DEFAULT 'queued',
  total_blocks  INT NOT NULL,
  blocks_done   INT NOT NULL DEFAULT 0,
  model_name    TEXT NOT NULL,                 -- 'hummingbird_v0', 'sana_block_v0'
  style         JSONB DEFAULT '{}'::jsonb,     -- style pack / conditioning
  script        JSONB DEFAULT '{}'::jsonb,     -- high-level script / shots
  artifacts     JSONB DEFAULT '[]'::jsonb,     -- list of output paths when done
  created_at    TIMESTAMPTZ DEFAULT now(),
  updated_at    TIMESTAMPTZ DEFAULT now(),
  finished_at   TIMESTAMPTZ,
  error_message TEXT
);

CREATE INDEX IF NOT EXISTS idx_video_jobs_status ON video_jobs(status);
CREATE INDEX IF NOT EXISTS idx_video_jobs_user ON video_jobs(user_id);

-- 2.3 Video blocks use existing `tasks` table with:
--   task_type = 'video_block'
--   payload = { job_id, block_index, num_frames, seed, model_name, style, script }

-- Create sites for video_block on GPU nodes
INSERT INTO sites (task_type, node_id, q_hat, intensity, congestion)
SELECT 'video_block', id, 0.0, 1.0, 0
FROM nodes
WHERE role = 'gpu'
ON CONFLICT (task_type, node_id) DO NOTHING;

-- =============================================================================
-- Views for Video Jobs
-- =============================================================================

-- Video job progress
CREATE OR REPLACE VIEW video_job_progress AS
SELECT
    vj.id,
    vj.user_id,
    vj.status,
    vj.total_blocks,
    vj.blocks_done,
    ROUND(100.0 * vj.blocks_done / NULLIF(vj.total_blocks, 0), 1) AS pct_done,
    vj.model_name,
    vj.created_at,
    EXTRACT(EPOCH FROM (COALESCE(vj.finished_at, now()) - vj.created_at)) AS elapsed_sec,
    vj.error_message
FROM video_jobs vj
ORDER BY vj.created_at DESC;

-- Video block queue status
CREATE OR REPLACE VIEW video_block_queue AS
SELECT
    COUNT(*) FILTER (WHERE status = 'pending') AS pending,
    COUNT(*) FILTER (WHERE status = 'in_progress') AS running,
    COUNT(*) FILTER (WHERE status = 'done') AS done,
    COUNT(*) FILTER (WHERE status = 'failed') AS failed
FROM tasks
WHERE task_type = 'video_block';

-- GPU node performance (frames/sec estimate)
CREATE OR REPLACE VIEW gpu_node_perf AS
SELECT
    n.hostname,
    n.gpu_load_pct,
    s.q_hat,
    s.intensity,
    s.congestion,
    s.last_update
FROM sites s
JOIN nodes n ON n.id = s.node_id
WHERE s.task_type = 'video_block'
  AND n.role = 'gpu'
ORDER BY s.intensity DESC;
