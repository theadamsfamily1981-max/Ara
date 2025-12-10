-- =============================================================================
-- Ara Hive Schema Extension - Hardware & Parts Picker
-- =============================================================================
-- Usage:
--   psql -U ara -d ara_hive -f db/schema_hardware.sql
-- =============================================================================

-- 2.1 Hardware inventory table - cataloged by hardware scout
CREATE TABLE IF NOT EXISTS hardware_inventory (
  id              SERIAL PRIMARY KEY,
  node_id         INT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
  device_type     TEXT NOT NULL,              -- 'gpu', 'fpga', 'tpu', 'cpu', 'npu'
  device_name     TEXT NOT NULL,              -- 'RTX 4090', 'Xilinx Alveo U250'
  vendor          TEXT NOT NULL,              -- 'nvidia', 'amd', 'intel', 'xilinx'

  -- Capabilities
  vram_gb         NUMERIC(8,2),               -- GPU VRAM
  compute_units   INT,                        -- CUDA cores, stream processors, etc.
  clock_mhz       INT,
  tdp_watts       INT,

  -- Current state
  available       BOOLEAN DEFAULT true,
  current_task_id BIGINT REFERENCES tasks(id),
  utilization_pct NUMERIC(5,2) DEFAULT 0,
  temp_celsius    NUMERIC(5,1),
  power_watts     NUMERIC(8,2),

  -- Cost model
  cost_per_hour   NUMERIC(10,4) DEFAULT 0,    -- USD or compute units
  priority        INT DEFAULT 0,               -- Higher = prefer this device

  -- Metadata
  driver_version  TEXT,
  capabilities    JSONB DEFAULT '{}'::jsonb,  -- CUDA version, features, etc.
  last_scout      TIMESTAMPTZ DEFAULT now(),

  UNIQUE(node_id, device_type, device_name)
);

CREATE INDEX IF NOT EXISTS idx_hardware_type ON hardware_inventory(device_type, available);
CREATE INDEX IF NOT EXISTS idx_hardware_node ON hardware_inventory(node_id);

-- 2.2 Parts picker jobs - requests for hardware allocation
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'parts_job_status') THEN
    CREATE TYPE parts_job_status AS ENUM ('pending', 'picking', 'allocated', 'released', 'failed');
  END IF;
END$$;

CREATE TABLE IF NOT EXISTS parts_jobs (
  id              BIGSERIAL PRIMARY KEY,
  requester       TEXT NOT NULL,              -- 'arasong', 'video_worker', etc.
  job_type        TEXT NOT NULL,              -- 'render', 'inference', 'training'

  -- Requirements
  min_vram_gb     NUMERIC(8,2),
  min_compute     INT,
  preferred_vendor TEXT,                      -- 'nvidia', 'amd', or NULL for any
  device_types    TEXT[] DEFAULT ARRAY['gpu'],

  -- Budget constraints
  max_cost_hour   NUMERIC(10,4),
  max_power_watts INT,
  deadline_at     TIMESTAMPTZ,

  -- Allocation result
  status          parts_job_status DEFAULT 'pending',
  allocated_hw    INT[],                      -- hardware_inventory.id array
  allocation_at   TIMESTAMPTZ,
  released_at     TIMESTAMPTZ,

  -- Metrics
  actual_cost     NUMERIC(10,4) DEFAULT 0,
  actual_power_wh NUMERIC(10,4) DEFAULT 0,
  duration_s      NUMERIC(10,2),

  -- Linking
  parent_task_id  BIGINT REFERENCES tasks(id),
  payload         JSONB DEFAULT '{}'::jsonb,

  created_at      TIMESTAMPTZ DEFAULT now(),
  updated_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_parts_jobs_status ON parts_jobs(status);
CREATE INDEX IF NOT EXISTS idx_parts_jobs_requester ON parts_jobs(requester);

-- 2.3 Hardware usage log - for cost tracking and capacity planning
CREATE TABLE IF NOT EXISTS hardware_usage_log (
  id              BIGSERIAL PRIMARY KEY,
  hardware_id     INT NOT NULL REFERENCES hardware_inventory(id),
  parts_job_id    BIGINT REFERENCES parts_jobs(id),
  task_id         BIGINT REFERENCES tasks(id),

  started_at      TIMESTAMPTZ DEFAULT now(),
  ended_at        TIMESTAMPTZ,

  -- Metrics during usage
  avg_utilization NUMERIC(5,2),
  avg_power_watts NUMERIC(8,2),
  peak_temp_c     NUMERIC(5,1),
  energy_wh       NUMERIC(10,4),

  -- Cost
  computed_cost   NUMERIC(10,4)
);

CREATE INDEX IF NOT EXISTS idx_hw_usage_hardware ON hardware_usage_log(hardware_id);
CREATE INDEX IF NOT EXISTS idx_hw_usage_time ON hardware_usage_log(started_at);

-- 2.4 Budget tracking per requester
CREATE TABLE IF NOT EXISTS budget_accounts (
  id              SERIAL PRIMARY KEY,
  account_name    TEXT UNIQUE NOT NULL,       -- 'arasong', 'video', 'training'

  -- Limits
  daily_limit     NUMERIC(10,4),              -- Cost per day
  weekly_limit    NUMERIC(10,4),
  monthly_limit   NUMERIC(10,4),
  power_limit_kwh NUMERIC(10,4),              -- Energy budget

  -- Current period usage
  period_start    DATE DEFAULT CURRENT_DATE,
  period_cost     NUMERIC(10,4) DEFAULT 0,
  period_energy   NUMERIC(10,4) DEFAULT 0,

  -- Lifetime
  total_cost      NUMERIC(12,4) DEFAULT 0,
  total_energy_kwh NUMERIC(12,4) DEFAULT 0,

  created_at      TIMESTAMPTZ DEFAULT now()
);

-- 2.5 View: Available hardware for allocation
CREATE OR REPLACE VIEW v_available_hardware AS
SELECT
  hi.*,
  n.hostname,
  n.role as node_role,
  n.last_heartbeat,
  CASE
    WHEN n.last_heartbeat > now() - INTERVAL '5 minutes' THEN true
    ELSE false
  END as node_online
FROM hardware_inventory hi
JOIN nodes n ON hi.node_id = n.id
WHERE hi.available = true
  AND hi.current_task_id IS NULL
  AND n.last_heartbeat > now() - INTERVAL '5 minutes';

-- 2.6 View: Current hardware utilization summary
CREATE OR REPLACE VIEW v_hardware_summary AS
SELECT
  device_type,
  COUNT(*) as total_devices,
  COUNT(*) FILTER (WHERE available AND current_task_id IS NULL) as available_count,
  SUM(vram_gb) as total_vram_gb,
  SUM(vram_gb) FILTER (WHERE available AND current_task_id IS NULL) as available_vram_gb,
  AVG(utilization_pct) as avg_utilization,
  SUM(power_watts) as total_power_watts
FROM hardware_inventory
GROUP BY device_type;

-- 2.7 Function: Pick best hardware for requirements
CREATE OR REPLACE FUNCTION pick_hardware(
  p_min_vram NUMERIC,
  p_device_types TEXT[],
  p_preferred_vendor TEXT,
  p_max_cost NUMERIC,
  p_limit INT DEFAULT 1
)
RETURNS TABLE(hardware_id INT, score NUMERIC) AS $$
BEGIN
  RETURN QUERY
  SELECT
    hi.id,
    (
      -- Score based on: availability, capability match, cost efficiency, priority
      100.0
      + hi.priority * 10.0
      + CASE WHEN hi.vendor = p_preferred_vendor THEN 20.0 ELSE 0.0 END
      + CASE WHEN hi.vram_gb >= p_min_vram * 1.5 THEN 10.0 ELSE 0.0 END
      - hi.cost_per_hour * 5.0
      - hi.utilization_pct * 0.5
    )::NUMERIC as score
  FROM hardware_inventory hi
  JOIN nodes n ON hi.node_id = n.id
  WHERE hi.available = true
    AND hi.current_task_id IS NULL
    AND hi.device_type = ANY(p_device_types)
    AND (p_min_vram IS NULL OR hi.vram_gb >= p_min_vram)
    AND (p_max_cost IS NULL OR hi.cost_per_hour <= p_max_cost)
    AND n.last_heartbeat > now() - INTERVAL '5 minutes'
  ORDER BY score DESC
  LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;
