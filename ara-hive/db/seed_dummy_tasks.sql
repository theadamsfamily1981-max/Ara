-- =============================================================================
-- Seed Dummy Tasks for Testing
-- =============================================================================
-- Usage: psql -U ara -d ara_hive -f db/seed_dummy_tasks.sql
-- =============================================================================

-- 1. Create sites for all worker nodes
INSERT INTO sites (task_type, node_id, q_hat, intensity, congestion)
SELECT 'dummy_cpu', id, 0.0, 1.0, 0
FROM nodes
WHERE role = 'worker'
ON CONFLICT (task_type, node_id) DO NOTHING;

-- 2. Seed 100 dummy tasks (500ms each)
INSERT INTO tasks (task_type, payload)
SELECT 'dummy_cpu', jsonb_build_object('work_ms', 500)
FROM generate_series(1, 100) g;

-- 3. Show what we created
SELECT 'Sites created:' AS info, count(*) FROM sites WHERE task_type = 'dummy_cpu';
SELECT 'Tasks pending:' AS info, count(*) FROM tasks WHERE status = 'pending';
