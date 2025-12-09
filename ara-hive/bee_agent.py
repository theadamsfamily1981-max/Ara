#!/usr/bin/env python3
"""
Ara Hive - Bee Agent
====================

A worker bee that:
1. Claims tasks from the shared queue (FOR UPDATE SKIP LOCKED)
2. Executes them (default: sleep, swap for real work)
3. Updates site pheromones (q_hat, intensity)

Bite 0: Just tasks table, simple queue drain
Bite 2+: Sites with waggle dance selection

Usage:
    export DB_DSN="dbname=ara_hive user=postgres host=<master-ip>"
    python bee_agent.py [--task-type dummy_cpu] [--node-id 1]
"""

import argparse
import logging
import os
import random
import socket
import time
from typing import Any, Dict, Optional, Tuple

import psycopg2
from psycopg2.extras import DictCursor

# =============================================================================
# Config
# =============================================================================

DB_DSN = os.getenv("DB_DSN", "dbname=ara_hive user=postgres")

# ABC parameters
ALPHA = 0.3              # EMA smoothing for q_hat
INTENSITY_BASE = 0.1     # Minimum intensity floor
INTENSITY_SCALE = 1.0    # How much q_hat affects intensity

# Polling
POLL_INTERVAL = 1.0      # Seconds between queue checks when idle
MAX_RETRIES = 3          # Max attempts per task before marking failed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("bee")


# =============================================================================
# Task Execution (swap this for real work)
# =============================================================================

def execute_task(task_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a task. Override this for real work.

    Default: sleep for payload["work_ms"] milliseconds.

    Returns:
        dict with result data
    """
    work_ms = int(payload.get("work_ms", 500))

    # Simulate work
    time.sleep(work_ms / 1000.0)

    return {
        "simulated_ms": work_ms,
        "message": "dummy task completed"
    }


# =============================================================================
# Queue Operations (Bite 0)
# =============================================================================

def claim_task(cur, task_type: str) -> Optional[Dict[str, Any]]:
    """
    Atomically claim the oldest pending task.
    Uses FOR UPDATE SKIP LOCKED to avoid contention.
    """
    cur.execute("""
        WITH next_task AS (
            SELECT id
            FROM tasks
            WHERE task_type = %s
              AND status = 'pending'
            ORDER BY id
            LIMIT 1
            FOR UPDATE SKIP LOCKED
        )
        UPDATE tasks
        SET status = 'in_progress',
            started_at = now(),
            attempt = attempt + 1
        FROM next_task
        WHERE tasks.id = next_task.id
        RETURNING tasks.id, tasks.payload, tasks.attempt;
    """, (task_type,))
    row = cur.fetchone()
    if row:
        return {"id": row["id"], "payload": row["payload"] or {}, "attempt": row["attempt"]}
    return None


def mark_done(cur, task_id: int, result: Dict[str, Any] = None):
    """Mark task as completed."""
    cur.execute("""
        UPDATE tasks
        SET status = 'done',
            finished_at = now(),
            result = %s
        WHERE id = %s;
    """, (psycopg2.extras.Json(result), task_id))


def mark_failed(cur, task_id: int, error: str):
    """Mark task as failed."""
    cur.execute("""
        UPDATE tasks
        SET status = 'failed',
            finished_at = now(),
            error = %s
        WHERE id = %s;
    """, (error, task_id))


def requeue_task(cur, task_id: int):
    """Put task back in queue for retry."""
    cur.execute("""
        UPDATE tasks
        SET status = 'pending',
            started_at = NULL
        WHERE id = %s;
    """, (task_id,))


# =============================================================================
# Site Operations (Bite 2+)
# =============================================================================

def pick_site(cur, task_type: str) -> Optional[int]:
    """
    Pick a site using waggle dance (intensity-weighted selection).
    Returns site_id or None if no sites exist.
    """
    cur.execute("""
        SELECT id, intensity
        FROM sites
        WHERE task_type = %s
        ORDER BY intensity DESC
        LIMIT 10;
    """, (task_type,))
    rows = cur.fetchall()

    if not rows:
        return None

    # Weighted random selection by intensity
    total = sum(max(float(r["intensity"]), 0.001) for r in rows)
    r = random.random() * total
    acc = 0.0
    for row in rows:
        w = max(float(row["intensity"]), 0.001)
        acc += w
        if acc >= r:
            return row["id"]
    return rows[0]["id"]


def increment_congestion(cur, site_id: int):
    """Mark that we're starting work on this site."""
    cur.execute("""
        UPDATE sites
        SET congestion = congestion + 1
        WHERE id = %s;
    """, (site_id,))


def update_site_metrics(cur, site_id: int, duration_s: float, success: bool = True):
    """
    Update site pheromones after job completion.

    - q_hat: EMA of reward (1/duration for success, 0 for failure)
    - intensity: INTENSITY_BASE + INTENSITY_SCALE * q_hat
    - congestion: decrement
    - job_count, total_time: accumulate
    """
    cur.execute("SELECT q_hat FROM sites WHERE id = %s FOR UPDATE", (site_id,))
    row = cur.fetchone()
    if not row:
        return

    q_old = float(row["q_hat"])
    reward = (1.0 / max(duration_s, 0.001)) if success else 0.0
    q_new = (1 - ALPHA) * q_old + ALPHA * reward
    intensity_new = INTENSITY_BASE + INTENSITY_SCALE * q_new

    cur.execute("""
        UPDATE sites
        SET q_hat = %s,
            intensity = %s,
            congestion = GREATEST(congestion - 1, 0),
            job_count = job_count + 1,
            total_time = total_time + %s,
            last_update = now()
        WHERE id = %s;
    """, (q_new, intensity_new, duration_s, site_id))


def claim_task_with_site(cur, task_type: str, site_id: int, node_id: int) -> Optional[Dict[str, Any]]:
    """
    Claim task and record which site/node is processing it.
    """
    cur.execute("""
        WITH next_task AS (
            SELECT id
            FROM tasks
            WHERE task_type = %s
              AND status = 'pending'
            ORDER BY id
            LIMIT 1
            FOR UPDATE SKIP LOCKED
        )
        UPDATE tasks
        SET status = 'in_progress',
            started_at = now(),
            attempt = attempt + 1,
            site_id = %s,
            node_id = %s
        FROM next_task
        WHERE tasks.id = next_task.id
        RETURNING tasks.id, tasks.payload, tasks.attempt;
    """, (task_type, site_id, node_id))
    row = cur.fetchone()
    if row:
        return {"id": row["id"], "payload": row["payload"] or {}, "attempt": row["attempt"]}
    return None


# =============================================================================
# Main Worker Loop
# =============================================================================

def worker_loop_simple(task_type: str):
    """
    Bite 0: Simple queue drain without sites.
    Just claim -> execute -> mark done.
    """
    log.info(f"Starting simple worker for task_type={task_type}")

    while True:
        conn = psycopg2.connect(DB_DSN)
        conn.autocommit = False
        cur = conn.cursor(cursor_factory=DictCursor)

        try:
            task = claim_task(cur, task_type)
            if not task:
                conn.rollback()
                time.sleep(POLL_INTERVAL)
                continue

            task_id = task["id"]
            payload = task["payload"]
            attempt = task["attempt"]

            log.info(f"Claimed task {task_id} (attempt {attempt})")

            t0 = time.time()
            try:
                result = execute_task(task_type, payload)
                duration = time.time() - t0
                mark_done(cur, task_id, result)
                conn.commit()
                log.info(f"Done task {task_id} in {duration:.3f}s")

            except Exception as e:
                duration = time.time() - t0
                log.error(f"Task {task_id} failed: {e}")

                if attempt >= MAX_RETRIES:
                    mark_failed(cur, task_id, str(e))
                    conn.commit()
                    log.warning(f"Task {task_id} permanently failed after {attempt} attempts")
                else:
                    requeue_task(cur, task_id)
                    conn.commit()
                    log.info(f"Task {task_id} requeued for retry")

        except Exception as e:
            log.error(f"Worker error: {e}")
            conn.rollback()
            time.sleep(POLL_INTERVAL)

        finally:
            conn.close()


def worker_loop_with_sites(task_type: str, node_id: int):
    """
    Bite 2+: Worker with site selection and pheromone updates.
    """
    hostname = socket.gethostname()
    log.info(f"Starting site-aware worker: host={hostname} node_id={node_id} task_type={task_type}")

    while True:
        conn = psycopg2.connect(DB_DSN)
        conn.autocommit = False
        cur = conn.cursor(cursor_factory=DictCursor)

        try:
            # Pick site using waggle dance
            site_id = pick_site(cur, task_type)
            if site_id is None:
                # No sites configured, fall back to simple mode
                log.warning("No sites found, using simple queue")
                task = claim_task(cur, task_type)
            else:
                increment_congestion(cur, site_id)
                task = claim_task_with_site(cur, task_type, site_id, node_id)

            if not task:
                conn.rollback()
                time.sleep(POLL_INTERVAL)
                continue

            task_id = task["id"]
            payload = task["payload"]
            attempt = task["attempt"]

            log.info(f"Claimed task {task_id} (site={site_id}, attempt {attempt})")

            t0 = time.time()
            success = False
            try:
                result = execute_task(task_type, payload)
                duration = time.time() - t0
                success = True
                mark_done(cur, task_id, result)
                log.info(f"Done task {task_id} in {duration:.3f}s")

            except Exception as e:
                duration = time.time() - t0
                log.error(f"Task {task_id} failed: {e}")

                if attempt >= MAX_RETRIES:
                    mark_failed(cur, task_id, str(e))
                    log.warning(f"Task {task_id} permanently failed after {attempt} attempts")
                else:
                    requeue_task(cur, task_id)
                    log.info(f"Task {task_id} requeued for retry")

            # Update site pheromones
            if site_id is not None:
                update_site_metrics(cur, site_id, duration, success)

            conn.commit()

        except Exception as e:
            log.error(f"Worker error: {e}")
            conn.rollback()
            time.sleep(POLL_INTERVAL)

        finally:
            conn.close()


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Ara Hive Bee Agent")
    parser.add_argument("--task-type", default="dummy_cpu",
                        help="Task type to process (default: dummy_cpu)")
    parser.add_argument("--node-id", type=int, default=None,
                        help="Node ID (enables site-aware mode)")
    parser.add_argument("--simple", action="store_true",
                        help="Force simple mode (no sites)")
    args = parser.parse_args()

    if args.simple or args.node_id is None:
        worker_loop_simple(args.task_type)
    else:
        worker_loop_with_sites(args.task_type, args.node_id)


if __name__ == "__main__":
    main()
