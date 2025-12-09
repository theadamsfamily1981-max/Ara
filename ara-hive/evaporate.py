#!/usr/bin/env python3
"""
Ara Hive - Pheromone Evaporation & Maintenance
==============================================

Run periodically (via cron or as daemon) to:
1. Evaporate site intensity for stale sites
2. Mark stale nodes as inactive
3. Cool overloaded sites (congestion relief)

Usage:
    # One-shot (for cron)
    python evaporate.py

    # Continuous daemon
    python evaporate.py --daemon --interval 60

    # Cron entry (every minute):
    * * * * * cd /path/to/ara-hive && /path/to/venv/bin/python evaporate.py >> /var/log/ara-evaporate.log 2>&1
"""

import argparse
import logging
import os
import time

import psycopg2
from psycopg2.extras import DictCursor

DB_DSN = os.getenv("DB_DSN", "dbname=ara_hive user=postgres")

# Evaporation parameters
EVAPORATION_DECAY = 0.95       # Multiply intensity by this
EVAPORATION_AGE_SEC = 30       # Sites not updated in this time evaporate
MIN_INTENSITY = 0.01           # Floor to prevent complete evaporation

# Stale node detection
STALE_NODE_AGE_SEC = 300       # 5 minutes without heartbeat = stale

# Congestion cooling
CONGESTION_THRESHOLD = 5       # Sites with congestion > this get cooled
CONGESTION_COOL_FACTOR = 0.8   # Multiply intensity by this when congested

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("evaporate")


def evaporate_intensity():
    """
    Decay intensity on sites that haven't been updated recently.
    This is the "forgetting" mechanism of the pheromone trail.
    """
    conn = psycopg2.connect(DB_DSN)
    conn.autocommit = True
    cur = conn.cursor()

    cur.execute("""
        UPDATE sites
        SET intensity = GREATEST(intensity * %s, %s)
        WHERE last_update < now() - interval '%s seconds';
    """, (EVAPORATION_DECAY, MIN_INTENSITY, EVAPORATION_AGE_SEC))

    affected = cur.rowcount
    conn.close()

    if affected > 0:
        log.info(f"Evaporated {affected} stale sites (decay={EVAPORATION_DECAY})")

    return affected


def mark_stale_nodes():
    """
    Mark nodes as inactive if they haven't sent a heartbeat recently.
    """
    conn = psycopg2.connect(DB_DSN)
    conn.autocommit = True
    cur = conn.cursor()

    cur.execute("""
        UPDATE nodes
        SET active = false
        WHERE active = true
          AND last_heartbeat < now() - interval '%s seconds';
    """, (STALE_NODE_AGE_SEC,))

    affected = cur.rowcount
    conn.close()

    if affected > 0:
        log.warning(f"Marked {affected} nodes as inactive (no heartbeat > {STALE_NODE_AGE_SEC}s)")

    return affected


def cool_congested_sites():
    """
    Reduce intensity on sites that are overloaded (high congestion).
    This discourages new jobs from piling onto busy sites.
    """
    conn = psycopg2.connect(DB_DSN)
    conn.autocommit = True
    cur = conn.cursor()

    cur.execute("""
        UPDATE sites
        SET intensity = GREATEST(intensity * %s, %s)
        WHERE congestion > %s;
    """, (CONGESTION_COOL_FACTOR, MIN_INTENSITY, CONGESTION_THRESHOLD))

    affected = cur.rowcount
    conn.close()

    if affected > 0:
        log.info(f"Cooled {affected} congested sites (congestion > {CONGESTION_THRESHOLD})")

    return affected


def reset_zombie_tasks():
    """
    Reset tasks that have been 'in_progress' for too long without completion.
    These are likely from crashed workers.
    """
    conn = psycopg2.connect(DB_DSN)
    conn.autocommit = True
    cur = conn.cursor()

    # Tasks stuck in_progress for > 10 minutes are zombies
    cur.execute("""
        UPDATE tasks
        SET status = 'pending',
            started_at = NULL,
            node_id = NULL,
            site_id = NULL
        WHERE status = 'in_progress'
          AND started_at < now() - interval '10 minutes';
    """)

    affected = cur.rowcount
    conn.close()

    if affected > 0:
        log.warning(f"Reset {affected} zombie tasks (stuck in_progress > 10min)")

    return affected


def get_hive_stats():
    """Get current hive statistics for logging."""
    conn = psycopg2.connect(DB_DSN)
    cur = conn.cursor(cursor_factory=DictCursor)

    # Node counts
    cur.execute("""
        SELECT
            COUNT(*) FILTER (WHERE active = true) AS active_nodes,
            COUNT(*) FILTER (WHERE active = false) AS inactive_nodes
        FROM nodes;
    """)
    nodes = cur.fetchone()

    # Task counts
    cur.execute("""
        SELECT
            COUNT(*) FILTER (WHERE status = 'pending') AS pending,
            COUNT(*) FILTER (WHERE status = 'in_progress') AS running,
            COUNT(*) FILTER (WHERE status = 'done') AS done,
            COUNT(*) FILTER (WHERE status = 'failed') AS failed
        FROM tasks;
    """)
    tasks = cur.fetchone()

    # Site stats
    cur.execute("""
        SELECT
            COUNT(*) AS site_count,
            ROUND(AVG(intensity)::numeric, 3) AS avg_intensity,
            ROUND(MAX(intensity)::numeric, 3) AS max_intensity,
            SUM(congestion) AS total_congestion
        FROM sites
        JOIN nodes ON nodes.id = sites.node_id
        WHERE nodes.active = true;
    """)
    sites = cur.fetchone()

    conn.close()

    return {
        "active_nodes": nodes["active_nodes"],
        "inactive_nodes": nodes["inactive_nodes"],
        "pending_tasks": tasks["pending"],
        "running_tasks": tasks["running"],
        "done_tasks": tasks["done"],
        "failed_tasks": tasks["failed"],
        "site_count": sites["site_count"] or 0,
        "avg_intensity": float(sites["avg_intensity"] or 0),
        "max_intensity": float(sites["max_intensity"] or 0),
        "total_congestion": sites["total_congestion"] or 0,
    }


def run_maintenance():
    """Run all maintenance tasks."""
    evaporated = evaporate_intensity()
    stale = mark_stale_nodes()
    cooled = cool_congested_sites()
    zombies = reset_zombie_tasks()

    stats = get_hive_stats()
    log.info(
        f"Hive: nodes={stats['active_nodes']}/{stats['active_nodes']+stats['inactive_nodes']} "
        f"tasks=p:{stats['pending_tasks']}/r:{stats['running_tasks']}/d:{stats['done_tasks']} "
        f"intensity={stats['avg_intensity']:.3f} (max {stats['max_intensity']:.3f})"
    )

    return evaporated + stale + cooled + zombies


def main():
    parser = argparse.ArgumentParser(description="Ara Hive maintenance (evaporation)")
    parser.add_argument("--daemon", action="store_true",
                        help="Run as continuous daemon")
    parser.add_argument("--interval", type=int, default=60,
                        help="Interval in seconds for daemon mode (default: 60)")
    parser.add_argument("--quiet", action="store_true",
                        help="Only log when changes occur")
    args = parser.parse_args()

    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    if args.daemon:
        log.info(f"Starting maintenance daemon (interval={args.interval}s)")
        while True:
            try:
                run_maintenance()
            except Exception as e:
                log.error(f"Maintenance error: {e}")
            time.sleep(args.interval)
    else:
        run_maintenance()


if __name__ == "__main__":
    main()
