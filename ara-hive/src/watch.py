#!/usr/bin/env python3
"""
Live Waggle Board Monitor - watch the hive in action.

Usage:
    python src/watch.py
    python src/watch.py --interval 2
"""

import os
import sys
import time
import argparse
import psycopg2
from psycopg2.extras import DictCursor

DB_DSN = os.getenv(
    "ARA_HIVE_DSN",
    "dbname=ara_hive user=ara password=ara host=127.0.0.1",
)


def clear():
    os.system('clear' if os.name == 'posix' else 'cls')


def get_stats(cur):
    # Queue stats
    cur.execute("""
        SELECT
            COUNT(*) FILTER (WHERE status = 'pending') AS pending,
            COUNT(*) FILTER (WHERE status = 'in_progress') AS running,
            COUNT(*) FILTER (WHERE status = 'done') AS done,
            COUNT(*) FILTER (WHERE status = 'failed') AS failed
        FROM tasks;
    """)
    tasks = cur.fetchone()

    # Node stats
    cur.execute("""
        SELECT COUNT(*) AS total,
               COUNT(*) FILTER (WHERE last_heartbeat > now() - interval '60 seconds') AS alive
        FROM nodes;
    """)
    nodes = cur.fetchone()

    # Waggle board
    cur.execute("""
        SELECT s.task_type, n.hostname, s.q_hat, s.intensity, s.congestion
        FROM sites s
        JOIN nodes n ON n.id = s.node_id
        ORDER BY s.intensity DESC
        LIMIT 10;
    """)
    sites = cur.fetchall()

    # Recent completions
    cur.execute("""
        SELECT id, task_type,
               EXTRACT(EPOCH FROM (finished_at - started_at))::numeric(6,3) AS duration
        FROM tasks
        WHERE status = 'done'
        ORDER BY finished_at DESC
        LIMIT 5;
    """)
    recent = cur.fetchall()

    return tasks, nodes, sites, recent


def display(tasks, nodes, sites, recent):
    clear()
    print("=" * 60)
    print("                    ARA HIVE MONITOR")
    print("=" * 60)
    print()

    # Queue
    print(f"QUEUE: {tasks['pending']} pending | {tasks['running']} running | {tasks['done']} done | {tasks['failed']} failed")
    print(f"NODES: {nodes['alive']}/{nodes['total']} alive")
    print()

    # Waggle board
    print("WAGGLE BOARD (top 10 sites by intensity):")
    print("-" * 60)
    print(f"{'Task Type':<15} {'Hostname':<15} {'q_hat':>8} {'Intensity':>10} {'Cong':>5}")
    print("-" * 60)
    for s in sites:
        print(f"{s['task_type']:<15} {s['hostname']:<15} {float(s['q_hat']):>8.3f} {float(s['intensity']):>10.3f} {s['congestion']:>5}")
    print()

    # Recent
    if recent:
        print("RECENT COMPLETIONS:")
        for r in recent:
            dur = float(r['duration']) if r['duration'] else 0
            print(f"  task {r['id']}: {r['task_type']} in {dur:.3f}s")
    print()
    print("[Ctrl+C to exit]")


def main():
    parser = argparse.ArgumentParser(description="Live hive monitor")
    parser.add_argument("--interval", type=float, default=1.0, help="Refresh interval (seconds)")
    args = parser.parse_args()

    print("Connecting to hive...")
    try:
        while True:
            conn = psycopg2.connect(DB_DSN)
            cur = conn.cursor(cursor_factory=DictCursor)
            tasks, nodes, sites, recent = get_stats(cur)
            conn.close()
            display(tasks, nodes, sites, recent)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nExiting.")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
