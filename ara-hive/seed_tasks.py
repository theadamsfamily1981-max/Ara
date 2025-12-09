#!/usr/bin/env python3
"""
Ara Hive - Task Seeder
======================

Seed the task queue with test tasks.

Usage:
    # Seed 20 dummy_cpu tasks
    python seed_tasks.py dummy_cpu 20

    # Seed with custom work_ms
    python seed_tasks.py dummy_cpu 50 --work-ms 1000

    # Continuous task injection (for load testing)
    python seed_tasks.py dummy_cpu 100 --continuous --rate 5
"""

import argparse
import json
import os
import random
import time

import psycopg2

DB_DSN = os.getenv("DB_DSN", "dbname=ara_hive user=postgres")


def seed_tasks(task_type: str, count: int, work_ms: int = None, randomize: bool = False):
    """Seed N tasks of given type."""
    conn = psycopg2.connect(DB_DSN)
    conn.autocommit = True
    cur = conn.cursor()

    for i in range(count):
        if work_ms is not None:
            ms = work_ms
        elif randomize:
            ms = random.randint(100, 2000)
        else:
            ms = 500

        payload = {"work_ms": ms}
        cur.execute("""
            INSERT INTO tasks (task_type, payload)
            VALUES (%s, %s);
        """, (task_type, json.dumps(payload)))

    conn.close()
    print(f"Seeded {count} tasks of type '{task_type}'")


def continuous_seed(task_type: str, total: int, rate: float, work_ms: int = None):
    """Continuously inject tasks at a given rate."""
    conn = psycopg2.connect(DB_DSN)
    conn.autocommit = True
    cur = conn.cursor()

    interval = 1.0 / rate if rate > 0 else 1.0
    injected = 0

    print(f"Injecting {total} tasks at {rate}/sec...")

    try:
        while injected < total:
            ms = work_ms if work_ms else random.randint(100, 1500)
            payload = {"work_ms": ms}
            cur.execute("""
                INSERT INTO tasks (task_type, payload)
                VALUES (%s, %s);
            """, (task_type, json.dumps(payload)))
            injected += 1

            if injected % 10 == 0:
                print(f"  Injected {injected}/{total}")

            time.sleep(interval)

    except KeyboardInterrupt:
        print(f"\nStopped. Injected {injected} tasks.")

    conn.close()


def show_queue_status():
    """Show current queue status."""
    conn = psycopg2.connect(DB_DSN)
    cur = conn.cursor()

    cur.execute("""
        SELECT task_type,
               COUNT(*) FILTER (WHERE status = 'pending') AS pending,
               COUNT(*) FILTER (WHERE status = 'in_progress') AS running,
               COUNT(*) FILTER (WHERE status = 'done') AS done,
               COUNT(*) FILTER (WHERE status = 'failed') AS failed
        FROM tasks
        GROUP BY task_type;
    """)

    rows = cur.fetchall()
    conn.close()

    print("\nQueue Status:")
    print("-" * 60)
    print(f"{'Task Type':<20} {'Pending':>10} {'Running':>10} {'Done':>10} {'Failed':>10}")
    print("-" * 60)
    for row in rows:
        print(f"{row[0]:<20} {row[1]:>10} {row[2]:>10} {row[3]:>10} {row[4]:>10}")


def main():
    parser = argparse.ArgumentParser(description="Seed tasks into Ara Hive")
    parser.add_argument("task_type", nargs="?", default="dummy_cpu",
                        help="Task type (default: dummy_cpu)")
    parser.add_argument("count", type=int, nargs="?", default=20,
                        help="Number of tasks to seed (default: 20)")
    parser.add_argument("--work-ms", type=int, default=None,
                        help="Work duration in ms (default: 500 or random)")
    parser.add_argument("--randomize", action="store_true",
                        help="Randomize work_ms between 100-2000")
    parser.add_argument("--continuous", action="store_true",
                        help="Continuously inject tasks")
    parser.add_argument("--rate", type=float, default=1.0,
                        help="Tasks per second in continuous mode (default: 1)")
    parser.add_argument("--status", action="store_true",
                        help="Just show queue status")
    args = parser.parse_args()

    if args.status:
        show_queue_status()
        return

    if args.continuous:
        continuous_seed(args.task_type, args.count, args.rate, args.work_ms)
    else:
        seed_tasks(args.task_type, args.count, args.work_ms, args.randomize)

    show_queue_status()


if __name__ == "__main__":
    main()
