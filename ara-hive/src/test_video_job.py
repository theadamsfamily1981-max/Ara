#!/usr/bin/env python3
"""
Test script: Submit a video job directly to Postgres.

Usage:
    export ARA_HIVE_DSN="dbname=ara_hive user=ara password=ara host=127.0.0.1"
    python src/test_video_job.py

This bypasses the HTTP API and directly inserts a job + blocks.
"""

import json
import math
import os
import psycopg2
from psycopg2.extras import DictCursor

DB_DSN = os.getenv(
    "ARA_HIVE_DSN",
    "dbname=ara_hive user=ara password=ara host=127.0.0.1",
)


def submit_test_job():
    """Submit a test video job with 8 blocks."""

    # Job params
    user_id = "test_user"
    model_name = "hummingbird_v0"
    duration_s = 4.0
    fps = 24
    block_frames = 12
    seed = 42

    total_frames = int(fps * duration_s)
    total_blocks = math.ceil(total_frames / block_frames)

    print(f"Submitting test video job:")
    print(f"  Duration: {duration_s}s @ {fps}fps = {total_frames} frames")
    print(f"  Block size: {block_frames} frames")
    print(f"  Total blocks: {total_blocks}")
    print()

    conn = psycopg2.connect(DB_DSN)
    conn.autocommit = False
    cur = conn.cursor(cursor_factory=DictCursor)

    try:
        # Insert job
        cur.execute("""
            INSERT INTO video_jobs (
                user_id, status, total_blocks, model_name, style, script
            )
            VALUES (%s, 'queued', %s, %s, '{}'::jsonb, '{}'::jsonb)
            RETURNING id;
        """, (user_id, total_blocks, model_name))

        job_id = cur.fetchone()["id"]
        print(f"Created job_id: {job_id}")

        # Insert blocks
        for block_idx in range(total_blocks):
            start_frame = block_idx * block_frames
            frames_this = min(block_frames, total_frames - start_frame)

            payload = {
                "job_id": job_id,
                "block_index": block_idx,
                "start_frame": start_frame,
                "num_frames": frames_this,
                "fps": fps,
                "model_name": model_name,
                "style": {},
                "script": {},
                "seed": seed + block_idx,
            }

            cur.execute("""
                INSERT INTO tasks (task_type, payload)
                VALUES ('video_block', %s::jsonb);
            """, (json.dumps(payload),))

        # Update job to running
        cur.execute("""
            UPDATE video_jobs
            SET status = 'running', updated_at = now()
            WHERE id = %s;
        """, (job_id,))

        conn.commit()
        print(f"Created {total_blocks} video_block tasks")

    except Exception as e:
        conn.rollback()
        print(f"Error: {e}")
        raise
    finally:
        conn.close()

    return job_id


def show_status():
    """Show current job and block status."""
    conn = psycopg2.connect(DB_DSN)
    cur = conn.cursor(cursor_factory=DictCursor)

    # Jobs
    cur.execute("""
        SELECT id, status, total_blocks, blocks_done, model_name
        FROM video_jobs
        ORDER BY id DESC
        LIMIT 5;
    """)
    jobs = cur.fetchall()

    print("\nVideo Jobs:")
    print("-" * 70)
    for j in jobs:
        pct = round(100 * j["blocks_done"] / max(j["total_blocks"], 1), 1)
        print(f"  job {j['id']}: {j['status']:<10} {j['blocks_done']}/{j['total_blocks']} ({pct}%) - {j['model_name']}")

    # Blocks
    cur.execute("""
        SELECT
            COUNT(*) FILTER (WHERE status = 'pending') AS pending,
            COUNT(*) FILTER (WHERE status = 'in_progress') AS running,
            COUNT(*) FILTER (WHERE status = 'done') AS done,
            COUNT(*) FILTER (WHERE status = 'failed') AS failed
        FROM tasks
        WHERE task_type = 'video_block';
    """)
    blocks = cur.fetchone()

    print("\nVideo Blocks:")
    print(f"  pending: {blocks['pending']}")
    print(f"  running: {blocks['running']}")
    print(f"  done:    {blocks['done']}")
    print(f"  failed:  {blocks['failed']}")

    conn.close()


if __name__ == "__main__":
    job_id = submit_test_job()
    show_status()
    print()
    print("Now run: python src/video_worker.py")
    print("And watch: python src/watch.py")
