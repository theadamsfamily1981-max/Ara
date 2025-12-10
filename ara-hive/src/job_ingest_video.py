#!/usr/bin/env python3
"""
Video Job Ingest Service
=========================

Accepts video job requests, creates video_jobs row, fans out blocks to tasks.

Usage:
    export ARA_HIVE_DSN="dbname=ara_hive user=ara password=ara host=<MASTER_IP>"
    uvicorn job_ingest_video:app --host 0.0.0.0 --port 8104

Or for testing:
    python job_ingest_video.py
"""

import json
import math
import os
from datetime import datetime
from typing import Optional

import psycopg2
from psycopg2.extras import DictCursor
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

DB_DSN = os.getenv(
    "ARA_HIVE_DSN",
    "dbname=ara_hive user=ara password=ara host=127.0.0.1",
)

app = FastAPI(title="Ara Video Job Ingest", version="0.1.0")


# =============================================================================
# Request Models
# =============================================================================

class VideoJobParams(BaseModel):
    user_id: str = "anon"
    request_id: Optional[str] = None
    model_name: str = "hummingbird_v0"
    duration_s: float = 5.0
    fps: int = 24
    block_frames: int = 16
    style: dict = {}
    script: dict = {}
    seed: int = 1234


class OuterEnvelope(BaseModel):
    """Wire protocol envelope (matches ara_saas)."""
    v: int = 1
    route_to: str
    message_type: str
    priority: float = 0.5
    payload_e2e: dict


class DirectJobRequest(BaseModel):
    """Direct job submission (bypassing router)."""
    params: VideoJobParams


# =============================================================================
# Database Operations
# =============================================================================

def create_video_job(params: VideoJobParams) -> dict:
    """Create video job and fan out blocks to tasks table."""

    total_frames = int(params.fps * params.duration_s)
    total_blocks = math.ceil(total_frames / params.block_frames)

    conn = psycopg2.connect(DB_DSN)
    conn.autocommit = False
    cur = conn.cursor(cursor_factory=DictCursor)

    try:
        # Insert video_jobs row
        cur.execute("""
            INSERT INTO video_jobs (
                user_id, request_id, status, total_blocks,
                model_name, style, script
            )
            VALUES (%s, %s, 'queued', %s, %s, %s::jsonb, %s::jsonb)
            RETURNING id;
        """, (
            params.user_id,
            params.request_id,
            total_blocks,
            params.model_name,
            json.dumps(params.style),
            json.dumps(params.script),
        ))
        job_row = cur.fetchone()
        job_id = job_row["id"]

        # Fan out blocks into tasks
        for block_idx in range(total_blocks):
            start_frame = block_idx * params.block_frames
            frames_this = min(params.block_frames, total_frames - start_frame)

            task_payload = {
                "job_id": job_id,
                "block_index": block_idx,
                "start_frame": start_frame,
                "num_frames": frames_this,
                "fps": params.fps,
                "model_name": params.model_name,
                "style": params.style,
                "script": params.script,
                "seed": params.seed + block_idx,
            }

            cur.execute("""
                INSERT INTO tasks (task_type, payload)
                VALUES ('video_block', %s::jsonb);
            """, (json.dumps(task_payload),))

        # Update job to running
        cur.execute("""
            UPDATE video_jobs
            SET status = 'running', updated_at = now()
            WHERE id = %s;
        """, (job_id,))

        conn.commit()

        return {
            "ok": True,
            "job_id": job_id,
            "total_blocks": total_blocks,
            "total_frames": total_frames,
            "estimated_duration_s": params.duration_s,
        }

    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()


def get_job_status(job_id: int) -> dict:
    """Get current status of a video job."""
    conn = psycopg2.connect(DB_DSN)
    cur = conn.cursor(cursor_factory=DictCursor)

    cur.execute("""
        SELECT id, user_id, status, total_blocks, blocks_done,
               model_name, created_at, updated_at, finished_at, error_message
        FROM video_jobs
        WHERE id = %s;
    """, (job_id,))

    row = cur.fetchone()
    conn.close()

    if not row:
        return None

    return {
        "job_id": row["id"],
        "user_id": row["user_id"],
        "status": row["status"],
        "total_blocks": row["total_blocks"],
        "blocks_done": row["blocks_done"],
        "pct_done": round(100.0 * row["blocks_done"] / max(row["total_blocks"], 1), 1),
        "model_name": row["model_name"],
        "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        "finished_at": row["finished_at"].isoformat() if row["finished_at"] else None,
        "error_message": row["error_message"],
    }


# =============================================================================
# API Endpoints
# =============================================================================

@app.post("/ingest")
async def ingest_envelope(envelope: OuterEnvelope):
    """
    Ingest via router envelope (wire protocol).

    Expected:
        route_to: "service:job_ingest_video"
        message_type: "job_submit"
        payload_e2e: { task_type: "video_job", params: {...} }
    """
    if envelope.message_type != "job_submit":
        raise HTTPException(status_code=400, detail="wrong_message_type")

    payload = envelope.payload_e2e
    if payload.get("task_type") != "video_job":
        raise HTTPException(status_code=400, detail="unsupported_task_type")

    params = VideoJobParams(**payload.get("params", {}))

    try:
        result = create_video_job(params)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/submit")
async def submit_direct(request: DirectJobRequest):
    """Direct job submission (bypassing router)."""
    try:
        result = create_video_job(request.params)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status/{job_id}")
async def job_status(job_id: int):
    """Get job status by ID."""
    status = get_job_status(job_id)
    if not status:
        raise HTTPException(status_code=404, detail="job_not_found")
    return status


@app.get("/queue")
async def queue_status():
    """Get overall video block queue status."""
    conn = psycopg2.connect(DB_DSN)
    cur = conn.cursor(cursor_factory=DictCursor)

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

    cur.execute("""
        SELECT
            COUNT(*) FILTER (WHERE status = 'queued') AS queued,
            COUNT(*) FILTER (WHERE status = 'running') AS running,
            COUNT(*) FILTER (WHERE status = 'completed') AS completed,
            COUNT(*) FILTER (WHERE status = 'failed') AS failed
        FROM video_jobs;
    """)
    jobs = cur.fetchone()

    conn.close()

    return {
        "jobs": dict(jobs),
        "blocks": dict(blocks),
    }


@app.get("/health")
async def health():
    """Health check."""
    try:
        conn = psycopg2.connect(DB_DSN)
        conn.close()
        return {"status": "ok", "db": "connected"}
    except Exception as e:
        return {"status": "error", "db": str(e)}


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    print(f"Starting video job ingest on port 8104...")
    print(f"DB: {DB_DSN}")
    uvicorn.run(app, host="0.0.0.0", port=8104)
