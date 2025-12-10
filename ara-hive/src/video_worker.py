#!/usr/bin/env python3
"""
Video Block Worker - GPU node agent for rendering video blocks.

Claims video_block tasks, runs model (stub for now), updates job progress.

Usage:
    export ARA_HIVE_DSN="dbname=ara_hive user=ara password=ara host=<MASTER_IP>"
    python src/video_worker.py

    # With GPU index (for multi-GPU nodes):
    python src/video_worker.py --gpu 0
"""

import argparse
import json
import os
import socket
import subprocess
import time
from pathlib import Path
from typing import Optional

import psycopg2
from psycopg2.extras import DictCursor

DB_DSN = os.getenv(
    "ARA_HIVE_DSN",
    "dbname=ara_hive user=ara password=ara host=127.0.0.1",
)

# Output directory for rendered blocks
OUTPUT_DIR = Path(os.getenv("ARA_VIDEO_OUTPUT", "/var/ara/video_blocks"))

# ABC parameters (same as bee_agent)
ALPHA = 0.3
INTENSITY_BASE = 0.1
INTENSITY_SCALE = 1.0


class VideoWorker:
    def __init__(self, task_type: str = "video_block", gpu_index: int = 0):
        self.task_type = task_type
        self.gpu_index = gpu_index
        self.hostname = socket.gethostname()
        self.node_id = self._get_node_id()

        # Ensure output directory exists
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        print(f"[video_worker] hostname={self.hostname} node_id={self.node_id} gpu={gpu_index}")

    def _get_node_id(self) -> int:
        """Get this node's ID from the database."""
        conn = psycopg2.connect(DB_DSN)
        conn.autocommit = True
        cur = conn.cursor(cursor_factory=DictCursor)
        cur.execute("SELECT id FROM nodes WHERE hostname=%s", (self.hostname,))
        row = cur.fetchone()
        conn.close()
        if not row:
            raise RuntimeError(f"Node {self.hostname} not registered in nodes table")
        return row["id"]

    def _get_gpu_utilization(self) -> float:
        """Get current GPU utilization via nvidia-smi."""
        try:
            result = subprocess.run(
                ["nvidia-smi",
                 f"--id={self.gpu_index}",
                 "--query-gpu=utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return float(result.stdout.strip())
        except Exception:
            pass
        return 0.0

    def _update_node_gpu_load(self):
        """Update this node's GPU load in the database."""
        gpu_load = self._get_gpu_utilization()
        conn = psycopg2.connect(DB_DSN)
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute("""
            UPDATE nodes
            SET gpu_load_pct = %s, last_heartbeat = now()
            WHERE id = %s;
        """, (gpu_load, self.node_id))
        conn.close()

    def _pick_site(self, cur) -> Optional[int]:
        """Pick best site for video_block, weighted by intensity."""
        cur.execute("""
            SELECT s.id, s.intensity
            FROM sites s
            JOIN nodes n ON n.id = s.node_id
            WHERE s.task_type = %s
              AND n.role = 'gpu'
            ORDER BY s.intensity DESC
            LIMIT 5;
        """, (self.task_type,))
        rows = cur.fetchall()

        if not rows:
            return None

        import random
        total = sum(max(float(r["intensity"]), 0.001) for r in rows)
        r = random.random() * total
        acc = 0.0
        for row in rows:
            w = max(float(row["intensity"]), 0.001)
            acc += w
            if acc >= r:
                return row["id"]
        return rows[0]["id"]

    def _claim_task(self, cur, site_id: int):
        """Atomically claim a pending video_block task."""
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
                site_id = %s,
                started_at = now(),
                attempt = attempt + 1,
                updated_at = now()
            FROM next_task
            WHERE tasks.id = next_task.id
            RETURNING tasks.id, tasks.payload;
        """, (self.task_type, site_id))
        return cur.fetchone()

    def _update_site_metrics(self, cur, site_id: int, duration_s: float, success: bool = True):
        """Update site pheromones after completing a block."""
        cur.execute(
            "SELECT q_hat FROM sites WHERE id=%s FOR UPDATE",
            (site_id,),
        )
        row = cur.fetchone()
        if not row:
            return

        q_old = float(row["q_hat"])
        # Reward = frames per second (higher = better)
        reward = (1.0 / max(duration_s, 0.001)) if success else 0.0
        q_new = (1 - ALPHA) * q_old + ALPHA * reward
        intensity_new = INTENSITY_BASE + INTENSITY_SCALE * q_new

        cur.execute("""
            UPDATE sites
            SET q_hat = %s,
                intensity = %s,
                congestion = GREATEST(congestion - 1, 0),
                last_update = now()
            WHERE id = %s;
        """, (q_new, intensity_new, site_id))

    def _update_job_progress(self, cur, job_id: int, artifact_path: str,
                             success: bool = True, error_msg: str = None):
        """Update video job progress after completing a block."""
        if success:
            cur.execute("""
                UPDATE video_jobs
                SET blocks_done = blocks_done + 1,
                    artifacts = artifacts || %s::jsonb,
                    updated_at = now()
                WHERE id = %s
                RETURNING total_blocks, blocks_done;
            """, (json.dumps([artifact_path]), job_id))
            row = cur.fetchone()

            if row and row["blocks_done"] >= row["total_blocks"]:
                cur.execute("""
                    UPDATE video_jobs
                    SET status = 'completed',
                        finished_at = now(),
                        updated_at = now()
                    WHERE id = %s;
                """, (job_id,))
        else:
            cur.execute("""
                UPDATE video_jobs
                SET status = 'failed',
                    error_message = %s,
                    updated_at = now()
                WHERE id = %s;
            """, (error_msg or "unknown error", job_id))

    def _render_block(self, payload: dict) -> str:
        """
        Render a video block. STUB for now - simulates GPU work.

        Replace this with actual model inference:
        - Load model to GPU (cache across calls)
        - Run blockwise diffusion/attention
        - Write frames to disk

        Returns: path to output artifact
        """
        job_id = payload["job_id"]
        block_index = payload["block_index"]
        num_frames = payload["num_frames"]
        model_name = payload.get("model_name", "stub")

        # Simulate GPU work: ~50ms per frame
        time_per_frame = 0.05
        time.sleep(num_frames * time_per_frame)

        # Write stub output (later: actual video/latent tensor)
        out_path = OUTPUT_DIR / f"job_{job_id:08d}_block_{block_index:04d}.json"

        output_data = {
            "job_id": job_id,
            "block_index": block_index,
            "num_frames": num_frames,
            "start_frame": payload.get("start_frame", 0),
            "model_name": model_name,
            "seed": payload.get("seed"),
            "rendered_at": time.time(),
            "gpu_index": self.gpu_index,
            "hostname": self.hostname,
        }

        with open(out_path, "w") as f:
            json.dump(output_data, f, indent=2)

        return str(out_path)

    def run_forever(self):
        """Main worker loop."""
        heartbeat_interval = 30
        last_heartbeat = 0

        while True:
            # Periodic heartbeat
            now = time.time()
            if now - last_heartbeat > heartbeat_interval:
                self._update_node_gpu_load()
                last_heartbeat = now

            conn = psycopg2.connect(DB_DSN)
            conn.autocommit = False
            cur = conn.cursor(cursor_factory=DictCursor)

            try:
                # Pick site using waggle dance
                site_id = self._pick_site(cur)
                if site_id is None:
                    conn.rollback()
                    time.sleep(1.0)
                    continue

                # Increment congestion
                cur.execute(
                    "UPDATE sites SET congestion = congestion + 1 WHERE id=%s",
                    (site_id,),
                )

                # Claim task
                task = self._claim_task(cur, site_id)
                if not task:
                    # No work, revert congestion
                    cur.execute(
                        "UPDATE sites SET congestion = GREATEST(congestion - 1, 0) WHERE id=%s",
                        (site_id,),
                    )
                    conn.commit()
                    time.sleep(1.0)
                    continue

                task_id = task["id"]
                payload = task["payload"]
                if isinstance(payload, str):
                    payload = json.loads(payload)

                job_id = payload["job_id"]
                block_index = payload["block_index"]

                print(f"[video_worker] job={job_id} block={block_index} claiming...")

                # Render the block
                t0 = time.time()
                try:
                    artifact_path = self._render_block(payload)
                    duration = time.time() - t0
                    success = True
                    error_msg = None
                except Exception as e:
                    duration = time.time() - t0
                    success = False
                    error_msg = str(e)
                    artifact_path = ""

                # Update task
                payload["artifact_path"] = artifact_path
                payload["duration_s"] = duration
                payload["success"] = success

                if success:
                    cur.execute("""
                        UPDATE tasks
                        SET status = 'done',
                            payload = %s::jsonb,
                            finished_at = now(),
                            updated_at = now()
                        WHERE id = %s;
                    """, (json.dumps(payload), task_id))
                else:
                    cur.execute("""
                        UPDATE tasks
                        SET status = 'failed',
                            payload = %s::jsonb,
                            finished_at = now(),
                            updated_at = now()
                        WHERE id = %s;
                    """, (json.dumps(payload), task_id))

                # Update site metrics
                self._update_site_metrics(cur, site_id, duration, success)

                # Update job progress
                self._update_job_progress(cur, job_id, artifact_path, success, error_msg)

                conn.commit()

                status = "done" if success else "FAILED"
                print(f"[video_worker] job={job_id} block={block_index} {status} in {duration:.2f}s")

            except Exception as e:
                print(f"[video_worker] error: {e}")
                conn.rollback()
                time.sleep(1.0)
            finally:
                conn.close()


def main():
    parser = argparse.ArgumentParser(description="Ara Video Block Worker")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index (default: 0)")
    args = parser.parse_args()

    worker = VideoWorker(task_type="video_block", gpu_index=args.gpu)
    worker.run_forever()


if __name__ == "__main__":
    main()
