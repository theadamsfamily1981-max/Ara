#!/usr/bin/env python3
"""
Parts Picker - Hardware allocation worker for the hive.

Processes parts_jobs requests:
1. Receives hardware requirements (VRAM, compute, vendor preference)
2. Queries available hardware inventory
3. Allocates optimal devices
4. Tracks usage and costs

Usage:
    export ARA_HIVE_DSN="dbname=ara_hive user=ara password=ara host=127.0.0.1"
    python src/parts_picker.py
"""

import os
import time
import json
import socket
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

try:
    import psycopg2
    from psycopg2.extras import DictCursor, Json
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False
    print("[parts_picker] Warning: psycopg2 not installed")

DB_DSN = os.getenv(
    "ARA_HIVE_DSN",
    "dbname=ara_hive user=ara password=ara host=127.0.0.1",
)


@dataclass
class AllocationRequest:
    """Request for hardware allocation."""
    job_id: int
    requester: str
    job_type: str
    min_vram_gb: Optional[float] = None
    min_compute: Optional[int] = None
    preferred_vendor: Optional[str] = None
    device_types: List[str] = None
    max_cost_hour: Optional[float] = None
    max_power_watts: Optional[int] = None
    deadline_at: Optional[str] = None

    @classmethod
    def from_row(cls, row: Dict) -> "AllocationRequest":
        return cls(
            job_id=row["id"],
            requester=row["requester"],
            job_type=row["job_type"],
            min_vram_gb=float(row["min_vram_gb"]) if row["min_vram_gb"] else None,
            min_compute=row["min_compute"],
            preferred_vendor=row["preferred_vendor"],
            device_types=row["device_types"] or ["gpu"],
            max_cost_hour=float(row["max_cost_hour"]) if row["max_cost_hour"] else None,
            max_power_watts=row["max_power_watts"],
            deadline_at=str(row["deadline_at"]) if row["deadline_at"] else None,
        )


@dataclass
class AllocationResult:
    """Result of hardware allocation."""
    success: bool
    hardware_ids: List[int]
    total_vram_gb: float = 0.0
    total_compute: int = 0
    estimated_cost_hour: float = 0.0
    estimated_power_watts: int = 0
    error: Optional[str] = None


class PartsPicker:
    """
    Hardware allocation engine.

    Uses a scoring algorithm to pick optimal hardware:
    - Matches requirements (VRAM, compute, vendor)
    - Optimizes for cost efficiency
    - Respects budget constraints
    - Considers current utilization
    """

    def __init__(self):
        self.hostname = socket.gethostname()

    def _score_hardware(self, hw: Dict, req: AllocationRequest) -> float:
        """
        Score a hardware device for a request.

        Higher score = better match.
        """
        score = 100.0

        # Vendor preference bonus
        if req.preferred_vendor and hw["vendor"] == req.preferred_vendor:
            score += 25.0

        # Excess VRAM bonus (can handle larger tasks)
        if req.min_vram_gb and hw["vram_gb"]:
            vram_ratio = hw["vram_gb"] / req.min_vram_gb
            if vram_ratio >= 2.0:
                score += 15.0
            elif vram_ratio >= 1.5:
                score += 10.0

        # Compute power bonus
        if req.min_compute and hw["compute_units"]:
            if hw["compute_units"] >= req.min_compute * 1.5:
                score += 10.0

        # Cost efficiency (lower cost = higher score)
        if hw["cost_per_hour"]:
            score -= float(hw["cost_per_hour"]) * 10.0

        # Prefer less utilized devices
        if hw["utilization_pct"]:
            score -= float(hw["utilization_pct"]) * 0.3

        # Priority bonus from inventory
        if hw["priority"]:
            score += hw["priority"] * 5.0

        # Penalty for high temperature
        if hw["temp_celsius"] and hw["temp_celsius"] > 80:
            score -= 20.0

        return score

    def find_hardware(self, cur, req: AllocationRequest) -> List[Tuple[int, Dict, float]]:
        """
        Find available hardware matching requirements.

        Returns list of (hardware_id, hardware_info, score).
        """
        # Build query
        conditions = [
            "hi.available = true",
            "hi.current_task_id IS NULL",
            "n.last_heartbeat > now() - INTERVAL '5 minutes'",
        ]
        params = []

        # Device type filter
        if req.device_types:
            conditions.append("hi.device_type = ANY(%s)")
            params.append(req.device_types)

        # VRAM requirement
        if req.min_vram_gb:
            conditions.append("(hi.vram_gb IS NULL OR hi.vram_gb >= %s)")
            params.append(req.min_vram_gb)

        # Compute requirement
        if req.min_compute:
            conditions.append("(hi.compute_units IS NULL OR hi.compute_units >= %s)")
            params.append(req.min_compute)

        # Cost constraint
        if req.max_cost_hour:
            conditions.append("(hi.cost_per_hour IS NULL OR hi.cost_per_hour <= %s)")
            params.append(req.max_cost_hour)

        # Power constraint
        if req.max_power_watts:
            conditions.append("(hi.tdp_watts IS NULL OR hi.tdp_watts <= %s)")
            params.append(req.max_power_watts)

        query = f"""
            SELECT
                hi.id, hi.device_type, hi.device_name, hi.vendor,
                hi.vram_gb, hi.compute_units, hi.clock_mhz, hi.tdp_watts,
                hi.cost_per_hour, hi.priority, hi.utilization_pct,
                hi.temp_celsius, hi.power_watts,
                n.hostname
            FROM hardware_inventory hi
            JOIN nodes n ON hi.node_id = n.id
            WHERE {' AND '.join(conditions)}
            ORDER BY hi.priority DESC, hi.vram_gb DESC NULLS LAST
            LIMIT 20
        """

        cur.execute(query, params)
        rows = cur.fetchall()

        # Score each device
        results = []
        for row in rows:
            hw = dict(row)
            score = self._score_hardware(hw, req)
            results.append((hw["id"], hw, score))

        # Sort by score descending
        results.sort(key=lambda x: x[2], reverse=True)
        return results

    def allocate(self, cur, req: AllocationRequest) -> AllocationResult:
        """
        Allocate hardware for a request.

        For now: allocates single best device.
        Future: multi-device allocation for large jobs.
        """
        candidates = self.find_hardware(cur, req)

        if not candidates:
            return AllocationResult(
                success=False,
                hardware_ids=[],
                error="No suitable hardware available"
            )

        # Pick best candidate
        hw_id, hw_info, score = candidates[0]

        # Mark as allocated
        cur.execute("""
            UPDATE hardware_inventory
            SET available = false,
                current_task_id = (SELECT parent_task_id FROM parts_jobs WHERE id = %s)
            WHERE id = %s
            RETURNING id
        """, (req.job_id, hw_id))

        if not cur.fetchone():
            return AllocationResult(
                success=False,
                hardware_ids=[],
                error="Failed to allocate hardware (race condition?)"
            )

        # Log allocation
        cur.execute("""
            INSERT INTO hardware_usage_log (hardware_id, parts_job_id, started_at)
            VALUES (%s, %s, now())
        """, (hw_id, req.job_id))

        return AllocationResult(
            success=True,
            hardware_ids=[hw_id],
            total_vram_gb=float(hw_info["vram_gb"]) if hw_info["vram_gb"] else 0,
            total_compute=hw_info["compute_units"] or 0,
            estimated_cost_hour=float(hw_info["cost_per_hour"]) if hw_info["cost_per_hour"] else 0,
            estimated_power_watts=hw_info["tdp_watts"] or 0,
        )

    def release(self, cur, job_id: int):
        """Release hardware allocated to a job."""
        # Get allocated hardware
        cur.execute("""
            SELECT allocated_hw FROM parts_jobs WHERE id = %s
        """, (job_id,))
        row = cur.fetchone()

        if not row or not row["allocated_hw"]:
            return

        hw_ids = row["allocated_hw"]

        # Mark hardware as available
        for hw_id in hw_ids:
            cur.execute("""
                UPDATE hardware_inventory
                SET available = true, current_task_id = NULL
                WHERE id = %s
            """, (hw_id,))

            # Close usage log
            cur.execute("""
                UPDATE hardware_usage_log
                SET ended_at = now()
                WHERE hardware_id = %s AND parts_job_id = %s AND ended_at IS NULL
            """, (hw_id, job_id))

        # Update job status
        cur.execute("""
            UPDATE parts_jobs
            SET status = 'released', released_at = now(), updated_at = now()
            WHERE id = %s
        """, (job_id,))

    def process_pending_jobs(self):
        """Process all pending parts_jobs."""
        if not HAS_PSYCOPG2:
            return

        conn = psycopg2.connect(DB_DSN)
        conn.autocommit = False
        cur = conn.cursor(cursor_factory=DictCursor)

        try:
            # Claim pending jobs
            cur.execute("""
                SELECT * FROM parts_jobs
                WHERE status = 'pending'
                ORDER BY created_at
                LIMIT 10
                FOR UPDATE SKIP LOCKED
            """)

            jobs = cur.fetchall()
            if not jobs:
                conn.rollback()
                return

            for job_row in jobs:
                req = AllocationRequest.from_row(job_row)
                print(f"[parts_picker] Processing job {req.job_id}: {req.requester}/{req.job_type}")

                # Mark as picking
                cur.execute("""
                    UPDATE parts_jobs SET status = 'picking', updated_at = now()
                    WHERE id = %s
                """, (req.job_id,))

                # Attempt allocation
                result = self.allocate(cur, req)

                if result.success:
                    # Update job with allocation
                    cur.execute("""
                        UPDATE parts_jobs
                        SET status = 'allocated',
                            allocated_hw = %s,
                            allocation_at = now(),
                            updated_at = now()
                        WHERE id = %s
                    """, (result.hardware_ids, req.job_id))

                    print(f"[parts_picker] Allocated hw={result.hardware_ids} "
                          f"vram={result.total_vram_gb}GB cost=${result.estimated_cost_hour}/hr")
                else:
                    # Mark as failed
                    cur.execute("""
                        UPDATE parts_jobs
                        SET status = 'failed',
                            payload = payload || %s,
                            updated_at = now()
                        WHERE id = %s
                    """, (Json({"error": result.error}), req.job_id))

                    print(f"[parts_picker] Failed: {result.error}")

            conn.commit()

        except Exception as e:
            print(f"[parts_picker] Error: {e}")
            conn.rollback()
        finally:
            conn.close()

    def run_forever(self, poll_interval: float = 1.0):
        """Main worker loop."""
        print(f"[parts_picker] Starting on {self.hostname}")
        while True:
            try:
                self.process_pending_jobs()
            except Exception as e:
                print(f"[parts_picker] Loop error: {e}")
            time.sleep(poll_interval)


# =============================================================================
# API Functions for direct use
# =============================================================================

def request_hardware(
    requester: str,
    job_type: str,
    min_vram_gb: Optional[float] = None,
    device_types: List[str] = None,
    preferred_vendor: Optional[str] = None,
    max_cost_hour: Optional[float] = None,
    parent_task_id: Optional[int] = None,
) -> int:
    """
    Submit a hardware allocation request.

    Returns the parts_job ID.
    """
    if not HAS_PSYCOPG2:
        raise RuntimeError("psycopg2 not available")

    conn = psycopg2.connect(DB_DSN)
    conn.autocommit = True
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO parts_jobs (
            requester, job_type, min_vram_gb, device_types,
            preferred_vendor, max_cost_hour, parent_task_id
        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
        RETURNING id
    """, (
        requester, job_type, min_vram_gb, device_types or ['gpu'],
        preferred_vendor, max_cost_hour, parent_task_id
    ))

    job_id = cur.fetchone()[0]
    conn.close()

    return job_id


def get_allocation(job_id: int, timeout_s: float = 30.0) -> Optional[Dict]:
    """
    Wait for allocation to complete and return result.

    Returns dict with hardware info or None if failed/timeout.
    """
    if not HAS_PSYCOPG2:
        return None

    start = time.time()
    while time.time() - start < timeout_s:
        conn = psycopg2.connect(DB_DSN)
        conn.autocommit = True
        cur = conn.cursor(cursor_factory=DictCursor)

        cur.execute("""
            SELECT status, allocated_hw, payload
            FROM parts_jobs WHERE id = %s
        """, (job_id,))
        row = cur.fetchone()
        conn.close()

        if not row:
            return None

        if row["status"] == "allocated":
            return {
                "status": "allocated",
                "hardware_ids": row["allocated_hw"],
            }
        elif row["status"] == "failed":
            return {
                "status": "failed",
                "error": row["payload"].get("error") if row["payload"] else "Unknown error",
            }

        time.sleep(0.5)

    return {"status": "timeout"}


def release_hardware(job_id: int):
    """Release hardware allocated to a job."""
    if not HAS_PSYCOPG2:
        return

    conn = psycopg2.connect(DB_DSN)
    conn.autocommit = False
    cur = conn.cursor(cursor_factory=DictCursor)

    try:
        picker = PartsPicker()
        picker.release(cur, job_id)
        conn.commit()
    finally:
        conn.close()


def main():
    picker = PartsPicker()
    picker.run_forever()


if __name__ == "__main__":
    main()
