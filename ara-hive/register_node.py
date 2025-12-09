#!/usr/bin/env python3
"""
Ara Hive - Node Registration & Heartbeat
=========================================

Register this machine as a hive node and send periodic heartbeats.

Usage:
    # One-shot registration
    python register_node.py worker

    # Continuous heartbeat (run as service)
    python register_node.py worker --heartbeat

    # With GPU metrics (requires nvidia-smi)
    python register_node.py gpu --heartbeat --gpu
"""

import argparse
import json
import logging
import os
import socket
import subprocess
import time
from typing import Dict, Optional, Tuple

import psutil
import psycopg2
from psycopg2.extras import DictCursor

DB_DSN = os.getenv("DB_DSN", "dbname=ara_hive user=postgres")
HEARTBEAT_INTERVAL = 30  # seconds

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("node")


# =============================================================================
# Metrics Collection
# =============================================================================

def get_cpu_metrics() -> Tuple[float, float]:
    """Get CPU load % and memory used %."""
    cpu = psutil.cpu_percent(interval=0.5)
    mem = psutil.virtual_memory().percent
    return cpu, mem


def get_gpu_metrics() -> Tuple[float, float]:
    """
    Get GPU load % and GPU memory % using nvidia-smi.
    Returns (0, 0) if nvidia-smi not available.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode != 0:
            return 0.0, 0.0

        # Parse first GPU (can extend for multi-GPU)
        line = result.stdout.strip().split("\n")[0]
        parts = [p.strip() for p in line.split(",")]
        gpu_util = float(parts[0])
        mem_used = float(parts[1])
        mem_total = float(parts[2])
        mem_pct = (mem_used / mem_total * 100) if mem_total > 0 else 0

        return gpu_util, mem_pct

    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        return 0.0, 0.0


def get_system_meta() -> Dict:
    """Collect system metadata."""
    return {
        "cpu_count": psutil.cpu_count(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "mem_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "platform": os.uname().sysname,
        "arch": os.uname().machine,
    }


# =============================================================================
# Database Operations
# =============================================================================

def register_node(
    hostname: str,
    role: str,
    cpu_load: float,
    mem_used: float,
    gpu_load: float = 0.0,
    gpu_mem: float = 0.0,
    meta: Dict = None
) -> int:
    """
    Register or update node in database.
    Returns node ID.
    """
    conn = psycopg2.connect(DB_DSN)
    conn.autocommit = True
    cur = conn.cursor(cursor_factory=DictCursor)

    cur.execute("""
        INSERT INTO nodes (hostname, role, cpu_load_pct, mem_used_pct, gpu_load_pct, gpu_mem_pct, meta)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (hostname) DO UPDATE
          SET role = EXCLUDED.role,
              cpu_load_pct = EXCLUDED.cpu_load_pct,
              mem_used_pct = EXCLUDED.mem_used_pct,
              gpu_load_pct = EXCLUDED.gpu_load_pct,
              gpu_mem_pct = EXCLUDED.gpu_mem_pct,
              meta = EXCLUDED.meta,
              last_heartbeat = now(),
              active = true
        RETURNING id;
    """, (hostname, role, cpu_load, mem_used, gpu_load, gpu_mem,
          psycopg2.extras.Json(meta or {})))

    row = cur.fetchone()
    node_id = row["id"]

    conn.close()
    return node_id


def deactivate_node(hostname: str):
    """Mark node as inactive (clean shutdown)."""
    conn = psycopg2.connect(DB_DSN)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("UPDATE nodes SET active = false WHERE hostname = %s", (hostname,))
    conn.close()


def ensure_sites_exist(node_id: int, task_types: list):
    """
    Ensure site entries exist for this node and given task types.
    Creates with default intensity if missing.
    """
    conn = psycopg2.connect(DB_DSN)
    conn.autocommit = True
    cur = conn.cursor()

    for task_type in task_types:
        cur.execute("""
            INSERT INTO sites (task_type, node_id, q_hat, intensity, congestion)
            VALUES (%s, %s, 0.0, 1.0, 0)
            ON CONFLICT (task_type, node_id) DO NOTHING;
        """, (task_type, node_id))

    conn.close()


# =============================================================================
# Main
# =============================================================================

def do_heartbeat(role: str, check_gpu: bool, task_types: list):
    """Single heartbeat: collect metrics and update DB."""
    hostname = socket.gethostname()

    cpu_load, mem_used = get_cpu_metrics()
    gpu_load, gpu_mem = get_gpu_metrics() if check_gpu else (0.0, 0.0)
    meta = get_system_meta()

    node_id = register_node(
        hostname=hostname,
        role=role,
        cpu_load=cpu_load,
        mem_used=mem_used,
        gpu_load=gpu_load,
        gpu_mem=gpu_mem,
        meta=meta
    )

    # Ensure sites exist for this node
    if task_types:
        ensure_sites_exist(node_id, task_types)

    return node_id, hostname, cpu_load, mem_used, gpu_load, gpu_mem


def main():
    parser = argparse.ArgumentParser(description="Register node in Ara Hive")
    parser.add_argument("role", choices=["worker", "gpu", "pi", "master"],
                        help="Node role")
    parser.add_argument("--heartbeat", action="store_true",
                        help="Run continuous heartbeat loop")
    parser.add_argument("--gpu", action="store_true",
                        help="Collect GPU metrics via nvidia-smi")
    parser.add_argument("--task-types", nargs="+", default=["dummy_cpu"],
                        help="Task types this node handles (creates sites)")
    parser.add_argument("--interval", type=int, default=HEARTBEAT_INTERVAL,
                        help=f"Heartbeat interval in seconds (default: {HEARTBEAT_INTERVAL})")
    args = parser.parse_args()

    hostname = socket.gethostname()

    if args.heartbeat:
        log.info(f"Starting heartbeat loop: host={hostname} role={args.role} interval={args.interval}s")
        try:
            while True:
                node_id, _, cpu, mem, gpu, gpu_mem = do_heartbeat(
                    args.role, args.gpu, args.task_types
                )
                log.info(f"Heartbeat: node_id={node_id} cpu={cpu:.1f}% mem={mem:.1f}% gpu={gpu:.1f}%")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            log.info("Shutting down, marking node inactive")
            deactivate_node(hostname)
    else:
        # One-shot registration
        node_id, _, cpu, mem, gpu, gpu_mem = do_heartbeat(
            args.role, args.gpu, args.task_types
        )
        print(f"Registered: node_id={node_id} hostname={hostname} role={args.role}")
        print(f"  CPU: {cpu:.1f}%  MEM: {mem:.1f}%  GPU: {gpu:.1f}%")
        print(f"  Sites created for: {args.task_types}")


if __name__ == "__main__":
    main()
