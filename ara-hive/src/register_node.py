#!/usr/bin/env python3
"""
Register this machine as a hive node.

Usage:
    export ARA_HIVE_DSN="dbname=ara_hive user=ara password=ara host=<MASTER_IP>"
    python src/register_node.py --role worker
"""

import socket
import os
import argparse
import psutil
import psycopg2
from psycopg2.extras import DictCursor

DB_DSN = os.getenv(
    "ARA_HIVE_DSN",
    "dbname=ara_hive user=ara password=ara host=127.0.0.1",
)


def get_node_metrics():
    hostname = socket.gethostname()
    cpu = psutil.cpu_percent(interval=0.5)
    mem = psutil.virtual_memory().percent
    # TODO: later use nvidia-smi / rocm-smi
    gpu = 0.0
    return hostname, cpu, mem, gpu


def main(role: str):
    hostname, cpu, mem, gpu = get_node_metrics()
    conn = psycopg2.connect(DB_DSN)
    conn.autocommit = True
    cur = conn.cursor(cursor_factory=DictCursor)

    cur.execute(
        """
        INSERT INTO nodes (hostname, role, cpu_load_pct, mem_used_pct, gpu_load_pct)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (hostname) DO UPDATE
          SET role = EXCLUDED.role,
              cpu_load_pct = EXCLUDED.cpu_load_pct,
              mem_used_pct = EXCLUDED.mem_used_pct,
              gpu_load_pct = EXCLUDED.gpu_load_pct,
              last_heartbeat = now()
        RETURNING id;
        """,
        (hostname, role, cpu, mem, gpu),
    )

    row = cur.fetchone()
    print(f"[register_node] {hostname=} id={row['id']} role={role}")

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", default="worker")
    args = parser.parse_args()
    main(args.role)
