#!/usr/bin/env python3
"""
Pheromone evaporation - decay intensity on stale sites.

Usage (one-shot for cron):
    python src/evaporate_sites.py

Cron entry (every minute):
    * * * * * cd /path/to/ara-hive && ARA_HIVE_DSN="..." venv/bin/python src/evaporate_sites.py
"""

import os
import psycopg2

DB_DSN = os.getenv(
    "ARA_HIVE_DSN",
    "dbname=ara_hive user=ara password=ara host=127.0.0.1",
)


def main():
    conn = psycopg2.connect(DB_DSN)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE sites
        SET intensity = intensity * 0.95
        WHERE last_update < now() - interval '30 seconds';
        """
    )
    updated = cur.rowcount
    conn.close()
    print(f"[evaporate_sites] updated {updated} rows")


if __name__ == "__main__":
    main()
