#!/usr/bin/env python3
"""
Bee Agent - Worker that claims and executes tasks.

Usage:
    export ARA_HIVE_DSN="dbname=ara_hive user=ara password=ara host=<MASTER_IP>"
    python src/register_node.py --role worker
    python src/bee_agent.py
"""

import time
import random
import os
import socket
import psycopg2
from psycopg2.extras import DictCursor

DB_DSN = os.getenv(
    "ARA_HIVE_DSN",
    "dbname=ara_hive user=ara password=ara host=127.0.0.1",
)

ALPHA = 0.3        # smoothing for q_hat
INTENSITY_BASE = 0.1
INTENSITY_SCALE = 1.0


class BeeAgent:
    def __init__(self, task_type: str = "dummy_cpu"):
        self.task_type = task_type
        self.hostname = socket.gethostname()
        self.node_id = self._get_node_id()

    def _get_node_id(self) -> int:
        conn = psycopg2.connect(DB_DSN)
        conn.autocommit = True
        cur = conn.cursor(cursor_factory=DictCursor)
        cur.execute("SELECT id FROM nodes WHERE hostname=%s", (self.hostname,))
        row = cur.fetchone()
        conn.close()
        if not row:
            raise RuntimeError(f"Node {self.hostname} not registered in nodes table")
        return row["id"]

    # --- Site selection ---

    def _pick_site(self, cur) -> int | None:
        """
        Pick a site for this task_type, weighted by intensity.
        For now it ignores node_id; later you can prefer local sites.
        """
        cur.execute(
            """
            SELECT id, intensity
            FROM sites
            WHERE task_type = %s
            ORDER BY intensity DESC
            LIMIT 5;
            """,
            (self.task_type,),
        )
        rows = cur.fetchall()
        if not rows:
            return None

        total = sum(max(float(r["intensity"]), 0.001) for r in rows)
        r = random.random() * total
        acc = 0.0
        for row in rows:
            w = max(float(row["intensity"]), 0.001)
            acc += w
            if acc >= r:
                return row["id"]
        return rows[0]["id"]

    # --- Task claim ---

    def _claim_task(self, cur, site_id: int):
        cur.execute(
            """
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
            """,
            (self.task_type, site_id),
        )
        return cur.fetchone()

    # --- Site metric update ---

    def _update_site_metrics(self, cur, site_id: int, duration_s: float, success: bool = True):
        cur.execute(
            "SELECT q_hat, intensity, congestion FROM sites WHERE id=%s FOR UPDATE",
            (site_id,),
        )
        row = cur.fetchone()
        if not row:
            return

        q_old = float(row["q_hat"])
        reward = 1.0 / max(duration_s, 0.001) if success else 0.0
        q_new = (1 - ALPHA) * q_old + ALPHA * reward
        intensity_new = INTENSITY_BASE + INTENSITY_SCALE * q_new

        cur.execute(
            """
            UPDATE sites
            SET q_hat = %s,
                intensity = %s,
                congestion = GREATEST(congestion - 1, 0),
                last_update = now()
            WHERE id = %s;
            """,
            (q_new, intensity_new, site_id),
        )

    # --- Main loop ---

    def run_forever(self):
        print(f"[bee_agent] hostname={self.hostname} node_id={self.node_id} task_type={self.task_type}")
        while True:
            conn = psycopg2.connect(DB_DSN)
            conn.autocommit = False
            cur = conn.cursor(cursor_factory=DictCursor)
            try:
                site_id = self._pick_site(cur)
                if site_id is None:
                    conn.rollback()
                    time.sleep(1)
                    continue

                # increment congestion
                cur.execute(
                    "UPDATE sites SET congestion = congestion + 1 WHERE id=%s",
                    (site_id,),
                )

                task = self._claim_task(cur, site_id)
                if not task:
                    # no work, revert congestion bump
                    cur.execute(
                        "UPDATE sites SET congestion = GREATEST(congestion - 1, 0) WHERE id=%s",
                        (site_id,),
                    )
                    conn.commit()
                    time.sleep(1)
                    continue

                task_id = task["id"]
                payload = task["payload"] or {}
                work_ms = int(payload.get("work_ms", 500))

                print(f"[bee_agent] node={self.hostname} task_id={task_id} work_ms={work_ms}")

                t0 = time.time()
                time.sleep(work_ms / 1000.0)  # simulate work
                duration = time.time() - t0

                # mark task done
                cur.execute(
                    """
                    UPDATE tasks
                    SET status = 'done',
                        finished_at = now(),
                        updated_at = now()
                    WHERE id = %s;
                    """,
                    (task_id,),
                )

                self._update_site_metrics(cur, site_id, duration_s=duration, success=True)
                conn.commit()

            except Exception as e:
                print("[bee_agent] worker error:", e)
                conn.rollback()
                time.sleep(1)
            finally:
                conn.close()


def main():
    agent = BeeAgent(task_type="dummy_cpu")
    agent.run_forever()


if __name__ == "__main__":
    main()
