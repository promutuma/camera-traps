"""
Project Configuration Manager
Supports multi-project surveys: metadata, indicator thresholds,
baseline locking, and JSON export.
"""

import sqlite3
import json
import pandas as pd
from datetime import datetime
from typing import Optional


DEFAULT_THRESHOLDS = {
    "min_trap_nights":        20,
    "min_functionality_pct":  70.0,
    "confidence_threshold":   0.75,
    "independence_window_min": 30,
    "rai_alert_below":        0.5,
    "min_ide_for_richness":   1,
}


class ProjectConfig:
    """
    SQLite-backed multi-project configuration store.

    Tables
    ------
    projects      — one row per survey project
    project_thresholds — key/value indicator settings per project
    project_baselines  — JSON snapshot locked as the reference baseline
    """

    def __init__(self, db_path: str = "wildlife_data.db"):
        self.db_path = db_path
        self._init_tables()

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def _init_tables(self):
        conn = self._conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                name         TEXT NOT NULL UNIQUE,
                survey_area  TEXT,
                lead_org     TEXT,
                start_date   TEXT,
                end_date     TEXT,
                notes        TEXT,
                is_active    INTEGER DEFAULT 0,
                created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS project_thresholds (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id   INTEGER NOT NULL,
                key          TEXT NOT NULL,
                value        TEXT NOT NULL,
                UNIQUE(project_id, key)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS project_baselines (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id   INTEGER NOT NULL UNIQUE,
                baseline_json TEXT NOT NULL,
                locked_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()

    # ------------------------------------------------------------------
    # Projects CRUD
    # ------------------------------------------------------------------

    def create_project(
        self,
        name: str,
        survey_area: str = "",
        lead_org: str = "",
        start_date: str = "",
        end_date: str = "",
        notes: str = "",
    ) -> Optional[int]:
        """Create a new project. Returns project id, or None if name exists."""
        conn = self._conn()
        try:
            cur = conn.execute("""
                INSERT INTO projects (name, survey_area, lead_org, start_date, end_date, notes)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (name.strip(), survey_area, lead_org, start_date, end_date, notes))
            project_id = cur.lastrowid
            conn.commit()
            # Seed default thresholds
            self._seed_thresholds(project_id, conn)
            conn.commit()
            return project_id
        except sqlite3.IntegrityError:
            return None
        finally:
            conn.close()

    def get_projects(self) -> pd.DataFrame:
        conn = self._conn()
        try:
            return pd.read_sql_query(
                "SELECT * FROM projects ORDER BY created_at DESC", conn
            )
        finally:
            conn.close()

    def update_project(self, project_id: int, **kwargs) -> bool:
        allowed = {"name", "survey_area", "lead_org", "start_date", "end_date", "notes"}
        fields = {k: v for k, v in kwargs.items() if k in allowed}
        if not fields:
            return False
        cols = ", ".join(f"{k} = ?" for k in fields)
        conn = self._conn()
        try:
            conn.execute(
                f"UPDATE projects SET {cols} WHERE id = ?",
                list(fields.values()) + [project_id],
            )
            conn.commit()
            return True
        finally:
            conn.close()

    def delete_project(self, project_id: int) -> bool:
        conn = self._conn()
        try:
            conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
            conn.execute("DELETE FROM project_thresholds WHERE project_id = ?", (project_id,))
            conn.execute("DELETE FROM project_baselines WHERE project_id = ?", (project_id,))
            conn.commit()
            return True
        finally:
            conn.close()

    def set_active_project(self, project_id: int) -> bool:
        conn = self._conn()
        try:
            conn.execute("UPDATE projects SET is_active = 0")
            conn.execute("UPDATE projects SET is_active = 1 WHERE id = ?", (project_id,))
            conn.commit()
            return True
        finally:
            conn.close()

    def get_active_project(self) -> Optional[dict]:
        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT * FROM projects WHERE is_active = 1 LIMIT 1"
            ).fetchone()
            if not row:
                return None
            cols = [d[0] for d in conn.execute("PRAGMA table_info(projects)").fetchall()]
            return dict(zip(cols, row))
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Thresholds
    # ------------------------------------------------------------------

    def _seed_thresholds(self, project_id: int, conn: sqlite3.Connection):
        for k, v in DEFAULT_THRESHOLDS.items():
            conn.execute("""
                INSERT OR IGNORE INTO project_thresholds (project_id, key, value)
                VALUES (?, ?, ?)
            """, (project_id, k, str(v)))

    def get_thresholds(self, project_id: int) -> dict:
        """Return threshold dict for a project, falling back to defaults."""
        conn = self._conn()
        try:
            rows = conn.execute(
                "SELECT key, value FROM project_thresholds WHERE project_id = ?",
                (project_id,),
            ).fetchall()
            result = dict(DEFAULT_THRESHOLDS)
            for k, v in rows:
                try:
                    result[k] = float(v) if "." in v else int(v)
                except (ValueError, TypeError):
                    result[k] = v
            return result
        finally:
            conn.close()

    def save_thresholds(self, project_id: int, thresholds: dict) -> bool:
        conn = self._conn()
        try:
            for k, v in thresholds.items():
                conn.execute("""
                    INSERT INTO project_thresholds (project_id, key, value)
                    VALUES (?, ?, ?)
                    ON CONFLICT(project_id, key) DO UPDATE SET value = excluded.value
                """, (project_id, k, str(v)))
            conn.commit()
            return True
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Baseline locking
    # ------------------------------------------------------------------

    def lock_baseline(self, project_id: int, baseline_data: dict) -> bool:
        """
        Freeze a snapshot of key metrics as the reference baseline.
        Overwrites any previously locked baseline for this project.
        """
        payload = json.dumps({
            **baseline_data,
            "locked_at": datetime.now().isoformat(),
        })
        conn = self._conn()
        try:
            conn.execute("""
                INSERT INTO project_baselines (project_id, baseline_json)
                VALUES (?, ?)
                ON CONFLICT(project_id) DO UPDATE SET
                    baseline_json = excluded.baseline_json,
                    locked_at     = CURRENT_TIMESTAMP
            """, (project_id, payload))
            conn.commit()
            return True
        finally:
            conn.close()

    def get_baseline(self, project_id: int) -> Optional[dict]:
        """Return locked baseline dict, or None if not yet locked."""
        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT baseline_json FROM project_baselines WHERE project_id = ?",
                (project_id,),
            ).fetchone()
            if row:
                return json.loads(row[0])
            return None
        finally:
            conn.close()

    def clear_baseline(self, project_id: int) -> bool:
        conn = self._conn()
        try:
            conn.execute(
                "DELETE FROM project_baselines WHERE project_id = ?", (project_id,)
            )
            conn.commit()
            return True
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_project(self, project_id: int) -> str:
        """Return a JSON string containing all project data."""
        projects_df = self.get_projects()
        proj_row = projects_df[projects_df["id"] == project_id]
        project_meta = proj_row.to_dict(orient="records")[0] if not proj_row.empty else {}
        thresholds = self.get_thresholds(project_id)
        baseline = self.get_baseline(project_id)

        export = {
            "export_version": "1.0",
            "exported_at":    datetime.now().isoformat(),
            "project":        project_meta,
            "thresholds":     thresholds,
            "baseline":       baseline,
        }
        return json.dumps(export, indent=2, default=str)

    def get_threshold_descriptions(self) -> dict:
        """Human-readable labels and units for each threshold key."""
        return {
            "min_trap_nights":         ("Min Trap Nights",         "nights",  1,    365),
            "min_functionality_pct":   ("Min Functionality %",     "%",       0.0,  100.0),
            "confidence_threshold":    ("AI Confidence Threshold", "0–1",     0.1,  1.0),
            "independence_window_min": ("Independence Window",     "minutes", 5,    120),
            "rai_alert_below":         ("RAI Alert Threshold",     "IDEs/TN", 0.0,  10.0),
            "min_ide_for_richness":    ("Min IDEs for Richness",   "IDEs",    1,    50),
        }
