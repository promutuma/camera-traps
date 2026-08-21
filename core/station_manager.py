"""
Station and Deployment Manager
Manages camera trap station registry, deployment history,
trap night calculation, and camera functionality rates.
"""

import sqlite3
import pandas as pd
from datetime import date, datetime, timedelta
from typing import Optional


STRATA = ["Wetland Habitat", "Watering Point", "Corridor", "Other"]


class StationManager:
    """
    Persistent station and deployment registry backed by the shared SQLite database.

    Station registry fields:
        station_id, gps_lat, gps_lon, habitat_stratum, camera_model,
        team_member, notes

    Deployment history fields (per station):
        id, station_id, deployment_date, retrieval_date,
        team_member, sd_card_id, camera_down_days, notes
    """

    def __init__(self, db_path: str = "wildlife_data.db"):
        self.db_path = db_path
        self._init_tables()

    @property
    def active_project_id(self) -> int:
        conn = self._conn()
        try:
            row = conn.execute("SELECT id FROM projects WHERE is_active = 1 LIMIT 1").fetchone()
            return row[0] if row else 1
        except Exception:
            return 1
        finally:
            conn.close()

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def _init_tables(self):
        conn = self._conn()
        cur = conn.cursor()

        # Migrate stations table if it exists but lacks project_id
        try:
            cur.execute("PRAGMA table_info(stations)")
            cols = [r[1] for r in cur.fetchall()]
            if cols and "project_id" not in cols:
                cur.execute("ALTER TABLE stations RENAME TO stations_old")
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS stations (
                        id            INTEGER PRIMARY KEY AUTOINCREMENT,
                        project_id    INTEGER DEFAULT 1,
                        station_id    TEXT NOT NULL,
                        gps_lat       REAL,
                        gps_lon       REAL,
                        habitat_stratum TEXT,
                        camera_model  TEXT,
                        team_member   TEXT,
                        notes         TEXT,
                        created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(project_id, station_id)
                    )
                """)
                cur.execute("""
                    INSERT INTO stations (project_id, station_id, gps_lat, gps_lon, habitat_stratum, camera_model, team_member, notes, created_at)
                    SELECT 1, station_id, gps_lat, gps_lon, habitat_stratum, camera_model, team_member, notes, created_at FROM stations_old
                """)
                cur.execute("DROP TABLE stations_old")
                conn.commit()
        except Exception as e:
            print(f"Error migrating stations table: {e}")

        # Migrate deployments table if it exists but lacks project_id
        try:
            cur.execute("PRAGMA table_info(deployments)")
            cols_dep = [r[1] for r in cur.fetchall()]
            if cols_dep and "project_id" not in cols_dep:
                cur.execute("ALTER TABLE deployments RENAME TO deployments_old")
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS deployments (
                        id               INTEGER PRIMARY KEY AUTOINCREMENT,
                        project_id       INTEGER DEFAULT 1,
                        station_id       TEXT NOT NULL,
                        deployment_date  TEXT,
                        retrieval_date   TEXT,
                        team_member      TEXT,
                        sd_card_id       TEXT,
                        camera_down_days INTEGER DEFAULT 0,
                        notes            TEXT,
                        created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                cur.execute("""
                    INSERT INTO deployments (id, project_id, station_id, deployment_date, retrieval_date, team_member, sd_card_id, camera_down_days, notes, created_at)
                    SELECT id, 1, station_id, deployment_date, retrieval_date, team_member, sd_card_id, camera_down_days, notes, created_at FROM deployments_old
                """)
                cur.execute("DROP TABLE deployments_old")
                conn.commit()
        except Exception as e:
            print(f"Error migrating deployments table: {e}")

        # Fresh creation fallback
        cur.execute("""
            CREATE TABLE IF NOT EXISTS stations (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id    INTEGER DEFAULT 1,
                station_id    TEXT NOT NULL,
                gps_lat       REAL,
                gps_lon       REAL,
                habitat_stratum TEXT,
                camera_model  TEXT,
                team_member   TEXT,
                notes         TEXT,
                created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(project_id, station_id)
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS deployments (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id       INTEGER DEFAULT 1,
                station_id       TEXT NOT NULL,
                deployment_date  TEXT,
                retrieval_date   TEXT,
                team_member      TEXT,
                sd_card_id       TEXT,
                camera_down_days INTEGER DEFAULT 0,
                notes            TEXT,
                created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        conn.close()

    # ------------------------------------------------------------------
    # Station CRUD
    # ------------------------------------------------------------------

    def add_station(
        self,
        station_id: str,
        gps_lat: Optional[float] = None,
        gps_lon: Optional[float] = None,
        habitat_stratum: str = "",
        camera_model: str = "",
        team_member: str = "",
        notes: str = "",
    ) -> bool:
        """Insert a new station. Returns False if station_id already exists in project."""
        conn = self._conn()
        proj_id = self.active_project_id
        try:
            conn.execute("""
                INSERT INTO stations
                    (project_id, station_id, gps_lat, gps_lon, habitat_stratum,
                     camera_model, team_member, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (proj_id, station_id.strip(), gps_lat, gps_lon, habitat_stratum,
                  camera_model, team_member, notes))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False
        finally:
            conn.close()

    def update_station(self, station_id: str, **kwargs) -> bool:
        """Update any subset of station fields by keyword argument. Backward-compatible: unknown keys ignored."""
        allowed = {"gps_lat", "gps_lon", "habitat_stratum",
                   "camera_model", "team_member", "notes"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return False
        sets = ", ".join(f"{k} = ?" for k in updates)
        proj_id = self.active_project_id
        vals = list(updates.values()) + [proj_id, station_id]
        conn = self._conn()
        try:
            conn.execute(f"UPDATE stations SET {sets} WHERE project_id = ? AND station_id = ?", vals)
            conn.commit()
            return True
        finally:
            conn.close()

    def delete_station(self, station_id: str) -> bool:
        """Delete a station and all its deployments."""
        conn = self._conn()
        proj_id = self.active_project_id
        try:
            conn.execute("DELETE FROM deployments WHERE project_id = ? AND station_id = ?", (proj_id, station_id))
            conn.execute("DELETE FROM stations WHERE project_id = ? AND station_id = ?", (proj_id, station_id))
            conn.commit()
            return True
        finally:
            conn.close()

    def get_stations(self) -> pd.DataFrame:
        """Return all stations as a DataFrame."""
        conn = self._conn()
        proj_id = self.active_project_id
        try:
            df = pd.read_sql_query(
                "SELECT * FROM stations WHERE project_id = ? ORDER BY station_id", conn, params=(proj_id,)
            )
            return df
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Deployment CRUD
    # ------------------------------------------------------------------

    def add_deployment(
        self,
        station_id: str,
        deployment_date: Optional[str] = None,
        retrieval_date: Optional[str] = None,
        team_member: str = "",
        sd_card_id: str = "",
        camera_down_days: int = 0,
        notes: str = "",
    ) -> int:
        """
        Add a deployment record. Returns the new row id, or -1 on error.
        deployment_date / retrieval_date: 'YYYY-MM-DD' strings or None.
        retrieval_date=None means the deployment is still active.
        """
        conn = self._conn()
        proj_id = self.active_project_id
        try:
            cur = conn.execute("""
                INSERT INTO deployments
                    (project_id, station_id, deployment_date, retrieval_date, team_member,
                     sd_card_id, camera_down_days, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (proj_id, station_id, deployment_date, retrieval_date,
                  team_member, sd_card_id, camera_down_days, notes))
            conn.commit()
            return cur.lastrowid
        except Exception as e:
            print(f"Error adding deployment: {e}")
            return -1
        finally:
            conn.close()

    def update_deployment(self, deployment_id: int, **kwargs) -> bool:
        allowed = {"deployment_date", "retrieval_date", "team_member",
                   "sd_card_id", "camera_down_days", "notes"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return False
        sets = ", ".join(f"{k} = ?" for k in updates)
        vals = list(updates.values()) + [deployment_id]
        conn = self._conn()
        try:
            conn.execute(f"UPDATE deployments SET {sets} WHERE id = ?", vals)
            conn.commit()
            return True
        finally:
            conn.close()

    def delete_deployment(self, deployment_id: int) -> bool:
        conn = self._conn()
        try:
            conn.execute("DELETE FROM deployments WHERE id = ?", (deployment_id,))
            conn.commit()
            return True
        finally:
            conn.close()

    def get_deployments(self, station_id: Optional[str] = None) -> pd.DataFrame:
        """Return deployments, optionally filtered to one station."""
        conn = self._conn()
        proj_id = self.active_project_id
        try:
            if station_id:
                df = pd.read_sql_query(
                    "SELECT * FROM deployments WHERE project_id = ? AND station_id = ? ORDER BY deployment_date",
                    conn, params=(proj_id, station_id)
                )
            else:
                df = pd.read_sql_query(
                    "SELECT * FROM deployments WHERE project_id = ? ORDER BY station_id, deployment_date",
                    conn, params=(proj_id,)
                )
            return df
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Derived metrics
    # ------------------------------------------------------------------

    def _deployment_nights(self, dep_date_str: Optional[str],
                           ret_date_str: Optional[str],
                           down_days: int) -> int:
        """
        Active trap nights for a single deployment.
        Uses today if retrieval_date is None (ongoing deployment).
        """
        today = date.today()
        try:
            dep = date.fromisoformat(dep_date_str) if dep_date_str else today
        except (ValueError, TypeError):
            dep = today
        try:
            ret = date.fromisoformat(ret_date_str) if ret_date_str else today
        except (ValueError, TypeError):
            ret = today

        raw = (ret - dep).days
        return max(0, raw - int(down_days or 0))

    def compute_trap_nights(self) -> dict:
        """
        Return {station_id: total_active_trap_nights} for all stations
        with at least one deployment record.
        """
        deps = self.get_deployments()
        if deps.empty:
            return {}

        result = {}
        for station_id, grp in deps.groupby("station_id"):
            total = sum(
                self._deployment_nights(
                    row.get("deployment_date"),
                    row.get("retrieval_date"),
                    row.get("camera_down_days", 0),
                )
                for _, row in grp.iterrows()
            )
            result[station_id] = total
        return result

    def compute_functionality_rates(self) -> dict:
        """
        Return {station_id: functionality_pct} where:
            functionality % = active_nights / total_deployment_nights × 100
        """
        deps = self.get_deployments()
        if deps.empty:
            return {}

        result = {}
        for station_id, grp in deps.groupby("station_id"):
            total_days = 0
            down_days = 0
            for _, row in grp.iterrows():
                dep_str = row.get("deployment_date")
                ret_str = row.get("retrieval_date")
                today = date.today()
                try:
                    dep = date.fromisoformat(dep_str) if dep_str else today
                    ret = date.fromisoformat(ret_str) if ret_str else today
                except (ValueError, TypeError):
                    continue
                total_days += max(0, (ret - dep).days)
                down_days += int(row.get("camera_down_days", 0) or 0)

            if total_days > 0:
                result[station_id] = round((total_days - down_days) / total_days * 100, 1)
            else:
                result[station_id] = 0.0
        return result

    def get_station_summary(self) -> pd.DataFrame:
        """
        One row per station combining registry info, trap nights,
        and functionality rate. Used as the primary station dashboard table.
        """
        stations = self.get_stations()
        if stations.empty:
            return pd.DataFrame()

        trap_nights = self.compute_trap_nights()
        func_rates = self.compute_functionality_rates()

        # Count deployments per station
        deps = self.get_deployments()
        dep_counts = (
            deps.groupby("station_id").size().to_dict()
            if not deps.empty else {}
        )

        stations["trap_nights"] = stations["station_id"].map(trap_nights).fillna(0).astype(int)
        stations["functionality_pct"] = stations["station_id"].map(func_rates).fillna(0)
        stations["deployment_count"] = stations["station_id"].map(dep_counts).fillna(0).astype(int)
        stations["status"] = stations.apply(self._station_status, axis=1)

        return stations

    def _station_status(self, row) -> str:
        """Derive a simple status label for a station row."""
        func = row.get("functionality_pct", 100)
        nights = row.get("trap_nights", 0)
        if func < 90:
            return "Low Functionality"
        if nights == 0:
            return "No Data"
        return "Active"
