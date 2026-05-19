"""
Community Observer Data Entry
Records field observer sightings (direct, track, scat, kill, vocalisation)
linked to the nearest camera station. Stored in its own SQLite table.
"""

import sqlite3
import pandas as pd
from datetime import datetime
from typing import Optional


OBSERVATION_TYPES = [
    "Direct Sighting",
    "Track / Footprint",
    "Scat / Dung",
    "Kill Site",
    "Vocalisation",
    "Camera Sighting (manual)",
    "Other",
]


class CommunityObserver:
    """
    Persistent store for community and field-observer wildlife records.

    Table: community_observations
        id, date, time, station_id, gps_lat, gps_lon,
        species, individual_count, observer_name,
        observation_type, notes, created_at
    """

    def __init__(self, db_path: str = "wildlife_data.db"):
        self.db_path = db_path
        self._init_tables()

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def _init_tables(self):
        conn = self._conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS community_observations (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                obs_date         TEXT,
                obs_time         TEXT,
                station_id       TEXT,
                gps_lat          REAL,
                gps_lon          REAL,
                species          TEXT NOT NULL,
                individual_count INTEGER DEFAULT 1,
                observer_name    TEXT,
                observation_type TEXT,
                notes            TEXT,
                created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_observation(
        self,
        species: str,
        obs_date: Optional[str] = None,
        obs_time: Optional[str] = None,
        station_id: Optional[str] = None,
        gps_lat: Optional[float] = None,
        gps_lon: Optional[float] = None,
        individual_count: int = 1,
        observer_name: str = "",
        observation_type: str = "Direct Sighting",
        notes: str = "",
    ) -> int:
        """Insert one observation. Returns the new row id."""
        conn = self._conn()
        try:
            cur = conn.execute("""
                INSERT INTO community_observations
                    (obs_date, obs_time, station_id, gps_lat, gps_lon,
                     species, individual_count, observer_name,
                     observation_type, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                obs_date, obs_time, station_id,
                gps_lat, gps_lon,
                species.strip(), individual_count,
                observer_name, observation_type, notes,
            ))
            conn.commit()
            return cur.lastrowid
        finally:
            conn.close()

    def delete_observation(self, obs_id: int) -> bool:
        conn = self._conn()
        try:
            conn.execute("DELETE FROM community_observations WHERE id = ?", (obs_id,))
            conn.commit()
            return True
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_observations(self, station_id: Optional[str] = None) -> pd.DataFrame:
        """Return all observations, optionally filtered by station."""
        conn = self._conn()
        try:
            if station_id:
                return pd.read_sql_query(
                    "SELECT * FROM community_observations WHERE station_id = ? ORDER BY obs_date DESC",
                    conn, params=(station_id,)
                )
            return pd.read_sql_query(
                "SELECT * FROM community_observations ORDER BY obs_date DESC", conn
            )
        finally:
            conn.close()

    def get_community_only_species(self, camera_species: set) -> pd.DataFrame:
        """
        Return community observations for species not detected by any camera.
        These are flagged as 'Community-only records' in the species inventory.

        Parameters
        ----------
        camera_species : set[str]
            Lowercase species names seen in camera detections.
        """
        obs = self.get_observations()
        if obs.empty:
            return pd.DataFrame()

        community_only = obs[
            ~obs["species"].str.lower().isin({s.lower() for s in camera_species})
        ].copy()
        community_only["flag"] = "Community-only record"
        return community_only

    def get_species_comparison(self, camera_species: set) -> dict:
        """
        Return a summary dict comparing community and camera records.

        Keys: camera_only, community_only, both
        Each value is a sorted list of species names.
        """
        obs = self.get_observations()
        community_set = set(obs["species"].str.strip().str.title().unique()) if not obs.empty else set()
        camera_set = {s.strip().title() for s in camera_species}

        return {
            "camera_only":    sorted(camera_set - community_set),
            "community_only": sorted(community_set - camera_set),
            "both":           sorted(camera_set & community_set),
        }

    def get_stats(self) -> dict:
        conn = self._conn()
        try:
            row = conn.execute("""
                SELECT
                    COUNT(*)           AS total,
                    COUNT(DISTINCT species) AS unique_species,
                    SUM(individual_count)   AS total_individuals
                FROM community_observations
            """).fetchone()
            return {
                "total": row[0] or 0,
                "unique_species": row[1] or 0,
                "total_individuals": row[2] or 0,
            }
        finally:
            conn.close()
