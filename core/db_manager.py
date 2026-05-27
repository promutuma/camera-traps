
import sqlite3
import pandas as pd
import json
from datetime import datetime
from pathlib import Path

class DatabaseManager:
    def __init__(self, db_path="wildlife_data.db"):
        self.db_path = db_path
        self._init_db()

    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self):
        """Initialize database with required tables and migrate existing schema."""
        conn = self.get_connection()
        cursor = conn.cursor()

        # Images table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                station_id TEXT DEFAULT 'Station-1',
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                capture_date TEXT,
                capture_time TEXT,
                temperature TEXT,
                day_night TEXT,
                brightness REAL,
                user_notes TEXT
            )
        ''')

        # Detections table (linked to images)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id INTEGER,
                detected_animal TEXT,
                confidence REAL,
                method TEXT,
                bbox TEXT,
                ide_id TEXT,
                bioclip_confidence REAL DEFAULT 0.0,
                speciesnet_confidence REAL DEFAULT 0.0,
                agreement TEXT,
                FOREIGN KEY (image_id) REFERENCES images (id)
            )
        ''')

        # Independence Events table — one row per IDE
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS independence_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ide_id TEXT UNIQUE NOT NULL,
                ide_group INTEGER,
                station_id TEXT,
                species TEXT,
                first_detection TIMESTAMP,
                last_detection TIMESTAMP,
                duration_minutes REAL,
                image_count INTEGER,
                max_confidence REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Jobs table — lightweight metadata so completed jobs survive restarts
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                total INTEGER DEFAULT 0,
                completed INTEGER DEFAULT 0,
                error TEXT,
                created_at REAL,
                finished_at REAL
            )
        ''')

        # Add new columns to existing tables if this is a schema migration
        self._migrate_columns(cursor, "images", [
            ("station_id", "TEXT DEFAULT 'Station-1'"),
        ])
        self._migrate_columns(cursor, "detections", [
            ("ide_id", "TEXT"),
            ("bioclip_confidence", "REAL DEFAULT 0.0"),
            ("speciesnet_confidence", "REAL DEFAULT 0.0"),
            ("agreement", "TEXT"),
            ("model_breakdown", "TEXT"),
        ])

        conn.commit()
        conn.close()

    def _migrate_columns(self, cursor, table: str, columns: list):
        """Add columns to an existing table if they don't already exist."""
        cursor.execute(f"PRAGMA table_info({table})")
        existing = {row[1] for row in cursor.fetchall()}
        for col_name, col_def in columns:
            if col_name not in existing:
                cursor.execute(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_def}")

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save_results(self, df):
        """
        Save processing results to database.

        Returns:
            int: Number of image records saved
        """
        if df is None or len(df) == 0:
            return 0

        conn = self.get_connection()
        cursor = conn.cursor()
        count = 0

        try:
            for _, row in df.iterrows():
                cursor.execute('''
                    INSERT INTO images (
                        filename, station_id, capture_date, capture_time,
                        temperature, day_night, brightness, user_notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    row['filename'],
                    row.get('station_id', 'Station-1'),
                    row.get('date'),
                    row.get('time'),
                    row.get('temperature'),
                    row.get('day_night'),
                    row.get('brightness', 0.0),
                    row.get('user_notes', '')
                ))

                image_id = cursor.lastrowid
                bbox_json = json.dumps(row.get('bbox')) if row.get('bbox') else None

                cursor.execute('''
                    INSERT INTO detections (
                        image_id, detected_animal, confidence, method, bbox, ide_id,
                        bioclip_confidence, speciesnet_confidence, agreement, model_breakdown
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    image_id,
                    row['detected_animal'],
                    row.get('detection_confidence', 0.0),
                    row.get('detection_method', 'Unknown'),
                    bbox_json,
                    row.get('ide_id'),
                    row.get('bioclip_confidence', 0.0),
                    row.get('speciesnet_confidence', 0.0),
                    row.get('agreement'),
                    json.dumps(row.get('model_breakdown', {})) if row.get('model_breakdown') else None,
                ))

                count += 1

            conn.commit()
            return count

        except Exception as e:
            conn.rollback()
            print(f"Error saving to database: {e}")
            raise e
        finally:
            conn.close()

    def save_independence_events(self, ide_summary: pd.DataFrame):
        """
        Persist an IDE summary DataFrame to the independence_events table.
        Rows with duplicate ide_id are ignored (INSERT OR IGNORE).

        Returns:
            int: Number of new rows inserted
        """
        if ide_summary is None or ide_summary.empty:
            return 0

        conn = self.get_connection()
        cursor = conn.cursor()
        count = 0

        try:
            for _, row in ide_summary.iterrows():
                first_dt = row.get("first_detection")
                last_dt = row.get("last_detection")

                cursor.execute('''
                    INSERT OR IGNORE INTO independence_events (
                        ide_id, ide_group, station_id, species,
                        first_detection, last_detection, duration_minutes,
                        image_count, max_confidence
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    row.get("ide_id"),
                    int(row.get("ide_group", 0)),
                    row.get("station_id"),
                    row.get("species"),
                    str(first_dt) if pd.notna(first_dt) else None,
                    str(last_dt) if pd.notna(last_dt) else None,
                    float(row.get("duration_minutes", 0)) if pd.notna(row.get("duration_minutes")) else None,
                    int(row.get("image_count", 1)),
                    float(row.get("max_confidence", 0)) if pd.notna(row.get("max_confidence")) else None,
                ))
                count += cursor.rowcount

            conn.commit()
            return count

        except Exception as e:
            conn.rollback()
            print(f"Error saving independence events: {e}")
            raise e
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def update_detection(self, detection_id: int, fields: dict):
        """Update a single detection row and/or its parent image row."""
        det_fields = {k: v for k, v in fields.items() if k == "detected_animal"}
        img_fields = {k: v for k, v in fields.items() if k in ("station_id", "user_notes")}
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            if det_fields:
                sets = ", ".join(f"{k} = ?" for k in det_fields)
                cursor.execute(f"UPDATE detections SET {sets} WHERE id = ?",
                               list(det_fields.values()) + [detection_id])
            if img_fields:
                row = cursor.execute(
                    "SELECT image_id FROM detections WHERE id = ?", [detection_id]
                ).fetchone()
                if row:
                    sets = ", ".join(f"{k} = ?" for k in img_fields)
                    cursor.execute(f"UPDATE images SET {sets} WHERE id = ?",
                                   list(img_fields.values()) + [row[0]])
            conn.commit()
        finally:
            conn.close()

    def get_history_df(self):
        """Retrieve full detection history as a flat DataFrame."""
        conn = self.get_connection()
        query = '''
            SELECT
                i.id, i.filename, i.station_id, i.processed_at,
                i.capture_date, i.capture_time, i.temperature,
                i.day_night, i.brightness, i.user_notes,
                d.id as detection_id,
                d.detected_animal, d.confidence as detection_confidence,
                d.method as detection_method, d.bbox, d.ide_id,
                d.bioclip_confidence, d.speciesnet_confidence, d.agreement,
                d.model_breakdown
            FROM images i
            JOIN detections d ON i.id = d.image_id
            ORDER BY i.processed_at DESC
        '''
        try:
            return pd.read_sql_query(query, conn)
        except Exception as e:
            print(f"Error fetching history: {e}")
            return pd.DataFrame()
        finally:
            conn.close()

    def get_independence_events_df(self):
        """Retrieve all stored independence events."""
        conn = self.get_connection()
        try:
            return pd.read_sql_query(
                "SELECT * FROM independence_events ORDER BY first_detection",
                conn
            )
        except Exception as e:
            print(f"Error fetching independence events: {e}")
            return pd.DataFrame()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Job persistence
    # ------------------------------------------------------------------

    def save_job(self, job) -> None:
        """Upsert a job's metadata row (called on completion or error)."""
        conn = self.get_connection()
        try:
            conn.execute(
                """
                INSERT INTO jobs (job_id, status, total, completed, error, created_at, finished_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    status     = excluded.status,
                    total      = excluded.total,
                    completed  = excluded.completed,
                    error      = excluded.error,
                    finished_at= excluded.finished_at
                """,
                (
                    job.job_id,
                    job.status,
                    job.total,
                    job.completed,
                    job.error,
                    job.created_at,
                    job.finished_at,
                ),
            )
            conn.commit()
        except Exception as exc:
            print(f"Warning: could not persist job {job.job_id}: {exc}")
        finally:
            conn.close()

    def load_recent_jobs(self, limit: int = 50) -> list:
        """Return metadata for the most recent completed/errored jobs."""
        conn = self.get_connection()
        try:
            cursor = conn.execute(
                """
                SELECT job_id, status, total, completed, error, created_at, finished_at
                FROM jobs
                ORDER BY finished_at DESC
                LIMIT ?
                """,
                (limit,),
            )
            cols = [d[0] for d in cursor.description]
            return [dict(zip(cols, row)) for row in cursor.fetchall()]
        except Exception as exc:
            print(f"Warning: could not load jobs: {exc}")
            return []
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def clear_history(self):
        """Clear all data from database."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM detections")
        cursor.execute("DELETE FROM images")
        cursor.execute("DELETE FROM independence_events")
        conn.commit()
        conn.close()
