
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
        return sqlite3.connect(self.db_path)

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

        # Add new columns to existing tables if this is a schema migration
        self._migrate_columns(cursor, "images", [
            ("station_id", "TEXT DEFAULT 'Station-1'"),
        ])
        self._migrate_columns(cursor, "detections", [
            ("ide_id", "TEXT"),
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
                        image_id, detected_animal, confidence, method, bbox, ide_id
                    ) VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    image_id,
                    row['detected_animal'],
                    row.get('detection_confidence', 0.0),
                    row.get('detection_method', 'Unknown'),
                    bbox_json,
                    row.get('ide_id'),
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

    def get_history_df(self):
        """Retrieve full detection history as a flat DataFrame."""
        conn = self.get_connection()
        query = '''
            SELECT
                i.id, i.filename, i.station_id, i.processed_at,
                i.capture_date, i.capture_time, i.temperature,
                i.day_night, i.brightness, i.user_notes,
                d.detected_animal, d.confidence as detection_confidence,
                d.method as detection_method, d.bbox, d.ide_id
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
