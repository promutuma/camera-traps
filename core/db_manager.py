
import sqlite3
import pandas as pd
import json
from datetime import datetime
from pathlib import Path

class DatabaseManager:
    def __init__(self, db_path="wildlife_data.db"):
        self.db_path = db_path
        self._init_db()

    @property
    def active_project_id(self) -> int:
        conn = self.get_connection()
        try:
            row = conn.execute("SELECT id FROM projects WHERE is_active = 1 LIMIT 1").fetchone()
            return row[0] if row else 1
        except Exception:
            return 1
        finally:
            conn.close()

    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self):
        """Initialize database with required tables and migrate existing schema."""
        conn = self.get_connection()
        cursor = conn.cursor()

        # Projects table
        cursor.execute('''
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
        ''')
        # Seed default project if empty
        cursor.execute("SELECT COUNT(*) FROM projects")
        if cursor.fetchone()[0] == 0:
            cursor.execute("INSERT INTO projects (id, name, is_active) VALUES (1, 'Default Project', 1)")
            conn.commit()

        # Images table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER DEFAULT 1,
                filename TEXT NOT NULL,
                station_id TEXT DEFAULT 'Station-1',
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                capture_date TEXT,
                capture_time TEXT,
                temperature TEXT,
                day_night TEXT,
                brightness REAL,
                user_notes TEXT,
                file_hash TEXT UNIQUE,
                file_size_bytes INTEGER,
                uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                has_animal BOOLEAN DEFAULT 0,
                file_tier TEXT DEFAULT 'valid',
                file_status TEXT DEFAULT 'available',
                marked_for_deletion_at TIMESTAMP,
                deleted_at TIMESTAMP,
                can_delete BOOLEAN DEFAULT 1,
                FOREIGN KEY (project_id) REFERENCES projects (id)
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
                is_exported BOOLEAN DEFAULT 0,
                exported_at TIMESTAMP,
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

        # Downloads audit table — track all downloads
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS downloads_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                download_id TEXT UNIQUE NOT NULL,
                download_type TEXT,
                image_count INTEGER,
                total_size_bytes INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                status TEXT DEFAULT 'preparing',
                error_message TEXT,
                file_hash TEXT
            )
        ''')

        # Download image mappings — which images were in which download
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS download_images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                download_id TEXT,
                image_id INTEGER,
                FOREIGN KEY (download_id) REFERENCES downloads_audit(download_id),
                FOREIGN KEY (image_id) REFERENCES images(id)
            )
        ''')

        # Exports table — track exports of results
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS exports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                export_id TEXT UNIQUE NOT NULL,
                export_type TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                image_count INTEGER,
                detection_count INTEGER,
                exported_detection_ids TEXT
            )
        ''')

        # Add new columns to existing tables if this is a schema migration
        self._migrate_columns(cursor, "images", [
            ("project_id", "INTEGER DEFAULT 1"),
            ("station_id", "TEXT DEFAULT 'Station-1'"),
            ("file_hash", "TEXT"),
            ("file_size_bytes", "INTEGER"),
            ("uploaded_at", "TIMESTAMP"),
            ("has_animal", "BOOLEAN DEFAULT 0"),
            ("file_tier", "TEXT DEFAULT 'valid'"),
            ("file_status", "TEXT DEFAULT 'available'"),
            ("marked_for_deletion_at", "TIMESTAMP"),
            ("deleted_at", "TIMESTAMP"),
            ("can_delete", "BOOLEAN DEFAULT 1"),
        ])
        self._migrate_columns(cursor, "detections", [
            ("ide_id", "TEXT"),
            ("bioclip_confidence", "REAL DEFAULT 0.0"),  # deprecated — always NULL for new rows
            ("speciesnet_confidence", "REAL DEFAULT 0.0"),
            ("agreement", "TEXT"),                       # deprecated — always NULL for new rows
            ("model_breakdown", "TEXT"),
            ("is_exported", "BOOLEAN DEFAULT 0"),
            ("exported_at", "TIMESTAMP"),
            ("scientific_name", "TEXT"),
            ("md_confidence", "REAL"),
            ("top_candidates", "TEXT"),
            ("taxonomy_hierarchy", "TEXT"),
            ("raw_model_output", "TEXT"),
        ])

        # Backfill uploaded_at for existing records
        cursor.execute("UPDATE images SET uploaded_at = CURRENT_TIMESTAMP WHERE uploaded_at IS NULL")

        # Create unique index for file_hash if it doesn't exist
        cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_images_file_hash ON images (file_hash)")

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
        active_project = self.active_project_id

        try:
            for _, row in df.iterrows():
                cursor.execute('''
                    INSERT INTO images (
                        filename, station_id, capture_date, capture_time,
                        temperature, day_night, brightness, user_notes, uploaded_at, project_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?)
                ''', (
                    row['filename'],
                    row.get('station_id', 'Station-1'),
                    row.get('date'),
                    row.get('time'),
                    row.get('temperature'),
                    row.get('day_night'),
                    row.get('brightness', 0.0),
                    row.get('user_notes', ''),
                    active_project
                ))

                image_id = cursor.lastrowid
                bbox_json = json.dumps(row.get('bbox')) if row.get('bbox') else None

                # Parse all SpeciesNet raw results (JSON-encoded taxonomy objects).
                # sn_raw_results is the full ranked list: [(raw_json_label, conf), ...]
                sn_raw = row.get('sn_raw_results') or []

                def _parse_sn_label(label):
                    """Parse a SpeciesNet JSON label string into a metadata dict."""
                    if isinstance(label, str) and label.startswith('{'):
                        try:
                            return json.loads(label)
                        except Exception:
                            pass
                    return {'display': str(label).strip(), 'common_name': str(label).strip()}

                # Top-1 structured fields
                scientific_name = None
                taxonomy_hierarchy = None
                if sn_raw:
                    top_meta = _parse_sn_label(sn_raw[0][0] if isinstance(sn_raw[0], (list, tuple)) else sn_raw[0])
                    scientific_name = top_meta.get('scientific_name') or None
                    hierarchy = top_meta.get('hierarchy')
                    if hierarchy:
                        taxonomy_hierarchy = json.dumps(hierarchy)

                # Full structured candidates list (all top_k results from SpeciesNet)
                top_candidates = None
                if sn_raw:
                    cands = []
                    for entry in sn_raw:
                        lbl, conf = (entry[0], entry[1]) if isinstance(entry, (list, tuple)) else (entry, 0.0)
                        meta = _parse_sn_label(lbl)
                        cands.append({
                            'id': meta.get('id'),
                            'common_name': meta.get('common_name') or meta.get('display') or str(lbl),
                            'scientific_name': meta.get('scientific_name') or None,
                            'hierarchy': meta.get('hierarchy') or [],
                            'confidence': round(float(conf), 4),
                        })
                    top_candidates = json.dumps(cands)

                raw_output = row.get('raw_model_output')
                raw_output_json = json.dumps(raw_output) if raw_output else None

                cursor.execute('''
                    INSERT INTO detections (
                        image_id, detected_animal, confidence, method, bbox, ide_id,
                        speciesnet_confidence, model_breakdown,
                        scientific_name, md_confidence, top_candidates, taxonomy_hierarchy,
                        raw_model_output
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    image_id,
                    row['detected_animal'],
                    row.get('detection_confidence', 0.0),
                    row.get('detection_method', 'Unknown'),
                    bbox_json,
                    row.get('ide_id'),
                    row.get('speciesnet_confidence', 0.0),
                    json.dumps(row.get('model_breakdown', {})) if row.get('model_breakdown') else None,
                    scientific_name,
                    row.get('md_confidence'),
                    top_candidates,
                    taxonomy_hierarchy,
                    raw_output_json,
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

    def delete_detection(self, detection_id: int):
        """Delete a detection record. If it's the last detection for an image, mark image for deletion."""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            # Get the image_id for this detection
            cursor.execute('SELECT image_id FROM detections WHERE id = ?', (detection_id,))
            result = cursor.fetchone()
            if not result:
                return False

            image_id = result[0]

            # Delete the detection
            cursor.execute('DELETE FROM detections WHERE id = ?', (detection_id,))

            # Check if this image has any remaining detections
            cursor.execute('SELECT COUNT(*) FROM detections WHERE image_id = ?', (image_id,))
            remaining = cursor.fetchone()[0]

            # If no detections left, mark image for deletion
            if remaining == 0:
                cursor.execute('''
                    UPDATE images SET marked_for_deletion_at = CURRENT_TIMESTAMP
                    WHERE id = ? AND marked_for_deletion_at IS NULL
                ''', (image_id,))

            conn.commit()
            return True
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def delete_detections_batch(self, detection_ids: list) -> int:
        """Delete multiple detections. Returns count deleted."""
        conn = self.get_connection()
        cursor = conn.cursor()
        deleted_count = 0
        try:
            for det_id in detection_ids:
                cursor.execute('SELECT image_id FROM detections WHERE id = ?', (det_id,))
                result = cursor.fetchone()
                if result:
                    image_id = result[0]
                    cursor.execute('DELETE FROM detections WHERE id = ?', (det_id,))
                    deleted_count += cursor.rowcount

                    # Mark image for deletion if no detections remain
                    cursor.execute('SELECT COUNT(*) FROM detections WHERE image_id = ?', (image_id,))
                    if cursor.fetchone()[0] == 0:
                        cursor.execute('''
                            UPDATE images SET marked_for_deletion_at = CURRENT_TIMESTAMP
                            WHERE id = ? AND marked_for_deletion_at IS NULL
                        ''', (image_id,))

            conn.commit()
            return deleted_count
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

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

    def count_images_by_station(self, station_id: str) -> int:
        """Return the number of detection records referencing this station_id."""
        conn = self.get_connection()
        try:
            row = conn.execute(
                "SELECT COUNT(*) FROM images WHERE station_id = ?", (station_id,)
            ).fetchone()
            return row[0] if row else 0
        finally:
            conn.close()

    def get_history_df(self, limit: int = None, offset: int = 0,
                       station_id: str = None, species: str = None,
                       day_night: str = None, min_conf: float = None,
                       max_conf: float = None):
        """Retrieve detection history as a flat DataFrame, optionally filtered and paginated."""
        conn = self.get_connection()
        where_parts = ["i.project_id = ?"]
        params = [self.active_project_id]
        if station_id:
            where_parts.append("i.station_id = ?")
            params.append(station_id)
        if species:
            where_parts.append("d.detected_animal LIKE ?")
            params.append(f"%{species}%")
        if day_night:
            where_parts.append("i.day_night = ?")
            params.append(day_night)
        if min_conf is not None:
            where_parts.append("d.confidence >= ?")
            params.append(min_conf)
        if max_conf is not None:
            where_parts.append("d.confidence <= ?")
            params.append(max_conf)
        where = "WHERE " + " AND ".join(where_parts)
        query = f'''
            SELECT
                i.id, i.filename, i.station_id, i.processed_at,
                i.capture_date, i.capture_time, i.temperature,
                i.day_night, i.brightness, i.user_notes,
                d.id as detection_id,
                d.detected_animal, d.confidence as detection_confidence,
                d.method as detection_method, d.bbox, d.ide_id,
                d.speciesnet_confidence, d.model_breakdown,
                d.scientific_name, d.md_confidence, d.top_candidates,
                d.taxonomy_hierarchy, d.raw_model_output
            FROM images i
            JOIN detections d ON i.id = d.image_id
            {where}
            ORDER BY i.processed_at DESC
        '''
        if limit is not None:
            query += ' LIMIT ? OFFSET ?'
            params.extend([limit, offset])
        try:
            return pd.read_sql_query(query, conn, params=params if params else None)
        except Exception as e:
            print(f"Error fetching history: {e}")
            return pd.DataFrame()
        finally:
            conn.close()

    def get_pending_review(self, confidence_threshold: float = 0.9, limit: int = 500, offset: int = 0):
        """Return detections below threshold that have not yet been actioned in review_actions."""
        conn = self.get_connection()
        query = '''
            SELECT
                i.id, i.filename, i.station_id,
                i.capture_date, i.capture_time, i.temperature,
                i.day_night, i.brightness, i.user_notes,
                d.id as detection_id,
                d.detected_animal, d.confidence as detection_confidence,
                d.method as detection_method, d.bbox, d.ide_id,
                d.speciesnet_confidence, d.model_breakdown,
                d.scientific_name, d.md_confidence, d.top_candidates,
                d.taxonomy_hierarchy, d.raw_model_output
            FROM images i
            JOIN detections d ON i.id = d.image_id
            LEFT JOIN review_actions ra ON ra.image_id = CAST(i.id AS TEXT)
            WHERE d.confidence < ?
              AND d.detected_animal NOT IN ('Empty', 'Person', 'Vehicle', 'Error')
              AND ra.id IS NULL
            ORDER BY d.confidence ASC
            LIMIT ? OFFSET ?
        '''
        try:
            return pd.read_sql_query(query, conn, params=[confidence_threshold, limit, offset])
        except Exception as e:
            print(f"Error fetching pending review: {e}")
            return pd.DataFrame()
        finally:
            conn.close()

    def get_detected_animal_for_image(self, image_id: int) -> str:
        """Return the top detected_animal label for an image (by confidence)."""
        conn = self.get_connection()
        try:
            row = conn.execute(
                "SELECT detected_animal FROM detections WHERE image_id = ? ORDER BY confidence DESC LIMIT 1",
                [image_id],
            ).fetchone()
            return row[0] if row else ""
        finally:
            conn.close()

    def get_image_ids_by_filenames(self, filenames: list) -> list:
        """Return [(image_id, filename)] for each filename found in the images table."""
        conn = self.get_connection()
        try:
            result = []
            for filename in filenames:
                row = conn.execute(
                    "SELECT id FROM images WHERE filename = ? ORDER BY id DESC LIMIT 1",
                    [filename],
                ).fetchone()
                if row:
                    result.append((row[0], filename))
            return result
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
    # File Management
    # ------------------------------------------------------------------

    def image_hash_exists(self, file_hash: str) -> bool:
        """Return True if a record with this SHA-256 hash already exists."""
        conn = self.get_connection()
        try:
            row = conn.execute(
                "SELECT 1 FROM images WHERE file_hash = ? LIMIT 1", (file_hash,)
            ).fetchone()
            return row is not None
        finally:
            conn.close()

    def update_image_file_info(self, image_id: int, file_hash: str, file_size_bytes: int, has_animal: bool, file_tier: str = 'valid'):
        """Update image with file metadata and tier classification."""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('''
                UPDATE images SET file_hash = ?, file_size_bytes = ?, has_animal = ?, file_tier = ?, uploaded_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (file_hash, file_size_bytes, has_animal, file_tier, image_id))
            conn.commit()
        finally:
            conn.close()

    def get_images_by_tier(self, tier: str, limit: int = 1000, offset: int = 0):
        """Get images by file tier — one row per unique filename (latest id wins).

        A single physical file can have multiple rows in `images` (one per
        detection crop). This query collapses them to avoid returning the same
        file multiple times in downloads and storage listings.
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        active_project = self.active_project_id
        try:
            cursor.execute('''
                SELECT MAX(id) as id, filename,
                       MAX(file_size_bytes) as file_size_bytes,
                       file_status, MAX(uploaded_at) as uploaded_at
                FROM images
                WHERE file_tier = ? AND file_status = 'available' AND project_id = ?
                GROUP BY filename
                ORDER BY MAX(uploaded_at) DESC
                LIMIT ? OFFSET ?
            ''', (tier, active_project, limit, offset))
            return cursor.fetchall()
        finally:
            conn.close()

    def get_storage_stats(self):
        """Get storage breakdown by tier — counts and sizes per unique filename."""
        conn = self.get_connection()
        cursor = conn.cursor()
        active_project = self.active_project_id
        try:
            cursor.execute('''
                SELECT
                    file_tier,
                    COUNT(DISTINCT filename) as count,
                    SUM(size_bytes) as total_bytes,
                    MIN(first_upload) as oldest_upload
                FROM (
                    SELECT file_tier, filename,
                           MAX(COALESCE(file_size_bytes, 0)) as size_bytes,
                           MIN(uploaded_at) as first_upload
                    FROM images
                    WHERE file_status = 'available' AND project_id = ?
                    GROUP BY file_tier, filename
                )
                GROUP BY file_tier
            ''', (active_project,))
            return cursor.fetchall()
        finally:
            conn.close()

    def mark_for_deletion(self, image_id: int):
        """Mark image for deletion (starts grace period)."""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('''
                UPDATE images SET marked_for_deletion_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (image_id,))
            conn.commit()
        finally:
            conn.close()

    def delete_image_file(self, image_id: int):
        """Mark image as deleted in database."""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('''
                UPDATE images SET file_status = 'deleted', deleted_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (image_id,))
            conn.commit()
        finally:
            conn.close()

    def mark_files_missing(self, image_ids: list) -> int:
        """Mark images whose files are not on disk with file_status='missing'."""
        if not image_ids:
            return 0
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            placeholders = ",".join("?" * len(image_ids))
            cursor.execute(
                f"UPDATE images SET file_status = 'missing' WHERE id IN ({placeholders})",
                image_ids,
            )
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()

    def get_available_image_filenames(self) -> list:
        """Return [(id, filename)] for all images with file_status='available'."""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT id, filename FROM images WHERE file_status = 'available'")
            return cursor.fetchall()
        finally:
            conn.close()

    def get_unregistered_station_ids(self) -> list:
        """Return station_ids used in images that have no entry in the stations table."""
        conn = self.get_connection()
        cursor = conn.cursor()
        active_project = self.active_project_id
        try:
            cursor.execute('''
                SELECT DISTINCT i.station_id, COUNT(*) as image_count
                FROM images i
                WHERE i.project_id = ?
                  AND i.station_id IS NOT NULL
                  AND i.station_id != ""
                  AND i.station_id NOT IN (
                      SELECT station_id FROM stations WHERE project_id = ?
                  )
                GROUP BY i.station_id
            ''', (active_project, active_project))
            return [{"station_id": r[0], "image_count": r[1]} for r in cursor.fetchall()]
        finally:
            conn.close()

    def reassign_images_station(self, from_station_id: str, to_station_id: str) -> int:
        """Reassign all images from one station_id to another. Returns updated row count."""
        conn = self.get_connection()
        cursor = conn.cursor()
        active_project = self.active_project_id
        try:
            cursor.execute(
                "UPDATE images SET station_id = ? WHERE station_id = ? AND project_id = ?",
                (to_station_id, from_station_id, active_project),
            )
            conn.commit()
            return cursor.rowcount
        finally:
            conn.close()

    def get_deletable_images(self, tier: str, days_old: int = 7):
        """Get images eligible for deletion (old enough, marked for deletion)."""
        conn = self.get_connection()
        cursor = conn.cursor()
        active_project = self.active_project_id
        try:
            cursor.execute('''
                SELECT id, filename, file_size_bytes
                FROM images
                WHERE file_tier = ?
                  AND file_status = 'available'
                  AND can_delete = 1
                  AND project_id = ?
                  AND datetime(marked_for_deletion_at) < datetime('now', ?)
            ''', (tier, active_project, f"-{days_old} days"))
            return cursor.fetchall()
        finally:
            conn.close()

    def get_images_for_deletion_warning(self, days_until_deletion: int = 3):
        """Get images about to be deleted (for user warnings)."""
        conn = self.get_connection()
        cursor = conn.cursor()
        active_project = self.active_project_id
        try:
            cursor.execute('''
                SELECT id, filename, marked_for_deletion_at
                FROM images
                WHERE file_status = 'available'
                  AND marked_for_deletion_at IS NOT NULL
                  AND project_id = ?
                  AND datetime(marked_for_deletion_at) > datetime('now', ?)
                  AND file_tier IN ('empty', 'low_conf')
            ''', (active_project, f"-{days_until_deletion} days"))
            return cursor.fetchall()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Downloads & Exports
    # ------------------------------------------------------------------

    def create_download(self, download_id: str, download_type: str, image_ids: list, total_size_bytes: int):
        """Create a download record and link images."""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO downloads_audit (download_id, download_type, image_count, total_size_bytes, status)
                VALUES (?, ?, ?, ?, 'preparing')
            ''', (download_id, download_type, len(image_ids), total_size_bytes))

            for image_id in image_ids:
                cursor.execute('''
                    INSERT INTO download_images (download_id, image_id)
                    VALUES (?, ?)
                ''', (download_id, image_id))

            conn.commit()
        finally:
            conn.close()

    def complete_download(self, download_id: str, file_hash: str = None):
        """Mark download as completed."""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('''
                UPDATE downloads_audit
                SET status = 'completed', completed_at = CURRENT_TIMESTAMP, file_hash = ?
                WHERE download_id = ?
            ''', (file_hash, download_id))
            conn.commit()
        finally:
            conn.close()

    def mark_exports_as_exported(self, detection_ids: list, export_type: str):
        """Mark detections as exported."""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            for det_id in detection_ids:
                cursor.execute('''
                    UPDATE detections
                    SET is_exported = 1, exported_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (det_id,))

            cursor.execute('''
                INSERT INTO exports (export_type, image_count, detection_count, exported_detection_ids)
                VALUES (?, ?, ?, ?)
            ''', (export_type, 0, len(detection_ids), json.dumps(detection_ids)))

            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Hash Management (for performance optimization)
    # ------------------------------------------------------------------

    def get_hash_stats(self):
        """Get hash statistics for storage optimization."""
        conn = self.get_connection()
        cursor = conn.cursor()
        active_project = self.active_project_id
        try:
            cursor.execute('''
                SELECT
                    COUNT(*) as total_images,
                    COALESCE(SUM(CASE WHEN file_hash IS NOT NULL THEN 1 ELSE 0 END), 0) as with_hashes,
                    COALESCE(SUM(CASE WHEN file_hash IS NULL THEN 1 ELSE 0 END), 0) as without_hashes,
                    COUNT(DISTINCT file_hash) as unique_hashes
                FROM images
                WHERE project_id = ?
            ''', (active_project,))
            result = cursor.fetchone()
            if result:
                return {
                    "total_images": result[0] or 0,
                    "with_hashes": result[1] or 0,
                    "without_hashes": result[2] or 0,
                    "unique_hashes": result[3] or 0,
                }
            return {"total_images": 0, "with_hashes": 0, "without_hashes": 0, "unique_hashes": 0}
        finally:
            conn.close()

    def clear_hashes_by_tier(self, tier: str):
        """Clear file hashes for a specific tier (reduces processing time)."""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('''
                UPDATE images SET file_hash = NULL
                WHERE file_tier = ?
            ''', (tier,))
            count = cursor.rowcount
            conn.commit()
            return count
        finally:
            conn.close()

    def clear_hashes_by_status(self, status: str):
        """Clear file hashes for images with specific status (deleted, archived)."""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('''
                UPDATE images SET file_hash = NULL
                WHERE file_status = ?
            ''', (status,))
            count = cursor.rowcount
            conn.commit()
            return count
        finally:
            conn.close()

    def clear_old_hashes(self, days_old: int = 30):
        """Clear hashes for old images (older than N days)."""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('''
                UPDATE images SET file_hash = NULL
                WHERE datetime(uploaded_at) < datetime('now', ?)
            ''', (f"-{days_old} days",))
            count = cursor.rowcount
            conn.commit()
            return count
        finally:
            conn.close()

    def clear_duplicate_hashes(self):
        """Clear hashes for duplicate images (keep only first occurrence of each hash)."""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            # Find duplicate hashes
            cursor.execute('''
                SELECT file_hash, MIN(id) as keep_id
                FROM images
                WHERE file_hash IS NOT NULL
                GROUP BY file_hash
                HAVING COUNT(*) > 1
            ''')
            duplicates = cursor.fetchall()

            cleared = 0
            for file_hash, keep_id in duplicates:
                cursor.execute('''
                    UPDATE images SET file_hash = NULL
                    WHERE file_hash = ? AND id != ?
                ''', (file_hash, keep_id))
                cleared += cursor.rowcount

            conn.commit()
            return cleared
        finally:
            conn.close()

    def clear_all_hashes(self):
        """Clear all file hashes (maximum performance gain)."""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('UPDATE images SET file_hash = NULL')
            count = cursor.rowcount
            conn.commit()
            return count
        finally:
            conn.close()

    def find_duplicate_files(self):
        """Find duplicate files by hash (shows which images could be deduplicated)."""
        conn = self.get_connection()
        cursor = conn.cursor()
        active_project = self.active_project_id
        try:
            cursor.execute('''
                SELECT file_hash, COUNT(*) as count, GROUP_CONCAT(id) as image_ids
                FROM images
                WHERE file_hash IS NOT NULL AND project_id = ?
                GROUP BY file_hash
                HAVING count > 1
                ORDER BY count DESC
            ''', (active_project,))
            return cursor.fetchall()
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

    def reset_database(self):
        """Reset database completely by truncating all tables."""
        conn = self.get_connection()
        cursor = conn.cursor()
        tables = [
            "detections",
            "images",
            "independence_events",
            "jobs",
            "downloads_audit",
            "download_images",
            "exports",
            "stations",
            "deployments",
            "community_observations"
        ]
        for table in tables:
            try:
                cursor.execute(f"DELETE FROM {table}")
            except Exception:
                pass
        conn.commit()
        conn.close()
