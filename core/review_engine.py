"""
HITL Review Engine
Manages the structured human-in-the-loop review queue:
  - Surfaces detections below a confidence threshold for expert review
  - Records accept / correct / reject actions with reviewer and timestamp
  - Stores the original AI prediction alongside every correction
  - Supports batch accept/reject on filtered groups
"""

import sqlite3
import pandas as pd
import json
from datetime import datetime
from typing import Optional

ACTION_ACCEPT  = "accept"
ACTION_CORRECT = "correct"
ACTION_REJECT  = "reject"

_VALID_ACTIONS = {ACTION_ACCEPT, ACTION_CORRECT, ACTION_REJECT}


class ReviewConflictError(Exception):
    """Raised when a detection has already been reviewed."""


class ReviewEngine:
    """
    Persistent HITL review store backed by the shared SQLite database.

    Tables managed here:
        review_actions  — one row per review decision (keyed by detection_id)
    """

    def __init__(self, db_path: str = "wildlife_data.db"):
        self.db_path = db_path
        self._init_tables()

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def _init_tables(self):
        conn = self._conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS review_actions (
                id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                detection_id       INTEGER,
                image_id           TEXT NOT NULL,
                filename           TEXT,
                action             TEXT NOT NULL,
                original_species   TEXT,
                corrected_species  TEXT,
                original_label     TEXT,
                corrected_label    TEXT,
                reviewer_id        TEXT DEFAULT 'anonymous',
                confidence         REAL,
                notes              TEXT,
                reviewed_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cols = {row[1] for row in conn.execute("PRAGMA table_info(review_actions)")}
        if "detection_id" not in cols:
            conn.execute("ALTER TABLE review_actions ADD COLUMN detection_id INTEGER")
        conn.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_review_actions_detection_id
            ON review_actions(detection_id) WHERE detection_id IS NOT NULL
        """)
        conn.commit()
        conn.close()

    # ------------------------------------------------------------------
    # Queue
    # ------------------------------------------------------------------

    def get_queue(
        self,
        df: pd.DataFrame,
        confidence_threshold: float = 0.9,
        reviewed_detection_ids: Optional[set] = None,
    ) -> pd.DataFrame:
        """
        Return a sub-DataFrame of detections that need review.
        """
        if df is None or df.empty:
            return pd.DataFrame()

        already_reviewed = self._get_reviewed_detection_ids()
        if reviewed_detection_ids:
            already_reviewed |= reviewed_detection_ids

        animals = df[df.get("primary_label", pd.Series(dtype=str)) == "Animal"].copy() \
            if "primary_label" in df.columns else df.copy()

        if "detection_confidence" in animals.columns:
            queue = animals[animals["detection_confidence"] < confidence_threshold]
        else:
            queue = animals.copy()

        id_col = "detection_id" if "detection_id" in queue.columns else "id"
        if id_col in queue.columns:
            queue = queue[~queue[id_col].isin(already_reviewed)]

        if "agreement" in queue.columns:
            priority_map = {"Low": 0, "Medium": 1, "High": 2}
            queue["priority_score"] = queue["agreement"].map(lambda x: priority_map.get(x, 3))

            sort_cols = ["priority_score"]
            sort_ascending = [True]

            if "detection_confidence" in queue.columns:
                sort_cols.append("detection_confidence")
                sort_ascending.append(True)

            queue = queue.sort_values(sort_cols, ascending=sort_ascending)
            queue = queue.drop(columns=["priority_score"])
        elif "detection_confidence" in queue.columns:
            queue = queue.sort_values("detection_confidence", ascending=True)

        return queue.reset_index(drop=True)

    def queue_size(self, df: pd.DataFrame, confidence_threshold: float = 0.9) -> int:
        return len(self.get_queue(df, confidence_threshold))

    # ------------------------------------------------------------------
    # Review actions
    # ------------------------------------------------------------------

    def _resolve_detection(self, detection_id: str) -> tuple:
        conn = self._conn()
        try:
            row = conn.execute(
                """
                SELECT d.id, d.image_id, d.detected_animal, d.confidence, i.filename
                FROM detections d
                JOIN images i ON i.id = d.image_id
                WHERE d.id = ?
                """,
                [detection_id],
            ).fetchone()
            if not row:
                raise ValueError(f"Detection {detection_id} not found")
            return row
        finally:
            conn.close()

    def accept(
        self,
        detection_id: str,
        filename: str = "",
        original_species: str = "",
        original_label: str = "Animal",
        confidence: float = 0.0,
        reviewer_id: str = "anonymous",
        notes: str = "",
        bbox: Optional[list] = None,
    ) -> int:
        """Accept the AI prediction as correct. Returns the new action row id."""
        det_id, image_id, species, conf, fname = self._resolve_detection(detection_id)
        conn = self._conn()
        try:
            if bbox is not None:
                conn.execute(
                    "UPDATE detections SET bbox = ? WHERE id = ?",
                    [json.dumps(bbox), det_id],
                )
                conn.commit()
        finally:
            conn.close()

        return self._record(
            detection_id=int(det_id),
            image_id=str(image_id),
            filename=filename or fname or "",
            action=ACTION_ACCEPT,
            original_species=original_species or species or "",
            corrected_species=original_species or species or "",
            original_label=original_label,
            corrected_label=original_label,
            confidence=confidence or conf or 0.0,
            reviewer_id=reviewer_id,
            notes=notes,
        )

    def correct(
        self,
        detection_id: str,
        original_species: str,
        corrected_species: str,
        filename: str = "",
        original_label: str = "Animal",
        corrected_label: str = "Animal",
        confidence: float = 0.0,
        reviewer_id: str = "anonymous",
        notes: str = "",
        bbox: Optional[list] = None,
    ) -> int:
        """Record a species correction. Returns the new action row id."""
        det_id, image_id, species, conf, fname = self._resolve_detection(detection_id)
        conn = self._conn()
        try:
            if bbox is not None:
                conn.execute(
                    "UPDATE detections SET detected_animal = ?, bbox = ? WHERE id = ?",
                    [corrected_species, json.dumps(bbox), det_id],
                )
            else:
                conn.execute(
                    "UPDATE detections SET detected_animal = ? WHERE id = ?",
                    [corrected_species, det_id],
                )
            conn.commit()
        finally:
            conn.close()

        return self._record(
            detection_id=int(det_id),
            image_id=str(image_id),
            filename=filename or fname or "",
            action=ACTION_CORRECT,
            original_species=original_species or species or "",
            corrected_species=corrected_species,
            original_label=original_label,
            corrected_label=corrected_label,
            confidence=confidence or conf or 0.0,
            reviewer_id=reviewer_id,
            notes=notes,
        )

    def reject(
        self,
        detection_id: str,
        filename: str = "",
        original_species: str = "",
        original_label: str = "Animal",
        confidence: float = 0.0,
        reviewer_id: str = "anonymous",
        notes: str = "",
    ) -> int:
        """Mark the detection as a false positive. Returns the new action row id."""
        det_id, image_id, species, conf, fname = self._resolve_detection(detection_id)
        conn = self._conn()
        try:
            conn.execute(
                "UPDATE detections SET detected_animal = 'Empty' WHERE id = ?",
                [det_id],
            )
            conn.commit()
        finally:
            conn.close()

        return self._record(
            detection_id=int(det_id),
            image_id=str(image_id),
            filename=filename or fname or "",
            action=ACTION_REJECT,
            original_species=original_species or species or "",
            corrected_species="",
            original_label=original_label,
            corrected_label="Empty",
            confidence=confidence or conf or 0.0,
            reviewer_id=reviewer_id,
            notes=notes,
        )

    def batch_accept(
        self,
        rows: list,
        reviewer_id: str = "anonymous",
    ) -> int:
        """Accept multiple detections at once."""
        count = 0
        for row in rows:
            det_id = row.get("detection_id") or row.get("id")
            if not det_id:
                continue
            self.accept(
                detection_id=str(det_id),
                filename=row.get("filename", ""),
                original_species=row.get("species_label", row.get("detected_animal", "")),
                original_label=row.get("primary_label", "Animal"),
                confidence=float(row.get("detection_confidence", 0)),
                reviewer_id=reviewer_id,
            )
            count += 1
        return count

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_actions_df(self) -> pd.DataFrame:
        """Return all review actions as a DataFrame."""
        conn = self._conn()
        try:
            return pd.read_sql_query(
                "SELECT * FROM review_actions ORDER BY reviewed_at DESC", conn
            )
        finally:
            conn.close()

    def get_corrections_df(self) -> pd.DataFrame:
        """Return only correction records (action = 'correct')."""
        conn = self._conn()
        try:
            return pd.read_sql_query(
                "SELECT * FROM review_actions WHERE action = 'correct' ORDER BY reviewed_at DESC",
                conn,
            )
        finally:
            conn.close()

    def get_stats(self) -> dict:
        """Return summary counts: total, accepted, corrected, rejected."""
        conn = self._conn()
        try:
            row = conn.execute("""
                SELECT
                    COUNT(*) AS total,
                    SUM(action = 'accept')  AS accepted,
                    SUM(action = 'correct') AS corrected,
                    SUM(action = 'reject')  AS rejected
                FROM review_actions
            """).fetchone()
            return {
                "total": row[0] or 0,
                "accepted": row[1] or 0,
                "corrected": row[2] or 0,
                "rejected": row[3] or 0,
            }
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_reviewed_detection_ids(self) -> set:
        conn = self._conn()
        try:
            rows = conn.execute(
                "SELECT DISTINCT detection_id FROM review_actions WHERE detection_id IS NOT NULL"
            ).fetchall()
            return {r[0] for r in rows}
        finally:
            conn.close()

    def _record(self, **kwargs) -> int:
        conn = self._conn()
        try:
            cur = conn.execute("""
                INSERT INTO review_actions (
                    detection_id, image_id, filename, action,
                    original_species, corrected_species,
                    original_label, corrected_label,
                    reviewer_id, confidence, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                kwargs.get("detection_id"),
                kwargs["image_id"],
                kwargs.get("filename", ""),
                kwargs["action"],
                kwargs.get("original_species", ""),
                kwargs.get("corrected_species", ""),
                kwargs.get("original_label", ""),
                kwargs.get("corrected_label", ""),
                kwargs.get("reviewer_id", "anonymous"),
                kwargs.get("confidence", 0.0),
                kwargs.get("notes", ""),
            ))
            conn.commit()
            return cur.lastrowid
        except sqlite3.IntegrityError as exc:
            raise ReviewConflictError("Detection already reviewed") from exc
        finally:
            conn.close()
