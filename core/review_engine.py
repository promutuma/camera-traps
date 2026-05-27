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


class ReviewEngine:
    """
    Persistent HITL review store backed by the shared SQLite database.

    Tables managed here:
        review_actions  — one row per review decision
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
        conn.commit()
        conn.close()

    # ------------------------------------------------------------------
    # Queue
    # ------------------------------------------------------------------

    def get_queue(
        self,
        df: pd.DataFrame,
        confidence_threshold: float = 0.9,
        reviewed_image_ids: Optional[set] = None,
    ) -> pd.DataFrame:
        """
        Return a sub-DataFrame of images that need review.

        Rules
        -----
        - primary_label == 'Animal'
        - detection_confidence < confidence_threshold
        - image_id not already reviewed (checked against DB + optional set)

        Parameters
        ----------
        df : pd.DataFrame
            Current session processed_data (one row per detection).
        confidence_threshold : float
            Images below this score enter the queue.
        reviewed_image_ids : set | None
            Extra set of image_ids to exclude (e.g. from the current session).

        Returns
        -------
        pd.DataFrame
            One row per unique image (highest-confidence detection kept),
            sorted by classifier disagreement priority (low/medium first) and confidence.
        """
        if df is None or df.empty:
            return pd.DataFrame()

        already_reviewed = self._get_reviewed_ids()
        if reviewed_image_ids:
            already_reviewed |= reviewed_image_ids

        animals = df[df.get("primary_label", pd.Series(dtype=str)) == "Animal"].copy() \
            if "primary_label" in df.columns else df.copy()

        if "detection_confidence" in animals.columns:
            queue = animals[animals["detection_confidence"] < confidence_threshold]
        else:
            queue = animals.copy()

        if "image_id" in queue.columns:
            queue = queue[~queue["image_id"].isin(already_reviewed)]

        # One row per image: keep the row with the highest confidence for display
        if "filepath" in queue.columns and "detection_confidence" in queue.columns:
            queue = (
                queue.sort_values("detection_confidence", ascending=False)
                .groupby("filepath", as_index=False)
                .first()
            )

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

    def accept(
        self,
        image_id: str,
        filename: str = "",
        original_species: str = "",
        original_label: str = "Animal",
        confidence: float = 0.0,
        reviewer_id: str = "anonymous",
        notes: str = "",
        bbox: Optional[list] = None,
    ) -> int:
        """Accept the AI prediction as correct. Returns the new action row id."""
        if bbox is not None:
            conn = self._conn()
            try:
                row = conn.execute(
                    "SELECT id FROM detections WHERE image_id = ? ORDER BY confidence DESC LIMIT 1",
                    [image_id]
                ).fetchone()
                if row:
                    conn.execute(
                        "UPDATE detections SET bbox = ? WHERE id = ?",
                        [json.dumps(bbox), row[0]]
                    )
                    conn.commit()
            finally:
                conn.close()

        return self._record(
            image_id=image_id,
            filename=filename,
            action=ACTION_ACCEPT,
            original_species=original_species,
            corrected_species=original_species,
            original_label=original_label,
            corrected_label=original_label,
            confidence=confidence,
            reviewer_id=reviewer_id,
            notes=notes,
        )

    def correct(
        self,
        image_id: str,
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
        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT id FROM detections WHERE image_id = ? ORDER BY confidence DESC LIMIT 1",
                [image_id]
            ).fetchone()
            if row:
                if bbox is not None:
                    conn.execute(
                        "UPDATE detections SET detected_animal = ?, bbox = ? WHERE id = ?",
                        [corrected_species, json.dumps(bbox), row[0]]
                    )
                else:
                    conn.execute(
                        "UPDATE detections SET detected_animal = ? WHERE id = ?",
                        [corrected_species, row[0]]
                    )
                conn.commit()
        finally:
            conn.close()

        return self._record(
            image_id=image_id,
            filename=filename,
            action=ACTION_CORRECT,
            original_species=original_species,
            corrected_species=corrected_species,
            original_label=original_label,
            corrected_label=corrected_label,
            confidence=confidence,
            reviewer_id=reviewer_id,
            notes=notes,
        )

    def reject(
        self,
        image_id: str,
        filename: str = "",
        original_species: str = "",
        original_label: str = "Animal",
        confidence: float = 0.0,
        reviewer_id: str = "anonymous",
        notes: str = "",
    ) -> int:
        """Mark the detection as a false positive. Returns the new action row id."""
        conn = self._conn()
        try:
            conn.execute(
                "UPDATE detections SET detected_animal = 'Empty' WHERE image_id = ?",
                [image_id]
            )
            conn.commit()
        finally:
            conn.close()

        return self._record(
            image_id=image_id,
            filename=filename,
            action=ACTION_REJECT,
            original_species=original_species,
            corrected_species="",
            original_label=original_label,
            corrected_label="Empty",
            confidence=confidence,
            reviewer_id=reviewer_id,
            notes=notes,
        )

    def batch_accept(
        self,
        rows: list,
        reviewer_id: str = "anonymous",
    ) -> int:
        """
        Accept multiple detections at once.

        Parameters
        ----------
        rows : list[dict]
            Each dict must contain at minimum: image_id, species_label,
            primary_label, detection_confidence, filename.

        Returns
        -------
        int : number of records saved
        """
        count = 0
        for row in rows:
            self.accept(
                image_id=row.get("image_id", ""),
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

    def _get_reviewed_ids(self) -> set:
        conn = self._conn()
        try:
            rows = conn.execute(
                "SELECT DISTINCT image_id FROM review_actions"
            ).fetchall()
            return {r[0] for r in rows}
        finally:
            conn.close()

    def _record(self, **kwargs) -> int:
        conn = self._conn()
        try:
            cur = conn.execute("""
                INSERT INTO review_actions (
                    image_id, filename, action,
                    original_species, corrected_species,
                    original_label, corrected_label,
                    reviewer_id, confidence, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
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
        finally:
            conn.close()
