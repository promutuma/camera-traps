"""
Automated QC Flag System
Evaluates camera trap data against quality-control rules and returns
a DataFrame of flagged records ready for display or export.
"""

import re
import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import Optional


# Flag codes match the spec (Section 4.9)
FLAG_INSUFFICIENT_EFFORT    = "INSUFFICIENT_EFFORT"
FLAG_LOW_FUNCTIONALITY      = "LOW_FUNCTIONALITY"
FLAG_POSSIBLE_FAILURE       = "POSSIBLE_FAILURE"
FLAG_MODEL_ERROR            = "MODEL_ERROR"
FLAG_OUT_OF_RANGE           = "OUT_OF_RANGE"
FLAG_DUPLICATE_STATION      = "DUPLICATE_STATION"
FLAG_TIMESTAMP_INCONSISTENCY = "TIMESTAMP_INCONSISTENCY"
FLAG_LOW_CONFIDENCE         = "LOW_CONFIDENCE"


def _safe_date(val) -> Optional[date]:
    try:
        return pd.to_datetime(val, dayfirst=True).date()
    except Exception:
        return None


class QCEngine:
    """
    Run all QC checks against processed detection data and an optional
    IDE summary. Returns a unified flags DataFrame.

    Parameters
    ----------
    min_trap_nights : int
        Stations with fewer trap nights than this are flagged INSUFFICIENT_EFFORT.
    min_functionality_pct : float
        Stations whose active-night rate drops below this (0–100) are flagged
        LOW_FUNCTIONALITY.
    min_zero_detection_nights : int
        Stations with zero detections but at least this many trap nights are
        flagged POSSIBLE_FAILURE.
    confidence_threshold : float
        Detections below this confidence get a LOW_CONFIDENCE flag.
    clock_drift_minutes : float
        Timestamps that deviate from the per-station median by more than this
        many minutes are flagged TIMESTAMP_INCONSISTENCY.
    survey_start : date | None
        Records before this date are flagged OUT_OF_RANGE.
    survey_end : date | None
        Records after this date are flagged OUT_OF_RANGE.
    """

    def __init__(
        self,
        min_trap_nights: int = 60,
        min_functionality_pct: float = 90.0,
        min_zero_detection_nights: int = 10,
        confidence_threshold: float = 0.35,
        clock_drift_minutes: float = 5.0,
        survey_start: Optional[date] = None,
        survey_end: Optional[date] = None,
    ):
        self.min_trap_nights = min_trap_nights
        self.min_functionality_pct = min_functionality_pct
        self.min_zero_detection_nights = min_zero_detection_nights
        self.confidence_threshold = confidence_threshold
        self.clock_drift_minutes = clock_drift_minutes
        self.survey_start = survey_start
        self.survey_end = survey_end

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_all_checks(
        self,
        df: pd.DataFrame,
        trap_nights: dict,
        ide_summary: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Run every QC check and return a single flags DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Processed detection data (session_state.processed_data or history).
        trap_nights : dict
            station_id → number of active trap nights.
        ide_summary : pd.DataFrame | None
            Output of IndependenceEngine.get_ide_summary(), if available.

        Returns
        -------
        pd.DataFrame with columns:
            flag_code, severity, station_id, detail, affected_value, checked_at
        """
        flags = []

        flags.extend(self._check_model_errors(df))
        flags.extend(self._check_low_confidence(df))
        flags.extend(self._check_out_of_range(df))
        flags.extend(self._check_duplicate_stations(df))
        flags.extend(self._check_timestamp_inconsistency(df))
        flags.extend(self._check_effort(df, trap_nights))
        flags.extend(self._check_zero_detections(df, trap_nights))

        result = pd.DataFrame(flags) if flags else pd.DataFrame(
            columns=["flag_code", "severity", "station_id", "detail",
                     "affected_value", "checked_at"]
        )
        return result

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _flag(self, code, severity, station_id, detail, affected_value=""):
        return {
            "flag_code": code,
            "severity": severity,
            "station_id": str(station_id),
            "detail": detail,
            "affected_value": str(affected_value),
            "checked_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    def _check_model_errors(self, df: pd.DataFrame) -> list:
        """Flag rows with null or zero confidence scores."""
        flags = []
        if "detection_confidence" not in df.columns:
            return flags
        bad = df[
            df["detection_confidence"].isna() |
            (df["detection_confidence"] == 0.0)
        ]
        for station in bad.get("station_id", pd.Series(dtype=str)).unique():
            count = len(bad[bad.get("station_id", pd.Series(dtype=str)) == station])
            flags.append(self._flag(
                FLAG_MODEL_ERROR, "Warning", station,
                f"{count} detection(s) have null/zero confidence score",
                count,
            ))
        if bad.empty is False and "station_id" not in df.columns:
            flags.append(self._flag(
                FLAG_MODEL_ERROR, "Warning", "Unknown",
                f"{len(bad)} detection(s) have null/zero confidence score",
                len(bad),
            ))
        return flags

    def _check_low_confidence(self, df: pd.DataFrame) -> list:
        """Flag animal detections below the confidence threshold."""
        flags = []
        if "detection_confidence" not in df.columns or "primary_label" not in df.columns:
            return flags
        animals = df[df["primary_label"] == "Animal"]
        low = animals[animals["detection_confidence"] < self.confidence_threshold]
        if low.empty:
            return flags

        station_col = "station_id" if "station_id" in low.columns else None
        if station_col:
            for station, grp in low.groupby(station_col):
                flags.append(self._flag(
                    FLAG_LOW_CONFIDENCE, "Info", station,
                    f"{len(grp)} animal detection(s) below confidence threshold "
                    f"({self.confidence_threshold})",
                    f"{len(grp)} detections",
                ))
        else:
            flags.append(self._flag(
                FLAG_LOW_CONFIDENCE, "Info", "All stations",
                f"{len(low)} animal detection(s) below confidence threshold "
                f"({self.confidence_threshold})",
                len(low),
            ))
        return flags

    def _check_out_of_range(self, df: pd.DataFrame) -> list:
        """Flag records whose capture date falls outside the survey period."""
        flags = []
        if not (self.survey_start or self.survey_end):
            return flags
        if "date" not in df.columns:
            return flags

        dates = df["date"].apply(_safe_date)
        station_col = "station_id" if "station_id" in df.columns else None

        out_mask = pd.Series(False, index=df.index)
        if self.survey_start:
            out_mask |= dates.apply(lambda d: d is not None and d < self.survey_start)
        if self.survey_end:
            out_mask |= dates.apply(lambda d: d is not None and d > self.survey_end)

        out_df = df[out_mask]
        if out_df.empty:
            return flags

        if station_col:
            for station, grp in out_df.groupby(station_col):
                flags.append(self._flag(
                    FLAG_OUT_OF_RANGE, "Warning", station,
                    f"{len(grp)} record(s) outside survey period "
                    f"({self.survey_start} – {self.survey_end})",
                    len(grp),
                ))
        else:
            flags.append(self._flag(
                FLAG_OUT_OF_RANGE, "Warning", "All stations",
                f"{len(out_df)} record(s) outside survey period",
                len(out_df),
            ))
        return flags

    def _check_duplicate_stations(self, df: pd.DataFrame) -> list:
        """Flag duplicate station IDs (same ID assigned to multiple distinct cameras)."""
        flags = []
        if "station_id" not in df.columns:
            return flags

        # Heuristic: same station_id, but filenames suggest different camera prefixes
        if "filename" not in df.columns:
            return flags

        def _prefix(fn):
            parts = str(fn).split("_")
            return parts[0] if len(parts) > 1 else str(fn)[:6]

        station_prefixes = (
            df.groupby("station_id")["filename"]
            .apply(lambda s: s.apply(_prefix).nunique())
        )
        dups = station_prefixes[station_prefixes > 1]
        for station, prefix_count in dups.items():
            flags.append(self._flag(
                FLAG_DUPLICATE_STATION, "Error", station,
                f"Station ID maps to {prefix_count} different filename prefixes — "
                "possible duplicate station assignment",
                prefix_count,
            ))
        return flags

    def _check_timestamp_inconsistency(self, df: pd.DataFrame) -> list:
        """
        Detect per-station clock drift: timestamps that deviate more than
        clock_drift_minutes from the station-level median for that date.
        """
        flags = []
        if "datetime_parsed" not in df.columns or "station_id" not in df.columns:
            return flags

        work = df[["station_id", "datetime_parsed"]].dropna(subset=["datetime_parsed"]).copy()
        if work.empty:
            return flags

        work["_date"] = pd.to_datetime(work["datetime_parsed"], errors="coerce").dt.date
        work["_ts"] = pd.to_datetime(work["datetime_parsed"], errors="coerce").astype(np.int64) // 10**9

        for (station, day), grp in work.groupby(["station_id", "_date"]):
            if len(grp) < 3:
                continue
            median_ts = grp["_ts"].median()
            drift = (grp["_ts"] - median_ts).abs() / 60  # minutes
            outliers = drift[drift > self.clock_drift_minutes]
            if not outliers.empty:
                flags.append(self._flag(
                    FLAG_TIMESTAMP_INCONSISTENCY, "Warning", station,
                    f"{len(outliers)} timestamp(s) on {day} deviate >"
                    f"{self.clock_drift_minutes} min from station median",
                    f"max drift: {drift.max():.1f} min",
                ))
        return flags

    def _check_effort(self, df: pd.DataFrame, trap_nights: dict) -> list:
        """Flag stations with insufficient trap nights."""
        flags = []
        stations = (
            df["station_id"].unique().tolist()
            if "station_id" in df.columns
            else list(trap_nights.keys())
        )
        for station in stations:
            nights = trap_nights.get(station, trap_nights.get(str(station), 0))
            if nights < self.min_trap_nights:
                flags.append(self._flag(
                    FLAG_INSUFFICIENT_EFFORT, "Warning", station,
                    f"Only {nights} trap night(s); minimum is {self.min_trap_nights}",
                    nights,
                ))
        return flags

    def _check_zero_detections(self, df: pd.DataFrame, trap_nights: dict) -> list:
        """Flag stations with zero animal detections despite adequate trap nights."""
        flags = []
        if "station_id" not in df.columns or "primary_label" not in df.columns:
            return flags

        animal_stations = set(
            df[df["primary_label"] == "Animal"]["station_id"].unique().tolist()
        )
        all_stations = set(df["station_id"].unique().tolist()) | set(trap_nights.keys())

        for station in all_stations:
            nights = trap_nights.get(station, 0)
            if station not in animal_stations and nights >= self.min_zero_detection_nights:
                flags.append(self._flag(
                    FLAG_POSSIBLE_FAILURE, "Error", station,
                    f"Zero animal detections after {nights} trap night(s)",
                    f"{nights} nights, 0 detections",
                ))
        return flags
