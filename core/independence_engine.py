"""
Independence Rule Engine
Groups camera trap detections into Independent Detection Events (IDEs)
using a configurable time-window rule (default: 30 minutes).
"""

import re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional


# Date/time format strings to try when parsing OCR output
_DATE_FORMATS = [
    "%d/%m/%Y", "%m/%d/%Y", "%Y-%m-%d",
    "%d-%m-%Y", "%Y/%m/%d", "%d.%m.%Y",
]
_TIME_FORMATS = ["%H:%M:%S", "%H:%M", "%I:%M:%S %p", "%I:%M %p"]


def _parse_datetime(date_str, time_str) -> Optional[datetime]:
    """Combine date and time strings into a datetime, trying multiple formats."""
    if not date_str or not time_str:
        return None
    date_str = str(date_str).strip()
    time_str = str(time_str).strip()
    for dfmt in _DATE_FORMATS:
        for tfmt in _TIME_FORMATS:
            try:
                return datetime.strptime(f"{date_str} {time_str}", f"{dfmt} {tfmt}")
            except ValueError:
                continue
    # Last resort: let pandas figure it out
    try:
        return pd.to_datetime(f"{date_str} {time_str}", dayfirst=True)
    except Exception:
        return None


def _extract_station_id(filename: str) -> str:
    """
    Try to extract station ID from a filename following the convention
    YYYYMMDD_StationID_DeploymentID or similar. Falls back to 'Station-1'.
    """
    if not filename:
        return "Station-1"
    stem = re.sub(r"\.\w+$", "", filename)  # strip extension
    parts = stem.split("_")
    # Convention: first part looks like a date (all digits, 8 chars)
    if len(parts) >= 2 and re.match(r"^\d{8}$", parts[0]):
        return parts[1]
    # No date prefix — use the first underscore-delimited token as station ID
    if len(parts) >= 2:
        return parts[0]
    return "Station-1"


def _sanitize(value: str) -> str:
    """Make a string safe to embed in an IDE ID."""
    return re.sub(r"[^A-Za-z0-9]", "_", str(value)).strip("_") or "Unknown"


class IndependenceEngine:
    """
    Applies the N-minute independence rule to a detection DataFrame.

    Required input columns (all can come from the existing processed_data):
        - filename       : image filename (used to infer station_id if absent)
        - date           : capture date string (from OCR)
        - time           : capture time string (from OCR)
        - species_label  : species name (cleaned string)
        - primary_label  : 'Animal' / 'Person' / 'Vehicle' / 'Empty'

    Optional input columns:
        - station_id     : pre-assigned station identifier

    Added output columns:
        - station_id     : extracted or provided station ID
        - datetime_parsed: combined datetime (NaT if unparseable)
        - ide_id         : unique IDE identifier string
        - ide_group      : sequential integer within the dataset (1-based)
    """

    def __init__(self, window_minutes: int = 30):
        self.window_minutes = window_minutes

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_ides(self, df: pd.DataFrame, default_station: str = "Station-1") -> pd.DataFrame:
        """
        Assign IDE IDs to a detection DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            One row per detection (may have multiple rows per image when
            multiple animals are detected in the same frame).
        default_station : str
            Station ID to use when it cannot be inferred from the filename.

        Returns
        -------
        pd.DataFrame
            Input DataFrame with four new columns added.
        """
        if df is None or df.empty:
            return df

        result = df.copy()

        # --- 1. Station ID --------------------------------------------------
        if "station_id" not in result.columns:
            if "filename" in result.columns:
                result["station_id"] = result["filename"].apply(
                    lambda fn: _extract_station_id(str(fn)) if fn else default_station
                )
            else:
                result["station_id"] = default_station

        # Fill blanks
        result["station_id"] = result["station_id"].fillna(default_station).replace("", default_station)

        # --- 2. Parse datetimes --------------------------------------------
        if "date" in result.columns and "time" in result.columns:
            result["datetime_parsed"] = result.apply(
                lambda r: _parse_datetime(r.get("date"), r.get("time")), axis=1
            )
        else:
            result["datetime_parsed"] = pd.NaT

        # --- 3. Apply independence rule per (station, species) group --------
        result["ide_id"] = ""
        result["ide_group"] = 0

        global_counter = [0]  # mutable counter shared across groups

        # Only apply to animal detections; non-animals get their own single-image IDEs
        animal_mask = result.get("primary_label", pd.Series(["Animal"] * len(result))) == "Animal"

        def _assign_group(sub_df):
            """Assign IDE group numbers within a (station, species) subset."""
            sub = sub_df.sort_values("datetime_parsed", na_position="last").copy()
            group_nums = []
            last_time = None
            current_group = None

            for _, row in sub.iterrows():
                dt = row["datetime_parsed"]
                if pd.isna(dt) or last_time is None or (dt - last_time) > timedelta(minutes=self.window_minutes):
                    global_counter[0] += 1
                    current_group = global_counter[0]
                last_time = dt if not pd.isna(dt) else last_time
                group_nums.append(current_group)

            sub["ide_group"] = group_nums
            return sub

        # Process animals by (station_id, species)
        animal_df = result[animal_mask].copy()
        if not animal_df.empty:
            species_col = "species_label" if "species_label" in animal_df.columns else "detected_animal"
            animal_df["_species_key"] = animal_df[species_col].fillna("Unknown").apply(
                lambda s: re.sub(r"\s+\d+(\.\d+)?", "", str(s)).strip() or "Unknown"
            )
            processed_parts = []
            for (station, species), grp in animal_df.groupby(["station_id", "_species_key"], sort=False):
                grp = _assign_group(grp)
                grp["ide_id"] = grp["ide_group"].apply(
                    lambda g: f"{_sanitize(station)}__{_sanitize(species)}__{g:05d}"
                )
                processed_parts.append(grp)

            if processed_parts:
                animal_processed = pd.concat(processed_parts)
                animal_df_cols = ["ide_id", "ide_group", "datetime_parsed", "station_id", "_species_key"]
                for col in [c for c in animal_df_cols if c in animal_processed.columns]:
                    result.loc[animal_processed.index, col] = animal_processed[col]

        # Non-animal detections: each image = its own event
        non_animal_mask = ~animal_mask
        for idx in result.index[non_animal_mask]:
            global_counter[0] += 1
            label = result.at[idx, "primary_label"] if "primary_label" in result.columns else "Unknown"
            station = result.at[idx, "station_id"]
            result.at[idx, "ide_group"] = global_counter[0]
            result.at[idx, "ide_id"] = f"{_sanitize(station)}__{_sanitize(label)}__{global_counter[0]:05d}"

        # Drop temporary column
        if "_species_key" in result.columns:
            result.drop(columns=["_species_key"], inplace=True)

        return result

    def get_ide_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return a summary DataFrame with one row per IDE.

        Columns: ide_id, ide_group, station_id, species, first_detection,
                 last_detection, duration_minutes, image_count, datetime_parsed
        """
        if df is None or df.empty or "ide_id" not in df.columns:
            return pd.DataFrame()

        species_col = "species_label" if "species_label" in df.columns else "detected_animal"

        agg = df.groupby("ide_id", sort=False).agg(
            ide_group=("ide_group", "first"),
            station_id=("station_id", "first"),
            species=(species_col, lambda s: re.sub(r"\s+\d+(\.\d+)?", "", str(s.iloc[0])).strip()),
            first_detection=("datetime_parsed", "min"),
            last_detection=("datetime_parsed", "max"),
            image_count=("filename", "count") if "filename" in df.columns else ("ide_id", "count"),
            max_confidence=("detection_confidence", "max") if "detection_confidence" in df.columns else ("ide_id", "count"),
        ).reset_index()

        agg["duration_minutes"] = (
            (agg["last_detection"] - agg["first_detection"])
            .dt.total_seconds()
            .div(60)
            .round(1)
        )
        return agg.sort_values("first_detection")

    def compute_rai(self, ide_summary: pd.DataFrame, trap_nights: dict) -> pd.DataFrame:
        """
        Compute Relative Abundance Index per species per station.

        RAI = Independent Detection Events / Total Camera Trap Nights

        Parameters
        ----------
        ide_summary : pd.DataFrame
            Output of get_ide_summary().
        trap_nights : dict
            Mapping of station_id → number of trap nights.

        Returns
        -------
        pd.DataFrame
            Columns: station_id, species, ide_count, trap_nights, rai
        """
        if ide_summary.empty:
            return pd.DataFrame()

        counts = (
            ide_summary.groupby(["station_id", "species"])
            .size()
            .reset_index(name="ide_count")
        )
        counts["trap_nights"] = counts["station_id"].map(trap_nights).fillna(1)
        counts["rai"] = (counts["ide_count"] / counts["trap_nights"]).round(4)
        return counts.sort_values(["station_id", "rai"], ascending=[True, False])

    # ------------------------------------------------------------------
    # Ecological Indicators
    # ------------------------------------------------------------------

    def compute_species_richness(self, ide_summary: pd.DataFrame) -> pd.DataFrame:
        """
        Species richness per station: count of unique wildlife species detected.

        Returns a DataFrame with columns:
            station_id, species_count, singleton_count, species_list
        Singleton species (detected in only one IDE) are flagged separately.
        """
        if ide_summary is None or ide_summary.empty:
            return pd.DataFrame()

        wildlife = ide_summary[~ide_summary["species"].str.lower().isin(
            ["empty", "person", "vehicle", "unknown"]
        )].copy()

        ide_per_species = (
            wildlife.groupby(["station_id", "species"])
            .size()
            .reset_index(name="ide_count")
        )

        richness = (
            ide_per_species.groupby("station_id")
            .apply(lambda g: pd.Series({
                "species_count": len(g),
                "singleton_count": int((g["ide_count"] == 1).sum()),
                "species_list": ", ".join(sorted(g["species"].tolist())),
            }))
            .reset_index()
        )
        return richness.sort_values("species_count", ascending=False)

    def compute_species_accumulation(self, ide_summary: pd.DataFrame) -> pd.DataFrame:
        """
        Species accumulation curve: cumulative unique species vs. cumulative IDEs.

        Returns a DataFrame with columns:
            cumulative_ides, cumulative_species
        Sorted by first_detection so the curve reflects discovery order over time.
        """
        if ide_summary is None or ide_summary.empty:
            return pd.DataFrame()

        wildlife = ide_summary[~ide_summary["species"].str.lower().isin(
            ["empty", "person", "vehicle", "unknown"]
        )].copy()

        wildlife = wildlife.sort_values("first_detection", na_position="last").reset_index(drop=True)

        seen = set()
        records = []
        for i, row in wildlife.iterrows():
            seen.add(row["species"])
            records.append({"cumulative_ides": i + 1, "cumulative_species": len(seen)})

        return pd.DataFrame(records)

    def compute_mean_group_size(self, enriched_df: pd.DataFrame) -> pd.DataFrame:
        """
        Mean group size per species per station.

        Group size is approximated as the number of animal bounding-box detections
        within a single IDE (one detection row = one identified individual).

        Returns a DataFrame with columns:
            station_id, species, mean_group_size, min_group, max_group,
            std_group, ide_count, outlier_ides
        Outliers: IDEs whose group size exceeds mean + 3*std.
        """
        if enriched_df is None or enriched_df.empty or "ide_id" not in enriched_df.columns:
            return pd.DataFrame()

        species_col = "species_label" if "species_label" in enriched_df.columns else "detected_animal"

        animals = enriched_df[
            enriched_df.get("primary_label", pd.Series(["Animal"] * len(enriched_df))) == "Animal"
        ].copy()

        if animals.empty:
            return pd.DataFrame()

        animals["_species_clean"] = animals[species_col].apply(
            lambda s: re.sub(r"\s+\d+(\.\d+)?", "", str(s)).strip()
        )

        group_sizes = (
            animals.groupby(["station_id", "ide_id", "_species_clean"])
            .size()
            .reset_index(name="group_size")
        )

        result = (
            group_sizes.groupby(["station_id", "_species_clean"])
            .agg(
                mean_group_size=("group_size", "mean"),
                min_group=("group_size", "min"),
                max_group=("group_size", "max"),
                std_group=("group_size", "std"),
                ide_count=("group_size", "count"),
            )
            .reset_index()
            .rename(columns={"_species_clean": "species"})
        )

        result["mean_group_size"] = result["mean_group_size"].round(2)
        result["std_group"] = result["std_group"].fillna(0).round(2)

        # Flag outlier IDEs (group_size > mean + 3*std)
        outlier_counts = []
        for _, row in result.iterrows():
            threshold = row["mean_group_size"] + 3 * row["std_group"]
            mask = (
                (group_sizes["station_id"] == row["station_id"]) &
                (group_sizes["_species_clean"] == row["species"]) &
                (group_sizes["group_size"] > threshold)
            )
            outlier_counts.append(int(mask.sum()))
        result["outlier_ides"] = outlier_counts

        return result.sort_values(["station_id", "mean_group_size"], ascending=[True, False])

    def compute_visitation_rate(
        self, ide_summary: pd.DataFrame, trap_nights: dict
    ) -> tuple:
        """
        Visitation rate per species per station + time-of-day heatmap.

        Returns
        -------
        visit_rate_df : pd.DataFrame
            Columns: station_id, species, visit_count, trap_nights,
                     visit_rate, diurnal_pct, nocturnal_pct
        heatmap_df : pd.DataFrame
            Index = species, columns = 2-hour time blocks (e.g. "06-08"),
            values = IDE count in that block.
        """
        if ide_summary is None or ide_summary.empty:
            return pd.DataFrame(), pd.DataFrame()

        wildlife = ide_summary[~ide_summary["species"].str.lower().isin(
            ["empty", "person", "vehicle", "unknown"]
        )].copy()

        if wildlife.empty:
            return pd.DataFrame(), pd.DataFrame()

        # --- Visit rate ---
        counts = (
            wildlife.groupby(["station_id", "species"])
            .size()
            .reset_index(name="visit_count")
        )
        counts["trap_nights"] = counts["station_id"].map(trap_nights).fillna(1)
        counts["visit_rate"] = (counts["visit_count"] / counts["trap_nights"]).round(4)

        # Diurnal / nocturnal split (06:00–18:00 = diurnal)
        if "first_detection" in wildlife.columns:
            wildlife["_hour"] = pd.to_datetime(
                wildlife["first_detection"], errors="coerce"
            ).dt.hour.fillna(-1)

            def _split(grp):
                total = len(grp)
                if total == 0:
                    return pd.Series({"diurnal_pct": 0.0, "nocturnal_pct": 0.0})
                diurnal = ((grp["_hour"] >= 6) & (grp["_hour"] < 18)).sum()
                return pd.Series({
                    "diurnal_pct": round(diurnal / total * 100, 1),
                    "nocturnal_pct": round((total - diurnal) / total * 100, 1),
                })

            split_df = (
                wildlife.groupby(["station_id", "species"])
                .apply(_split)
                .reset_index()
            )
            counts = counts.merge(split_df, on=["station_id", "species"], how="left")
        else:
            counts["diurnal_pct"] = None
            counts["nocturnal_pct"] = None

        visit_rate_df = counts.sort_values(["station_id", "visit_rate"], ascending=[True, False])

        # --- Time-of-day heatmap (2-hour bins × species) ---
        heatmap_df = pd.DataFrame()
        if "first_detection" in wildlife.columns and "_hour" in wildlife.columns:
            bins = list(range(0, 25, 2))
            labels = [f"{h:02d}-{h+2:02d}" for h in range(0, 24, 2)]
            wildlife["_time_bin"] = pd.cut(
                wildlife["_hour"],
                bins=bins,
                labels=labels,
                right=False,
                include_lowest=True,
            )
            heatmap_df = (
                wildlife.groupby(["species", "_time_bin"], observed=True)
                .size()
                .unstack(fill_value=0)
            )

        return visit_rate_df, heatmap_df
