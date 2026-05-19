"""
Corridor Movement Analysis
Analyses directed wildlife movement between paired camera stations.

Assumptions:
  - Stations with GPS coordinates can be connected if within max_distance_m.
  - A directed movement event is when the same species appears at station A,
    then station B (or vice versa) within movement_window_hours.
  - Passage frequency = directed movements per 100 trap-nights per pair.
  - Bottleneck = pair whose passage frequency is below the 25th percentile
    of all pairs that recorded at least one passage.
"""

import math
import pandas as pd
import numpy as np
from typing import Optional


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return distance in metres between two WGS-84 coordinates."""
    R = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


class CorridorAnalyzer:
    """
    Analyses directional wildlife movement between adjacent camera stations.

    Parameters
    ----------
    max_distance_m : float
        Maximum inter-station distance to consider a corridor pair (default 1 000 m).
    movement_window_hours : float
        Maximum time gap to infer a directed movement event (default 24 h).
    bottleneck_percentile : float
        Pairs below this passage-frequency percentile are flagged as bottlenecks.
    """

    def __init__(
        self,
        max_distance_m: float = 1000.0,
        movement_window_hours: float = 24.0,
        bottleneck_percentile: float = 25.0,
    ):
        self.max_distance_m = max_distance_m
        self.movement_window_hours = movement_window_hours
        self.bottleneck_percentile = bottleneck_percentile

    # ------------------------------------------------------------------
    # Station pairs
    # ------------------------------------------------------------------

    def compute_station_pairs(self, stations_df: pd.DataFrame) -> pd.DataFrame:
        """
        Return all station pairs within max_distance_m.

        Parameters
        ----------
        stations_df : pd.DataFrame
            Must contain columns: station_id, gps_lat, gps_lon.

        Returns
        -------
        pd.DataFrame with columns:
            station_a, station_b, lat_a, lon_a, lat_b, lon_b, distance_m
        """
        rows = []
        valid = stations_df.dropna(subset=["gps_lat", "gps_lon"]).copy()
        valid = valid[(valid["gps_lat"] != 0) | (valid["gps_lon"] != 0)]

        sids = valid["station_id"].tolist()
        for i in range(len(sids)):
            for j in range(i + 1, len(sids)):
                a = valid.iloc[i]
                b = valid.iloc[j]
                try:
                    dist = _haversine_m(
                        float(a["gps_lat"]), float(a["gps_lon"]),
                        float(b["gps_lat"]), float(b["gps_lon"]),
                    )
                except (TypeError, ValueError):
                    continue
                if dist <= self.max_distance_m:
                    rows.append({
                        "station_a":  str(a["station_id"]),
                        "station_b":  str(b["station_id"]),
                        "lat_a":      float(a["gps_lat"]),
                        "lon_a":      float(a["gps_lon"]),
                        "lat_b":      float(b["gps_lat"]),
                        "lon_b":      float(b["gps_lon"]),
                        "distance_m": round(dist, 1),
                    })

        if rows:
            return pd.DataFrame(rows).sort_values("distance_m").reset_index(drop=True)
        return pd.DataFrame(columns=["station_a", "station_b", "lat_a", "lon_a",
                                     "lat_b", "lon_b", "distance_m"])

    # ------------------------------------------------------------------
    # Directed flows
    # ------------------------------------------------------------------

    def compute_corridor_flows(
        self,
        ide_summary: pd.DataFrame,
        station_pairs: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Detect directed movement events between paired stations.

        For each species × pair, find all cases where the species appeared at
        one station and then the other within movement_window_hours. Both
        directions (A→B and B→A) are counted separately.

        Parameters
        ----------
        ide_summary : pd.DataFrame
            One row per IDE; must have: station_id, species, first_detection.
        station_pairs : pd.DataFrame
            Output of compute_station_pairs().

        Returns
        -------
        pd.DataFrame with columns:
            station_a, station_b, distance_m, species,
            flow_a_to_b, flow_b_to_a, total_passages,
            dominant_direction, asymmetry_ratio
        """
        if ide_summary is None or ide_summary.empty or station_pairs is None or station_pairs.empty:
            return pd.DataFrame()

        # Parse first_detection to datetime
        events = ide_summary[["station_id", "species", "first_detection"]].copy()
        events["dt"] = pd.to_datetime(events["first_detection"], errors="coerce")
        events = events.dropna(subset=["dt"])
        events["station_id"] = events["station_id"].astype(str)

        window_td = pd.Timedelta(hours=self.movement_window_hours)
        rows = []

        for _, pair in station_pairs.iterrows():
            sa, sb = str(pair["station_a"]), str(pair["station_b"])
            ev_a = events[events["station_id"] == sa]
            ev_b = events[events["station_id"] == sb]
            if ev_a.empty or ev_b.empty:
                continue

            all_species = set(ev_a["species"].unique()) | set(ev_b["species"].unique())

            for sp in all_species:
                sp_a = ev_a[ev_a["species"] == sp].sort_values("dt")
                sp_b = ev_b[ev_b["species"] == sp].sort_values("dt")
                if sp_a.empty or sp_b.empty:
                    continue

                a_to_b = _count_directed(sp_a["dt"].values, sp_b["dt"].values, window_td)
                b_to_a = _count_directed(sp_b["dt"].values, sp_a["dt"].values, window_td)
                total = a_to_b + b_to_a
                if total == 0:
                    continue

                dominant = sa if a_to_b >= b_to_a else sb
                asym = round(abs(a_to_b - b_to_a) / total, 3) if total > 0 else 0.0

                rows.append({
                    "station_a":        sa,
                    "station_b":        sb,
                    "distance_m":       round(pair["distance_m"], 1),
                    "species":          sp,
                    "flow_a_to_b":      a_to_b,
                    "flow_b_to_a":      b_to_a,
                    "total_passages":   total,
                    "dominant_direction": f"{dominant} →",
                    "asymmetry_ratio":  asym,
                })

        if rows:
            return pd.DataFrame(rows).sort_values(
                ["total_passages"], ascending=False
            ).reset_index(drop=True)
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # Passage frequency
    # ------------------------------------------------------------------

    def compute_passage_frequency(
        self,
        corridor_flows: pd.DataFrame,
        trap_nights_map: dict,
    ) -> pd.DataFrame:
        """
        Add a passage_rate column = total_passages / mean_trap_nights_of_pair * 100.

        Parameters
        ----------
        trap_nights_map : dict
            {station_id: trap_nights}  (from StationManager.compute_trap_nights())
        """
        if corridor_flows is None or corridor_flows.empty:
            return pd.DataFrame()

        df = corridor_flows.copy()

        def _rate(row):
            tn_a = trap_nights_map.get(str(row["station_a"]), 0)
            tn_b = trap_nights_map.get(str(row["station_b"]), 0)
            mean_tn = (tn_a + tn_b) / 2
            if mean_tn <= 0:
                return 0.0
            return round(row["total_passages"] / mean_tn * 100, 3)

        df["passage_rate_per_100tn"] = df.apply(_rate, axis=1)
        return df.sort_values("passage_rate_per_100tn", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Bottleneck identification
    # ------------------------------------------------------------------

    def identify_bottlenecks(self, corridor_flows: pd.DataFrame) -> pd.DataFrame:
        """
        Flag pairs whose total_passages is below the bottleneck_percentile.

        Returns the subset of corridor_flows that are bottlenecks, with an
        extra 'bottleneck' column set to True.
        """
        if corridor_flows is None or corridor_flows.empty:
            return pd.DataFrame()

        active = corridor_flows[corridor_flows["total_passages"] > 0].copy()
        if active.empty:
            return pd.DataFrame()

        threshold = np.percentile(active["total_passages"].values, self.bottleneck_percentile)
        bottlenecks = active[active["total_passages"] <= threshold].copy()
        bottlenecks["bottleneck"] = True
        return bottlenecks.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Corridor utilisation
    # ------------------------------------------------------------------

    def compute_corridor_utilisation(
        self,
        corridor_flows: pd.DataFrame,
        station_pairs: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Per-species utilisation: % of corridor pairs that recorded the species.

        Returns DataFrame: species, pairs_used, total_pairs, utilisation_pct
        """
        if station_pairs is None or station_pairs.empty:
            return pd.DataFrame()

        total_pairs = len(station_pairs)

        if corridor_flows is None or corridor_flows.empty:
            return pd.DataFrame()

        util = (
            corridor_flows.groupby("species")
            .apply(lambda g: len(g[["station_a", "station_b"]]
                               .drop_duplicates()))
            .reset_index()
        )
        util.columns = ["species", "pairs_used"]
        util["total_pairs"] = total_pairs
        util["utilisation_pct"] = (util["pairs_used"] / total_pairs * 100).round(1)
        return util.sort_values("utilisation_pct", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Combined summary
    # ------------------------------------------------------------------

    def get_full_summary(
        self,
        ide_summary: pd.DataFrame,
        stations_df: pd.DataFrame,
        trap_nights_map: Optional[dict] = None,
    ) -> dict:
        """
        Run the full corridor analysis pipeline.

        Returns
        -------
        dict with keys:
            pairs, flows, flows_with_rate, bottlenecks, utilisation
        """
        pairs = self.compute_station_pairs(stations_df)
        flows = self.compute_corridor_flows(ide_summary, pairs)
        flows_rate = self.compute_passage_frequency(flows, trap_nights_map or {})
        bottlenecks = self.identify_bottlenecks(flows)
        utilisation = self.compute_corridor_utilisation(flows, pairs)

        return {
            "pairs":           pairs,
            "flows":           flows,
            "flows_with_rate": flows_rate,
            "bottlenecks":     bottlenecks,
            "utilisation":     utilisation,
        }


# ------------------------------------------------------------------
# Internal helper
# ------------------------------------------------------------------

def _count_directed(
    source_dts: np.ndarray,
    target_dts: np.ndarray,
    window: pd.Timedelta,
) -> int:
    """
    Count directed movement events: each source event followed by ≥1 target
    event within window_td, counting only the first such follow-up per source.
    """
    count = 0
    for src in source_dts:
        for tgt in target_dts:
            diff = pd.Timestamp(tgt) - pd.Timestamp(src)
            if pd.Timedelta(0) < diff <= window:
                count += 1
                break
    return count
