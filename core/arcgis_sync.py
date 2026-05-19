"""
ArcGIS Portal / Online Sync
Pushes validated station locations and detection events to an
ArcGIS Online or ArcGIS Enterprise hosted feature layer using the REST API.
No Esri SDK required — uses only the built-in `urllib` and `json` modules
plus the `requests` library (already a Streamlit dependency).
"""

import json
import sqlite3
import pandas as pd
from datetime import datetime
from typing import Optional

try:
    import requests as _requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False


class ArcGISSync:
    """
    Authenticate to ArcGIS Online / Portal and push feature data.

    Parameters
    ----------
    portal_url : str
        Root URL of the ArcGIS portal, e.g. "https://www.arcgis.com"
        or "https://your-org.maps.arcgis.com".
    db_path : str
        Path to the shared SQLite database (for sync history).
    """

    def __init__(self, portal_url: str = "https://www.arcgis.com", db_path: str = "wildlife_data.db"):
        self.portal_url = portal_url.rstrip("/")
        self.db_path = db_path
        self._token: Optional[str] = None
        self._token_expires: Optional[int] = None
        self._init_tables()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def _init_tables(self):
        conn = self._conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS arcgis_sync_log (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                sync_type    TEXT,
                layer_url    TEXT,
                records_sent INTEGER,
                status       TEXT,
                message      TEXT,
                synced_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    def authenticate(self, username: str, password: str, expiration: int = 60) -> bool:
        """
        Generate a short-lived token from username + password.
        Returns True on success.
        """
        if not _HAS_REQUESTS:
            raise ImportError("The 'requests' package is required for ArcGIS sync.")

        url = f"{self.portal_url}/sharing/rest/generateToken"
        payload = {
            "username":   username,
            "password":   password,
            "referer":    self.portal_url,
            "expiration": expiration,
            "f":          "json",
        }
        try:
            resp = _requests.post(url, data=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if "token" in data:
                self._token = data["token"]
                self._token_expires = data.get("expires")
                return True
            return False
        except Exception as exc:
            raise ConnectionError(f"ArcGIS authentication failed: {exc}") from exc

    def authenticate_with_token(self, token: str) -> None:
        """Use a pre-generated API token (e.g. from ArcGIS Online API key)."""
        self._token = token

    @property
    def is_authenticated(self) -> bool:
        return bool(self._token)

    # ------------------------------------------------------------------
    # Push stations
    # ------------------------------------------------------------------

    def push_stations(
        self,
        stations_df: pd.DataFrame,
        layer_url: str,
    ) -> dict:
        """
        Push station point features to an ArcGIS feature layer.

        Parameters
        ----------
        stations_df : pd.DataFrame
            Output of StationManager.get_station_summary().
            Must contain gps_lat, gps_lon, station_id.
        layer_url : str
            REST endpoint of the target feature layer, e.g.:
            "https://services.arcgis.com/.../FeatureServer/0"

        Returns
        -------
        dict : {"added": int, "updated": int, "errors": list}
        """
        self._require_auth()
        features = []
        for _, row in stations_df.iterrows():
            try:
                lat = float(row["gps_lat"])
                lon = float(row["gps_lon"])
                if lat == 0.0 and lon == 0.0:
                    continue
            except (TypeError, ValueError):
                continue

            features.append({
                "geometry": {"x": lon, "y": lat, "spatialReference": {"wkid": 4326}},
                "attributes": {
                    "station_id":       str(row.get("station_id", "")),
                    "habitat_stratum":  str(row.get("habitat_stratum", "")),
                    "camera_model":     str(row.get("camera_model", "")),
                    "trap_nights":      int(row.get("trap_nights", 0) or 0),
                    "functionality_pct": float(row.get("functionality_pct", 0) or 0),
                    "status":           str(row.get("status", "")),
                },
            })

        result = self._add_features(layer_url, features)
        self._log("stations", layer_url, len(features), result)
        return result

    # ------------------------------------------------------------------
    # Push detections / IDEs
    # ------------------------------------------------------------------

    def push_detections(
        self,
        ide_summary: pd.DataFrame,
        stations_df: pd.DataFrame,
        layer_url: str,
    ) -> dict:
        """
        Push IDE (Independent Detection Event) records as point features.

        Parameters
        ----------
        ide_summary : pd.DataFrame
            Output of IndependenceEngine.get_ide_summary().
        stations_df : pd.DataFrame
            Used to look up GPS coordinates by station_id.
        layer_url : str
            REST endpoint of the target feature layer.
        """
        self._require_auth()

        station_coords: dict = {}
        if stations_df is not None and not stations_df.empty:
            for _, s in stations_df.iterrows():
                try:
                    lat, lon = float(s["gps_lat"]), float(s["gps_lon"])
                    if not (lat == 0.0 and lon == 0.0):
                        station_coords[str(s["station_id"])] = (lat, lon)
                except (TypeError, ValueError, KeyError):
                    pass

        features = []
        for _, row in ide_summary.iterrows():
            sid = str(row.get("station_id", ""))
            coords = station_coords.get(sid)
            if not coords:
                continue
            lat, lon = coords
            features.append({
                "geometry": {"x": lon, "y": lat, "spatialReference": {"wkid": 4326}},
                "attributes": {
                    "ide_id":           str(row.get("ide_id", "")),
                    "station_id":       sid,
                    "species":          str(row.get("species", "")),
                    "image_count":      int(row.get("image_count", 1) or 1),
                    "max_confidence":   float(row.get("max_confidence", 0) or 0),
                    "first_detection":  str(row.get("first_detection", "")),
                    "last_detection":   str(row.get("last_detection", "")),
                    "duration_minutes": float(row.get("duration_minutes", 0) or 0),
                },
            })

        result = self._add_features(layer_url, features)
        self._log("detections", layer_url, len(features), result)
        return result

    # ------------------------------------------------------------------
    # Sync log
    # ------------------------------------------------------------------

    def get_sync_log(self) -> pd.DataFrame:
        conn = self._conn()
        try:
            return pd.read_sql_query(
                "SELECT * FROM arcgis_sync_log ORDER BY synced_at DESC", conn
            )
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_auth(self):
        if not self._token:
            raise PermissionError("Not authenticated. Call authenticate() or authenticate_with_token() first.")

    def _add_features(self, layer_url: str, features: list) -> dict:
        if not features:
            return {"added": 0, "updated": 0, "errors": ["No features to push."]}

        url = f"{layer_url.rstrip('/')}/addFeatures"
        payload = {
            "features": json.dumps(features),
            "f":        "json",
            "token":    self._token,
        }
        try:
            resp = _requests.post(url, data=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            added  = sum(1 for r in data.get("addResults", []) if r.get("success"))
            errors = [r.get("error", {}) for r in data.get("addResults", []) if not r.get("success")]
            return {"added": added, "updated": 0, "errors": errors}
        except Exception as exc:
            return {"added": 0, "updated": 0, "errors": [str(exc)]}

    def _log(self, sync_type: str, layer_url: str, records_sent: int, result: dict):
        status = "success" if not result.get("errors") else "partial"
        message = str(result.get("errors", ""))[:500]
        conn = self._conn()
        try:
            conn.execute("""
                INSERT INTO arcgis_sync_log (sync_type, layer_url, records_sent, status, message)
                VALUES (?, ?, ?, ?, ?)
            """, (sync_type, layer_url, records_sent, status, message))
            conn.commit()
        finally:
            conn.close()
