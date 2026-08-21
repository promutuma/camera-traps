"""
Spatial Outputs and Map Data
Exports station locations and detection events as GeoJSON, Shapefile, and KML.
Provides pydeck layer definitions for the built-in map viewer.
"""

import io
import json
import zipfile
import pandas as pd
from datetime import datetime
from typing import Optional
from xml.etree.ElementTree import Element, SubElement, tostring

# WGS84 geographic coordinate system projection string
_WGS84_PRJ = (
    'GEOGCS["GCS_WGS_1984",'
    'DATUM["D_WGS_1984",'
    'SPHEROID["WGS_1984",6378137.0,298.257223563]],'
    'PRIMEM["Greenwich",0.0],'
    'UNIT["Degree",0.0174532925199433]]'
)


# Stratum → colour (RGB) for map rendering
_STRATUM_COLOURS = {
    "Wetland Habitat": [0, 120, 200],
    "Watering Point":  [0, 180, 100],
    "Corridor":        [220, 140, 0],
    "Other":           [150, 150, 150],
}
_DEFAULT_COLOUR = [100, 100, 200]


class SpatialExporter:
    """
    Converts station registry and detection DataFrames into GeoJSON
    FeatureCollections and pydeck-ready layer data.
    """

    # ------------------------------------------------------------------
    # GeoJSON export
    # ------------------------------------------------------------------

    def stations_to_geojson(self, stations_df: pd.DataFrame) -> str:
        """
        Convert the station summary DataFrame to a GeoJSON FeatureCollection
        of Point features. Stations without coordinates are skipped.

        Properties per feature:
            station_id, habitat_stratum, camera_model, team_member,
            trap_nights, functionality_pct, deployment_count, status
        """
        features = []
        for _, row in stations_df.iterrows():
            lat = row.get("gps_lat")
            lon = row.get("gps_lon")
            if lat is None or lon is None:
                continue
            try:
                lat, lon = float(lat), float(lon)
            except (TypeError, ValueError):
                continue
            if lat == 0.0 and lon == 0.0:
                continue

            props = {
                "station_id":       str(row.get("station_id", "")),
                "habitat_stratum":  str(row.get("habitat_stratum", "")),
                "camera_model":     str(row.get("camera_model", "")),
                "team_member":      str(row.get("team_member", "")),
                "trap_nights":      int(row.get("trap_nights", 0)),
                "functionality_pct": float(row.get("functionality_pct", 0)),
                "deployment_count": int(row.get("deployment_count", 0)),
                "status":           str(row.get("status", "")),
            }
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": props,
            })

        return json.dumps({"type": "FeatureCollection", "features": features}, indent=2)

    def detections_to_geojson(
        self,
        df: pd.DataFrame,
        stations_df: Optional[pd.DataFrame] = None,
        ide_summary: Optional[pd.DataFrame] = None,
    ) -> str:
        """
        Convert detection records to a GeoJSON FeatureCollection.
        Each detection event is placed at its station's GPS coordinates.
        If ide_summary is provided, one feature per IDE is created;
        otherwise one feature per detection row.

        Properties per feature:
            station_id, camera_id, species, detection_confidence, date, time,
            day_night, ide_id (if available)
        """
        # Build station → (lat, lon) lookup
        station_coords = {}
        if stations_df is not None and not stations_df.empty:
            for _, s in stations_df.iterrows():
                try:
                    lat, lon = float(s["gps_lat"]), float(s["gps_lon"])
                    if not (lat == 0.0 and lon == 0.0):
                        station_coords[str(s["station_id"])] = (lat, lon)
                except (TypeError, ValueError, KeyError):
                    pass

        features = []
        source = ide_summary if (ide_summary is not None and not ide_summary.empty) else df

        for _, row in source.iterrows():
            sid = str(row.get("station_id", ""))
            coords = station_coords.get(sid)
            if not coords:
                continue
            lat, lon = coords

            if ide_summary is not None and not ide_summary.empty:
                props = {
                    "station_id": sid,
                    "species":    str(row.get("species", "")),
                    "ide_id":     str(row.get("ide_id", "")),
                    "image_count": int(row.get("image_count", 1)),
                    "max_confidence": float(row.get("max_confidence", 0) or 0),
                    "first_detection": str(row.get("first_detection", "")),
                }
            else:
                props = {
                    "station_id":  sid,
                    "camera_id":   str(row.get("camera_id", "") or ""),
                    "species":     str(row.get("species_label", row.get("detected_animal", ""))),
                    "confidence":  float(row.get("detection_confidence", 0) or 0),
                    "date":        str(row.get("capture_date", "") or ""),
                    "time":        str(row.get("capture_time", "") or ""),
                    "day_night":   str(row.get("day_night", "")),
                    "primary_label": str(row.get("primary_label", "")),
                }

            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": props,
            })

        return json.dumps({"type": "FeatureCollection", "features": features}, indent=2)

    # ------------------------------------------------------------------
    # pydeck layer data
    # ------------------------------------------------------------------

    def stations_layer_data(self, stations_df: pd.DataFrame) -> list:
        """
        Return a list of dicts ready for a pydeck ScatterplotLayer.
        Columns: lat, lon, station_id, stratum, colour (RGB list), radius
        """
        rows = []
        for _, row in stations_df.iterrows():
            try:
                lat = float(row["gps_lat"])
                lon = float(row["gps_lon"])
                if lat == 0.0 and lon == 0.0:
                    continue
            except (TypeError, ValueError, KeyError):
                continue

            stratum = str(row.get("habitat_stratum", "Other"))
            colour = _STRATUM_COLOURS.get(stratum, _DEFAULT_COLOUR)
            rows.append({
                "lat": lat,
                "lon": lon,
                "station_id": str(row.get("station_id", "")),
                "stratum": stratum,
                "trap_nights": int(row.get("trap_nights", 0)),
                "colour": colour,
                "radius": 300,
            })
        return rows

    def detections_layer_data(
        self,
        ide_summary: pd.DataFrame,
        stations_df: pd.DataFrame,
    ) -> list:
        """
        Return a list of dicts ready for a pydeck HeatmapLayer.
        Columns: lat, lon, weight (= IDE count per station)
        """
        if ide_summary is None or ide_summary.empty or stations_df is None or stations_df.empty:
            return []

        station_coords = {}
        for _, s in stations_df.iterrows():
            try:
                lat, lon = float(s["gps_lat"]), float(s["gps_lon"])
                if not (lat == 0.0 and lon == 0.0):
                    station_coords[str(s["station_id"])] = (lat, lon)
            except (TypeError, ValueError, KeyError):
                pass

        ide_counts = ide_summary.groupby("station_id").size().to_dict()

        rows = []
        for sid, count in ide_counts.items():
            coords = station_coords.get(str(sid))
            if not coords:
                continue
            rows.append({"lat": coords[0], "lon": coords[1], "weight": count})
        return rows

    # ------------------------------------------------------------------
    # Flat CSV with coordinates
    # ------------------------------------------------------------------

    def detections_to_csv(
        self,
        df: pd.DataFrame,
        stations_df: Optional[pd.DataFrame] = None,
    ) -> str:
        """
        Flat CSV of all detections, with lat/lon columns from the station registry.
        """
        out = df.copy()

        if stations_df is not None and not stations_df.empty:
            coord_df = stations_df[["station_id", "gps_lat", "gps_lon"]].copy()
            coord_df.columns = ["station_id", "latitude", "longitude"]
            if "station_id" in out.columns:
                out = out.merge(coord_df, on="station_id", how="left")

        # Drop binary/complex columns that don't export cleanly
        drop_cols = [c for c in ["bbox", "species_data", "filepath"] if c in out.columns]
        out = out.drop(columns=drop_cols, errors="ignore")
        return out.to_csv(index=False)

    # ------------------------------------------------------------------
    # Shapefile export (.zip of .shp / .shx / .dbf / .prj)
    # ------------------------------------------------------------------

    def stations_to_shapefile(self, stations_df: pd.DataFrame) -> bytes:
        """
        Return a ZIP archive (bytes) containing a point shapefile of stations.
        Fields: station_id, stratum, camera_mod, trap_nights, status.
        Requires the `pyshp` package (import name: shapefile).
        """
        try:
            import shapefile as sf
        except ImportError:
            raise ImportError("Install pyshp: pip install pyshp")

        shp = io.BytesIO()
        shx = io.BytesIO()
        dbf = io.BytesIO()
        w = sf.Writer(shp=shp, shx=shx, dbf=dbf, shapeType=sf.POINT)
        w.field("station_id",  "C", 50)
        w.field("stratum",     "C", 50)
        w.field("camera_mod",  "C", 40)
        w.field("trap_nights", "N", 10)
        w.field("status",      "C", 30)

        has_shapes = False
        for _, row in stations_df.iterrows():
            try:
                lat = float(row["gps_lat"])
                lon = float(row["gps_lon"])
                if lat == 0.0 and lon == 0.0:
                    continue
            except (TypeError, ValueError):
                continue
            w.point(lon, lat)
            w.record(
                str(row.get("station_id", ""))[:50],
                str(row.get("habitat_stratum", ""))[:50],
                str(row.get("camera_model", ""))[:40],
                int(row.get("trap_nights", 0) or 0),
                str(row.get("status", ""))[:30],
            )
            has_shapes = True

        if not has_shapes:
            w.null()
            w.record("", "", "", 0, "")

        w.close()
        return self._pack_shapefile_zip(shp, shx, dbf, "stations")

    def detections_to_shapefile(
        self,
        ide_summary: pd.DataFrame,
        stations_df: pd.DataFrame,
    ) -> bytes:
        """
        Return a ZIP archive (bytes) of a point shapefile of IDE events.
        Fields: ide_id, station_id, species, image_count, max_conf, first_det.
        """
        try:
            import shapefile as sf
        except ImportError:
            raise ImportError("Install pyshp: pip install pyshp")

        station_coords = self._build_coord_lookup(stations_df)

        shp = io.BytesIO()
        shx = io.BytesIO()
        dbf = io.BytesIO()
        w = sf.Writer(shp=shp, shx=shx, dbf=dbf, shapeType=sf.POINT)
        w.field("ide_id",      "C", 60)
        w.field("station_id",  "C", 50)
        w.field("species",     "C", 80)
        w.field("img_count",   "N", 10)
        w.field("max_conf",    "F", 10, 4)
        w.field("first_det",   "C", 30)

        has_shapes = False
        for _, row in ide_summary.iterrows():
            sid = str(row.get("station_id", ""))
            coords = station_coords.get(sid)
            if not coords:
                continue
            lat, lon = coords
            w.point(lon, lat)
            w.record(
                str(row.get("ide_id", ""))[:60],
                sid[:50],
                str(row.get("species", ""))[:80],
                int(row.get("image_count", 1) or 1),
                float(row.get("max_confidence", 0) or 0),
                str(row.get("first_detection", ""))[:30],
            )
            has_shapes = True

        if not has_shapes:
            w.null()
            w.record("", "", "", 0, 0.0, "")

        w.close()
        return self._pack_shapefile_zip(shp, shx, dbf, "detections")

    def _pack_shapefile_zip(self, shp: io.BytesIO, shx: io.BytesIO, dbf: io.BytesIO, basename: str) -> bytes:
        """Write a pyshp Writer to an in-memory ZIP and return raw bytes."""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(f"{basename}.shp", shp.getvalue())
            zf.writestr(f"{basename}.shx", shx.getvalue())
            zf.writestr(f"{basename}.dbf", dbf.getvalue())
            zf.writestr(f"{basename}.prj", _WGS84_PRJ)
        return buf.getvalue()

    # ------------------------------------------------------------------
    # KML export
    # ------------------------------------------------------------------

    def stations_to_kml(self, stations_df: pd.DataFrame) -> str:
        """Return a KML string with one Placemark per station."""
        kml = Element("kml", xmlns="http://www.opengis.net/kml/2.2")
        doc = SubElement(kml, "Document")
        SubElement(doc, "name").text = "Camera Stations"

        for _, row in stations_df.iterrows():
            try:
                lat = float(row["gps_lat"])
                lon = float(row["gps_lon"])
                if lat == 0.0 and lon == 0.0:
                    continue
            except (TypeError, ValueError):
                continue

            pm = SubElement(doc, "Placemark")
            SubElement(pm, "name").text = str(row.get("station_id", ""))
            desc_parts = [
                f"Stratum: {row.get('habitat_stratum', '')}",
                f"Camera: {row.get('camera_model', '')}",
                f"Trap nights: {row.get('trap_nights', '')}",
                f"Status: {row.get('status', '')}",
            ]
            SubElement(pm, "description").text = "\n".join(desc_parts)
            pt = SubElement(pm, "Point")
            SubElement(pt, "coordinates").text = f"{lon},{lat},0"

        return '<?xml version="1.0" encoding="UTF-8"?>\n' + tostring(kml, encoding="unicode")

    def detections_to_kml(
        self,
        ide_summary: pd.DataFrame,
        stations_df: pd.DataFrame,
    ) -> str:
        """Return a KML string with one Placemark per IDE event."""
        station_coords = self._build_coord_lookup(stations_df)

        kml = Element("kml", xmlns="http://www.opengis.net/kml/2.2")
        doc = SubElement(kml, "Document")
        SubElement(doc, "name").text = "Wildlife Detection Events"

        for _, row in ide_summary.iterrows():
            sid = str(row.get("station_id", ""))
            coords = station_coords.get(sid)
            if not coords:
                continue
            lat, lon = coords

            pm = SubElement(doc, "Placemark")
            SubElement(pm, "name").text = str(row.get("species", "Unknown"))
            desc_parts = [
                f"Station: {sid}",
                f"IDE: {row.get('ide_id', '')}",
                f"Images: {row.get('image_count', '')}",
                f"Max confidence: {row.get('max_confidence', '')}",
                f"First detection: {row.get('first_detection', '')}",
            ]
            SubElement(pm, "description").text = "\n".join(desc_parts)
            pt = SubElement(pm, "Point")
            SubElement(pt, "coordinates").text = f"{lon},{lat},0"

        return '<?xml version="1.0" encoding="UTF-8"?>\n' + tostring(kml, encoding="unicode")

    # ------------------------------------------------------------------
    # Router adapter methods (accept raw detection history df)
    # ------------------------------------------------------------------

    def to_geojson(self, df: pd.DataFrame, stations_df: Optional[pd.DataFrame] = None) -> dict:
        result = self.detections_to_geojson(df, stations_df)
        return json.loads(result) if isinstance(result, str) else result

    def to_shapefile_bytes(self, df: pd.DataFrame, stations_df: Optional[pd.DataFrame] = None) -> bytes:
        try:
            import shapefile as sf
        except ImportError:
            raise ImportError("Install pyshp: pip install pyshp")

        has_coords = stations_df is not None and not stations_df.empty
        
        shp = io.BytesIO()
        shx = io.BytesIO()
        dbf = io.BytesIO()
        w = sf.Writer(shp=shp, shx=shx, dbf=dbf, shapeType=sf.POINT if has_coords else sf.NULL)

        keep = ["station_id", "camera_id", "detected_animal", "detection_confidence",
                "capture_date", "capture_time", "day_night", "ide_id"]
        cols = [c for c in keep if c in df.columns]
        for col in cols:
            w.field(col[:10], "C", 50)

        station_coords = self._build_coord_lookup(stations_df) if has_coords else {}

        has_shapes = False
        for _, row in df.iterrows():
            sid = str(row.get("station_id", ""))
            coords = station_coords.get(sid) if has_coords else None
            if has_coords and coords:
                lat, lon = coords
                w.point(lon, lat)
                has_shapes = True
            elif has_coords:
                continue  # Skip stations with missing coords to avoid invalid POINT geom
            else:
                w.null()
                has_shapes = True
            w.record(*[str(row.get(c, ""))[:50] for c in cols])

        if not has_shapes:
            w.null()
            w.record(*["" for _ in cols])

        w.close()
        return self._pack_shapefile_zip(shp, shx, dbf, "detections")

    def to_kml(self, df: pd.DataFrame, stations_df: Optional[pd.DataFrame] = None) -> str:
        from xml.etree.ElementTree import Element, SubElement, tostring
        kml = Element("kml", xmlns="http://www.opengis.net/kml/2.2")
        doc = SubElement(kml, "Document")
        SubElement(doc, "name").text = "Wildlife Detections"

        # If stations_df is provided, merge it to include coordinates in df
        out = df.copy()
        if stations_df is not None and not stations_df.empty:
            coord_df = stations_df[["station_id", "gps_lat", "gps_lon"]].copy()
            coord_df.columns = ["station_id", "latitude", "longitude"]
            if "station_id" in out.columns:
                out = out.merge(coord_df, on="station_id", how="left")

        lat_col = next((c for c in ["gps_lat", "lat", "latitude"] if c in out.columns), None)
        lon_col = next((c for c in ["gps_lon", "lon", "longitude"] if c in out.columns), None)

        for _, row in out.iterrows():
            pm = SubElement(doc, "Placemark")
            SubElement(pm, "name").text = str(row.get("detected_animal", row.get("species_label", "Unknown")))
            SubElement(pm, "description").text = "\n".join([
                f"Station: {row.get('station_id', '')}",
                f"Camera: {row.get('camera_id', '') or 'Unknown'}",
                f"Confidence: {row.get('detection_confidence', '')}",
                f"Date: {row.get('capture_date', '')}",
                f"Time: {row.get('capture_time', '')}",
            ])
            if lat_col and lon_col:
                try:
                    lat, lon = float(row[lat_col]), float(row[lon_col])
                    if not (lat == 0.0 and lon == 0.0):
                        pt = SubElement(pm, "Point")
                        SubElement(pt, "coordinates").text = f"{lon},{lat},0"
                except (TypeError, ValueError):
                    pass
        return '<?xml version="1.0" encoding="UTF-8"?>\n' + tostring(kml, encoding="unicode")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_coord_lookup(self, stations_df: Optional[pd.DataFrame]) -> dict:
        """Return {station_id: (lat, lon)} for stations with valid coordinates."""
        coords = {}
        if stations_df is None or stations_df.empty:
            return coords
        for _, s in stations_df.iterrows():
            try:
                lat, lon = float(s["gps_lat"]), float(s["gps_lon"])
                if not (lat == 0.0 and lon == 0.0):
                    coords[str(s["station_id"])] = (lat, lon)
            except (TypeError, ValueError, KeyError):
                pass
        return coords
