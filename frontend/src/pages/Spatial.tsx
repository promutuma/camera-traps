import { useEffect, useState } from "react";
import { MapContainer, TileLayer, CircleMarker, Popup } from "react-leaflet";
import { getSpatialGeoJSON } from "../api/client";

type Feature = {
  type: string;
  geometry: { type: string; coordinates: [number, number] };
  properties: Record<string, unknown>;
};

export default function Spatial() {
  const [features, setFeatures] = useState<Feature[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getSpatialGeoJSON()
      .then((gj) => setFeatures(gj?.features ?? []))
      .catch(() => setFeatures([]))
      .finally(() => setLoading(false));
  }, []);

  const center: [number, number] = features.length > 0
    ? [features[0].geometry.coordinates[1], features[0].geometry.coordinates[0]]
    : [0, 20];

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-slate-800">Spatial & Map Viewer</h1>
        <div className="flex gap-2">
          <a href="/api/spatial/export/geojson" className="px-4 py-2 bg-green-600 text-white text-sm rounded-lg hover:bg-green-700">GeoJSON</a>
          <a href="/api/spatial/export/csv" className="px-4 py-2 bg-slate-600 text-white text-sm rounded-lg hover:bg-slate-700">CSV</a>
          <a href="/api/spatial/export/shapefile" className="px-4 py-2 bg-slate-600 text-white text-sm rounded-lg hover:bg-slate-700">Shapefile</a>
          <a href="/api/spatial/export/kml" className="px-4 py-2 bg-slate-600 text-white text-sm rounded-lg hover:bg-slate-700">KML</a>
        </div>
      </div>

      {loading ? (
        <div className="text-center py-12 text-slate-400">Loading map…</div>
      ) : features.length === 0 ? (
        <div className="text-center py-12 text-slate-400">
          No georeferenced detections. Add stations with coordinates to see them on the map.
        </div>
      ) : (
        <div className="bg-white rounded-xl border border-slate-200 overflow-hidden" style={{ height: 480 }}>
          <MapContainer center={center} zoom={10} style={{ height: "100%", width: "100%" }}>
            <TileLayer
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
              attribution="© OpenStreetMap contributors"
            />
            {features.map((f, i) => {
              const [lon, lat] = f.geometry.coordinates;
              const p = f.properties;
              return (
                <CircleMarker key={i} center={[lat, lon]} radius={8} color="#16a34a" fillOpacity={0.7}>
                  <Popup>
                    <div className="text-sm space-y-1">
                      <p className="font-semibold">{String(p.detected_animal ?? p.species ?? "Unknown")}</p>
                      <p>{String(p.filename ?? "")}</p>
                      <p className="text-slate-500">{String(p.station_id ?? "")} · {String(p.day_night ?? "")}</p>
                      {p.detection_confidence != null && (
                        <p className="text-slate-500">Conf: {Number(p.detection_confidence).toFixed(2)}</p>
                      )}
                    </div>
                  </Popup>
                </CircleMarker>
              );
            })}
          </MapContainer>
        </div>
      )}

      <div className="text-sm text-slate-500">
        {features.length} georeferenced detection(s) shown
      </div>
    </div>
  );
}
