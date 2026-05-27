import { useEffect, useState } from "react";
import { MapContainer, TileLayer, CircleMarker, Popup } from "react-leaflet";
import { getSpatialGeoJSON } from "../api/client";

type Feature = {
  type: string;
  geometry: { type: string; coordinates: [number, number] };
  properties: Record<string, unknown>;
};

const PALETTE = [
  "#10b981", // emerald
  "#3b82f6", // blue
  "#f59e0b", // amber
  "#ef4444", // red
  "#8b5cf6", // violet
  "#06b6d4", // cyan
  "#ec4899", // pink
  "#6366f1", // indigo
  "#14b8a6", // teal
  "#f43f5e", // rose
];

export default function Spatial() {
  const [features, setFeatures] = useState<Feature[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedSpecies, setSelectedSpecies] = useState<string[]>([]);
  const [speciesColors, setSpeciesColors] = useState<Record<string, string>>({});

  useEffect(() => {
    getSpatialGeoJSON()
      .then((gj) => {
        const feats = gj?.features ?? [];
        setFeatures(feats);

        // Extract unique species and build color map
        const spSet = new Set<string>();
        feats.forEach((f: Feature) => {
          const sp = String(f.properties.detected_animal ?? f.properties.species ?? "Unknown");
          spSet.add(sp);
        });

        const uniqueSp = Array.from(spSet);
        const colorMap: Record<string, string> = {};
        uniqueSp.forEach((sp, idx) => {
          colorMap[sp] = PALETTE[idx % PALETTE.length];
        });

        setSpeciesColors(colorMap);
        setSelectedSpecies(uniqueSp);
      })
      .catch(() => {
        setFeatures([]);
      })
      .finally(() => setLoading(false));
  }, []);

  const displayedFeatures = features.filter((f) => {
    const sp = String(f.properties.detected_animal ?? f.properties.species ?? "Unknown");
    return selectedSpecies.includes(sp);
  });

  const center: [number, number] = features.length > 0
    ? [features[0].geometry.coordinates[1], features[0].geometry.coordinates[0]]
    : [0, 20];

  const handleToggleSpecies = (sp: string) => {
    setSelectedSpecies((prev) =>
      prev.includes(sp) ? prev.filter((s) => s !== sp) : [...prev, sp]
    );
  };

  const handleToggleAll = () => {
    const allSp = Object.keys(speciesColors);
    if (selectedSpecies.length === allSp.length) {
      setSelectedSpecies([]);
    } else {
      setSelectedSpecies(allSp);
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-800 dark:text-slate-100">Spatial & Map Viewer</h1>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">Visualize and filter species distribution by camera trap location.</p>
        </div>
        <div className="flex gap-2">
          <a href="/api/spatial/export/geojson" className="px-3.5 py-1.5 bg-emerald-600 text-white text-xs font-semibold rounded-lg hover:bg-emerald-700 shadow-sm transition">GeoJSON</a>
          <a href="/api/spatial/export/csv" className="px-3.5 py-1.5 bg-slate-700 text-white text-xs font-semibold rounded-lg hover:bg-slate-800 shadow-sm transition">CSV</a>
          <a href="/api/spatial/export/shapefile" className="px-3.5 py-1.5 bg-slate-700 text-white text-xs font-semibold rounded-lg hover:bg-slate-800 shadow-sm transition">Shapefile</a>
          <a href="/api/spatial/export/kml" className="px-3.5 py-1.5 bg-slate-700 text-white text-xs font-semibold rounded-lg hover:bg-slate-800 shadow-sm transition">KML</a>
        </div>
      </div>

      {loading ? (
        <div className="text-center py-12 text-slate-400">Loading map…</div>
      ) : features.length === 0 ? (
        <div className="text-center py-12 text-slate-400">
          No georeferenced detections. Add stations with coordinates to see them on the map.
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl overflow-hidden shadow-sm">
          {/* Map area */}
          <div className="lg:col-span-3 h-[500px] relative">
            <MapContainer center={center} zoom={10} style={{ height: "100%", width: "100%" }}>
              <TileLayer
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                attribution="© OpenStreetMap contributors"
              />
              {displayedFeatures.map((f, i) => {
                const [lon, lat] = f.geometry.coordinates;
                const p = f.properties;
                const spName = String(p.detected_animal ?? p.species ?? "Unknown");
                const markerColor = speciesColors[spName] || "#16a34a";
                return (
                  <CircleMarker
                    key={i}
                    center={[lat, lon]}
                    radius={9}
                    color={markerColor}
                    fillColor={markerColor}
                    fillOpacity={0.65}
                    weight={2}
                  >
                    <Popup>
                      <div className="text-sm space-y-1">
                        <p className="font-semibold text-slate-800">{spName}</p>
                        <p className="text-xs text-slate-500 font-mono">{String(p.filename ?? "")}</p>
                        <p className="text-xs text-slate-500">
                          Station: <span className="font-semibold">{String(p.station_id ?? "")}</span> · {String(p.day_night ?? "")}
                        </p>
                        {typeof p.detection_confidence === "number" && (
                          <p className="text-xs text-slate-500">Confidence: {Math.round(p.detection_confidence * 100)}%</p>
                        )}
                      </div>
                    </Popup>
                  </CircleMarker>
                );
              })}
            </MapContainer>
          </div>

          {/* Sidebar Legend */}
          <div className="p-4 flex flex-col justify-between border-t lg:border-t-0 lg:border-l border-slate-200 dark:border-slate-800 h-[500px]">
            <div className="space-y-4 overflow-y-auto pr-1">
              <div className="flex items-center justify-between">
                <h3 className="font-bold text-sm text-slate-700 dark:text-slate-300">Species Legend</h3>
                <button
                  onClick={handleToggleAll}
                  className="text-xs font-semibold text-emerald-600 dark:text-emerald-450 hover:underline cursor-pointer"
                >
                  {selectedSpecies.length === Object.keys(speciesColors).length ? "Deselect All" : "Select All"}
                </button>
              </div>
              <div className="space-y-2">
                {Object.entries(speciesColors).map(([sp, color]) => {
                  const count = features.filter((f) => String(f.properties.detected_animal ?? f.properties.species ?? "Unknown") === sp).length;
                  const isChecked = selectedSpecies.includes(sp);
                  return (
                    <label
                      key={sp}
                      className="flex items-center justify-between p-2 rounded-lg border border-slate-100 dark:border-slate-800 hover:bg-slate-50 dark:hover:bg-slate-800/50 cursor-pointer text-xs"
                    >
                      <div className="flex items-center gap-2">
                        <input
                          type="checkbox"
                          checked={isChecked}
                          onChange={() => handleToggleSpecies(sp)}
                          className="rounded text-emerald-600 focus:ring-emerald-500 border-slate-300 w-3.5 h-3.5"
                        />
                        <span className="w-3.5 h-3.5 rounded-full shrink-0" style={{ backgroundColor: color }} />
                        <span className="font-medium text-slate-700 dark:text-slate-350">{sp}</span>
                      </div>
                      <span className="text-[10px] text-slate-400 font-mono font-bold bg-slate-100 dark:bg-slate-800 px-1.5 py-0.5 rounded-full">
                        {count}
                      </span>
                    </label>
                  );
                })}
              </div>
            </div>

            <div className="pt-3 border-t border-slate-200 dark:border-slate-800 text-xs text-slate-400">
              <p className="font-medium">{displayedFeatures.length} of {features.length} markers shown</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
