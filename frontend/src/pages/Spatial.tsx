import { useEffect, useState } from "react";
import { MapContainer, TileLayer, CircleMarker, Popup, useMap } from "react-leaflet";
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

// SVG Icons
const DownloadIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.8} stroke="currentColor" className="w-3.5 h-3.5">
    <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5M16.5 12L12 16.5m0 0L7.5 12m4.5 4.5V3" />
  </svg>
);

const GlobeIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-5 h-5">
    <path strokeLinecap="round" strokeLinejoin="round" d="M12 21a9.004 9.004 0 008.716-6.747M12 21a9.004 9.004 0 01-8.716-6.747M12 21V9.75M3.284 14.253A8.966 8.966 0 0112 3.75c3.81 0 7.06 2.372 8.35 5.75m-18.066 4.753A9.043 9.043 0 0012 15.75c2.31 0 4.418-.867 6.012-2.3m-15.312-3.14a8.995 8.995 0 0118.066 0M3.284 10.25a8.966 8.966 0 008.716 5.5M12 9.75V3.75" />
  </svg>
);

function FitBounds({ features }: { features: Feature[] }) {
  const map = useMap();
  useEffect(() => {
    if (features.length === 0) return;
    const lats = features.map((f) => f.geometry.coordinates[1]);
    const lons = features.map((f) => f.geometry.coordinates[0]);
    map.fitBounds(
      [[Math.min(...lats), Math.min(...lons)], [Math.max(...lats), Math.max(...lons)]],
      { padding: [40, 40], maxZoom: 13 }
    );
  }, [features, map]);
  return null;
}

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

  const handleExport = (format: string) => {
    if (features.length === 0) return;
    // Trigger download securely
    window.location.href = `/api/spatial/export/${format}`;
  };

  return (
    <div className="space-y-4 animate-fade-in">
      {/* Header and Toolkit */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-slate-900 dark:text-white">
            Spatial Map Viewer
          </h1>
          <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
            Map spatial distribution patterns of classified wildlife detections across Ethiopia monitoring stations.
          </p>
        </div>

        {/* Action Export Toolkit (Programmatic buttons with validation) */}
        <div className="flex flex-wrap gap-2">
          <button
            onClick={() => handleExport("geojson")}
            disabled={features.length === 0}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-emerald-600 hover:bg-emerald-700 disabled:bg-slate-100 dark:disabled:bg-slate-800 text-white disabled:text-slate-400 dark:disabled:text-slate-600 text-xs font-semibold rounded-xl transition shadow-sm hover:shadow cursor-pointer disabled:cursor-not-allowed border border-transparent disabled:border-slate-200/20"
            title="Download GeoJSON features"
          >
            <DownloadIcon />
            GeoJSON
          </button>
          <button
            onClick={() => handleExport("csv")}
            disabled={features.length === 0}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-slate-900 hover:bg-slate-850 dark:bg-white dark:hover:bg-slate-100 text-white dark:text-slate-900 disabled:opacity-40 text-xs font-semibold rounded-xl transition shadow-sm hover:shadow cursor-pointer disabled:cursor-not-allowed border border-transparent"
            title="Download CSV coordinate catalog"
          >
            <DownloadIcon />
            CSV
          </button>
          <button
            onClick={() => handleExport("shapefile")}
            disabled={features.length === 0}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-slate-900 hover:bg-slate-850 dark:bg-white dark:hover:bg-slate-100 text-white dark:text-slate-900 disabled:opacity-40 text-xs font-semibold rounded-xl transition shadow-sm hover:shadow cursor-pointer disabled:cursor-not-allowed border border-transparent"
            title="Download ESRI Shapefile archive"
          >
            <DownloadIcon />
            Shapefile
          </button>
          <button
            onClick={() => handleExport("kml")}
            disabled={features.length === 0}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-slate-900 hover:bg-slate-850 dark:bg-white dark:hover:bg-slate-100 text-white dark:text-slate-900 disabled:opacity-40 text-xs font-semibold rounded-xl transition shadow-sm hover:shadow cursor-pointer disabled:cursor-not-allowed border border-transparent"
            title="Download Google Earth KML"
          >
            <DownloadIcon />
            KML
          </button>
        </div>
      </div>

      {loading ? (
        <div className="text-center py-20 text-slate-400 dark:text-slate-500 animate-pulse">
          Initializing geographic map layers...
        </div>
      ) : features.length === 0 ? (
        <div className="text-center py-24 text-slate-400 dark:text-slate-550 bg-white dark:bg-slate-900 border border-slate-200/60 dark:border-slate-800/80 rounded-2xl shadow-sm space-y-2">
          <div className="text-2xl"><span className="material-symbols-outlined text-2xl select-none">pin_drop</span></div>
          <p className="font-semibold text-sm text-slate-700 dark:text-slate-300">No Georeferenced Coordinates Available</p>
          <p className="text-xs max-w-sm mx-auto text-slate-400 dark:text-slate-500 leading-relaxed">
            Please register stations with valid latitude/longitude coordinate bounds and analyze images before viewing spatial layers.
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 bg-white dark:bg-slate-900 border border-slate-200/60 dark:border-slate-800/80 rounded-2xl overflow-hidden shadow-sm">
          {/* Leaflet Map panel */}
          <div className="lg:col-span-3 h-[520px] relative z-10">
            <MapContainer center={[0, 20]} zoom={3} style={{ height: "100%", width: "100%" }}>
              <TileLayer
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                attribution="© OpenStreetMap contributors"
              />
              <FitBounds features={displayedFeatures} />
              {displayedFeatures.map((f, i) => {
                const [lon, lat] = f.geometry.coordinates;
                const p = f.properties;
                const spName = String(p.detected_animal ?? p.species ?? "Unknown");
                const markerColor = speciesColors[spName] || "#10b981";
                return (
                  <CircleMarker
                    key={i}
                    center={[lat, lon]}
                    radius={10}
                    color={markerColor}
                    fillColor={markerColor}
                    fillOpacity={0.6}
                    weight={2}
                  >
                    <Popup>
                      <div className="text-xs space-y-1.5 font-sans p-1">
                        <div className="flex justify-between items-center border-b border-slate-100 pb-1.5 mb-1 gap-4">
                          <span className="font-bold text-slate-900">{spName}</span>
                          <span className="text-[9px] font-bold text-emerald-600 bg-emerald-50 px-2 py-0.5 rounded-full border border-emerald-100">
                            {typeof p.detection_confidence === "number" ? `${Math.round(p.detection_confidence * 100)}% Match` : "Verified"}
                          </span>
                        </div>
                        <p className="text-slate-500 font-medium">
                          Location: <span className="font-bold text-slate-700">{String(p.station_id ?? "—")}</span>
                        </p>
                        <p className="text-slate-500 font-medium font-mono text-[10px]">
                          File: {String(p.filename ?? "")}
                        </p>
                        <p className="text-[10px] text-slate-400 capitalize">
                          Period: {String(p.day_night ?? "unknown")}
                        </p>
                      </div>
                    </Popup>
                  </CircleMarker>
                );
              })}
            </MapContainer>
          </div>

          {/* Interactive Legend sidebar */}
          <div className="p-5 flex flex-col justify-between border-t lg:border-t-0 lg:border-l border-slate-200/60 dark:border-slate-800/80 h-[520px]">
            <div className="space-y-4 overflow-y-auto pr-1">
              <div className="flex items-center justify-between border-b border-slate-150/40 dark:border-slate-800 pb-3">
                <div className="flex items-center gap-1.5 text-slate-800 dark:text-slate-200">
                  <span className="text-emerald-500">
                    <GlobeIcon />
                  </span>
                  <h3 className="font-bold text-xs uppercase tracking-wider">
                    Species Legend
                  </h3>
                </div>
                <button
                  onClick={handleToggleAll}
                  className="text-xs font-semibold text-emerald-600 dark:text-emerald-450 hover:underline cursor-pointer"
                >
                  {selectedSpecies.length === Object.keys(speciesColors).length ? "Deselect All" : "Select All"}
                </button>
              </div>
              
              <div className="space-y-2.5">
                {Object.entries(speciesColors).map(([sp, color]) => {
                  const count = features.filter((f) => String(f.properties.detected_animal ?? f.properties.species ?? "Unknown") === sp).length;
                  const isChecked = selectedSpecies.includes(sp);
                  return (
                    <label
                      key={sp}
                      className="flex items-center justify-between p-2 rounded-xl border border-slate-100 dark:border-slate-850 hover:bg-slate-50 dark:hover:bg-slate-950 cursor-pointer text-xs transition"
                    >
                      <div className="flex items-center gap-2">
                        <input
                          type="checkbox"
                          checked={isChecked}
                          onChange={() => handleToggleSpecies(sp)}
                          className="rounded text-emerald-600 focus:ring-emerald-500/20 border-slate-200 dark:border-slate-800 w-3.5 h-3.5 cursor-pointer"
                        />
                        <span className="w-2.5 h-2.5 rounded-full shrink-0 shadow-sm" style={{ backgroundColor: color }} />
                        <span className="font-bold text-slate-700 dark:text-slate-350">{sp}</span>
                      </div>
                      <span className="text-[10px] text-slate-400 font-mono font-bold bg-slate-100 dark:bg-slate-950 px-2 py-0.5 rounded-full border border-slate-200/20 dark:border-slate-800/40">
                        {count}
                      </span>
                    </label>
                  );
                })}
              </div>
            </div>

            <div className="pt-3 border-t border-slate-150/40 dark:border-slate-800 text-[11px] text-slate-400 font-semibold">
              <p>{displayedFeatures.length} of {features.length} coordinate markers active</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
