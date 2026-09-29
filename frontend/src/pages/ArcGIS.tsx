import { useEffect, useState } from "react";
import { pushArcGIS, getArcGISStatus } from "../api/client";

export default function ArcGIS() {
  const [status, setStatus] = useState<string | null>(null);
  const [form, setForm] = useState({ url: "", token: "", layer_id: "0" });
  const [pushing, setPushing] = useState(false);
  const [result, setResult] = useState<string | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    getArcGISStatus().then((s) => setStatus(s.status)).catch(() => {});
  }, []);

  const handlePush = async () => {
    if (!form.url || !form.token) {
      setError("URL and token are required.");
      return;
    }
    setPushing(true);
    setError("");
    setResult(null);
    try {
      const res = await pushArcGIS({
        url: form.url,
        token: form.token,
        layer_id: Number(form.layer_id),
      });
      setResult(JSON.stringify(res, null, 2));
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Push failed");
    } finally {
      setPushing(false);
    }
  };

  return (
    <div className="max-w-3xl mx-auto space-y-6 animate-fade-in">
      <div>
        <h1 className="text-2xl font-bold text-slate-900 dark:text-white">ArcGIS Sync & Spatial Exports</h1>
        <p className="text-slate-500 dark:text-slate-400 text-sm mt-1">
          Push to ArcGIS Online / Enterprise or download GIS spatial files.
        </p>
      </div>

      {status && (
        <div className="bg-slate-50 dark:bg-slate-950/40 border border-slate-200 dark:border-slate-800 rounded-xl px-4 py-3 text-sm text-slate-600 dark:text-slate-400">
          Last sync status: <span className="font-semibold text-emerald-600 dark:text-emerald-450">{status}</span>
        </div>
      )}

      {/* Push form */}
      <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6 space-y-5 shadow-sm">
        <h2 className="font-bold text-base text-slate-855 dark:text-slate-200">Push to ArcGIS Feature Layer</h2>
        <div className="space-y-4">
          <div>
            <label className="block text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-1.5">
              Feature Layer URL
            </label>
            <input
              value={form.url}
              onChange={(e) => setForm((f) => ({ ...f, url: e.target.value }))}
              placeholder="https://services.arcgis.com/…/FeatureServer/0"
              className="block w-full border border-slate-300 dark:border-slate-800 rounded-lg px-3.5 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 bg-white dark:bg-slate-950 text-slate-800 dark:text-slate-100 placeholder-slate-400 transition"
            />
          </div>
          <div>
            <label className="block text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-1.5">
              ArcGIS Token
            </label>
            <input
              type="password"
              value={form.token}
              onChange={(e) => setForm((f) => ({ ...f, token: e.target.value }))}
              placeholder="••••••••••••••••"
              className="block w-full border border-slate-300 dark:border-slate-800 rounded-lg px-3.5 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 bg-white dark:bg-slate-950 text-slate-800 dark:text-slate-100 placeholder-slate-400 transition"
            />
          </div>
          <div>
            <label className="block text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-1.5">
              Layer ID
            </label>
            <input
              type="number"
              value={form.layer_id}
              onChange={(e) => setForm((f) => ({ ...f, layer_id: e.target.value }))}
              className="block w-24 border border-slate-300 dark:border-slate-800 rounded-lg px-3.5 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 bg-white dark:bg-slate-950 text-slate-800 dark:text-slate-100 placeholder-slate-400 transition"
            />
          </div>
        </div>

        {error && (
          <div className="bg-red-50 dark:bg-red-950/20 border border-red-200 dark:border-red-900/30 text-red-700 dark:text-red-400 rounded-lg p-3 text-sm">
            {error}
          </div>
        )}

        <button
          onClick={handlePush}
          disabled={pushing}
          className="px-5 py-2.5 bg-blue-600 hover:bg-blue-750 text-white text-sm font-semibold rounded-lg shadow-sm hover:shadow transition disabled:bg-slate-300 dark:disabled:bg-slate-800 dark:disabled:text-slate-500 cursor-pointer"
        >
          {pushing ? "Pushing to ArcGIS…" : "Push to ArcGIS"}
        </button>

        {result && (
          <div className="space-y-2">
            <p className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase">Response payload</p>
            <pre className="text-xs bg-slate-50 dark:bg-slate-950/60 rounded-xl border border-slate-100 dark:border-slate-850 p-4 overflow-x-auto max-h-40 font-mono text-slate-700 dark:text-slate-400">
              {result}
            </pre>
          </div>
        )}
      </div>

      {/* Export buttons */}
      <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6 space-y-4 shadow-sm">
        <h2 className="font-bold text-base text-slate-800 dark:text-slate-200">Download Spatial Files</h2>
        <div className="flex flex-wrap gap-3">
          <a
            href="/api/arcgis/export/geojson"
            className="px-4 py-2 bg-emerald-600 hover:bg-emerald-700 text-white text-sm font-semibold rounded-lg shadow-sm hover:shadow transition"
          >
            GeoJSON
          </a>
          <a
            href="/api/arcgis/export/shapefile"
            className="px-4 py-2 bg-slate-700 hover:bg-slate-800 text-white text-sm font-semibold rounded-lg shadow-sm hover:shadow transition"
          >
            Shapefile (ZIP)
          </a>
          <a
            href="/api/arcgis/export/kml"
            className="px-4 py-2 bg-slate-700 hover:bg-slate-800 text-white text-sm font-semibold rounded-lg shadow-sm hover:shadow transition"
          >
            KML
          </a>
        </div>
      </div>
    </div>
  );
}

