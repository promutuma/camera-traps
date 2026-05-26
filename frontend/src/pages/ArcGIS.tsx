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
    if (!form.url || !form.token) { setError("URL and token are required."); return; }
    setPushing(true);
    setError("");
    setResult(null);
    try {
      const res = await pushArcGIS({ url: form.url, token: form.token, layer_id: Number(form.layer_id) });
      setResult(JSON.stringify(res, null, 2));
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Push failed");
    } finally {
      setPushing(false);
    }
  };

  return (
    <div className="max-w-2xl space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-slate-800">ArcGIS Sync & Spatial Exports</h1>
        <p className="text-slate-500 text-sm mt-1">Push to ArcGIS Online / Enterprise or download GIS files.</p>
      </div>

      {status && (
        <div className="bg-slate-50 border border-slate-200 rounded-lg px-4 py-2 text-sm text-slate-600">
          Last sync status: <span className="font-medium">{status}</span>
        </div>
      )}

      {/* Push form */}
      <div className="bg-white rounded-xl border border-slate-200 p-4 space-y-4">
        <h2 className="font-semibold text-slate-700">Push to ArcGIS</h2>
        <div className="space-y-3">
          <label className="block text-sm text-slate-700">
            Feature Layer URL
            <input value={form.url} onChange={(e) => setForm((f) => ({ ...f, url: e.target.value }))}
              placeholder="https://services.arcgis.com/…/FeatureServer"
              className="mt-1 block w-full border border-slate-300 rounded px-3 py-2 text-sm" />
          </label>
          <label className="block text-sm text-slate-700">
            ArcGIS Token
            <input type="password" value={form.token} onChange={(e) => setForm((f) => ({ ...f, token: e.target.value }))}
              className="mt-1 block w-full border border-slate-300 rounded px-3 py-2 text-sm" />
          </label>
          <label className="block text-sm text-slate-700">
            Layer ID
            <input type="number" value={form.layer_id} onChange={(e) => setForm((f) => ({ ...f, layer_id: e.target.value }))}
              className="mt-1 block w-24 border border-slate-300 rounded px-3 py-2 text-sm" />
          </label>
        </div>
        {error && <p className="text-red-600 text-sm">{error}</p>}
        <button onClick={handlePush} disabled={pushing}
          className="px-4 py-2 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-700 disabled:bg-slate-300">
          {pushing ? "Pushing…" : "Push to ArcGIS"}
        </button>
        {result && (
          <pre className="text-xs bg-slate-50 rounded p-3 overflow-x-auto max-h-40">{result}</pre>
        )}
      </div>

      {/* Export buttons */}
      <div className="bg-white rounded-xl border border-slate-200 p-4 space-y-3">
        <h2 className="font-semibold text-slate-700">Download Spatial Files</h2>
        <div className="flex flex-wrap gap-3">
          <a href="/api/arcgis/export/geojson" className="px-4 py-2 bg-green-600 text-white text-sm rounded-lg hover:bg-green-700">GeoJSON</a>
          <a href="/api/arcgis/export/shapefile" className="px-4 py-2 bg-slate-600 text-white text-sm rounded-lg hover:bg-slate-700">Shapefile (ZIP)</a>
          <a href="/api/arcgis/export/kml" className="px-4 py-2 bg-slate-600 text-white text-sm rounded-lg hover:bg-slate-700">KML</a>
        </div>
      </div>
    </div>
  );
}
