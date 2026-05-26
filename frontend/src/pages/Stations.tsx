import { useEffect, useState } from "react";
import { getStations, addStation, deleteStation, getDeployments, addDeployment } from "../api/client";

type Row = Record<string, unknown>;

export default function Stations() {
  const [tab, setTab] = useState<"stations" | "deployments">("stations");
  const [stations, setStations] = useState<Row[]>([]);
  const [deployments, setDeployments] = useState<Row[]>([]);
  const [loading, setLoading] = useState(true);
  const [newStation, setNewStation] = useState({ station_id: "", latitude: "", longitude: "", stratum: "", habitat: "", notes: "" });
  const [newDep, setNewDep] = useState({ station_id: "", start_date: "", end_date: "", camera_id: "", notes: "" });
  const [adding, setAdding] = useState(false);

  const load = async () => {
    setLoading(true);
    const [s, d] = await Promise.all([getStations(), getDeployments()]);
    setStations(Array.isArray(s) ? s : []);
    setDeployments(Array.isArray(d) ? d : []);
    setLoading(false);
  };
  useEffect(() => { load(); }, []);

  const handleAddStation = async () => {
    if (!newStation.station_id) return;
    setAdding(true);
    await addStation({
      ...newStation,
      latitude: newStation.latitude ? Number(newStation.latitude) : null,
      longitude: newStation.longitude ? Number(newStation.longitude) : null,
    });
    setNewStation({ station_id: "", latitude: "", longitude: "", stratum: "", habitat: "", notes: "" });
    await load();
    setAdding(false);
  };

  const handleAddDeployment = async () => {
    if (!newDep.station_id || !newDep.start_date || !newDep.end_date) return;
    setAdding(true);
    await addDeployment(newDep);
    setNewDep({ station_id: "", start_date: "", end_date: "", camera_id: "", notes: "" });
    await load();
    setAdding(false);
  };

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-bold text-slate-800">Stations & Deployments</h1>

      <div className="flex gap-2">
        {(["stations", "deployments"] as const).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`px-4 py-2 rounded-lg text-sm font-medium capitalize ${
              tab === t ? "bg-green-600 text-white" : "bg-white border border-slate-200 text-slate-600 hover:bg-slate-50"
            }`}
          >
            {t}
          </button>
        ))}
      </div>

      {loading ? (
        <div className="text-center py-12 text-slate-400">Loading…</div>
      ) : tab === "stations" ? (
        <>
          {/* Add station form */}
          <div className="bg-white rounded-xl border border-slate-200 p-4">
            <h2 className="font-semibold text-slate-700 mb-3">Add Station</h2>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
              {[["station_id", "Station ID *"], ["latitude", "Latitude"], ["longitude", "Longitude"], ["stratum", "Stratum"], ["habitat", "Habitat"], ["notes", "Notes"]].map(([key, label]) => (
                <input
                  key={key}
                  placeholder={label}
                  value={newStation[key as keyof typeof newStation]}
                  onChange={(e) => setNewStation((s) => ({ ...s, [key]: e.target.value }))}
                  className="border border-slate-300 rounded px-3 py-2 text-sm"
                />
              ))}
            </div>
            <button
              onClick={handleAddStation}
              disabled={adding || !newStation.station_id}
              className="mt-3 px-4 py-2 bg-green-600 text-white text-sm rounded-lg hover:bg-green-700 disabled:bg-slate-300"
            >
              Add Station
            </button>
          </div>

          {/* Stations table */}
          {stations.length === 0 ? (
            <div className="text-center py-12 text-slate-400">No stations registered yet.</div>
          ) : (
            <div className="bg-white rounded-xl border border-slate-200 overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="bg-slate-50 border-b border-slate-200">
                  <tr>
                    {["station_id", "latitude", "longitude", "stratum", "habitat", "notes", ""].map((c, i) => (
                      <th key={i} className="text-left px-4 py-3 font-medium text-slate-600">{c}</th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {stations.map((s, i) => (
                    <tr key={i} className="hover:bg-slate-50">
                      {["station_id", "latitude", "longitude", "stratum", "habitat", "notes"].map((c) => (
                        <td key={c} className="px-4 py-2 text-slate-700">{String(s[c] ?? "")}</td>
                      ))}
                      <td className="px-4 py-2">
                        <button
                          onClick={() => deleteStation(String(s.station_id)).then(load)}
                          className="text-red-400 hover:text-red-600 text-xs"
                        >
                          Remove
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </>
      ) : (
        <>
          {/* Add deployment form */}
          <div className="bg-white rounded-xl border border-slate-200 p-4">
            <h2 className="font-semibold text-slate-700 mb-3">Add Deployment</h2>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
              {[["station_id", "Station ID *"], ["start_date", "Start Date * (YYYY-MM-DD)"], ["end_date", "End Date * (YYYY-MM-DD)"], ["camera_id", "Camera ID"], ["notes", "Notes"]].map(([key, label]) => (
                <input
                  key={key}
                  placeholder={label}
                  value={newDep[key as keyof typeof newDep]}
                  onChange={(e) => setNewDep((d) => ({ ...d, [key]: e.target.value }))}
                  className="border border-slate-300 rounded px-3 py-2 text-sm"
                />
              ))}
            </div>
            <button
              onClick={handleAddDeployment}
              disabled={adding}
              className="mt-3 px-4 py-2 bg-green-600 text-white text-sm rounded-lg hover:bg-green-700 disabled:bg-slate-300"
            >
              Add Deployment
            </button>
          </div>

          {deployments.length === 0 ? (
            <div className="text-center py-12 text-slate-400">No deployments recorded.</div>
          ) : (
            <div className="bg-white rounded-xl border border-slate-200 overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="bg-slate-50 border-b border-slate-200">
                  <tr>
                    {["station_id", "start_date", "end_date", "camera_id", "notes"].map((c) => (
                      <th key={c} className="text-left px-4 py-3 font-medium text-slate-600 capitalize">{c.replace(/_/g, " ")}</th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {deployments.map((d, i) => (
                    <tr key={i} className="hover:bg-slate-50">
                      {["station_id", "start_date", "end_date", "camera_id", "notes"].map((c) => (
                        <td key={c} className="px-4 py-2 text-slate-700">{String(d[c] ?? "")}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}
    </div>
  );
}
