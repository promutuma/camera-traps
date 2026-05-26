import { useEffect, useState } from "react";
import { getObservations, addObservation, deleteObservation, getCrosscheck } from "../api/client";

type Row = Record<string, unknown>;

const OBS_TYPES = ["Animal", "Track", "Scat", "Camera Malfunction", "Human Activity", "Other"];

export default function Community() {
  const [obs, setObs] = useState<Row[]>([]);
  const [crosscheck, setCrosscheck] = useState<Row[]>([]);
  const [loading, setLoading] = useState(true);
  const [tab, setTab] = useState<"entry" | "crosscheck">("entry");
  const [form, setForm] = useState({
    observer_name: "", observation_type: "Animal", species: "",
    count: "", latitude: "", longitude: "", date: "", time: "", notes: "",
  });

  const load = async () => {
    setLoading(true);
    const [o, c] = await Promise.all([getObservations(), getCrosscheck()]);
    setObs(Array.isArray(o) ? o : []);
    setCrosscheck(Array.isArray(c) ? c : []);
    setLoading(false);
  };
  useEffect(() => { load(); }, []);

  const handleAdd = async () => {
    if (!form.observer_name) return;
    await addObservation({
      ...form,
      count: form.count ? Number(form.count) : null,
      latitude: form.latitude ? Number(form.latitude) : null,
      longitude: form.longitude ? Number(form.longitude) : null,
    });
    setForm({ observer_name: "", observation_type: "Animal", species: "", count: "", latitude: "", longitude: "", date: "", time: "", notes: "" });
    load();
  };

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-bold text-slate-800">Community Observer</h1>

      <div className="flex gap-2">
        {(["entry", "crosscheck"] as const).map((t) => (
          <button key={t} onClick={() => setTab(t)}
            className={`px-4 py-2 rounded-lg text-sm font-medium ${tab === t ? "bg-green-600 text-white" : "bg-white border border-slate-200 text-slate-600 hover:bg-slate-50"}`}>
            {t === "entry" ? "Data Entry" : "Cross-Verification"}
          </button>
        ))}
      </div>

      {loading ? <div className="text-center py-12 text-slate-400">Loading…</div> : tab === "entry" ? (
        <>
          {/* Form */}
          <div className="bg-white rounded-xl border border-slate-200 p-4">
            <h2 className="font-semibold text-slate-700 mb-3">Record Observation</h2>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
              <input placeholder="Observer Name *" value={form.observer_name} onChange={(e) => setForm((f) => ({ ...f, observer_name: e.target.value }))} className="border border-slate-300 rounded px-3 py-2 text-sm" />
              <select value={form.observation_type} onChange={(e) => setForm((f) => ({ ...f, observation_type: e.target.value }))} className="border border-slate-300 rounded px-3 py-2 text-sm">
                {OBS_TYPES.map((t) => <option key={t}>{t}</option>)}
              </select>
              <input placeholder="Species" value={form.species} onChange={(e) => setForm((f) => ({ ...f, species: e.target.value }))} className="border border-slate-300 rounded px-3 py-2 text-sm" />
              <input placeholder="Count" type="number" value={form.count} onChange={(e) => setForm((f) => ({ ...f, count: e.target.value }))} className="border border-slate-300 rounded px-3 py-2 text-sm" />
              <input placeholder="Latitude" value={form.latitude} onChange={(e) => setForm((f) => ({ ...f, latitude: e.target.value }))} className="border border-slate-300 rounded px-3 py-2 text-sm" />
              <input placeholder="Longitude" value={form.longitude} onChange={(e) => setForm((f) => ({ ...f, longitude: e.target.value }))} className="border border-slate-300 rounded px-3 py-2 text-sm" />
              <input placeholder="Date (YYYY-MM-DD)" value={form.date} onChange={(e) => setForm((f) => ({ ...f, date: e.target.value }))} className="border border-slate-300 rounded px-3 py-2 text-sm" />
              <input placeholder="Time (HH:MM)" value={form.time} onChange={(e) => setForm((f) => ({ ...f, time: e.target.value }))} className="border border-slate-300 rounded px-3 py-2 text-sm" />
              <input placeholder="Notes" value={form.notes} onChange={(e) => setForm((f) => ({ ...f, notes: e.target.value }))} className="border border-slate-300 rounded px-3 py-2 text-sm" />
            </div>
            <button onClick={handleAdd} disabled={!form.observer_name} className="mt-3 px-4 py-2 bg-green-600 text-white text-sm rounded-lg hover:bg-green-700 disabled:bg-slate-300">
              Add Observation
            </button>
          </div>

          {/* Table */}
          {obs.length === 0 ? <div className="text-center py-8 text-slate-400">No observations recorded.</div> : (
            <div className="bg-white rounded-xl border border-slate-200 overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="bg-slate-50 border-b border-slate-200">
                  <tr>{["observer_name", "observation_type", "species", "count", "date", "notes", ""].map((c, i) => <th key={i} className="text-left px-4 py-3 font-medium text-slate-600">{c}</th>)}</tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {obs.map((row, i) => (
                    <tr key={i} className="hover:bg-slate-50">
                      {["observer_name", "observation_type", "species", "count", "date", "notes"].map((c) => <td key={c} className="px-4 py-2 text-slate-700">{String(row[c] ?? "")}</td>)}
                      <td className="px-4 py-2"><button onClick={() => deleteObservation(Number(row.id ?? i)).then(load)} className="text-red-400 hover:text-red-600 text-xs">Remove</button></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </>
      ) : (
        crosscheck.length === 0 ? <div className="text-center py-12 text-slate-400">No cross-verification data. Add observations and process camera images first.</div> : (
          <div className="bg-white rounded-xl border border-slate-200 overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-slate-50 border-b border-slate-200">
                <tr>{["species", "camera_detections", "observer_sightings", "match"].map((c) => <th key={c} className="text-left px-4 py-3 font-medium text-slate-600 capitalize">{c.replace(/_/g, " ")}</th>)}</tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {crosscheck.map((row, i) => (
                  <tr key={i} className="hover:bg-slate-50">
                    {["species", "camera_detections", "observer_sightings", "match"].map((c) => <td key={c} className="px-4 py-2 text-slate-700">{String(row[c] ?? "")}</td>)}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )
      )}
    </div>
  );
}
