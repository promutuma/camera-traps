import { useEffect, useState } from "react";
import { getProject, updateProject } from "../api/client";

type ProjectData = Record<string, unknown>;

export default function ProjectConfig() {
  const [data, setData] = useState<ProjectData | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [form, setForm] = useState({ project_name: "", survey_area: "", notes: "" });

  useEffect(() => {
    getProject()
      .then((d) => {
        setData(d);
        setForm({
          project_name: String(d.project_name ?? ""),
          survey_area: String(d.survey_area ?? ""),
          notes: String(d.notes ?? ""),
        });
      })
      .finally(() => setLoading(false));
  }, []);

  const handleSave = async () => {
    setSaving(true);
    await updateProject(form);
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
    setSaving(false);
  };

  const handleLockBaseline = async () => {
    await fetch("/api/project/baseline/lock", { method: "POST" });
    alert("Baseline locked.");
  };

  if (loading) return <div className="text-center py-12 text-slate-400">Loading…</div>;

  return (
    <div className="max-w-2xl space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-slate-800">Project Configuration</h1>
        <p className="text-slate-500 text-sm mt-1">Manage survey projects, configure thresholds, and lock reference baselines.</p>
      </div>

      <div className="bg-white rounded-xl border border-slate-200 p-4 space-y-4">
        <h2 className="font-semibold text-slate-700">Project Details</h2>
        <div className="space-y-3">
          <label className="block text-sm text-slate-700">
            Project Name
            <input value={form.project_name} onChange={(e) => setForm((f) => ({ ...f, project_name: e.target.value }))}
              className="mt-1 block w-full border border-slate-300 rounded px-3 py-2 text-sm" />
          </label>
          <label className="block text-sm text-slate-700">
            Survey Area
            <input value={form.survey_area} onChange={(e) => setForm((f) => ({ ...f, survey_area: e.target.value }))}
              className="mt-1 block w-full border border-slate-300 rounded px-3 py-2 text-sm" />
          </label>
          <label className="block text-sm text-slate-700">
            Notes
            <textarea value={form.notes} onChange={(e) => setForm((f) => ({ ...f, notes: e.target.value }))}
              rows={3} className="mt-1 block w-full border border-slate-300 rounded px-3 py-2 text-sm" />
          </label>
        </div>
        <div className="flex gap-3 items-center">
          <button onClick={handleSave} disabled={saving}
            className="px-4 py-2 bg-green-600 text-white text-sm rounded-lg hover:bg-green-700 disabled:bg-slate-300">
            {saving ? "Saving…" : "Save"}
          </button>
          {saved && <span className="text-green-600 text-sm">✓ Saved</span>}
        </div>
      </div>

      <div className="bg-white rounded-xl border border-slate-200 p-4 space-y-4">
        <h2 className="font-semibold text-slate-700">Current Configuration</h2>
        <pre className="text-xs bg-slate-50 rounded p-3 overflow-x-auto max-h-64">
          {JSON.stringify(data, null, 2)}
        </pre>
      </div>

      <div className="bg-white rounded-xl border border-slate-200 p-4 space-y-3">
        <h2 className="font-semibold text-slate-700">Actions</h2>
        <div className="flex gap-3">
          <button onClick={handleLockBaseline}
            className="px-4 py-2 bg-amber-500 text-white text-sm rounded-lg hover:bg-amber-600">
            Lock Reference Baseline
          </button>
          <a href="/api/project/export"
            className="px-4 py-2 bg-slate-600 text-white text-sm rounded-lg hover:bg-slate-700">
            Export Config JSON
          </a>
        </div>
      </div>
    </div>
  );
}
