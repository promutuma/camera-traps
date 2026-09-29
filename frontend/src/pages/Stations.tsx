import React, { useEffect, useState } from "react";
import { MapContainer, TileLayer, Marker, Popup, useMap } from "react-leaflet";
import { getStationSummary, addStation, updateStation, deleteStation, getDeployments, addDeployment, deleteDeployment, getOrphanStations, reassignStation, getStationMap, assignCamera, getCameras, addCamera, updateCamera, deleteCamera } from "../api/client";

type StationFeature = {
  type: string;
  geometry: { type: string; coordinates: [number, number] };
  properties: Record<string, unknown>;
};

function FitStationBounds({ features }: { features: StationFeature[] }) {
  const map = useMap();
  useEffect(() => {
    if (features.length === 0) return;
    if (features.length === 1) {
      map.setView([features[0].geometry.coordinates[1], features[0].geometry.coordinates[0]], 10);
      return;
    }
    const lats = features.map((f) => f.geometry.coordinates[1]);
    const lons = features.map((f) => f.geometry.coordinates[0]);
    map.fitBounds(
      [[Math.min(...lats), Math.min(...lons)], [Math.max(...lats), Math.max(...lons)]],
      { padding: [40, 40], maxZoom: 13 }
    );
  }, [features, map]);
  return null;
}

type Row = Record<string, unknown>;

// SVG Icons
const MapPinIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.8} stroke="currentColor" className="w-5 h-5">
    <path strokeLinecap="round" strokeLinejoin="round" d="M15 10.5a3 3 0 11-6 0 3 3 0 016 0z" />
    <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 10.5c0 7.142-7.5 11.25-7.5 11.25S4.5 17.642 4.5 10.5a7.5 7.5 0 1115 0z" />
  </svg>
);

const CalendarIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.8} stroke="currentColor" className="w-5 h-5">
    <path strokeLinecap="round" strokeLinejoin="round" d="M6.75 3v2.25M17.25 3v2.25M3 18.75V7.5a2.25 2.25 0 012.25-2.25h13.5A2.25 2.25 0 0121 7.5v11.25m-18 0A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75m-18 0v-7.5A2.25 2.25 0 015.25 9h13.5A2.25 2.25 0 0121 11.25v7.5m-9-6h.008v.008H12v-.008zM12 15h.008v.008H12V15zm0 2.25h.008v.008H12v-.008zM9.75 15h.008v.008H9.75V15zm0 2.25h.008v.008H9.75v-.008zM7.5 15h.008v.008H7.5V15zm0 2.25h.008v.008H7.5v-.008zm6.75-4.5h.008v.008h-.008v-.008zm0 2.25h.008v.008h-.008V15zm0 2.25h.008v.008h-.008v-.008zm2.25-4.5h.008v.008H16.5v-.008zm0 2.25h.008v.008H16.5V15z" />
  </svg>
);

const ChartBarIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.8} stroke="currentColor" className="w-5 h-5">
    <path strokeLinecap="round" strokeLinejoin="round" d="M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v5.625c0 .621-.504 1.125-1.125 1.125h-2.25A1.125 1.125 0 013 18.75v-5.625zM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v10.125c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V8.625zM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v14.625c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 01-1.125-1.125V4.125z" />
  </svg>
);

const ActivityIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.8} stroke="currentColor" className="w-5 h-5">
    <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 13.5l10.5-11.25L12 10.5h8.25L9.75 21.75 12 13.5H3.75z" />
  </svg>
);

const STRATA_OPTIONS = [
  "Wetland Habitat",
  "Watering Point",
  "Corridor",
  "Other"
];

export default function Stations() {
  const [tab, setTab] = useState<"stations" | "deployments" | "cameras" | "map">("stations");
  const [stations, setStations] = useState<Row[]>([]);
  const [deployments, setDeployments] = useState<Row[]>([]);
  const [cameras, setCameras] = useState<Row[]>([]);
  const [loading, setLoading] = useState(true);
  const [adding, setAdding] = useState(false);
  const [mapFeatures, setMapFeatures] = useState<StationFeature[] | null>(null);
  const [mapLoading, setMapLoading] = useState(false);

  // Forms
  const [newStation, setNewStation] = useState({ station_id: "", latitude: "", longitude: "", stratum: "", habitat: "", camera_model: "", team_member: "", notes: "" });
  const [newDep, setNewDep] = useState({ station_id: "", start_date: "", end_date: "", camera_id: "", camera_down_days: "", notes: "" });
  const [newCamera, setNewCamera] = useState({ camera_id: "", model: "", serial_number: "", notes: "" });

  // Inline edit state for cameras
  const [editingCameraId, setEditingCameraId] = useState<string | null>(null);
  const [editCameraDraft, setEditCameraDraft] = useState<Record<string, string>>({});
  const [cameraErrors, setCameraErrors] = useState<string[]>([]);

  // Inline edit state: station_id of the row being edited, and its draft values
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editDraft, setEditDraft] = useState<Record<string, string>>({});

  // Quick "assign camera to station" inline editor — separate from the main
  // edit form since it's a distinct action (creates/updates a deployment,
  // not a station field) and needs to be immediately discoverable.
  const [cameraEditId, setCameraEditId] = useState<string | null>(null);
  const [cameraDraft, setCameraDraft] = useState("");
  const [assigningCamera, setAssigningCamera] = useState(false);

  // Orphan station IDs (used by images but not registered)
  const [orphans, setOrphans] = useState<{ station_id: string; image_count: number }[]>([]);
  const [reassigning, setReassigning] = useState<string | null>(null);
  const [reassignTarget, setReassignTarget] = useState<Record<string, string>>({});

  // Validations
  const [stationErrors, setStationErrors] = useState<string[]>([]);
  const [depErrors, setDepErrors] = useState<string[]>([]);

  const load = async () => {
    setLoading(true);
    try {
      const [s, d, o, c] = await Promise.all([getStationSummary(), getDeployments(), getOrphanStations(), getCameras()]);
      
      // Normalize stations schema differences (gps_lat/gps_lon and habitat_stratum from backend)
      const mappedStations = (Array.isArray(s) ? s : []).map((r) => ({
        station_id: r.station_id,
        latitude: r.latitude ?? r.gps_lat ?? "",
        longitude: r.longitude ?? r.gps_lon ?? "",
        stratum: r.stratum ?? r.habitat_stratum ?? "",
        camera_model: r.camera_model ?? "",
        team_member: r.team_member ?? "",
        notes: r.notes ?? "",
        current_camera_id: r.current_camera_id ?? "",
        trap_nights: r.trap_nights ?? 0,
        functionality_pct: r.functionality_pct ?? 0,
        deployment_count: r.deployment_count ?? 0,
        status: r.status ?? "No Data"
      }));

      setStations(mappedStations);
      setDeployments(Array.isArray(d) ? d : []);
      setOrphans(Array.isArray(o) ? o : []);
      setCameras(Array.isArray(c) ? c : []);
      setMapFeatures(null); // stale — refetched next time the Map tab is opened
    } catch (e) {
      console.error("Failed to load stations data", e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  useEffect(() => {
    if (tab !== "map" || mapFeatures !== null) return;
    setMapLoading(true);
    getStationMap()
      .then((gj) => setMapFeatures(gj?.features ?? []))
      .catch(() => setMapFeatures([]))
      .finally(() => setMapLoading(false));
  }, [tab, mapFeatures]);

  // Inline Validation for Stations Form
  useEffect(() => {
    const errs: string[] = [];
    if (newStation.station_id) {
      const exists = stations.some(
        (s) => String(s.station_id).trim().toLowerCase() === newStation.station_id.trim().toLowerCase()
      );
      if (exists) {
        errs.push(`Station ID "${newStation.station_id}" is already registered.`);
      }
    }
    
    if (newStation.latitude) {
      const lat = Number(newStation.latitude);
      if (isNaN(lat) || lat < -90 || lat > 90) {
        errs.push("Latitude must be a valid number between -90 and 90.");
      }
    }

    if (newStation.longitude) {
      const lon = Number(newStation.longitude);
      if (isNaN(lon) || lon < -180 || lon > 180) {
        errs.push("Longitude must be a valid number between -180 and 180.");
      }
    }
    setStationErrors(errs);
  }, [newStation, stations]);

  // Inline Validation for Deployments Form
  useEffect(() => {
    const errs: string[] = [];
    if (newDep.start_date && newDep.end_date) {
      const start = new Date(newDep.start_date);
      const end = new Date(newDep.end_date);
      if (end < start) {
        errs.push("End Date cannot be chronologically before Start Date.");
      }
    }
    setDepErrors(errs);
  }, [newDep]);

  // Inline Validation for Camera Form
  useEffect(() => {
    const errs: string[] = [];
    if (newCamera.camera_id) {
      const exists = cameras.some(
        (c) => String(c.camera_id).trim().toLowerCase() === newCamera.camera_id.trim().toLowerCase()
      );
      if (exists) {
        errs.push(`Camera ID "${newCamera.camera_id}" is already registered.`);
      }
    }
    setCameraErrors(errs);
  }, [newCamera, cameras]);

  const handleAddStation = async () => {
    if (!newStation.station_id || stationErrors.length > 0) return;
    setAdding(true);
    try {
      await addStation({
        ...newStation,
        latitude: newStation.latitude ? Number(newStation.latitude) : null,
        longitude: newStation.longitude ? Number(newStation.longitude) : null,
      });
      setNewStation({ station_id: "", latitude: "", longitude: "", stratum: "", habitat: "", camera_model: "", team_member: "", notes: "" });
      await load();
    } catch (e) {
      console.error(e);
    } finally {
      setAdding(false);
    }
  };

  const handleDeleteStation = async (id: string) => {
    try {
      const res: any = await deleteStation(id);
      if (res?.ok === false && (res?.orphaned_detections ?? 0) > 0) {
        const count = res.orphaned_detections;
        const ok = window.confirm(
          `Station "${id}" has ${count} detection record${count === 1 ? "" : "s"} referencing it.\n\nThose records will keep their station label but will no longer appear on the spatial map.\n\nDelete anyway?`
        );
        if (!ok) return;
        await deleteStation(id, true);
      }
      await load();
    } catch (e) {
      console.error(e);
    }
  };

  const handleEditStart = (s: Row) => {
    setEditingId(String(s.station_id));
    setEditDraft({
      latitude:     String(s.latitude ?? ""),
      longitude:    String(s.longitude ?? ""),
      stratum:      String(s.stratum ?? ""),
      camera_model: String(s.camera_model ?? ""),
      team_member:  String(s.team_member ?? ""),
      notes:        String(s.notes ?? ""),
    });
  };

  const handleEditSave = async (station_id: string) => {
    try {
      await updateStation(station_id, {
        latitude:     editDraft.latitude     ? Number(editDraft.latitude)  : null,
        longitude:    editDraft.longitude    ? Number(editDraft.longitude) : null,
        stratum:      editDraft.stratum      || null,
        camera_model: editDraft.camera_model || null,
        team_member:  editDraft.team_member  || null,
        notes:        editDraft.notes        || null,
      });
      setEditingId(null);
      await load();
    } catch (e) {
      console.error(e);
    }
  };

  const handleReassign = async (fromId: string) => {
    const toId = reassignTarget[fromId];
    if (!toId) return;
    setReassigning(fromId);
    try {
      await reassignStation(fromId, toId);
      await load();
    } catch (e) {
      console.error(e);
    } finally {
      setReassigning(null);
    }
  };

  const handleAssignCamera = async (stationId: string) => {
    if (!cameraDraft.trim()) return;
    setAssigningCamera(true);
    try {
      await assignCamera(stationId, cameraDraft.trim());
      setCameraEditId(null);
      setCameraDraft("");
      await load();
    } catch (e) {
      console.error(e);
    } finally {
      setAssigningCamera(false);
    }
  };

  const handleAddDeployment = async () => {
    if (!newDep.station_id || !newDep.start_date || depErrors.length > 0) return;
    setAdding(true);
    try {
      await addDeployment({
        ...newDep,
        end_date: newDep.end_date || undefined,
        camera_down_days: newDep.camera_down_days ? parseInt(newDep.camera_down_days, 10) : undefined,
      });
      setNewDep({ station_id: "", start_date: "", end_date: "", camera_id: "", camera_down_days: "", notes: "" });
      await load();
    } catch (e) {
      console.error(e);
    } finally {
      setAdding(false);
    }
  };

  const handleAddCamera = async () => {
    if (!newCamera.camera_id || cameraErrors.length > 0) return;
    setAdding(true);
    try {
      await addCamera(newCamera);
      setNewCamera({ camera_id: "", model: "", serial_number: "", notes: "" });
      await load();
    } catch (e) {
      console.error(e);
    } finally {
      setAdding(false);
    }
  };

  const handleDeleteCamera = async (cameraId: string) => {
    try {
      await deleteCamera(cameraId);
      await load();
    } catch (e) {
      console.error(e);
    }
  };

  const handleCameraEditStart = (c: Row) => {
    setEditingCameraId(String(c.camera_id));
    setEditCameraDraft({
      model: String(c.model ?? ""),
      serial_number: String(c.serial_number ?? ""),
      status: String(c.status ?? "active"),
      notes: String(c.notes ?? ""),
    });
  };

  const handleCameraEditSave = async (cameraId: string) => {
    try {
      await updateCamera(cameraId, editCameraDraft);
      setEditingCameraId(null);
      await load();
    } catch (e) {
      console.error(e);
    }
  };

  // KPIs
  const totalStations = stations.length;
  const activeDeploymentsCount = deployments.filter((d) => {
    if (!d.end_date) return true;
    const now = new Date();
    const end = new Date(String(d.end_date));
    return end >= now;
  }).length;
  const totalTrapNights = stations.reduce((sum, s) => sum + (Number(s.trap_nights) || 0), 0);
  const avgFunctionality = stations.length > 0
    ? (stations.reduce((sum, s) => sum + (Number(s.functionality_pct) || 0), 0) / stations.length).toFixed(1)
    : "0.0";

  const getStatusBadge = (status: string) => {
    switch (String(status)) {
      case "Active":
        return (
          <span className="px-2.5 py-1 text-[11px] font-semibold bg-emerald-50 text-emerald-700 dark:bg-emerald-950/40 dark:text-emerald-400 rounded-full border border-emerald-100 dark:border-emerald-900/50">
            Active
          </span>
        );
      case "Low Functionality":
        return (
          <span className="px-2.5 py-1 text-[11px] font-semibold bg-amber-50 text-amber-700 dark:bg-amber-950/40 dark:text-amber-400 rounded-full border border-amber-100 dark:border-amber-900/50">
            Low Func
          </span>
        );
      case "No Data":
      default:
        return (
          <span className="px-2.5 py-1 text-[11px] font-semibold bg-slate-100 text-slate-500 dark:bg-slate-800 dark:text-slate-400 rounded-full border border-slate-200 dark:border-slate-700/60">
            Inactive
          </span>
        );
    }
  };

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Page Header */}
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-slate-900 dark:text-white">
          Stations & Deployments
        </h1>
        <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
          Register camera trap setups, track functional deployment intervals, and analyze metrics by stratum.
        </p>
      </div>

      {/* Orphan Station Banner */}
      {orphans.length > 0 && (
        <div className="bg-amber-50 dark:bg-amber-950/30 border border-amber-200 dark:border-amber-800/50 rounded-xl p-4 space-y-3">
          <div className="flex items-start gap-3">
            <svg xmlns="http://www.w3.org/2000/svg" className="w-5 h-5 text-amber-600 dark:text-amber-400 mt-0.5 shrink-0" fill="none" viewBox="0 0 24 24" strokeWidth={1.8} stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z" />
            </svg>
            <div>
              <p className="text-sm font-semibold text-amber-800 dark:text-amber-300">Unregistered station IDs found in image data</p>
              <p className="text-xs text-amber-700 dark:text-amber-400 mt-0.5">
                These images won't appear on the spatial map or GIS exports until reassigned to a registered station.
              </p>
            </div>
          </div>
          <div className="space-y-2">
            {orphans.map((o) => (
              <div key={o.station_id} className="flex flex-wrap items-center gap-2">
                <span className="text-xs font-mono bg-amber-100 dark:bg-amber-900/40 text-amber-800 dark:text-amber-300 px-2 py-1 rounded border border-amber-200 dark:border-amber-700/50">
                  {o.station_id}
                </span>
                <span className="text-xs text-amber-700 dark:text-amber-400">{o.image_count} image{o.image_count !== 1 ? "s" : ""} →</span>
                <select
                  value={reassignTarget[o.station_id] ?? ""}
                  onChange={(e) => setReassignTarget((t) => ({ ...t, [o.station_id]: e.target.value }))}
                  className="text-xs border border-amber-300 dark:border-amber-700 rounded px-2 py-1 bg-white dark:bg-slate-900 text-slate-800 dark:text-slate-200 focus:outline-none focus:ring-1 focus:ring-amber-500"
                >
                  <option value="">— pick a station —</option>
                  {stations.map((s) => (
                    <option key={String(s.station_id)} value={String(s.station_id)}>{String(s.station_id)}</option>
                  ))}
                </select>
                <button
                  onClick={() => handleReassign(o.station_id)}
                  disabled={!reassignTarget[o.station_id] || reassigning === o.station_id}
                  className="text-xs px-3 py-1 bg-amber-600 hover:bg-amber-700 disabled:bg-amber-300 dark:disabled:bg-amber-900 text-white font-semibold rounded transition cursor-pointer disabled:cursor-not-allowed"
                >
                  {reassigning === o.station_id ? "Reassigning…" : "Reassign"}
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* KPI Stats Panel */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-white dark:bg-slate-900 p-5 rounded-2xl border border-slate-200/60 dark:border-slate-800/80 shadow-sm flex items-center gap-4 hover:border-slate-300 dark:hover:border-slate-700 transition">
          <div className="p-3 bg-emerald-50 dark:bg-emerald-950/40 text-emerald-600 dark:text-emerald-400 rounded-xl">
            <MapPinIcon />
          </div>
          <div>
            <div className="text-[11px] font-bold text-slate-400 dark:text-slate-550 uppercase tracking-wider">
              Total Stations
            </div>
            <div className="text-2xl font-bold text-slate-800 dark:text-white mt-0.5">
              {totalStations}
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-slate-900 p-5 rounded-2xl border border-slate-200/60 dark:border-slate-800/80 shadow-sm flex items-center gap-4 hover:border-slate-300 dark:hover:border-slate-700 transition">
          <div className="p-3 bg-emerald-50 dark:bg-emerald-950/40 text-emerald-600 dark:text-emerald-400 rounded-xl">
            <CalendarIcon />
          </div>
          <div>
            <div className="text-[11px] font-bold text-slate-400 dark:text-slate-550 uppercase tracking-wider">
              Active Deployments
            </div>
            <div className="text-2xl font-bold text-slate-800 dark:text-white mt-0.5">
              {activeDeploymentsCount}
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-slate-900 p-5 rounded-2xl border border-slate-200/60 dark:border-slate-800/80 shadow-sm flex items-center gap-4 hover:border-slate-300 dark:hover:border-slate-700 transition">
          <div className="p-3 bg-emerald-50 dark:bg-emerald-950/40 text-emerald-600 dark:text-emerald-400 rounded-xl">
            <ChartBarIcon />
          </div>
          <div>
            <div className="text-[11px] font-bold text-slate-400 dark:text-slate-555 uppercase tracking-wider">
              Total Trap Nights
            </div>
            <div className="text-2xl font-bold text-slate-800 dark:text-white mt-0.5">
              {totalTrapNights}
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-slate-900 p-5 rounded-2xl border border-slate-200/60 dark:border-slate-800/80 shadow-sm flex items-center gap-4 hover:border-slate-300 dark:hover:border-slate-700 transition">
          <div className="p-3 bg-emerald-50 dark:bg-emerald-950/40 text-emerald-600 dark:text-emerald-400 rounded-xl">
            <ActivityIcon />
          </div>
          <div>
            <div className="text-[11px] font-bold text-slate-400 dark:text-slate-555 uppercase tracking-wider">
              Avg Functionality
            </div>
            <div className="text-2xl font-bold text-slate-800 dark:text-white mt-0.5">
              {avgFunctionality}%
            </div>
          </div>
        </div>
      </div>

      {/* Segmented Control Switcher */}
      <div className="flex bg-slate-100 dark:bg-slate-950 p-1 rounded-xl border border-slate-200/50 dark:border-slate-900/60 max-w-sm">
        {(["stations", "deployments", "cameras", "map"] as const).map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`flex-1 text-center py-2 px-4 rounded-lg text-sm font-semibold capitalize transition-all cursor-pointer ${
              tab === t
                ? "bg-white dark:bg-slate-800 text-emerald-600 dark:text-emerald-400 shadow-sm"
                : "text-slate-500 dark:text-slate-400 hover:text-slate-800 dark:hover:text-slate-200"
            }`}
          >
            {t}
          </button>
        ))}
      </div>

      {loading ? (
        <div className="text-center py-16 text-slate-400 dark:text-slate-555 animate-pulse">
          Loading camera trap records…
        </div>
      ) : tab === "stations" ? (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 items-start">
          {/* Add Station Sidebar */}
          <div className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200/60 dark:border-slate-800/80 p-5 space-y-4 shadow-sm lg:col-span-1">
            <div>
              <h2 className="font-bold text-base text-slate-900 dark:text-white">
                Register Camera Station
              </h2>
              <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
                Register a new monitoring location in the network.
              </p>
            </div>

            <div className="space-y-3">
              <div>
                <label className="block text-xs font-semibold text-slate-500 dark:text-slate-400 mb-1">
                  Station ID *
                </label>
                <input
                  placeholder="e.g. ST-01"
                  value={newStation.station_id}
                  onChange={(e) => setNewStation((s) => ({ ...s, station_id: e.target.value }))}
                  className="w-full border border-slate-200 dark:border-slate-800 rounded-xl px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500/20 focus:border-emerald-500 bg-slate-50/50 dark:bg-slate-950 text-slate-800 dark:text-slate-100 transition"
                />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-xs font-semibold text-slate-500 dark:text-slate-400 mb-1">
                    Latitude
                  </label>
                  <input
                    placeholder="e.g. 8.23"
                    type="number"
                    step="0.000001"
                    value={newStation.latitude}
                    onChange={(e) => setNewStation((s) => ({ ...s, latitude: e.target.value }))}
                    className="w-full border border-slate-200 dark:border-slate-800 rounded-xl px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500/20 focus:border-emerald-500 bg-slate-50/50 dark:bg-slate-955 text-slate-800 dark:text-slate-100 transition"
                  />
                </div>
                <div>
                  <label className="block text-xs font-semibold text-slate-500 dark:text-slate-400 mb-1">
                    Longitude
                  </label>
                  <input
                    placeholder="e.g. 34.62"
                    type="number"
                    step="0.000001"
                    value={newStation.longitude}
                    onChange={(e) => setNewStation((s) => ({ ...s, longitude: e.target.value }))}
                    className="w-full border border-slate-200 dark:border-slate-800 rounded-xl px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500/20 focus:border-emerald-500 bg-slate-50/50 dark:bg-slate-955 text-slate-800 dark:text-slate-100 transition"
                  />
                </div>
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-500 dark:text-slate-400 mb-1">
                  Stratum Class
                </label>
                <select
                  value={newStation.stratum}
                  onChange={(e) => setNewStation((s) => ({ ...s, stratum: e.target.value }))}
                  className="w-full border border-slate-200 dark:border-slate-800 rounded-xl px-3 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500/20 focus:border-emerald-500 bg-slate-50/50 dark:bg-slate-950 text-slate-800 dark:text-slate-100 cursor-pointer transition"
                >
                  <option value="">Select Stratum Type</option>
                  {STRATA_OPTIONS.map((opt) => (
                    <option key={opt} value={opt}>
                      {opt}
                    </option>
                  ))}
                </select>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-xs font-semibold text-slate-500 dark:text-slate-400 mb-1">
                    Camera Model
                  </label>
                  <input
                    placeholder="e.g. Bushnell Core"
                    value={newStation.camera_model}
                    onChange={(e) => setNewStation((s) => ({ ...s, camera_model: e.target.value }))}
                    className="w-full border border-slate-200 dark:border-slate-800 rounded-xl px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500/20 focus:border-emerald-500 bg-slate-50/50 dark:bg-slate-950 text-slate-800 dark:text-slate-100 transition"
                  />
                </div>
                <div>
                  <label className="block text-xs font-semibold text-slate-500 dark:text-slate-400 mb-1">
                    Team Member
                  </label>
                  <input
                    placeholder="e.g. J. Kamau"
                    value={newStation.team_member}
                    onChange={(e) => setNewStation((s) => ({ ...s, team_member: e.target.value }))}
                    className="w-full border border-slate-200 dark:border-slate-800 rounded-xl px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500/20 focus:border-emerald-500 bg-slate-50/50 dark:bg-slate-950 text-slate-800 dark:text-slate-100 transition"
                  />
                </div>
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-500 dark:text-slate-400 mb-1">
                  Notes
                </label>
                <textarea
                  placeholder="Notes, features, vegetation..."
                  rows={2}
                  value={newStation.notes}
                  onChange={(e) => setNewStation((s) => ({ ...s, notes: e.target.value }))}
                  className="w-full border border-slate-200 dark:border-slate-800 rounded-xl px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500/20 focus:border-emerald-500 bg-slate-50/50 dark:bg-slate-950 text-slate-800 dark:text-slate-100 transition"
                />
              </div>
            </div>

            {/* Error notifications */}
            {stationErrors.length > 0 && (
              <div className="bg-red-50 dark:bg-red-950/20 border border-red-200 dark:border-red-800/40 rounded-xl p-3 space-y-1">
                {stationErrors.map((err, i) => (
                  <p key={i} className="text-xs text-red-600 dark:text-red-400 flex items-center gap-1">
                    <span className="material-symbols-outlined text-sm leading-none select-none">error</span> {err}
                  </p>
                ))}
              </div>
            )}

            <button
              onClick={handleAddStation}
              disabled={adding || !newStation.station_id || stationErrors.length > 0}
              className="w-full py-2.5 bg-emerald-600 hover:bg-emerald-700 disabled:bg-slate-100 dark:disabled:bg-slate-800 text-white disabled:text-slate-400 dark:disabled:text-slate-600 text-sm font-semibold rounded-xl transition shadow-sm hover:shadow cursor-pointer"
            >
              {adding ? "Adding..." : "Add Station"}
            </button>
          </div>

          {/* Stations Table Column */}
          <div className="lg:col-span-2 space-y-4">
            {stations.length === 0 ? (
              <div className="text-center py-20 text-slate-400 dark:text-slate-550 bg-white dark:bg-slate-900 border border-slate-200/60 dark:border-slate-800/80 rounded-2xl shadow-sm">
                No camera stations registered yet. Register a station to get started.
              </div>
            ) : (
              <div className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200/60 dark:border-slate-800/80 overflow-hidden shadow-sm">
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead className="bg-slate-50/50 dark:bg-slate-955/20 border-b border-slate-100 dark:border-slate-800">
                      <tr>
                        {["Station ID", "Coordinates", "Camera", "Stratum", "Deployments", "Trap Nights", "Func Rate", "Status", ""].map((header) => (
                          <th
                            key={header}
                            className="text-left px-4 py-3.5 font-bold text-slate-400 dark:text-slate-500 uppercase tracking-wider text-[10px]"
                          >
                            {header}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100 dark:divide-slate-800/80">
                      {stations.map((s, i) => {
                        const sid = String(s.station_id);
                        const isEditing = editingId === sid;
                        return (
                          <React.Fragment key={i}>
                            <tr className="hover:bg-slate-50/50 dark:hover:bg-slate-950/20 transition-colors">
                              <td className="px-4 py-3.5 font-bold text-slate-800 dark:text-slate-200">
                                {sid}
                              </td>
                              <td className="px-4 py-3.5 text-xs text-slate-500 dark:text-slate-400 font-mono">
                                {s.latitude && s.longitude
                                  ? `${Number(s.latitude).toFixed(4)}, ${Number(s.longitude).toFixed(4)}`
                                  : "No Coordinates"}
                              </td>
                              <td className="px-4 py-3.5 text-xs">
                                {cameraEditId === sid ? (
                                  <div className="flex items-center gap-1.5">
                                    <input
                                      autoFocus
                                      placeholder="e.g. CAM-42"
                                      value={cameraDraft}
                                      onChange={(e) => setCameraDraft(e.target.value)}
                                      onKeyDown={(e) => {
                                        if (e.key === "Enter") handleAssignCamera(sid);
                                        if (e.key === "Escape") { setCameraEditId(null); setCameraDraft(""); }
                                      }}
                                      className="w-24 border border-slate-200 dark:border-slate-700 rounded-lg px-2 py-1 text-xs bg-white dark:bg-slate-900 text-slate-800 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-emerald-500/20 focus:border-emerald-500"
                                    />
                                    <button
                                      onClick={() => handleAssignCamera(sid)}
                                      disabled={assigningCamera || !cameraDraft.trim()}
                                      className="text-emerald-600 hover:text-emerald-700 dark:text-emerald-400 disabled:opacity-40 cursor-pointer"
                                      title="Save"
                                    >
                                      <span className="material-symbols-outlined text-sm select-none">check</span>
                                    </button>
                                    <button
                                      onClick={() => { setCameraEditId(null); setCameraDraft(""); }}
                                      className="text-slate-400 hover:text-slate-600 dark:hover:text-slate-300 cursor-pointer"
                                      title="Cancel"
                                    >
                                      <span className="material-symbols-outlined text-sm select-none">close</span>
                                    </button>
                                  </div>
                                ) : (
                                  <button
                                    onClick={() => { setCameraEditId(sid); setCameraDraft(String(s.current_camera_id || "")); }}
                                    className="flex items-center gap-1 font-mono text-slate-700 dark:text-slate-300 hover:text-emerald-600 dark:hover:text-emerald-400 cursor-pointer group"
                                    title="Assign a camera to this station"
                                  >
                                    {s.current_camera_id ? String(s.current_camera_id) : <span className="text-slate-400 italic font-sans">Unassigned</span>}
                                    <span className="material-symbols-outlined text-xs opacity-0 group-hover:opacity-100 transition-opacity select-none">edit</span>
                                  </button>
                                )}
                              </td>
                              <td className="px-4 py-3.5 text-slate-600 dark:text-slate-350 text-xs">
                                {String(s.stratum || "—")}
                              </td>
                              <td className="px-4 py-3.5 text-center text-slate-700 dark:text-slate-300 font-medium">
                                {Number(s.deployment_count)}
                              </td>
                              <td className="px-4 py-3.5 text-center text-slate-700 dark:text-slate-300 font-medium">
                                {Number(s.trap_nights)}
                              </td>
                              <td className="px-4 py-3.5">
                                <div className="flex items-center gap-2">
                                  <span className={`text-xs font-semibold ${
                                    Number(s.functionality_pct) >= 90
                                      ? "text-emerald-600 dark:text-emerald-400"
                                      : Number(s.functionality_pct) > 0
                                      ? "text-amber-600 dark:text-amber-400"
                                      : "text-slate-400"
                                  }`}>
                                    {Number(s.functionality_pct).toFixed(0)}%
                                  </span>
                                  {Number(s.functionality_pct) > 0 && (
                                    <div className="w-12 h-1.5 bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden">
                                      <div
                                        className={`h-full rounded-full ${
                                          Number(s.functionality_pct) >= 90 ? "bg-emerald-500" : "bg-amber-500"
                                        }`}
                                        style={{ width: `${s.functionality_pct}%` }}
                                      />
                                    </div>
                                  )}
                                </div>
                              </td>
                              <td className="px-4 py-3.5">
                                {getStatusBadge(String(s.status))}
                              </td>
                              <td className="px-4 py-3.5 text-right whitespace-nowrap">
                                <button
                                  onClick={() => isEditing ? setEditingId(null) : handleEditStart(s)}
                                  className="text-emerald-600 hover:text-emerald-700 dark:text-emerald-400 dark:hover:text-emerald-300 font-semibold text-xs cursor-pointer transition hover:underline mr-3"
                                >
                                  {isEditing ? "Cancel" : "Edit"}
                                </button>
                                <button
                                  onClick={() => handleDeleteStation(sid)}
                                  className="text-red-500 hover:text-red-600 dark:hover:text-red-400 font-semibold text-xs cursor-pointer transition hover:underline"
                                >
                                  Remove
                                </button>
                              </td>
                            </tr>
                            {isEditing && (
                              <tr className="bg-emerald-50/40 dark:bg-emerald-950/10 border-b border-emerald-100 dark:border-emerald-900/30">
                                <td colSpan={9} className="px-4 py-4">
                                  <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                                    <div>
                                      <label className="block text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1">Latitude</label>
                                      <input
                                        type="number" step="0.000001"
                                        value={editDraft.latitude}
                                        onChange={(e) => setEditDraft((d) => ({ ...d, latitude: e.target.value }))}
                                        className="w-full border border-slate-200 dark:border-slate-700 rounded-lg px-3 py-1.5 text-sm bg-white dark:bg-slate-900 text-slate-800 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-emerald-500/20 focus:border-emerald-500"
                                      />
                                    </div>
                                    <div>
                                      <label className="block text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1">Longitude</label>
                                      <input
                                        type="number" step="0.000001"
                                        value={editDraft.longitude}
                                        onChange={(e) => setEditDraft((d) => ({ ...d, longitude: e.target.value }))}
                                        className="w-full border border-slate-200 dark:border-slate-700 rounded-lg px-3 py-1.5 text-sm bg-white dark:bg-slate-900 text-slate-800 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-emerald-500/20 focus:border-emerald-500"
                                      />
                                    </div>
                                    <div>
                                      <label className="block text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1">Stratum</label>
                                      <select
                                        value={editDraft.stratum}
                                        onChange={(e) => setEditDraft((d) => ({ ...d, stratum: e.target.value }))}
                                        className="w-full border border-slate-200 dark:border-slate-700 rounded-lg px-3 py-1.5 text-sm bg-white dark:bg-slate-900 text-slate-800 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-emerald-500/20 focus:border-emerald-500 cursor-pointer"
                                      >
                                        <option value="">— Select —</option>
                                        {STRATA_OPTIONS.map((o) => <option key={o} value={o}>{o}</option>)}
                                      </select>
                                    </div>
                                    <div>
                                      <label className="block text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1">Camera Model</label>
                                      <input
                                        value={editDraft.camera_model}
                                        onChange={(e) => setEditDraft((d) => ({ ...d, camera_model: e.target.value }))}
                                        className="w-full border border-slate-200 dark:border-slate-700 rounded-lg px-3 py-1.5 text-sm bg-white dark:bg-slate-900 text-slate-800 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-emerald-500/20 focus:border-emerald-500"
                                      />
                                    </div>
                                    <div>
                                      <label className="block text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1">Team Member</label>
                                      <input
                                        value={editDraft.team_member}
                                        onChange={(e) => setEditDraft((d) => ({ ...d, team_member: e.target.value }))}
                                        className="w-full border border-slate-200 dark:border-slate-700 rounded-lg px-3 py-1.5 text-sm bg-white dark:bg-slate-900 text-slate-800 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-emerald-500/20 focus:border-emerald-500"
                                      />
                                    </div>
                                    <div>
                                      <label className="block text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1">Notes</label>
                                      <input
                                        value={editDraft.notes}
                                        onChange={(e) => setEditDraft((d) => ({ ...d, notes: e.target.value }))}
                                        className="w-full border border-slate-200 dark:border-slate-700 rounded-lg px-3 py-1.5 text-sm bg-white dark:bg-slate-900 text-slate-800 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-emerald-500/20 focus:border-emerald-500"
                                      />
                                    </div>
                                  </div>
                                  <div className="mt-3 flex gap-2">
                                    <button
                                      onClick={() => handleEditSave(sid)}
                                      className="px-4 py-1.5 bg-emerald-600 hover:bg-emerald-700 text-white text-xs font-semibold rounded-lg transition cursor-pointer"
                                    >
                                      Save Changes
                                    </button>
                                    <button
                                      onClick={() => setEditingId(null)}
                                      className="px-4 py-1.5 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-300 text-xs font-semibold rounded-lg transition cursor-pointer hover:bg-slate-50 dark:hover:bg-slate-750"
                                    >
                                      Cancel
                                    </button>
                                  </div>
                                </td>
                              </tr>
                            )}
                          </React.Fragment>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        </div>
      ) : tab === "deployments" ? (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 items-start">
          {/* Add Deployment Sidebar */}
          <div className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200/60 dark:border-slate-800/80 p-5 space-y-4 shadow-sm lg:col-span-1">
            <div>
              <h2 className="font-bold text-base text-slate-900 dark:text-white">
                Log Camera Deployment
              </h2>
              <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
                Record duration and specs for a camera trap deployment.
              </p>
            </div>

            <div className="space-y-3">
              <div>
                <label className="block text-xs font-semibold text-slate-500 dark:text-slate-400 mb-1">
                  Select Station *
                </label>
                <select
                  value={newDep.station_id}
                  onChange={(e) => setNewDep((d) => ({ ...d, station_id: e.target.value }))}
                  className="w-full border border-slate-200 dark:border-slate-800 rounded-xl px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500/20 focus:border-emerald-500 bg-slate-50/50 dark:bg-slate-955 text-slate-800 dark:text-slate-100 transition cursor-pointer"
                >
                  <option value="">Select Location</option>
                  {stations.map((s) => (
                    <option key={String(s.station_id)} value={String(s.station_id)}>
                      {String(s.station_id)}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-500 dark:text-slate-400 mb-1">
                  Start Date *
                </label>
                <input
                  type="date"
                  value={newDep.start_date}
                  onChange={(e) => setNewDep((d) => ({ ...d, start_date: e.target.value }))}
                  className="w-full border border-slate-200 dark:border-slate-800 rounded-xl px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500/20 focus:border-emerald-500 bg-slate-50/50 dark:bg-slate-950 text-slate-800 dark:text-slate-100 transition"
                />
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-500 dark:text-slate-400 mb-1">
                  End Date <span className="font-normal opacity-60">(leave blank for active deployment)</span>
                </label>
                <input
                  type="date"
                  value={newDep.end_date}
                  onChange={(e) => setNewDep((d) => ({ ...d, end_date: e.target.value }))}
                  className="w-full border border-slate-200 dark:border-slate-800 rounded-xl px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500/20 focus:border-emerald-500 bg-slate-50/50 dark:bg-slate-950 text-slate-800 dark:text-slate-100 transition"
                />
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-500 dark:text-slate-400 mb-1">
                  Camera Hardware ID
                </label>
                <input
                  placeholder="e.g. CAM-42"
                  value={newDep.camera_id}
                  onChange={(e) => setNewDep((d) => ({ ...d, camera_id: e.target.value }))}
                  className="w-full border border-slate-200 dark:border-slate-800 rounded-xl px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500/20 focus:border-emerald-500 bg-slate-50/50 dark:bg-slate-950 text-slate-800 dark:text-slate-100 transition"
                />
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-500 dark:text-slate-400 mb-1">
                  Camera Down Days
                </label>
                <input
                  type="number"
                  min="0"
                  placeholder="0"
                  value={newDep.camera_down_days}
                  onChange={(e) => setNewDep((d) => ({ ...d, camera_down_days: e.target.value }))}
                  className="w-full border border-slate-200 dark:border-slate-800 rounded-xl px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500/20 focus:border-emerald-500 bg-slate-50/50 dark:bg-slate-950 text-slate-800 dark:text-slate-100 transition"
                />
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-500 dark:text-slate-400 mb-1">
                  Remarks / Notes
                </label>
                <textarea
                  placeholder="SD Card ID, batteries status..."
                  rows={2}
                  value={newDep.notes}
                  onChange={(e) => setNewDep((d) => ({ ...d, notes: e.target.value }))}
                  className="w-full border border-slate-200 dark:border-slate-800 rounded-xl px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500/20 focus:border-emerald-500 bg-slate-50/50 dark:bg-slate-955 text-slate-800 dark:text-slate-100 transition"
                />
              </div>
            </div>

            {/* Error notifications */}
            {depErrors.length > 0 && (
              <div className="bg-red-50 dark:bg-red-950/20 border border-red-200 dark:border-red-800/40 rounded-xl p-3 space-y-1">
                {depErrors.map((err, i) => (
                  <p key={i} className="text-xs text-red-600 dark:text-red-400 flex items-center gap-1">
                    <span className="material-symbols-outlined text-sm leading-none select-none">error</span> {err}
                  </p>
                ))}
              </div>
            )}

            <button
              onClick={handleAddDeployment}
              disabled={adding || !newDep.station_id || !newDep.start_date || depErrors.length > 0}
              className="w-full py-2.5 bg-emerald-600 hover:bg-emerald-700 disabled:bg-slate-100 dark:disabled:bg-slate-800 text-white disabled:text-slate-400 dark:disabled:text-slate-600 text-sm font-semibold rounded-xl transition shadow-sm hover:shadow cursor-pointer"
            >
              {adding ? "Adding..." : "Add Deployment"}
            </button>
          </div>

          {/* Deployments Table Column */}
          <div className="lg:col-span-2 space-y-4">
            {deployments.length === 0 ? (
              <div className="text-center py-20 text-slate-400 dark:text-slate-550 bg-white dark:bg-slate-900 border border-slate-200/60 dark:border-slate-800/80 rounded-2xl shadow-sm">
                No deployments recorded yet. Log a deployment to track status.
              </div>
            ) : (
              <div className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200/60 dark:border-slate-800/80 overflow-hidden shadow-sm">
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead className="bg-slate-50/50 dark:bg-slate-955/20 border-b border-slate-100 dark:border-slate-800">
                      <tr>
                        {["Station ID", "Start Date", "End Date", "Camera Hardware", "Down Days", "Notes", ""].map((header) => (
                          <th
                            key={header}
                            className="text-left px-4 py-3.5 font-bold text-slate-400 dark:text-slate-550 uppercase tracking-wider text-[10px]"
                          >
                            {header}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100 dark:divide-slate-800/80">
                      {deployments.map((d, i) => (
                        <tr key={i} className="hover:bg-slate-50/50 dark:hover:bg-slate-950/20 transition-colors">
                          <td className="px-4 py-3.5 font-bold text-slate-800 dark:text-slate-200">
                            {String(d.station_id)}
                          </td>
                          <td className="px-4 py-3.5 text-xs text-slate-500 dark:text-slate-400 font-mono">
                            {String(d.start_date || "—")}
                          </td>
                          <td className="px-4 py-3.5 text-xs text-slate-500 dark:text-slate-400 font-mono">
                            {String(d.end_date || "Active")}
                          </td>
                          <td className="px-4 py-3.5 text-xs text-slate-700 dark:text-slate-300 font-mono">
                            {String(d.camera_id || "—")}
                          </td>
                          <td className="px-4 py-3.5 text-center text-xs text-slate-700 dark:text-slate-300">
                            {Number(d.camera_down_days) > 0 ? (
                              <span className="text-amber-600 dark:text-amber-400 font-semibold">{Number(d.camera_down_days)}</span>
                            ) : "—"}
                          </td>
                          <td className="px-4 py-3.5 text-slate-500 dark:text-slate-400 text-xs max-w-xs truncate">
                            {String(d.notes || "—")}
                          </td>
                          <td className="px-4 py-3.5 text-right">
                            <button
                              onClick={() => deleteDeployment(Number(d.id)).then(load)}
                              className="text-red-500 hover:text-red-600 dark:hover:text-red-400 font-semibold text-xs cursor-pointer transition hover:underline"
                            >
                              Remove
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        </div>
      ) : tab === "cameras" ? (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 items-start">
          {/* Register Camera Sidebar */}
          <div className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200/60 dark:border-slate-800/80 p-5 space-y-4 shadow-sm lg:col-span-1">
            <div>
              <h2 className="font-bold text-base text-slate-900 dark:text-white">
                Register Camera
              </h2>
              <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
                Add a physical camera unit to the equipment registry.
              </p>
            </div>

            <div className="space-y-3">
              <div>
                <label className="block text-xs font-semibold text-slate-500 dark:text-slate-400 mb-1">
                  Camera ID *
                </label>
                <input
                  placeholder="e.g. CAM-42"
                  value={newCamera.camera_id}
                  onChange={(e) => setNewCamera((c) => ({ ...c, camera_id: e.target.value }))}
                  className="w-full border border-slate-200 dark:border-slate-800 rounded-xl px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500/20 focus:border-emerald-500 bg-slate-50/50 dark:bg-slate-950 text-slate-800 dark:text-slate-100 transition"
                />
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-500 dark:text-slate-400 mb-1">
                  Model
                </label>
                <input
                  placeholder="e.g. Bushnell Core"
                  value={newCamera.model}
                  onChange={(e) => setNewCamera((c) => ({ ...c, model: e.target.value }))}
                  className="w-full border border-slate-200 dark:border-slate-800 rounded-xl px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500/20 focus:border-emerald-500 bg-slate-50/50 dark:bg-slate-950 text-slate-800 dark:text-slate-100 transition"
                />
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-500 dark:text-slate-400 mb-1">
                  Serial Number
                </label>
                <input
                  placeholder="e.g. SN-882910"
                  value={newCamera.serial_number}
                  onChange={(e) => setNewCamera((c) => ({ ...c, serial_number: e.target.value }))}
                  className="w-full border border-slate-200 dark:border-slate-800 rounded-xl px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500/20 focus:border-emerald-500 bg-slate-50/50 dark:bg-slate-950 text-slate-800 dark:text-slate-100 transition"
                />
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-500 dark:text-slate-400 mb-1">
                  Notes
                </label>
                <textarea
                  placeholder="Condition, accessories..."
                  rows={2}
                  value={newCamera.notes}
                  onChange={(e) => setNewCamera((c) => ({ ...c, notes: e.target.value }))}
                  className="w-full border border-slate-200 dark:border-slate-800 rounded-xl px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500/20 focus:border-emerald-500 bg-slate-50/50 dark:bg-slate-950 text-slate-800 dark:text-slate-100 transition"
                />
              </div>
            </div>

            {cameraErrors.length > 0 && (
              <div className="bg-red-50 dark:bg-red-950/20 border border-red-200 dark:border-red-800/40 rounded-xl p-3 space-y-1">
                {cameraErrors.map((err, i) => (
                  <p key={i} className="text-xs text-red-600 dark:text-red-400 flex items-center gap-1">
                    <span className="material-symbols-outlined text-sm leading-none select-none">error</span> {err}
                  </p>
                ))}
              </div>
            )}

            <button
              onClick={handleAddCamera}
              disabled={adding || !newCamera.camera_id || cameraErrors.length > 0}
              className="w-full py-2.5 bg-emerald-600 hover:bg-emerald-700 disabled:bg-slate-100 dark:disabled:bg-slate-800 text-white disabled:text-slate-400 dark:disabled:text-slate-600 text-sm font-semibold rounded-xl transition shadow-sm hover:shadow cursor-pointer"
            >
              {adding ? "Adding..." : "Register Camera"}
            </button>
          </div>

          {/* Cameras Table Column */}
          <div className="lg:col-span-2 space-y-4">
            {cameras.length === 0 ? (
              <div className="text-center py-20 text-slate-400 dark:text-slate-550 bg-white dark:bg-slate-900 border border-slate-200/60 dark:border-slate-800/80 rounded-2xl shadow-sm">
                No cameras registered yet. Register a camera to make it selectable at upload time.
              </div>
            ) : (
              <div className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200/60 dark:border-slate-800/80 overflow-hidden shadow-sm">
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead className="bg-slate-50/50 dark:bg-slate-955/20 border-b border-slate-100 dark:border-slate-800">
                      <tr>
                        {["Camera ID", "Model", "Serial", "Status", "Current Station", "Notes", ""].map((header) => (
                          <th
                            key={header}
                            className="text-left px-4 py-3.5 font-bold text-slate-400 dark:text-slate-500 uppercase tracking-wider text-[10px]"
                          >
                            {header}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100 dark:divide-slate-800/80">
                      {cameras.map((c, i) => {
                        const cid = String(c.camera_id);
                        const isEditing = editingCameraId === cid;
                        return (
                          <React.Fragment key={i}>
                            <tr className="hover:bg-slate-50/50 dark:hover:bg-slate-950/20 transition-colors">
                              <td className="px-4 py-3.5 font-bold text-slate-800 dark:text-slate-200 font-mono">
                                {cid}
                              </td>
                              <td className="px-4 py-3.5 text-xs text-slate-600 dark:text-slate-350">
                                {String(c.model || "—")}
                              </td>
                              <td className="px-4 py-3.5 text-xs text-slate-500 dark:text-slate-400 font-mono">
                                {String(c.serial_number || "—")}
                              </td>
                              <td className="px-4 py-3.5">
                                <span className={`px-2.5 py-1 text-[11px] font-semibold rounded-full border ${
                                  c.status === "active"
                                    ? "bg-emerald-50 text-emerald-700 dark:bg-emerald-950/40 dark:text-emerald-400 border-emerald-100 dark:border-emerald-900/50"
                                    : c.status === "retired"
                                    ? "bg-red-50 text-red-700 dark:bg-red-950/40 dark:text-red-400 border-red-100 dark:border-red-900/50"
                                    : "bg-slate-100 text-slate-500 dark:bg-slate-800 dark:text-slate-400 border-slate-200 dark:border-slate-700/60"
                                }`}>
                                  {String(c.status || "active")}
                                </span>
                              </td>
                              <td className="px-4 py-3.5 text-xs text-slate-700 dark:text-slate-300 font-mono">
                                {String(c.current_station_id || "Unassigned")}
                              </td>
                              <td className="px-4 py-3.5 text-slate-500 dark:text-slate-400 text-xs max-w-xs truncate">
                                {String(c.notes || "—")}
                              </td>
                              <td className="px-4 py-3.5 text-right whitespace-nowrap">
                                <button
                                  onClick={() => isEditing ? setEditingCameraId(null) : handleCameraEditStart(c)}
                                  className="text-emerald-600 hover:text-emerald-700 dark:text-emerald-400 dark:hover:text-emerald-300 font-semibold text-xs cursor-pointer transition hover:underline mr-3"
                                >
                                  {isEditing ? "Cancel" : "Edit"}
                                </button>
                                <button
                                  onClick={() => handleDeleteCamera(cid)}
                                  className="text-red-500 hover:text-red-600 dark:hover:text-red-400 font-semibold text-xs cursor-pointer transition hover:underline"
                                >
                                  Remove
                                </button>
                              </td>
                            </tr>
                            {isEditing && (
                              <tr className="bg-emerald-50/40 dark:bg-emerald-950/10 border-b border-emerald-100 dark:border-emerald-900/30">
                                <td colSpan={7} className="px-4 py-4">
                                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                                    <div>
                                      <label className="block text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1">Model</label>
                                      <input
                                        value={editCameraDraft.model}
                                        onChange={(e) => setEditCameraDraft((d) => ({ ...d, model: e.target.value }))}
                                        className="w-full border border-slate-200 dark:border-slate-700 rounded-lg px-3 py-1.5 text-sm bg-white dark:bg-slate-900 text-slate-800 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-emerald-500/20 focus:border-emerald-500"
                                      />
                                    </div>
                                    <div>
                                      <label className="block text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1">Serial Number</label>
                                      <input
                                        value={editCameraDraft.serial_number}
                                        onChange={(e) => setEditCameraDraft((d) => ({ ...d, serial_number: e.target.value }))}
                                        className="w-full border border-slate-200 dark:border-slate-700 rounded-lg px-3 py-1.5 text-sm bg-white dark:bg-slate-900 text-slate-800 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-emerald-500/20 focus:border-emerald-500"
                                      />
                                    </div>
                                    <div>
                                      <label className="block text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1">Status</label>
                                      <select
                                        value={editCameraDraft.status}
                                        onChange={(e) => setEditCameraDraft((d) => ({ ...d, status: e.target.value }))}
                                        className="w-full border border-slate-200 dark:border-slate-700 rounded-lg px-3 py-1.5 text-sm bg-white dark:bg-slate-900 text-slate-800 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-emerald-500/20 focus:border-emerald-500 cursor-pointer"
                                      >
                                        <option value="active">active</option>
                                        <option value="spare">spare</option>
                                        <option value="retired">retired</option>
                                      </select>
                                    </div>
                                    <div>
                                      <label className="block text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1">Notes</label>
                                      <input
                                        value={editCameraDraft.notes}
                                        onChange={(e) => setEditCameraDraft((d) => ({ ...d, notes: e.target.value }))}
                                        className="w-full border border-slate-200 dark:border-slate-700 rounded-lg px-3 py-1.5 text-sm bg-white dark:bg-slate-900 text-slate-800 dark:text-slate-100 focus:outline-none focus:ring-2 focus:ring-emerald-500/20 focus:border-emerald-500"
                                      />
                                    </div>
                                  </div>
                                  <div className="mt-3 flex gap-2">
                                    <button
                                      onClick={() => handleCameraEditSave(cid)}
                                      className="px-4 py-1.5 bg-emerald-600 hover:bg-emerald-700 text-white text-xs font-semibold rounded-lg transition cursor-pointer"
                                    >
                                      Save Changes
                                    </button>
                                    <button
                                      onClick={() => setEditingCameraId(null)}
                                      className="px-4 py-1.5 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-300 text-xs font-semibold rounded-lg transition cursor-pointer hover:bg-slate-50 dark:hover:bg-slate-750"
                                    >
                                      Cancel
                                    </button>
                                  </div>
                                </td>
                              </tr>
                            )}
                          </React.Fragment>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        </div>
      ) : (
        <div className="bg-white dark:bg-slate-900 border border-slate-200/60 dark:border-slate-800/80 rounded-2xl overflow-hidden shadow-sm">
          {mapLoading ? (
            <div className="text-center py-24 text-slate-400 dark:text-slate-500 animate-pulse">
              Loading station coordinates…
            </div>
          ) : !mapFeatures || mapFeatures.length === 0 ? (
            <div className="text-center py-24 text-slate-400 dark:text-slate-550 space-y-2">
              <span className="material-symbols-outlined text-2xl select-none">pin_drop</span>
              <p className="font-semibold text-sm text-slate-700 dark:text-slate-300">No Georeferenced Stations</p>
              <p className="text-xs max-w-sm mx-auto text-slate-400 dark:text-slate-500 leading-relaxed">
                Add latitude/longitude to a station to see it here.
              </p>
            </div>
          ) : (
            <div className="h-[520px]">
              <MapContainer center={[0, 20]} zoom={3} style={{ height: "100%", width: "100%" }}>
                <TileLayer
                  url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                  attribution="© OpenStreetMap contributors"
                />
                <FitStationBounds features={mapFeatures} />
                {mapFeatures.map((f, i) => {
                  const [lon, lat] = f.geometry.coordinates;
                  const p = f.properties;
                  return (
                    <Marker key={i} position={[lat, lon]}>
                      <Popup>
                        <div className="text-xs space-y-1 font-sans p-1">
                          <p className="font-bold text-slate-900">{String(p.station_id ?? "—")}</p>
                          <p className="text-slate-500">Stratum: {String(p.stratum || "—")}</p>
                          <p className="text-slate-500">Camera: {String(p.current_camera_id || "Unassigned")} <span className="text-slate-400">({String(p.camera_model || "unknown model")})</span></p>
                          <p className="text-slate-500">Trap nights: {String(p.trap_nights ?? 0)}</p>
                          <p className="text-slate-500">Status: {String(p.status || "—")}</p>
                        </div>
                      </Popup>
                    </Marker>
                  );
                })}
              </MapContainer>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
