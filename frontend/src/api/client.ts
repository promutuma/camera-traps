import axios, { AxiosError } from "axios";

const api = axios.create({ baseURL: "/api" });

// Global error handler — surfaces network/server errors as a console warning
// and re-throws so individual callers can still handle them if needed.
api.interceptors.response.use(
  (res) => res,
  (err: AxiosError) => {
    const status = err.response?.status;
    if (status === 503) {
      console.warn("[API] Server unavailable — AI models may still be loading.");
    } else if (status === 413) {
      console.warn("[API] Upload rejected — file too large.");
    } else if (!err.response) {
      console.warn("[API] Network error — is the backend running?", err.message);
    }
    return Promise.reject(err);
  }
);

export default api;

// ── Config ──────────────────────────────────────────────────────────────────
export const getConfig = () => api.get("/config").then((r) => r.data);
export const updateConfig = (patch: Record<string, unknown>) =>
  api.patch("/config", patch).then((r) => r.data);
export const getModelStatus = () =>
  api.get("/config/status").then((r) => r.data);

// ── Images / Jobs ────────────────────────────────────────────────────────────
export const uploadImages = (files: File[]) => {
  const form = new FormData();
  files.forEach((f) => form.append("files", f));
  return api.post("/images/upload", form).then((r) => r.data);
};
export const startProcessing = (jobId: string, stationId?: string, cameraId?: string) => {
  const params: Record<string, string> = {};
  if (stationId) params.station_id = stationId;
  if (cameraId) params.camera_id = cameraId;
  return api.post(`/images/process/${jobId}`, null, Object.keys(params).length ? { params } : {}).then((r) => r.data);
};
export const pollJob = (jobId: string) =>
  api.get(`/images/job/${jobId}`).then((r) => r.data);
export const getJobResults = (jobId: string) =>
  api.get(`/images/results/${jobId}`).then((r) => r.data);
export const imageFileUrl = (jobId: string, filename: string) =>
  `/api/images/file/${jobId}/${encodeURIComponent(filename)}`;
/** Full original image — use for lightbox / review panels where fine detail matters. */
export const storedImageUrl = (filename: string) =>
  `/api/images/stored/${encodeURIComponent(filename)}`;
/** Cached thumbnail — w is the max dimension in pixels (64–2560). */
export const storedThumbUrl = (filename: string, w = 800) =>
  `/api/images/thumb/${encodeURIComponent(filename)}?w=${w}`;

// ── Results ──────────────────────────────────────────────────────────────────
export const getResults = (params?: Record<string, string | number>) =>
  api.get("/results", { params }).then((r) => r.data as { total: number; limit: number; offset: number; items: Record<string, unknown>[] });
export const updateResult = (imageId: number, patch: Record<string, unknown>) =>
  api.patch(`/results/${imageId}`, patch).then((r) => r.data);
export const deleteResult = (detectionId: number) =>
  api.delete(`/results/${detectionId}`).then((r) => r.data);
export const deleteResults = (detectionIds: number[]) =>
  api.delete("/results", { params: { detection_ids: detectionIds.join(",") } }).then((r) => r.data);
export const exportExcel = () => `/api/results/export/excel`;
export const exportCsv = () => `/api/results/export/csv`;

// ── Statistics ───────────────────────────────────────────────────────────────
export const getStats = () => api.get("/stats/summary").then((r) => r.data);

// ── History ──────────────────────────────────────────────────────────────────
export const getHistory = () => api.get("/history").then((r) => r.data);
export const clearHistory = () => api.delete("/history").then((r) => r.data);

// ── Diagnostics ──────────────────────────────────────────────────────────────
export const inspectImage = (file: File) => {
  const form = new FormData();
  form.append("file", file);
  return api.post("/diagnostics/inspect", form).then((r) => r.data);
};

// ── Ecological ───────────────────────────────────────────────────────────────
export const getIDE = () => api.get("/ecological/ide").then((r) => r.data);
export const getRAI = (trap_nights: number) =>
  api.get("/ecological/rai", { params: { trap_nights } }).then((r) => r.data);
export const getTimeline = () =>
  api.get("/ecological/timeline").then((r) => r.data);
export const getRichness = () =>
  api.get("/ecological/richness").then((r) => r.data);
export const getAccumulation = () =>
  api.get("/ecological/accumulation").then((r) => r.data);
export const getGroupSize = () =>
  api.get("/ecological/group-size").then((r) => r.data);
export const getVisitation = (trap_nights: number) =>
  api.get("/ecological/visitation", { params: { trap_nights } }).then((r) => r.data);

// ── QC ───────────────────────────────────────────────────────────────────────
export const getQCFlags = () => api.get("/qc/flags").then((r) => r.data);
export const getQCSummary = () => api.get("/qc/summary").then((r) => r.data);

// ── Stations ─────────────────────────────────────────────────────────────────
export const getStations = () => api.get("/stations").then((r) => r.data);
export const addStation = (body: Record<string, unknown>) =>
  api.post("/stations", body).then((r) => r.data);
export const updateStation = (id: string, body: Record<string, unknown>) =>
  api.patch(`/stations/${id}`, body).then((r) => r.data);
export const deleteStation = (id: string, force = false) =>
  api.delete(`/stations/${id}${force ? "?force=true" : ""}`).then((r) => r.data);
export const getDeployments = () =>
  api.get("/stations/deployments").then((r) => r.data);
export const addDeployment = (body: Record<string, unknown>) =>
  api.post("/stations/deployments", body).then((r) => r.data);
export const deleteDeployment = (id: number) =>
  api.delete(`/stations/deployments/${id}`).then((r) => r.data);
export const getStationSummary = () =>
  api.get("/stations/summary").then((r) => r.data);
export const getStationMap = () =>
  api.get("/stations/map").then((r) => r.data);
export const assignCamera = (stationId: string, cameraId: string) =>
  api.post(`/stations/${encodeURIComponent(stationId)}/camera`, undefined, { params: { camera_id: cameraId } }).then((r) => r.data);

// ── Camera Registry ──────────────────────────────────────────────────────────
export const getCameras = () => api.get("/cameras").then((r) => r.data);
export const addCamera = (body: { camera_id: string; model?: string; serial_number?: string; status?: string; notes?: string }) =>
  api.post("/cameras", body).then((r) => r.data);
export const updateCamera = (cameraId: string, body: Record<string, unknown>) =>
  api.patch(`/cameras/${encodeURIComponent(cameraId)}`, body).then((r) => r.data);
export const deleteCamera = (cameraId: string) =>
  api.delete(`/cameras/${encodeURIComponent(cameraId)}`).then((r) => r.data);
export const getOrphanStations = () =>
  api.get("/stations/orphans").then((r) => r.data);
export const reassignStation = (fromStation: string, toStation: string) =>
  api.post(`/stations/reassign?from_station=${encodeURIComponent(fromStation)}&to_station=${encodeURIComponent(toStation)}`).then((r) => r.data);

// ── Review Queue ─────────────────────────────────────────────────────────────
export const getReviewQueue = () =>
  api.get("/review/queue").then((r) => r.data);
export const confirmDetection = (id: number, body: Record<string, unknown>) =>
  api.post(`/review/confirm/${id}`, body).then((r) => r.data);
export const correctDetection = (id: number, body: Record<string, unknown>) =>
  api.post(`/review/correct/${id}`, body).then((r) => r.data);
export const flagDetection = (id: number, body: Record<string, unknown>) =>
  api.post(`/review/flag/${id}`, body).then((r) => r.data);
export const flagByFilenames = (
  filenames: string[],
  reviewer_id: string,
  notes = "Flagged during upload review",
) =>
  api
    .post("/review/flag-by-filenames", { filenames, reviewer_id, notes })
    .then((r) => r.data);
export const getReviewLog = () =>
  api.get("/review/log").then((r) => r.data);
export const getPrivacyAudit = () =>
  api.get("/review/privacy-audit").then((r) => r.data);

// ── HITL Retraining ──────────────────────────────────────────────────────────
export const getRetrainPreview = () =>
  api.get("/retrain/preview").then((r) => r.data as { available_corrections: number; distinct_species: number });
export const runRetrain = (reviewer_id: string) =>
  api.post("/retrain/run", { reviewer_id }).then((r) => r.data);
export const getRetrainStatus = () =>
  api.get("/retrain/status").then((r) => r.data);
export const getRetrainHistory = () =>
  api.get("/retrain/history").then((r) => r.data);
export const activateRetrainRun = (jobId: string) =>
  api.post(`/retrain/activate/${jobId}`).then((r) => r.data);
export const deactivateRetrain = () =>
  api.post("/retrain/deactivate").then((r) => r.data);

// ── Community ────────────────────────────────────────────────────────────────
export const getObservations = () =>
  api.get("/community/observations").then((r) => r.data);
export const addObservation = (body: Record<string, unknown>) =>
  api.post("/community/observations", body).then((r) => r.data);
export const deleteObservation = (id: number) =>
  api.delete(`/community/observations/${id}`).then((r) => r.data);
export const getCrosscheck = () =>
  api.get("/community/crosscheck").then((r) => r.data);

// ── Spatial ──────────────────────────────────────────────────────────────────
export const getSpatialGeoJSON = () =>
  api.get("/spatial/geojson").then((r) => r.data);

// ── Species ──────────────────────────────────────────────────────────────────
export const getSpecies = () => api.get("/species").then((r) => r.data);
export const lookupSpecies = (name: string) =>
  api.get(`/species/lookup/${encodeURIComponent(name)}`).then((r) => r.data);
export const resolveSynonym = (name: string) =>
  api.get(`/species/synonyms/${encodeURIComponent(name)}`).then((r) => r.data);
export const addSpecies = (name: string, commonName = "") =>
  api.post("/species/add", undefined, { params: { name, common_name: commonName } }).then((r) => r.data);

// ── Corridor ─────────────────────────────────────────────────────────────────
export const getCorridorPairs = (max_km = 50) =>
  api.get("/corridor/pairs", { params: { max_km } }).then((r) => r.data);
export const getMovements = (max_km = 50) =>
  api.get("/corridor/movements", { params: { max_km } }).then((r) => r.data);
export const getBottlenecks = (max_km = 50) =>
  api.get("/corridor/bottlenecks", { params: { max_km } }).then((r) => r.data);
export const getUtilisation = (max_km = 50) =>
  api.get("/corridor/utilisation", { params: { max_km } }).then((r) => r.data);

// ── Project Config ───────────────────────────────────────────────────────────
export const getProject = () => api.get("/project").then((r) => r.data);
export const updateProject = (body: Record<string, unknown>) =>
  api.patch("/project", body).then((r) => r.data);
export const listProjects = () => api.get("/project/list").then((r) => r.data as Record<string, unknown>[]);
export const setActiveProject = (projectId: number) =>
  api.post(`/project/active/${projectId}`).then((r) => r.data);
export const createProject = (params: { name: string; survey_area?: string; notes?: string }) =>
  api.post("/project/create", undefined, { params }).then((r) => r.data);
export const deleteProject = (projectId: number) =>
  api.delete(`/project/${projectId}`).then((r) => r.data);

// ── ArcGIS ───────────────────────────────────────────────────────────────────
export const pushArcGIS = (body: Record<string, unknown>) =>
  api.post("/arcgis/push", body).then((r) => r.data);
export const getArcGISStatus = () =>
  api.get("/arcgis/status").then((r) => r.data);

// ── Storage Management ────────────────────────────────────────────────────────
export const getStorageStatus = () =>
  api.get("/storage/status").then((r) => r.data);
export const getStorageWarnings = () =>
  api.get("/storage/warnings").then((r) => r.data);
export const getDeletionPreview = (tier: string, daysOld: number = 7) =>
  api.get("/storage/deletion-preview", { params: { tier, days_old: daysOld } }).then((r) => r.data);
export const createBatchDownload = (tier: string, includeMetadata: boolean = true) =>
  api.post("/storage/downloads/batch", undefined, { params: { tier, include_metadata: includeMetadata } }).then((r) => r.data);
export const cleanupImages = (action: string, daysOld: number = 7, dryRun: boolean = true) =>
  api.post("/storage/cleanup", undefined, { params: { action, days_old: daysOld, dry_run: dryRun } }).then((r) => r.data);
export const markForDeletion = (imageId: number) =>
  api.post(`/storage/mark-for-deletion/${imageId}`).then((r) => r.data);

// ── Hash Management (Performance Optimization) ──────────────────────────────
export const getHashStats = () =>
  api.get("/storage/hash-stats").then((r) => r.data);
export const clearHashes = (strategy: string) =>
  api.post("/storage/clear-hashes", undefined, { params: { strategy } }).then((r) => r.data);

// ── Database Reset ──────────────────────────────────────────────────────────
export const resetDatabase = () =>
  api.post("/storage/reset-db", null, { params: { confirm: true } }).then((r) => r.data);

