import { useCallback, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { uploadImages, startProcessing, pollJob, getJobResults } from "../api/client";

type JobState = {
  jobId: string;
  status: string;
  total: number;
  completed: number;
  error?: string;
};

export default function Upload() {
  const [files, setFiles] = useState<File[]>([]);
  const [dragging, setDragging] = useState(false);
  const [job, setJob] = useState<JobState | null>(null);
  const [log, setLog] = useState<string[]>([]);
  const inputRef = useRef<HTMLInputElement>(null);
  const navigate = useNavigate();

  const addLog = (msg: string) => setLog((prev) => [...prev, msg]);

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const dropped = Array.from(e.dataTransfer.files).filter((f) =>
      /\.(jpe?g|png)$/i.test(f.name)
    );
    setFiles((prev) => [...prev, ...dropped]);
  }, []);

  const onFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const selected = Array.from(e.target.files).filter((f) =>
        /\.(jpe?g|png)$/i.test(f.name)
      );
      setFiles((prev) => [...prev, ...selected]);
    }
  };

  const removeFile = (i: number) =>
    setFiles((prev) => prev.filter((_, idx) => idx !== i));

  const handleProcess = async () => {
    if (!files.length) return;
    setLog([]);
    try {
      addLog(`Uploading ${files.length} file(s)…`);
      const { job_id } = await uploadImages(files);
      addLog("Files saved. Starting AI analysis pipeline…");

      await startProcessing(job_id);
      setJob({ jobId: job_id, status: "running", total: files.length, completed: 0 });

      const interval = setInterval(async () => {
        const status = await pollJob(job_id);
        setJob(status);
        if (status.status === "done") {
          clearInterval(interval);
          addLog("Processing complete! Storing results…");
          await getJobResults(job_id);
          addLog("Done. Redirecting to Results view…");
          setTimeout(() => navigate("/results"), 1200);
        } else if (status.status === "error") {
          clearInterval(interval);
          addLog(`Error: ${status.error}`);
        }
      }, 1500);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      addLog(`Pipeline failed: ${msg}`);
    }
  };

  const pct = job && job.total > 0 ? Math.round((job.completed / job.total) * 100) : 0;
  const processing = job?.status === "running";

  return (
    <div className="max-w-4xl mx-auto space-y-8">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-extrabold tracking-tight text-slate-900">
            Upload & Process
          </h1>
          <p className="text-slate-500 mt-1.5 text-sm">
            Import images from camera trap deployments to run detection, OCR, and classification.
          </p>
        </div>
        <div className="flex items-center gap-1.5 px-3 py-1 bg-emerald-50 rounded-full border border-emerald-200">
          <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
          <span className="text-xs font-semibold text-emerald-700">Pipeline Active</span>
        </div>
      </div>

      {/* Main Grid: Upload Area & Log Dashboard */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-2 space-y-6">
          {/* Drop zone */}
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
            onClick={() => inputRef.current?.click()}
            className={`border-2 border-dashed rounded-2xl p-10 text-center cursor-pointer transition-all duration-300 relative overflow-hidden group ${
              dragging
                ? "border-emerald-500 bg-emerald-50/60 shadow-inner scale-[0.99]"
                : "border-slate-300 bg-white hover:border-emerald-400 hover:bg-emerald-50/20 hover:shadow-md"
            }`}
          >
            <div className="flex flex-col items-center">
              <span
                className={`material-symbols-outlined text-5xl mb-3 select-none transition-transform duration-300 ${
                  dragging ? "scale-110 text-emerald-600 animate-bounce" : "text-slate-400 group-hover:text-emerald-500"
                }`}
              >
                cloud_upload
              </span>
              <p className="text-slate-700 font-semibold text-base">
                Drag & drop camera images here
              </p>
              <p className="text-slate-400 text-xs mt-1.5">
                Supports JPG, JPEG, and PNG formats. Multi-file selection enabled.
              </p>
              <button
                type="button"
                className="mt-5 px-4 py-2 bg-slate-900 text-white rounded-lg text-xs font-semibold hover:bg-slate-800 transition-colors shadow-sm"
              >
                Browse Files
              </button>
            </div>
            <input
              ref={inputRef}
              type="file"
              accept=".jpg,.jpeg,.png"
              multiple
              className="hidden"
              onChange={onFileInput}
            />
          </div>

          {/* Selected File Grid */}
          {files.length > 0 && (
            <div className="bg-white rounded-2xl border border-slate-200 shadow-sm overflow-hidden">
              <div className="px-5 py-3.5 bg-slate-50 border-b border-slate-100 flex justify-between items-center">
                <span className="font-semibold text-slate-700 text-sm flex items-center gap-1.5">
                  <span className="material-symbols-outlined text-lg text-slate-500 select-none">
                    photo_library
                  </span>
                  {files.length} image(s) selected
                </span>
                <button
                  onClick={() => setFiles([])}
                  disabled={processing}
                  className="text-xs font-semibold text-slate-400 hover:text-red-500 transition-colors disabled:opacity-50"
                >
                  Clear All
                </button>
              </div>
              <div className="p-4 max-h-72 overflow-y-auto custom-scrollbar">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  {files.map((f, i) => (
                    <div
                      key={i}
                      className="flex items-center justify-between p-2.5 rounded-xl border border-slate-100 bg-slate-50/50 hover:bg-slate-50 transition-colors group"
                    >
                      <div className="flex items-center gap-2.5 overflow-hidden">
                        <span className="material-symbols-outlined text-slate-400 select-none shrink-0 text-xl">
                          image
                        </span>
                        <div className="overflow-hidden">
                          <p className="text-xs font-semibold text-slate-700 truncate max-w-[160px]">
                            {f.name}
                          </p>
                          <p className="text-[10px] text-slate-400">
                            {(f.size / 1024).toFixed(0)} KB
                          </p>
                        </div>
                      </div>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          removeFile(i);
                        }}
                        disabled={processing}
                        className="text-slate-300 hover:text-red-500 p-1 rounded-full hover:bg-red-50 transition-all shrink-0 disabled:opacity-50"
                      >
                        <span className="material-symbols-outlined text-base select-none leading-none block">
                          close
                        </span>
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* Action button */}
          <button
            onClick={handleProcess}
            disabled={!files.length || processing}
            className="w-full py-3.5 px-6 bg-emerald-600 hover:bg-emerald-700 disabled:bg-slate-200 disabled:text-slate-400 text-white font-semibold rounded-2xl transition-all duration-150 shadow-md hover:shadow-lg flex items-center justify-center gap-2 cursor-pointer disabled:cursor-not-allowed"
          >
            {processing ? (
              <>
                <span className="material-symbols-outlined text-lg select-none animate-spin">
                  sync
                </span>
                Processing Pipeline...
              </>
            ) : (
              <>
                <span className="material-symbols-outlined text-lg select-none">
                  play_circle
                </span>
                Start AI Analysis
              </>
            )}
          </button>
        </div>

        {/* Console / Log panel */}
        <div className="space-y-6">
          {/* Progress widget */}
          {job && (
            <div className="bg-white rounded-2xl border border-slate-200 p-5 space-y-3.5 shadow-sm">
              <div className="flex justify-between items-center text-xs">
                <span className="font-bold text-slate-700">
                  {job.status === "done"
                    ? "Analysis Complete"
                    : job.status === "error"
                    ? "Process Failed"
                    : "Running Inference..."}
                </span>
                <span className="font-mono font-bold text-emerald-600 text-sm">{pct}%</span>
              </div>
              <div className="w-full bg-slate-100 rounded-full h-2 overflow-hidden">
                <div
                  className={`h-2 rounded-full transition-all duration-300 ${
                    job.status === "error" ? "bg-red-500" : "bg-emerald-500"
                  }`}
                  style={{ width: `${job.status === "done" ? 100 : pct}%` }}
                />
              </div>
              <div className="flex justify-between items-center text-[10px] text-slate-400 font-medium">
                <span>Total: {job.total} images</span>
                <span>Processed: {job.completed}</span>
              </div>
            </div>
          )}

          {/* Log panel */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl overflow-hidden flex flex-col h-80 shadow-md">
            <div className="px-4 py-2.5 bg-slate-950 border-b border-slate-800/80 flex items-center justify-between">
              <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                Pipeline Logs
              </span>
              <button
                onClick={() => setLog([])}
                className="text-[10px] text-slate-500 hover:text-slate-300 font-semibold"
              >
                Clear
              </button>
            </div>
            <div className="p-4 flex-1 overflow-y-auto custom-scrollbar font-mono text-[11px] text-emerald-400 space-y-1.5 bg-slate-900/90 leading-relaxed">
              {log.length === 0 ? (
                <div className="text-slate-500 italic">Logs will appear here once processing begins...</div>
              ) : (
                log.map((l, i) => (
                  <div key={i} className="flex gap-2">
                    <span className="text-emerald-700 select-none">&gt;</span>
                    <span>{l}</span>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
