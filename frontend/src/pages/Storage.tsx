import { useEffect, useState } from "react";
import { getStorageStatus, getStorageWarnings, createBatchDownload, getHashStats, clearHashes, cleanupImages, getDeletionPreview } from "../api/client";

type StorageStats = {
  total_mb: number;
  breakdown: Record<string, { count: number; size_mb: number; oldest_upload: string }>;
  timestamp: string;
};

type DeletionWarning = {
  pending_deletion_count: number;
  images: Array<{ id: number; filename: string; marked_for_deletion_at: string }>;
};

type DeletionPreview = {
  tier: string;
  will_delete: number;
  total_size_mb: number;
  earliest_uploaded: string | null;
  latest_uploaded: string | null;
  warning: string;
};

type HashStats = {
  hashes: {
    total_images: number;
    with_hashes: number;
    without_hashes: number;
    unique_hashes: number;
  };
  duplicates: Array<{ hash: string; count: number; image_ids: number[] }>;
  potential_savings_mb: number;
};

// SVG Icons
const DatabaseIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.8} stroke="currentColor" className="w-5 h-5">
    <path strokeLinecap="round" strokeLinejoin="round" d="M20.25 6.375c0 2.278-3.694 4.125-8.25 4.125S3.75 8.653 3.75 6.375m16.5 0c0-2.278-3.694-4.125-8.25-4.125S3.75 4.097 3.75 6.375m16.5 0v11.25c0 2.278-3.694 4.125-8.25 4.125s-8.25-1.847-8.25-4.125V6.375m16.5 0v3.75m-16.5-3.75v3.75m16.5 0v3.75C20.25 16.153 16.556 18 12 18s-8.25-1.847-8.25-4.125v-3.75m16.5 0v3.75m-16.5-3.75v3.75" />
  </svg>
);

const TrashIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.8} stroke="currentColor" className="w-4 h-4">
    <path strokeLinecap="round" strokeLinejoin="round" d="M14.74 9l-.346 9m-4.788 0L9.26 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 01-2.244 2.077H8.084a2.25 2.25 0 01-2.244-2.077L4.772 5.79m14.456 0a48.108 48.108 0 00-3.478-.397m-12 .562c.34-.059.68-.114 1.022-.165m0 0a48.11 48.11 0 013.478-.397m7.5 0v-.916c0-1.18-.91-2.164-2.09-2.201a51.964 51.964 0 00-3.32 0c-1.18.037-2.09 1.022-2.09 2.201v.916m7.5 0a48.667 48.667 0 00-7.5 0" />
  </svg>
);

const DownloadIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.8} stroke="currentColor" className="w-4 h-4">
    <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5M16.5 12L12 16.5m0 0L7.5 12m4.5 4.5V3" />
  </svg>
);

const SparklesIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.8} stroke="currentColor" className="w-5 h-5">
    <path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904L9 21L7.188 15.904L2 15L7.188 14.096L9 9L9.813 14.096L15 15L9.813 15.904Z" />
    <path strokeLinecap="round" strokeLinejoin="round" d="M19.071 4.929a10 10 0 00-14.142 0M12 3v3m0 12v3M3 12h3m12 0h3" />
  </svg>
);

const ShieldIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.8} stroke="currentColor" className="w-5 h-5">
    <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75m-3-7.036A11.959 11.959 0 013.598 6 11.99 11.99 0 003 9.749c0 5.592 3.824 10.29 9 11.623 5.176-1.332 9-6.03 9-11.622 0-1.31-.21-2.571-.598-3.751h-.152c-3.196 0-6.1-1.248-8.25-3.285Z" />
  </svg>
);

const InfoIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.8} stroke="currentColor" className="w-5 h-5">
    <path strokeLinecap="round" strokeLinejoin="round" d="M11.25 11.25l.041-.02a.75.75 0 111.085 1.085l-.041.02a.75.75 0 01-1.085-1.085zM12 18.75A6.75 6.75 0 1012 5.25a6.75 6.75 0 000 13.5z" />
  </svg>
);

const RefreshIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.8} stroke="currentColor" className="w-4 h-4 animate-spin">
    <path strokeLinecap="round" strokeLinejoin="round" d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0l3.181 3.183a8.25 8.25 0 0013.803-3.7M4.031 9.865a8.25 8.25 0 0113.803-3.7l3.181 3.182m0-4.991v4.99" />
  </svg>
);

export default function Storage() {
  const [stats, setStats] = useState<StorageStats | null>(null);
  const [warnings, setWarnings] = useState<DeletionWarning | null>(null);
  const [hashStats, setHashStats] = useState<HashStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [downloading, setDownloading] = useState(false);
  const [clearingHashes, setClearingHashes] = useState(false);

  // In-page confirmations to replace ugly browser alert/confirm sequences
  const [confirmStrategy, setConfirmStrategy] = useState<string | null>(null);
  const [notice, setNotice] = useState<{ type: "success" | "error"; text: string } | null>(null);

  // Manual empty-image cleanup
  const [cleanupDays, setCleanupDays] = useState(30);
  const [cleanupPreview, setCleanupPreview] = useState<{ deleted_count: number; freed_mb: number } | null>(null);
  const [cleanupStep, setCleanupStep] = useState<"idle" | "previewing" | "confirming" | "running">("idle");

  // Purge of images already marked-for-deletion past their grace period
  const [purgePreview, setPurgePreview] = useState<DeletionPreview | null>(null);
  const [purgeStep, setPurgeStep] = useState<"idle" | "previewing" | "confirming" | "running">("idle");

  const handleCleanupPreview = async () => {
    setCleanupStep("previewing");
    setCleanupPreview(null);
    try {
      const res = await cleanupImages("delete_empty", cleanupDays, true);
      setCleanupPreview(res);
      setCleanupStep("confirming");
    } catch {
      setNotice({ type: "error", text: "Failed to preview cleanup." });
      setCleanupStep("idle");
    }
  };

  const handleCleanupRun = async () => {
    setCleanupStep("running");
    try {
      const res = await cleanupImages("delete_empty", cleanupDays, false);
      setNotice({
        type: "success",
        text: `Deleted ${res.deleted_count} empty image${res.deleted_count !== 1 ? "s" : ""}, freed ${res.freed_mb?.toFixed(1) ?? "0"} MB.`,
      });
      setCleanupPreview(null);
      setCleanupStep("idle");
      fetchData();
    } catch {
      setNotice({ type: "error", text: "Cleanup failed." });
      setCleanupStep("idle");
    }
  };

  // Grace-period purge: images an operator already flagged via "mark for
  // deletion" (see the Results page), whose 7-day grace window has elapsed.
  // The tier value only matters for the "empty" branch on the backend; any
  // other value previews the same marked-for-deletion query cleanup runs.
  const handlePurgePreview = async () => {
    setPurgeStep("previewing");
    setPurgePreview(null);
    try {
      const res = await getDeletionPreview("valid", 7);
      setPurgePreview(res);
      setPurgeStep("confirming");
    } catch {
      setNotice({ type: "error", text: "Failed to preview pending deletions." });
      setPurgeStep("idle");
    }
  };

  const handlePurgeRun = async () => {
    setPurgeStep("running");
    try {
      const res = await cleanupImages("delete_marked", 7, false);
      setNotice({
        type: "success",
        text: `Purged ${res.deleted_count ?? 0} image${(res.deleted_count ?? 0) !== 1 ? "s" : ""} past their deletion grace period, freed ${res.freed_mb?.toFixed(1) ?? "0"} MB.`,
      });
      setPurgePreview(null);
      setPurgeStep("idle");
      fetchData();
    } catch {
      setNotice({ type: "error", text: "Purge failed." });
      setPurgeStep("idle");
    }
  };

  const fetchData = async () => {
    setError(null);
    try {
      const [statsData, warningsData, hashData] = await Promise.all([
        getStorageStatus(),
        getStorageWarnings(),
        getHashStats().catch(() => null),
      ]);
      setStats(statsData);
      setWarnings(warningsData);
      if (hashData) setHashStats(hashData);
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : "Failed to fetch storage data";
      console.error("Storage fetch error:", err);
      setError(errorMsg);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    setLoading(true);
    fetchData();
    const interval = setInterval(fetchData, 60000); // Refresh every minute
    return () => clearInterval(interval);
  }, []);

  const handleDownloadTier = async (tier: string) => {
    setDownloading(true);
    setNotice(null);
    try {
      const result = await createBatchDownload(tier);
      if (!result?.download_id) {
        throw new Error("No download ID returned");
      }
      await new Promise((resolve) => setTimeout(resolve, 500));
      window.location.href = `/api/storage/downloads/${result.download_id}/stream`;
      setNotice({ type: "success", text: `Triggered download for ${tier.replace("_", " ")} tier.` });
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : "Failed to create download";
      setNotice({ type: "error", text: `Download failed: ${errorMsg}` });
    } finally {
      setDownloading(false);
    }
  };

  const handleClearHashes = async (strategy: string) => {
    setClearingHashes(true);
    setNotice(null);
    try {
      const result = await clearHashes(strategy);
      const msg = `Cleared ${result.cleared} hashes. ${result.purpose} ${result.warning ? `WARNING: ${result.warning}` : ""}`;
      setNotice({ type: "success", text: msg });
      
      // Refresh stats
      const newHashStats = await getHashStats();
      if (newHashStats) setHashStats(newHashStats);
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : "Unknown error";
      setNotice({ type: "error", text: `Failed to clear hashes: ${errorMsg}` });
    } finally {
      setClearingHashes(false);
      setConfirmStrategy(null);
    }
  };

  if (error) {
    return (
      <div className="max-w-5xl mx-auto p-6">
        <h1 className="text-2xl font-bold tracking-tight text-slate-900 dark:text-white">
          Storage Management
        </h1>
        <div className="bg-red-50 dark:bg-red-950/20 border border-red-200 dark:border-red-900/30 rounded-2xl p-6 mt-6">
          <h2 className="text-lg font-semibold text-red-900 dark:text-red-200 flex items-center gap-2">
            <span className="material-symbols-outlined text-red-500 text-xl select-none">error</span> Failed to Load Storage Data
          </h2>
          <p className="text-red-800 dark:text-red-300 mt-2 text-sm">{error}</p>
          <button
            onClick={() => {
              setLoading(true);
              fetchData();
            }}
            className="mt-4 px-4 py-2 bg-red-600 hover:bg-red-750 text-white rounded-xl text-sm font-semibold transition cursor-pointer shadow-sm"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  if (loading || !stats) {
    return (
      <div className="max-w-5xl mx-auto p-6 space-y-6">
        <h1 className="text-2xl font-bold tracking-tight text-slate-900 dark:text-white">
          Storage Management
        </h1>
        <div className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 p-6 animate-pulse space-y-4">
          <div className="h-6 bg-slate-200 dark:bg-slate-800 rounded w-1/4"></div>
          <div className="h-4 bg-slate-200 dark:bg-slate-800 rounded w-1/3"></div>
          <div className="h-3 bg-slate-200 dark:bg-slate-800 rounded-full w-full"></div>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {[1, 2, 3].map((i) => (
            <div key={i} className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 p-5 animate-pulse space-y-3">
              <div className="h-5 bg-slate-200 dark:bg-slate-800 rounded w-1/2"></div>
              <div className="h-4 bg-slate-200 dark:bg-slate-800 rounded w-3/4"></div>
              <div className="h-8 bg-slate-200 dark:bg-slate-800 rounded w-full"></div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  const formatSize = (mb: number) => {
    if (mb > 1024) return `${(mb / 1024).toFixed(1)} GB`;
    return `${mb.toFixed(0)} MB`;
  };

  const limit = 100 * 1024; // 100 GB limit
  const percent = Math.min(Math.round((stats.total_mb / limit) * 100), 100);
  const percentColor = percent > 90 ? "bg-red-500" : percent > 70 ? "bg-amber-500" : "bg-emerald-500";

  return (
    <div className="max-w-5xl mx-auto p-6 space-y-6 animate-fade-in">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-slate-900 dark:text-white">
          Storage Management
        </h1>
        <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
          Monitor system cache, deduplicate redundant animal records, and download empty/valid frame archives.
        </p>
      </div>

      {/* Alert Notices */}
      {notice && (
        <div className={`p-4 rounded-xl border flex items-start gap-3 animate-fade-in ${
          notice.type === "success"
            ? "bg-emerald-50 dark:bg-emerald-950/20 border-emerald-200 dark:border-emerald-800/40 text-emerald-800 dark:text-emerald-350"
            : "bg-red-50 dark:bg-red-950/20 border-red-200 dark:border-red-800/40 text-red-800 dark:text-red-350"
        }`}>
          <div className="mt-0.5 flex items-center justify-center">
            {notice.type === "success" ? (
              <span className="material-symbols-outlined text-emerald-500 text-sm select-none">check_circle</span>
            ) : (
              <span className="material-symbols-outlined text-red-500 text-sm select-none">warning</span>
            )}
          </div>
          <div className="flex-1 text-xs font-semibold leading-relaxed">
            {notice.text}
          </div>
          <button onClick={() => setNotice(null)} className="text-xs font-bold hover:underline cursor-pointer">
            Dismiss
          </button>
        </div>
      )}

      {/* Storage Gauge */}
      <div className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200/60 dark:border-slate-800/80 p-6 shadow-sm">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <span className="text-slate-500 dark:text-slate-400">
              <DatabaseIcon />
            </span>
            <h2 className="font-bold text-slate-900 dark:text-white text-base">
              System Storage Capacity
            </h2>
          </div>
          <span className={`text-sm font-mono font-bold ${percent > 90 ? "text-red-500" : percent > 70 ? "text-amber-500" : "text-emerald-500"}`}>
            {formatSize(stats.total_mb)} / 100.0 GB
          </span>
        </div>

        {/* Progress Bar Container */}
        <div className="relative">
          <div className="h-4 bg-slate-100 dark:bg-slate-950 rounded-full overflow-hidden border border-slate-200/30 dark:border-slate-800/30">
            <div
              className={`h-full rounded-full transition-all duration-500 ${percentColor}`}
              style={{ width: `${percent}%` }}
            />
          </div>
          <div className="mt-2 flex justify-between text-xs text-slate-400 dark:text-slate-500 font-semibold">
            <span>{percent}% Used</span>
            <span>100 GB Safe Capacity Limit</span>
          </div>
        </div>

        {percent > 70 && (
          <div className="mt-4 p-3.5 bg-amber-50/70 dark:bg-amber-950/20 rounded-xl border border-amber-200/40 dark:border-amber-900/30 flex items-start gap-2">
            <span className="material-symbols-outlined text-amber-600 dark:text-amber-400 text-sm select-none mt-0.5">warning</span>
            <p className="text-xs text-amber-800 dark:text-amber-300 font-medium">
              Disk utilization is high ({percent}%). Please download archival zip files and purge raw duplicates or empty frame folders to avoid processing crashes.
            </p>
          </div>
        )}
      </div>

      {/* Storage Breakdown Tiers */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {["empty", "low_conf", "valid"].map((tier) => {
          const tierData = stats.breakdown[tier];
          if (!tierData) return null;

          const config: Record<string, { label: string; bg: string; border: string; text: string; icon: any; desc: string }> = {
            empty: {
              label: "Empty Frames",
              bg: "from-slate-50 to-slate-100/50 dark:from-slate-900 dark:to-slate-955/30",
              border: "border-slate-200/60 dark:border-slate-800/80",
              text: "text-slate-700 dark:text-slate-350",
              icon: (
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.8} stroke="currentColor" className="w-5 h-5 text-slate-500">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M2.25 15.75l5.159-5.159a2.25 2.25 0 013.182 0l5.159 5.159m-1.5-1.5l1.409-1.409a2.25 2.25 0 013.182 0l2.909 2.909m-18 3.75h16.5a1.5 1.5 0 001.5-1.5V6a1.5 1.5 0 00-1.5-1.5H3.75A1.5 1.5 0 002.25 6v12a1.5 1.5 0 001.5 1.5zm10.5-11.25h.008v.008h-.008V8.25zm.375 0a.375.375 0 11-.75 0 .375.375 0 01.75 0z" />
                </svg>
              ),
              desc: "No fauna detected (flagged for automatic deletion)",
            },
            low_conf: {
              label: "Low Confidence",
              bg: "from-white to-amber-50/10 dark:from-slate-900 dark:to-slate-900/30",
              border: "border-slate-200/60 dark:border-slate-800/80 hover:border-amber-200 dark:hover:border-amber-900/30",
              text: "text-amber-800 dark:text-amber-400",
              icon: (
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.8} stroke="currentColor" className="w-5 h-5 text-amber-500">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z" />
                </svg>
              ),
              desc: "Confidence score 0.2 - 0.4. Manual check recommended.",
            },
            valid: {
              label: "Valid Detections",
              bg: "from-white to-emerald-50/10 dark:from-slate-900 dark:to-slate-900/30",
              border: "border-slate-200/60 dark:border-slate-800/80 hover:border-emerald-200 dark:hover:border-emerald-900/30",
              text: "text-emerald-800 dark:text-emerald-400",
              icon: (
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.8} stroke="currentColor" className="w-5 h-5 text-emerald-500">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75M21 12c0 1.268-.63 2.39-1.593 3.068a3.745 3.745 0 01-1.043 3.296 3.745 3.745 0 01-3.296 1.043A3.745 3.745 0 0110 21a3.745 3.745 0 01-3.296-1.043 3.745 3.745 0 01-1.043-3.296A3.745 3.745 0 013 12c0-1.268.63-2.39 1.593-3.068a3.745 3.745 0 011.043-3.296 3.745 3.745 0 013.296-1.043A3.745 3.745 0 0114 3c1.268 0 2.39.63 3.068 1.593a3.746 3.746 0 013.296 1.043 3.746 3.746 0 011.043 3.296A3.745 3.745 0 0121 12z" />
                </svg>
              ),
              desc: "Confidence score > 0.4. Animals verified by classifier.",
            },
          };

          const ui = config[tier];

          return (
            <div
              key={tier}
              className={`bg-gradient-to-b rounded-2xl border p-5 shadow-sm transition flex flex-col justify-between min-h-[220px] ${ui.bg} ${ui.border}`}
            >
              <div className="space-y-2">
                <div className="flex justify-between items-start">
                  <h3 className="font-bold text-slate-800 dark:text-white text-sm">
                    {ui.label}
                  </h3>
                  <span>{ui.icon}</span>
                </div>
                <p className="text-xs text-slate-500 dark:text-slate-400">
                  {ui.desc}
                </p>
              </div>

              <div className="my-4 space-y-1.5 border-t border-slate-100 dark:border-slate-800 pt-3">
                <div className="flex justify-between text-xs font-semibold">
                  <span className="text-slate-400">File Count</span>
                  <span className="text-slate-800 dark:text-slate-200 font-mono">{(tierData.count ?? 0).toLocaleString()}</span>
                </div>
                <div className="flex justify-between text-xs font-semibold">
                  <span className="text-slate-400">Disk Weight</span>
                  <span className="text-slate-800 dark:text-slate-200 font-mono">{formatSize(tierData.size_mb)}</span>
                </div>
              </div>

              <button
                onClick={() => handleDownloadTier(tier)}
                disabled={downloading || tierData.count === 0}
                className="w-full flex items-center justify-center gap-2 py-2 px-3 bg-slate-900 hover:bg-slate-850 dark:bg-white dark:hover:bg-slate-100 text-white dark:text-slate-900 disabled:opacity-40 disabled:cursor-not-allowed text-xs font-bold rounded-xl transition cursor-pointer shadow-sm"
              >
                {downloading ? (
                  <>
                    <RefreshIcon />
                    Preparing ZIP...
                  </>
                ) : (
                  <>
                    <DownloadIcon />
                    Download Tier ZIP ({formatSize(tierData.size_mb)})
                  </>
                )}
              </button>
            </div>
          );
        })}
      </div>

      {/* Manual Empty-Image Cleanup */}
      <div className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200/60 dark:border-slate-800/80 p-6 shadow-sm space-y-4">
        <div className="flex items-center gap-2 border-b border-slate-100 dark:border-slate-800 pb-4">
          <span className="material-symbols-outlined text-red-400 select-none">delete_sweep</span>
          <div>
            <h3 className="font-bold text-slate-900 dark:text-white text-base">Delete Empty Images</h3>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
              Permanently remove images where no animal was detected. Only images older than the chosen age are affected.
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3 flex-wrap">
          <label className="text-xs font-semibold text-slate-500 dark:text-slate-400 shrink-0">Delete empty images older than</label>
          <div className="flex items-center gap-2">
            <input
              type="number"
              min={1}
              max={365}
              value={cleanupDays}
              onChange={(e) => { setCleanupDays(Number(e.target.value)); setCleanupStep("idle"); setCleanupPreview(null); }}
              disabled={cleanupStep === "running"}
              className="w-20 border border-slate-300 dark:border-slate-700 rounded-lg px-2.5 py-1.5 text-sm text-center font-mono focus:outline-none focus:ring-1 focus:ring-red-400 bg-white dark:bg-slate-950 text-slate-800 dark:text-slate-200 disabled:opacity-50"
            />
            <span className="text-xs font-semibold text-slate-500 dark:text-slate-400">days</span>
          </div>
        </div>

        {cleanupStep === "idle" && (
          <button
            onClick={handleCleanupPreview}
            className="flex items-center gap-2 px-4 py-2 bg-slate-100 hover:bg-slate-200 dark:bg-slate-800 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-300 text-sm font-semibold rounded-xl transition cursor-pointer"
          >
            <span className="material-symbols-outlined text-base select-none">preview</span>
            Preview what will be deleted
          </button>
        )}

        {cleanupStep === "previewing" && (
          <div className="flex items-center gap-2 text-sm text-slate-500 dark:text-slate-400">
            <span className="material-symbols-outlined text-base animate-spin select-none">sync</span>
            Scanning…
          </div>
        )}

        {cleanupStep === "confirming" && cleanupPreview && (
          <div className="space-y-3">
            {cleanupPreview.deleted_count === 0 ? (
              <div className="flex items-center gap-2 px-4 py-3 bg-emerald-50 dark:bg-emerald-950/30 border border-emerald-200 dark:border-emerald-900/40 rounded-xl text-sm text-emerald-700 dark:text-emerald-400 font-medium">
                <span className="material-symbols-outlined text-base select-none">check_circle</span>
                No empty images older than {cleanupDays} days — nothing to delete.
              </div>
            ) : (
              <div className="p-4 bg-red-50 dark:bg-red-950/20 border border-red-200 dark:border-red-900/40 rounded-xl space-y-3">
                <p className="text-sm font-semibold text-red-700 dark:text-red-400">
                  <span className="material-symbols-outlined text-base select-none align-middle mr-1">warning</span>
                  This will permanently delete{" "}
                  <strong>{cleanupPreview.deleted_count}</strong> image{cleanupPreview.deleted_count !== 1 ? "s" : ""}{" "}
                  and free <strong>{cleanupPreview.freed_mb?.toFixed(1) ?? "0"} MB</strong>. This cannot be undone.
                </p>
                <div className="flex gap-2">
                  <button
                    onClick={handleCleanupRun}
                    className="flex items-center gap-1.5 px-4 py-2 bg-red-600 hover:bg-red-700 text-white text-sm font-bold rounded-lg transition cursor-pointer shadow-sm"
                  >
                    <span className="material-symbols-outlined text-base select-none">delete_forever</span>
                    Delete {cleanupPreview.deleted_count} images
                  </button>
                  <button
                    onClick={() => { setCleanupStep("idle"); setCleanupPreview(null); }}
                    className="px-4 py-2 bg-slate-100 dark:bg-slate-800 hover:bg-slate-200 dark:hover:bg-slate-700 text-slate-600 dark:text-slate-300 text-sm font-semibold rounded-lg transition cursor-pointer"
                  >
                    Cancel
                  </button>
                </div>
              </div>
            )}
            {cleanupPreview.deleted_count > 0 && (
              <button
                onClick={() => { setCleanupStep("idle"); setCleanupPreview(null); }}
                className="text-xs text-slate-400 hover:text-slate-600 dark:hover:text-slate-300 cursor-pointer underline"
              >
                Cancel
              </button>
            )}
          </div>
        )}

        {cleanupStep === "running" && (
          <div className="flex items-center gap-2 text-sm text-red-600 dark:text-red-400 font-medium">
            <span className="material-symbols-outlined text-base animate-spin select-none">sync</span>
            Deleting…
          </div>
        )}
      </div>

      {/* Pending Deletions Alarm Panel */}
      {warnings && warnings.pending_deletion_count > 0 && (
        <div className="bg-gradient-to-r from-red-50 to-red-100/50 dark:from-red-950/20 dark:to-red-900/10 border border-red-200/50 dark:border-red-800/40 rounded-2xl p-5 shadow-sm space-y-3">
          <div className="flex items-center gap-1.5 text-red-700 dark:text-red-400 font-bold text-sm">
            <span className="material-symbols-outlined text-red-550 dark:text-red-450 text-base select-none">schedule</span>
            Files Scheduled for Deletion Cache Purge
          </div>
          <p className="text-xs text-red-850 dark:text-red-300 leading-relaxed font-semibold">
            {warnings.pending_deletion_count} classified empty image{warnings.pending_deletion_count > 1 ? "s" : ""} will be deleted permanently in 7 days to free space. Below is a sample preview of files being dropped:
          </p>

          <div className="bg-white/50 dark:bg-slate-950/30 border border-red-200/30 dark:border-red-800/30 rounded-xl p-3.5 space-y-1.5 max-h-36 overflow-y-auto font-mono text-[11px] text-red-700 dark:text-red-300">
            {warnings.images.slice(0, 5).map((img) => (
              <div key={img.id} className="flex justify-between items-center">
                <span>{img.filename}</span>
                <span className="text-[10px] text-red-500 font-sans">
                  Marked: {new Date(img.marked_for_deletion_at).toLocaleDateString()}
                </span>
              </div>
            ))}
            {warnings.images.length > 5 && (
              <p className="text-[10px] text-red-500 italic mt-1 font-sans">
                ...and {warnings.images.length - 5} more files.
              </p>
            )}
          </div>

          {purgeStep === "idle" && (
            <button
              onClick={handlePurgePreview}
              className="flex items-center gap-2 px-4 py-2 bg-red-100/70 hover:bg-red-200/70 dark:bg-red-900/30 dark:hover:bg-red-900/50 text-red-700 dark:text-red-350 text-xs font-bold rounded-xl transition cursor-pointer"
            >
              <span className="material-symbols-outlined text-base select-none">preview</span>
              Preview grace-period purge
            </button>
          )}

          {purgeStep === "previewing" && (
            <div className="flex items-center gap-2 text-xs text-red-600 dark:text-red-400 font-semibold">
              <span className="material-symbols-outlined text-base animate-spin select-none">sync</span>
              Scanning…
            </div>
          )}

          {purgeStep === "confirming" && purgePreview && (
            <div className="space-y-2">
              {purgePreview.will_delete === 0 ? (
                <div className="flex items-center gap-2 px-4 py-3 bg-white/60 dark:bg-slate-950/30 border border-red-200/30 dark:border-red-800/30 rounded-xl text-sm text-red-700 dark:text-red-350 font-medium">
                  <span className="material-symbols-outlined text-base select-none">check_circle</span>
                  Nothing has passed its 7-day grace period yet.
                </div>
              ) : (
                <div className="p-4 bg-white/60 dark:bg-slate-950/40 border border-red-300/40 dark:border-red-800/40 rounded-xl space-y-3">
                  <p className="text-sm font-bold text-red-800 dark:text-red-300">
                    <span className="material-symbols-outlined text-base select-none align-middle mr-1">warning</span>
                    This will permanently delete <strong>{purgePreview.will_delete}</strong> file
                    {purgePreview.will_delete !== 1 ? "s" : ""} and free <strong>{purgePreview.total_size_mb.toFixed(1)} MB</strong>. This cannot be undone.
                  </p>
                  <div className="flex gap-2">
                    <button
                      onClick={handlePurgeRun}
                      className="flex items-center gap-1.5 px-4 py-2 bg-red-600 hover:bg-red-700 text-white text-sm font-bold rounded-lg transition cursor-pointer shadow-sm"
                    >
                      <span className="material-symbols-outlined text-base select-none">delete_forever</span>
                      Purge {purgePreview.will_delete} file{purgePreview.will_delete !== 1 ? "s" : ""}
                    </button>
                    <button
                      onClick={() => { setPurgeStep("idle"); setPurgePreview(null); }}
                      className="px-4 py-2 bg-white/60 dark:bg-slate-900 hover:bg-white dark:hover:bg-slate-800 text-red-700 dark:text-red-350 text-sm font-semibold rounded-lg transition cursor-pointer"
                    >
                      Cancel
                    </button>
                  </div>
                </div>
              )}
              {purgePreview.will_delete === 0 && (
                <button
                  onClick={() => { setPurgeStep("idle"); setPurgePreview(null); }}
                  className="text-xs text-red-500 hover:text-red-700 dark:hover:text-red-300 cursor-pointer underline"
                >
                  Dismiss
                </button>
              )}
            </div>
          )}

          {purgeStep === "running" && (
            <div className="flex items-center gap-2 text-sm text-red-600 dark:text-red-400 font-medium">
              <span className="material-symbols-outlined text-base animate-spin select-none">sync</span>
              Purging…
            </div>
          )}
        </div>
      )}

      {/* Maintenance & Optimization Panel */}
      {hashStats && (
        <div className="bg-white dark:bg-slate-900 border border-slate-200/60 dark:border-slate-800/80 rounded-2xl p-6 shadow-sm space-y-6">
          <div className="flex items-start justify-between flex-wrap gap-2 border-b border-slate-100 dark:border-slate-800 pb-4">
            <div className="flex items-center gap-2">
              <span className="text-emerald-500">
                <SparklesIcon />
              </span>
              <div>
                <h3 className="font-bold text-slate-900 dark:text-white text-base">
                  Maintenance & Cache Management
                </h3>
                <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
                  Optimize internal indices, purge cached image hashes, and release redundant disk pages.
                </p>
              </div>
            </div>
          </div>

          {/* Stats Row */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {[
              { label: "Total Tracked Images", val: (hashStats.hashes.total_images ?? 0).toLocaleString(), color: "text-slate-800 dark:text-slate-100" },
              { label: "Indexed with Hashes", val: (hashStats.hashes.with_hashes ?? 0).toLocaleString(), color: "text-emerald-600 dark:text-emerald-450" },
              { label: "Pending Hash Index", val: (hashStats.hashes.without_hashes ?? 0).toLocaleString(), color: "text-slate-400 dark:text-slate-500" },
              { label: "Unique Signatures", val: (hashStats.hashes.unique_hashes ?? 0).toLocaleString(), color: "text-slate-800 dark:text-slate-100" },
            ].map((stat, i) => (
              <div key={i} className="bg-slate-50/50 dark:bg-slate-950 p-4 rounded-xl border border-slate-100 dark:border-slate-900/60">
                <p className="text-[10px] uppercase tracking-wider font-bold text-slate-400 dark:text-slate-500">
                  {stat.label}
                </p>
                <p className={`text-xl font-bold mt-1 font-mono ${stat.color}`}>
                  {stat.val}
                </p>
              </div>
            ))}
          </div>

          {/* Deduplication Card Notification */}
          {hashStats.duplicates.length > 0 && (
            <div className="bg-emerald-50/30 dark:bg-emerald-950/10 border border-emerald-200/40 dark:border-emerald-900/30 rounded-xl p-4 space-y-3">
              <div className="flex justify-between items-center flex-wrap gap-2 text-xs font-semibold text-emerald-800 dark:text-emerald-450">
                <span className="flex items-center gap-1.5">
                  <ShieldIcon />
                  Deduplication Scan Complete: Found {hashStats.duplicates.length} duplicate file signatures.
                </span>
                <span className="font-mono bg-emerald-50 dark:bg-emerald-950/40 px-2 py-0.5 rounded border border-emerald-200/30 dark:border-emerald-800/30">
                  Recoverable Space: ~{hashStats.potential_savings_mb.toFixed(0)} MB
                </span>
              </div>
              <details className="text-xs text-slate-500 dark:text-slate-400">
                <summary className="cursor-pointer font-bold hover:underline">View duplicate checksums</summary>
                <div className="mt-2.5 space-y-1 max-h-36 overflow-y-auto font-mono text-[10px] pl-3 border-l-2 border-emerald-500/20">
                  {hashStats.duplicates.slice(0, 5).map((dup, idx) => (
                    <div key={idx} className="flex justify-between text-slate-655 dark:text-slate-400">
                      <span>{dup.hash}</span>
                      <span className="font-sans font-semibold">{dup.count} references</span>
                    </div>
                  ))}
                  {hashStats.duplicates.length > 5 && (
                    <p className="font-sans italic text-slate-400 mt-1">...and {hashStats.duplicates.length - 5} more sets.</p>
                  )}
                </div>
              </details>
            </div>
          )}

          {/* Action grid with inline confirmation state */}
          <div className="space-y-4">
            <h4 className="text-xs font-bold text-slate-400 dark:text-slate-500 uppercase tracking-wider">
              Optimize Storage Operations
            </h4>

            {confirmStrategy ? (
              <div className="bg-amber-50/40 dark:bg-amber-950/20 border border-amber-200/50 dark:border-amber-900/30 p-4 rounded-xl space-y-3">
                <p className="text-xs font-semibold text-amber-800 dark:text-amber-300 leading-relaxed flex items-start gap-1">
                  <span className="material-symbols-outlined text-amber-500 text-sm leading-none select-none">warning</span>
                  <span>
                    Are you sure you want to run the <strong>"{confirmStrategy.replace("_", " ")}"</strong> cleaning strategy?
                    <br />
                    This clears matching image hash caches. Processing speed may be briefly impacted while new hashes recalculate on future uploads.
                  </span>
                </p>
                <div className="flex gap-2">
                  <button
                    onClick={() => handleClearHashes(confirmStrategy)}
                    disabled={clearingHashes}
                    className="px-3.5 py-1.5 bg-amber-600 hover:bg-amber-700 text-white font-semibold text-xs rounded-lg shadow transition cursor-pointer"
                  >
                    {clearingHashes ? "Clearing..." : "Yes, Proceed"}
                  </button>
                  <button
                    onClick={() => setConfirmStrategy(null)}
                    className="px-3.5 py-1.5 bg-slate-100 hover:bg-slate-200 dark:bg-slate-800 dark:hover:bg-slate-750 text-slate-700 dark:text-slate-350 font-semibold text-xs rounded-lg transition cursor-pointer"
                  >
                    Cancel
                  </button>
                </div>
              </div>
            ) : (
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
                <button
                  onClick={() => setConfirmStrategy("empty")}
                  className="flex items-center justify-center gap-1.5 px-3 py-2 border border-slate-200/60 dark:border-slate-800/80 hover:bg-slate-50 dark:hover:bg-slate-950 text-slate-700 dark:text-slate-300 text-xs font-semibold rounded-xl transition cursor-pointer"
                  title="Purge cache signature for empty frames"
                >
                  <TrashIcon />
                  Clear Empty Hashes
                </button>
                <button
                  onClick={() => setConfirmStrategy("duplicates")}
                  disabled={hashStats.duplicates.length === 0}
                  className="flex items-center justify-center gap-1.5 px-3 py-2 border border-slate-200/60 dark:border-slate-800/80 hover:bg-slate-50 dark:hover:bg-slate-950 text-slate-700 dark:text-slate-300 disabled:opacity-40 disabled:cursor-not-allowed text-xs font-semibold rounded-xl transition cursor-pointer"
                  title="Consolidate duplicate file records"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.8} stroke="currentColor" className="w-4 h-4">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M19 7.5v3m0 0v3m0-3h3m-3 0h-3m-2.25-4.125a3.375 3.375 0 11-6.75 0 3.375 3.375 0 016.75 0zM4 19.235v-.11a6.375 6.375 0 0112.75 0v.109A12.318 12.318 0 0110.374 21c-2.331 0-4.512-.645-6.374-1.766z" />
                  </svg>
                  Deduplicate Hashes
                </button>
                <button
                  onClick={() => setConfirmStrategy("old_30d")}
                  className="flex items-center justify-center gap-1.5 px-3 py-2 border border-slate-200/60 dark:border-slate-800/80 hover:bg-slate-50 dark:hover:bg-slate-950 text-slate-700 dark:text-slate-300 text-xs font-semibold rounded-xl transition cursor-pointer"
                  title="Clear signatures older than 30 days"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.8} stroke="currentColor" className="w-4 h-4">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M12 6v6h4.5m4.5 0a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  Clear Old (30d+)
                </button>
                <button
                  onClick={() => setConfirmStrategy("all")}
                  className="flex items-center justify-center gap-1.5 px-3 py-2 bg-red-50 hover:bg-red-100/60 dark:bg-red-950/20 dark:hover:bg-red-900/20 border border-red-200/40 dark:border-red-800/30 text-red-650 dark:text-red-400 text-xs font-bold rounded-xl transition cursor-pointer"
                  title="Flush total hash cache index"
                >
                  <TrashIcon />
                  Clear All Hashes
                </button>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Explanatory Info Box */}
      <div className="bg-blue-50/50 dark:bg-blue-950/10 border border-blue-200/40 dark:border-blue-900/30 rounded-2xl p-5 shadow-sm flex gap-3.5 items-start">
        <span className="text-blue-500 dark:text-blue-450 mt-0.5">
          <InfoIcon />
        </span>
        <div>
          <h3 className="font-bold text-blue-900 dark:text-blue-300 text-sm">
            Automatic Retention & Archival Policies
          </h3>
          <ul className="text-xs text-blue-800/85 dark:text-blue-400 mt-2 space-y-1.5 font-semibold">
            <li>
              • <strong className="text-blue-900 dark:text-blue-300">Empty Frames:</strong> Kept until you manually delete them using the "Delete Empty Images" panel above. Download the empty tier ZIP first to keep a local backup.
            </li>
            <li>
              • <strong className="text-blue-900 dark:text-blue-300">Low Confidence:</strong> Kept indefinitely in reviewer inbox. Purged only after explicit deletion command (7-day safety grace).
            </li>
            <li>
              • <strong className="text-blue-900 dark:text-blue-300">Valid Detections:</strong> Persisted permanently. Recommended practice is to download periodic backups before manual cleaning.
            </li>
            <li>
              • <strong className="text-blue-900 dark:text-blue-300">Database Schema:</strong> Numeric counts, classifications, and historical metadata are kept forever for logs, even if raw images are deleted.
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
}
