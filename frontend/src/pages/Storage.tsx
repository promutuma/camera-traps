import { useEffect, useState } from "react";
import { getStorageStatus, getStorageWarnings, getDeletionPreview, createBatchDownload, getHashStats, clearHashes } from "../api/client";

type StorageStats = {
  total_mb: number;
  breakdown: Record<string, { count: number; size_mb: number; oldest_upload: string }>;
  timestamp: string;
};

type DeletionWarning = {
  pending_deletion_count: number;
  images: Array<{ id: number; filename: string; marked_for_deletion_at: string }>;
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

export default function Storage() {
  const [stats, setStats] = useState<StorageStats | null>(null);
  const [warnings, setWarnings] = useState<DeletionWarning | null>(null);
  const [hashStats, setHashStats] = useState<HashStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedTier, setSelectedTier] = useState<string | null>(null);
  const [downloading, setDownloading] = useState(false);
  const [clearingHashes, setClearingHashes] = useState(false);

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      setError(null);
      try {
        const [statsData, warningsData, hashData] = await Promise.all([
          getStorageStatus(),
          getStorageWarnings(),
          getHashStats().catch(() => null), // Hash stats optional
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

    fetchData();
    const interval = setInterval(fetchData, 60000); // Refresh every minute
    return () => clearInterval(interval);
  }, []);

  const handleDownloadTier = async (tier: string) => {
    setDownloading(true);
    try {
      const result = await createBatchDownload(tier);
      if (!result?.download_id) {
        throw new Error("No download ID returned");
      }
      // Wait a moment for server to prepare ZIP, then trigger download
      await new Promise((resolve) => setTimeout(resolve, 500));
      window.location.href = `/api/storage/downloads/${result.download_id}/stream`;
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : "Failed to create download";
      console.error("Download failed:", err);
      alert(`Download failed: ${errorMsg}`);
    } finally {
      setDownloading(false);
    }
  };

  const handleClearHashes = async (strategy: string) => {
    if (!window.confirm(`Clear hashes with strategy: ${strategy}?\n\nThis reduces processing time but disables deduplication detection until new hashes are calculated.`)) {
      return;
    }

    setClearingHashes(true);
    try {
      const result = await clearHashes(strategy);
      const message = `Cleared ${result.cleared} hashes\n\n${result.purpose}${result.warning ? `\n\n⚠️ ${result.warning}` : ""}`;
      alert(message);
      // Refresh hash stats
      const newHashStats = await getHashStats();
      if (newHashStats) setHashStats(newHashStats);
    } catch (err) {
      const errorMsg = err instanceof Error ? err.message : "Unknown error";
      console.error("Clear hashes failed:", err);
      alert(`Failed to clear hashes: ${errorMsg}`);
    } finally {
      setClearingHashes(false);
    }
  };

  if (error) {
    return (
      <div className="max-w-6xl mx-auto p-6">
        <h1 className="text-2xl font-bold mb-6 text-slate-900 dark:text-white">
          Storage Management
        </h1>
        <div className="bg-red-50 dark:bg-red-950/20 border border-red-200 dark:border-red-900/30 rounded-xl p-6">
          <div className="flex items-center gap-3 mb-2">
            <span className="material-symbols-outlined text-red-600 dark:text-red-400 text-2xl">
              error
            </span>
            <h2 className="text-lg font-semibold text-red-900 dark:text-red-200">
              Failed to Load Storage Data
            </h2>
          </div>
          <p className="text-red-800 dark:text-red-300 mb-4">{error}</p>
          <button
            onClick={() => window.location.reload()}
            className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  if (loading || !stats) {
    return (
      <div className="max-w-6xl mx-auto p-6">
        <h1 className="text-2xl font-bold mb-6 text-slate-900 dark:text-white">
          Storage Management
        </h1>
        <div className="space-y-6">
          {/* Gauge skeleton */}
          <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6 animate-pulse">
            <div className="h-8 bg-slate-200 dark:bg-slate-800 rounded w-1/3 mb-4"></div>
            <div className="h-3 bg-slate-200 dark:bg-slate-800 rounded-full mb-4"></div>
            <div className="h-4 bg-slate-200 dark:bg-slate-800 rounded w-1/4"></div>
          </div>

          {/* Tier cards skeleton */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {[1, 2, 3].map((i) => (
              <div
                key={i}
                className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-4 animate-pulse"
              >
                <div className="h-6 bg-slate-200 dark:bg-slate-800 rounded w-1/2 mb-3"></div>
                <div className="space-y-2">
                  <div className="h-4 bg-slate-200 dark:bg-slate-800 rounded"></div>
                  <div className="h-8 bg-slate-200 dark:bg-slate-800 rounded"></div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  const formatSize = (mb: number) => {
    if (mb > 1024) return `${(mb / 1024).toFixed(1)} GB`;
    return `${mb.toFixed(0)} MB`;
  };

  const getStoragePercent = () => {
    const limit = 100 * 1024; // 100 GB default
    return Math.round((stats.total_mb / limit) * 100);
  };

  const percent = getStoragePercent();
  const percentColor = percent > 90 ? "bg-red-500" : percent > 70 ? "bg-amber-500" : "bg-emerald-500";

  return (
    <div className="max-w-6xl mx-auto p-6">
      <h1 className="text-2xl font-bold mb-6 text-slate-900 dark:text-white">
        Storage Management
      </h1>

      {/* Storage Gauge */}
      <div className="bg-white dark:bg-slate-900 rounded-xl border border-slate-200 dark:border-slate-800 p-6 mb-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-slate-900 dark:text-white">
            Total Storage
          </h2>
          <span className={`text-sm font-mono ${percent > 90 ? "text-red-600" : percent > 70 ? "text-amber-600" : "text-emerald-600"}`}>
            {formatSize(stats.total_mb)} / 100 GB
          </span>
        </div>

        {/* Progress bar */}
        <div className="h-3 bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full transition-all ${percentColor}`}
            style={{ width: `${Math.min(percent, 100)}%` }}
          />
        </div>
        <div className="mt-2 flex justify-between text-xs text-slate-500">
          <span>{percent}%</span>
          <span>100 GB limit</span>
        </div>

        {percent > 70 && (
          <div className="mt-4 p-3 bg-amber-50 dark:bg-amber-950/20 rounded-lg border border-amber-200 dark:border-amber-900/30">
            <p className="text-sm text-amber-800 dark:text-amber-200">
              ⚠️ Storage above {percent > 90 ? "90%" : "70%"} — consider downloading and deleting files
            </p>
          </div>
        )}
      </div>

      {/* Storage Breakdown by Tier */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        {["empty", "low_conf", "valid"].map((tier) => {
          const tierData = stats.breakdown[tier];
          if (!tierData) return null;

          const tierLabels: Record<string, { label: string; icon: string; color: string; description: string }> = {
            empty: {
              label: "Empty Frames",
              icon: "hide_image",
              color: "bg-slate-50 dark:bg-slate-800/50 border-slate-200 dark:border-slate-700",
              description: "No animals detected",
            },
            low_conf: {
              label: "Low Confidence",
              icon: "warning",
              color: "bg-amber-50 dark:bg-amber-950/20 border-amber-200 dark:border-amber-900/30",
              description: "0.2-0.4 confidence",
            },
            valid: {
              label: "Valid Detections",
              icon: "check_circle",
              color: "bg-emerald-50 dark:bg-emerald-950/20 border-emerald-200 dark:border-emerald-900/30",
              description: "High confidence animals",
            },
          };

          const info = tierLabels[tier];

          return (
            <div
              key={tier}
              className={`rounded-xl border p-4 ${info.color}`}
            >
              <div className="flex items-start justify-between mb-3">
                <div>
                  <p className="text-sm font-semibold text-slate-900 dark:text-white">
                    {info.label}
                  </p>
                  <p className="text-xs text-slate-500 dark:text-slate-400">
                    {info.description}
                  </p>
                </div>
                <span className="material-symbols-outlined text-lg text-slate-400">
                  {info.icon}
                </span>
              </div>

              <div className="space-y-2 mb-3">
                <div className="flex justify-between text-sm">
                  <span className="text-slate-600 dark:text-slate-300">Count:</span>
                  <span className="font-mono font-semibold">{tierData.count}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-slate-600 dark:text-slate-300">Size:</span>
                  <span className="font-mono font-semibold">{formatSize(tierData.size_mb)}</span>
                </div>
              </div>

              <button
                onClick={() => handleDownloadTier(tier)}
                disabled={downloading || tierData.count === 0}
                className={`w-full py-2 px-3 rounded-lg text-sm font-medium transition-colors ${
                  downloading || tierData.count === 0
                    ? "opacity-50 cursor-not-allowed"
                    : "bg-slate-900 dark:bg-white text-white dark:text-slate-900 hover:bg-slate-800 dark:hover:bg-slate-100"
                }`}
              >
                {downloading ? (
                  <>
                    <span className="material-symbols-outlined text-sm align-middle">sync</span>
                    {" "}Downloading...
                  </>
                ) : (
                  <>
                    <span className="material-symbols-outlined text-sm align-middle">download</span>
                    {" "}Download ({formatSize(tierData.size_mb)})
                  </>
                )}
              </button>
            </div>
          );
        })}
      </div>

      {/* Pending Deletions */}
      {warnings && warnings.pending_deletion_count > 0 && (
        <div className="bg-red-50 dark:bg-red-950/20 border border-red-200 dark:border-red-900/30 rounded-xl p-4 mb-6">
          <h3 className="font-semibold text-red-900 dark:text-red-200 mb-2">
            ⏰ Files Pending Deletion
          </h3>
          <p className="text-sm text-red-800 dark:text-red-300 mb-3">
            {warnings.pending_deletion_count} image{warnings.pending_deletion_count > 1 ? "s" : ""} will be automatically deleted in 7 days.
            Download them now if needed.
          </p>
          <div className="space-y-1 max-h-40 overflow-y-auto">
            {warnings.images.slice(0, 5).map((img) => (
              <div
                key={img.id}
                className="text-xs text-red-700 dark:text-red-300 flex items-center gap-2"
              >
                <span className="material-symbols-outlined text-sm">timer</span>
                {img.filename}
              </div>
            ))}
            {warnings.images.length > 5 && (
              <p className="text-xs text-red-600 dark:text-red-400 italic">
                ...and {warnings.images.length - 5} more
              </p>
            )}
          </div>
        </div>
      )}

      {/* Hash Management - Performance Optimization */}
      {hashStats && (
        <div className="bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-xl p-6 mb-6">
          <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-4 flex items-center gap-2">
            <span className="material-symbols-outlined">speed</span>
            Hash Management (Performance Optimization)
          </h3>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
            <div className="bg-white dark:bg-slate-900 rounded-lg p-3">
              <p className="text-xs text-slate-500 dark:text-slate-400">Total Images</p>
              <p className="text-lg font-bold text-slate-900 dark:text-white">{hashStats.hashes.total_images}</p>
            </div>
            <div className="bg-white dark:bg-slate-900 rounded-lg p-3">
              <p className="text-xs text-slate-500 dark:text-slate-400">With Hashes</p>
              <p className="text-lg font-bold text-emerald-600">{hashStats.hashes.with_hashes}</p>
            </div>
            <div className="bg-white dark:bg-slate-900 rounded-lg p-3">
              <p className="text-xs text-slate-500 dark:text-slate-400">Without Hashes</p>
              <p className="text-lg font-bold text-slate-600 dark:text-slate-300">{hashStats.hashes.without_hashes}</p>
            </div>
            <div className="bg-white dark:bg-slate-900 rounded-lg p-3">
              <p className="text-xs text-slate-500 dark:text-slate-400">Unique Hashes</p>
              <p className="text-lg font-bold text-slate-900 dark:text-white">{hashStats.hashes.unique_hashes}</p>
            </div>
          </div>

          {hashStats.duplicates.length > 0 && (
            <div className="mb-4 p-3 bg-amber-50 dark:bg-amber-950/20 rounded-lg border border-amber-200 dark:border-amber-900/30">
              <p className="text-sm text-amber-900 dark:text-amber-200 mb-2">
                📋 <strong>Deduplication Opportunity:</strong> Found {hashStats.duplicates.length} duplicate file hash{hashStats.duplicates.length > 1 ? "es" : ""}
                (could save ~{hashStats.potential_savings_mb.toFixed(0)} MB)
              </p>
              <details className="text-xs text-amber-800 dark:text-amber-300">
                <summary className="cursor-pointer font-semibold">View duplicates</summary>
                <div className="mt-2 space-y-1 max-h-40 overflow-y-auto">
                  {hashStats.duplicates.slice(0, 5).map((dup, idx) => (
                    <div key={idx} className="font-mono text-xs">
                      {dup.hash} × {dup.count} images
                    </div>
                  ))}
                  {hashStats.duplicates.length > 5 && (
                    <p className="italic">...and {hashStats.duplicates.length - 5} more</p>
                  )}
                </div>
              </details>
            </div>
          )}

          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <button
              onClick={() => handleClearHashes("empty")}
              disabled={clearingHashes}
              className="px-4 py-2 rounded-lg text-sm font-medium bg-slate-900 dark:bg-white text-white dark:text-slate-900 hover:bg-slate-800 dark:hover:bg-slate-100 disabled:opacity-50 transition-colors"
              title="Clear hashes from empty frames"
            >
              <span className="material-symbols-outlined text-sm align-middle">delete_outline</span>
              {" "}Clear Empty Hashes
            </button>
            <button
              onClick={() => handleClearHashes("duplicates")}
              disabled={clearingHashes || hashStats.duplicates.length === 0}
              className="px-4 py-2 rounded-lg text-sm font-medium bg-slate-900 dark:bg-white text-white dark:text-slate-900 hover:bg-slate-800 dark:hover:bg-slate-100 disabled:opacity-50 transition-colors"
              title="Keep only first occurrence of duplicate hashes"
            >
              <span className="material-symbols-outlined text-sm align-middle">delete_sweep</span>
              {" "}Deduplicate Hashes
            </button>
            <button
              onClick={() => handleClearHashes("old_30d")}
              disabled={clearingHashes}
              className="px-4 py-2 rounded-lg text-sm font-medium bg-amber-600 dark:bg-amber-500 text-white hover:bg-amber-700 dark:hover:bg-amber-600 disabled:opacity-50 transition-colors"
              title="Clear hashes older than 30 days"
            >
              <span className="material-symbols-outlined text-sm align-middle">history</span>
              {" "}Clear Old (30d+)
            </button>
            <button
              onClick={() => handleClearHashes("all")}
              disabled={clearingHashes}
              className="px-4 py-2 rounded-lg text-sm font-medium bg-red-600 dark:bg-red-500 text-white hover:bg-red-700 dark:hover:bg-red-600 disabled:opacity-50 transition-colors"
              title="Clear all hashes - maximum performance gain"
            >
              <span className="material-symbols-outlined text-sm align-middle">clear_all</span>
              {" "}Clear All Hashes
            </button>
          </div>

          <p className="text-xs text-slate-500 dark:text-slate-400 mt-3">
            💡 Clearing hashes reduces processing overhead but disables duplicate detection until new hashes are calculated. Choose the strategy that best fits your needs.
          </p>
        </div>
      )}

      {/* Info Box */}
      <div className="bg-blue-50 dark:bg-blue-950/20 border border-blue-200 dark:border-blue-900/30 rounded-xl p-4">
        <h3 className="font-semibold text-blue-900 dark:text-blue-200 mb-2">
          💡 How File Management Works
        </h3>
        <ul className="text-sm text-blue-800 dark:text-blue-300 space-y-1">
          <li>
            • <strong>Empty Frames:</strong> Automatically deleted after 7 days. Download first if needed.
          </li>
          <li>
            • <strong>Low Confidence:</strong> Available indefinitely until you mark for deletion, then 7-day grace period.
          </li>
          <li>
            • <strong>Valid Detections:</strong> Stay until exported. After export, can be deleted to save space.
          </li>
          <li>
            • <strong>Metadata:</strong> Always kept in the database for searching, even after files are deleted.
          </li>
        </ul>
      </div>
    </div>
  );
}
