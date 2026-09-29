import { useState } from "react";

interface UsernameModalProps {
  onSubmit: (username: string) => Promise<void>;
}

export default function UsernameModal({ onSubmit }: UsernameModalProps) {
  const [username, setUsername] = useState("");
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = username.trim();
    if (!trimmed) {
      setError("Please enter your name");
      return;
    }
    setSubmitting(true);
    setError("");
    try {
      await onSubmit(trimmed);
    } catch {
      setError("Could not start session. Is the backend running?");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/70 backdrop-blur-sm p-4">
      <div className="w-full max-w-md rounded-xl border border-slate-700 bg-slate-900 p-6 shadow-2xl">
        <h2 className="text-lg font-semibold text-white mb-1">Welcome to ViumbeLens</h2>
        <p className="text-sm text-slate-400 mb-5">
          Enter your name to attribute reviews and actions. No password required.
        </p>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label htmlFor="username" className="block text-xs font-semibold text-slate-400 mb-1">
              Your name
            </label>
            <input
              id="username"
              type="text"
              autoFocus
              maxLength={64}
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="e.g. Jane"
              className="block w-full rounded-lg border border-slate-700 bg-slate-800/50 text-white px-3 py-2 text-sm focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500"
            />
          </div>
          {error && <p className="text-sm text-red-400">{error}</p>}
          <button
            type="submit"
            disabled={submitting}
            className="w-full rounded-lg bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white font-medium py-2 text-sm transition-colors"
          >
            {submitting ? "Starting…" : "Continue"}
          </button>
        </form>
      </div>
    </div>
  );
}
