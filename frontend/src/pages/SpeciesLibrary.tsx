import { useEffect, useState } from "react";
import { getSpecies, lookupSpecies, resolveSynonym, addSpecies } from "../api/client";

type Row = Record<string, unknown>;

// Inline SVG Taxon Icons
const MammalIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-6 h-6">
    <path strokeLinecap="round" strokeLinejoin="round" d="M15.362 5.214A8.252 8.252 0 0112 21 8.25 8.25 0 016.038 7.048 8.287 8.287 0 009 9.6a8.983 8.983 0 013.361-6.867 8.21 8.21 0 006.101 2.48z" />
    <path strokeLinecap="round" strokeLinejoin="round" d="M12 18a3.75 3.75 0 00.495-7.467 5.99 5.99 0 00-1.925 3.546 5.974 5.974 0 01-2.133-1A3.75 3.75 0 0012 18z" />
  </svg>
);

const BirdIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-6 h-6">
    <path strokeLinecap="round" strokeLinejoin="round" d="M6 12L3.269 3.126A59.768 59.768 0 0121.485 12 59.77 59.77 0 013.27 20.876L5.999 12zm0 0h7.5" />
  </svg>
);

const ReptileIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-6 h-6">
    <path strokeLinecap="round" strokeLinejoin="round" d="M15.362 5.214A8.252 8.252 0 0112 21 8.25 8.25 0 016.038 7.048 8.287 8.287 0 009 9.6a8.983 8.983 0 013.361-6.867 8.21 8.21 0 006.101 2.48z" />
  </svg>
);

const InsectIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-6 h-6">
    <path strokeLinecap="round" strokeLinejoin="round" d="M12 21a9.004 9.004 0 008.716-6.747M12 21a9.004 9.004 0 01-8.716-6.747M12 21V9.75M3.284 14.253A8.966 8.966 0 0112 3.75c3.81 0 7.06 2.372 8.35 5.75m-18.066 4.753A9.043 9.043 0 0012 15.75c2.31 0 4.418-.867 6.012-2.3m-15.312-3.14a8.995 8.995 0 0118.066 0M3.284 10.25a8.966 8.966 0 008.716 5.5M12 9.75V3.75" />
  </svg>
);

const LeafIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-6 h-6">
    <path strokeLinecap="round" strokeLinejoin="round" d="M12 3v18M12 3a9 9 0 00-9 9v1.5a9 9 0 009 9M12 3a9 9 0 019 9v1.5a9 9 0 01-9 9" />
  </svg>
);

function getTaxonIcon(className?: string) {
  const c = String(className ?? "").toLowerCase();
  if (c.includes("mammal")) return <MammalIcon />;
  if (c.includes("aves") || c.includes("bird")) return <BirdIcon />;
  if (c.includes("reptil") || c.includes("snake") || c.includes("lizard")) return <ReptileIcon />;
  if (c.includes("amphib") || c.includes("frog")) return <ReptileIcon />;
  if (c.includes("insect") || c.includes("bug") || c.includes("arach")) return <InsectIcon />;
  return <LeafIcon />;
}

export default function SpeciesLibrary() {
  const [species, setSpecies] = useState<Row[]>([]);
  const [loading, setLoading] = useState(true);
  
  const [lookupQuery, setLookupQuery] = useState("");
  const [lookupResult, setLookupResult] = useState<Row | null>(null);
  const [lookupError, setLookupError] = useState("");
  
  const [synonymQuery, setSynonymQuery] = useState("");
  const [synonymResult, setSynonymResult] = useState<Row | null>(null);
  
  const [filter, setFilter] = useState("");
  const [selectedSp, setSelectedSp] = useState<Row | null>(null);

  const [newSciName, setNewSciName] = useState("");
  const [newCommonName, setNewCommonName] = useState("");
  const [addBusy, setAddBusy] = useState(false);
  const [addError, setAddError] = useState("");
  const [addSuccess, setAddSuccess] = useState("");

  const refreshSpecies = () => getSpecies().then(setSpecies);

  useEffect(() => {
    refreshSpecies().finally(() => setLoading(false));
  }, []);

  const handleAddSpecies = async () => {
    if (!newSciName.trim()) return;
    setAddBusy(true);
    setAddError("");
    setAddSuccess("");
    try {
      await addSpecies(newSciName.trim(), newCommonName.trim());
      setAddSuccess(`Added "${newCommonName.trim() || newSciName.trim()}" to the library.`);
      setNewSciName("");
      setNewCommonName("");
      await refreshSpecies();
    } catch (err: any) {
      setAddError(err?.response?.data?.detail || "Failed to add species.");
    } finally {
      setAddBusy(false);
    }
  };

  const handleLookup = async () => {
    if (!lookupQuery) return;
    setLookupError("");
    setLookupResult(null);
    try {
      const result = await lookupSpecies(lookupQuery);
      setLookupResult(result);
      setSelectedSp(result);
    } catch {
      setLookupError(`Species "${lookupQuery}" not found in canonical reference library.`);
    }
  };

  const handleSynonym = async () => {
    if (!synonymQuery) return;
    try {
      const result = await resolveSynonym(synonymQuery);
      setSynonymResult(result);
    } catch (e) {
      console.error(e);
    }
  };

  const filtered = species.filter((s) => {
    const term = filter.toLowerCase();
    if (!term) return true;
    const name = String(s.name ?? s.scientific_name ?? "").toLowerCase();
    const common = String(s.common_name ?? "").toLowerCase();
    const order = String(s.order ?? "").toLowerCase();
    const family = String(s.family ?? "").toLowerCase();
    return name.includes(term) || common.includes(term) || order.includes(term) || family.includes(term);
  });

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Page Header */}
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-slate-900 dark:text-white">
          Species Reference Library
        </h1>
        <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
          Explore wildlife baseline taxonomy, match taxonomic synonyms, and examine Ethiopia Gambella regional IUCN classifications.
        </p>
      </div>

      {/* Lookup, Synonym Resolver, and Add Species Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
        {/* Quick Species Lookup */}
        <div className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200/60 dark:border-slate-800/80 p-5 space-y-4 shadow-sm flex flex-col justify-between">
          <div className="space-y-1">
            <h2 className="font-bold text-slate-900 dark:text-white text-sm">
              Canonical Reference Lookup
            </h2>
            <p className="text-xs text-slate-400 dark:text-slate-500">
              Query GBIF/national baseline taxonomic database files by canonical scientific name.
            </p>
          </div>
          <div className="flex gap-2">
            <input
              placeholder="e.g. Panthera leo"
              value={lookupQuery}
              onChange={(e) => setLookupQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleLookup()}
              className="flex-1 border border-slate-200 dark:border-slate-800 rounded-xl px-3 py-2 text-xs focus:outline-none focus:ring-2 focus:ring-emerald-500/20 focus:border-emerald-500 bg-slate-50/50 dark:bg-slate-950 text-slate-800 dark:text-slate-100 transition"
            />
            <button
              onClick={handleLookup}
              className="px-4 py-2 bg-emerald-600 hover:bg-emerald-700 text-white text-xs font-semibold rounded-xl shadow-sm transition cursor-pointer"
            >
              Search
            </button>
          </div>

          {lookupError && (
            <p className="text-red-500 dark:text-red-400 text-xs font-semibold mt-1">
              <span className="material-symbols-outlined text-sm align-middle mr-1 select-none">warning</span> {lookupError}
            </p>
          )}

          {/* Pretty Lookup Result Summary (Replaces raw JSON dump) */}
          {lookupResult && (
            <div className="bg-emerald-50/30 dark:bg-emerald-950/15 border border-emerald-200/30 dark:border-emerald-800/40 rounded-xl p-4 space-y-2 text-xs">
              <div className="flex justify-between items-center text-[10px] font-bold uppercase tracking-wider text-emerald-700 dark:text-emerald-400 border-b border-emerald-200/20 dark:border-emerald-800/30 pb-2">
                <span>Database Match</span>
                <span className="font-mono">Score: 100%</span>
              </div>
              <div className="space-y-1.5 font-medium">
                <div className="flex justify-between">
                  <span className="text-slate-400">Common Name</span>
                  <span className="text-slate-850 dark:text-slate-200 font-semibold">{String(lookupResult.common_name ?? "Vernacular unmapped")}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Scientific Name</span>
                  <span className="text-slate-850 dark:text-slate-200 font-semibold italic">{String(lookupResult.name ?? lookupResult.scientific_name ?? "—")}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Taxonomic Class</span>
                  <span className="text-slate-850 dark:text-slate-200 font-semibold">{String(lookupResult.class ?? lookupResult.category ?? "Mammalia")}</span>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Synonym Resolver */}
        <div className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200/60 dark:border-slate-800/80 p-5 space-y-4 shadow-sm flex flex-col justify-between">
          <div className="space-y-1">
            <h2 className="font-bold text-slate-900 dark:text-white text-sm">
              Synonym & Vernacular Resolver
            </h2>
            <p className="text-xs text-slate-400 dark:text-slate-555">
              Normalize regional naming variations or synonyms into approved canonical names.
            </p>
          </div>
          <div className="flex gap-2">
            <input
              placeholder="e.g. Leo, Lion or alternate binomial"
              value={synonymQuery}
              onChange={(e) => setSynonymQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSynonym()}
              className="flex-1 border border-slate-200 dark:border-slate-800 rounded-xl px-3 py-2 text-xs focus:outline-none focus:ring-2 focus:ring-emerald-500/20 focus:border-emerald-500 bg-slate-50/50 dark:bg-slate-950 text-slate-800 dark:text-slate-100 transition"
            />
            <button
              onClick={handleSynonym}
              className="px-4 py-2 bg-emerald-600 hover:bg-emerald-700 text-white text-xs font-semibold rounded-xl shadow-sm transition cursor-pointer"
            >
              Resolve
            </button>
          </div>

          {synonymResult && (
            <div className="bg-slate-50 dark:bg-slate-950 border border-slate-100 dark:border-slate-800 rounded-xl p-3.5 flex items-center justify-between text-xs font-semibold">
              <span className="text-slate-500 dark:text-slate-400 font-mono">"{String(synonymResult.input ?? "")}"</span>
              <span className="text-emerald-500 dark:text-emerald-400 flex items-center justify-center">
                <span className="material-symbols-outlined text-sm leading-none select-none">arrow_forward</span>
              </span>
              <span className="font-bold text-emerald-600 dark:text-emerald-400 bg-emerald-50 dark:bg-emerald-950/40 px-2.5 py-1 rounded-full border border-emerald-100 dark:border-emerald-900/40">
                {String(synonymResult.canonical ?? "No canonical entry found")}
              </span>
            </div>
          )}
        </div>

        {/* Add Species */}
        <div className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200/60 dark:border-slate-800/80 p-5 space-y-4 shadow-sm flex flex-col justify-between">
          <div className="space-y-1">
            <h2 className="font-bold text-slate-900 dark:text-white text-sm">
              Add Species to Library
            </h2>
            <p className="text-xs text-slate-400 dark:text-slate-500">
              Register a species not yet in the canonical reference library.
            </p>
          </div>
          <div className="space-y-2">
            <input
              placeholder="Scientific name (required), e.g. Panthera pardus"
              value={newSciName}
              onChange={(e) => setNewSciName(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleAddSpecies()}
              className="w-full border border-slate-200 dark:border-slate-800 rounded-xl px-3 py-2 text-xs focus:outline-none focus:ring-2 focus:ring-emerald-500/20 focus:border-emerald-500 bg-slate-50/50 dark:bg-slate-950 text-slate-800 dark:text-slate-100 transition"
            />
            <input
              placeholder="Common name (optional), e.g. Leopard"
              value={newCommonName}
              onChange={(e) => setNewCommonName(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleAddSpecies()}
              className="w-full border border-slate-200 dark:border-slate-800 rounded-xl px-3 py-2 text-xs focus:outline-none focus:ring-2 focus:ring-emerald-500/20 focus:border-emerald-500 bg-slate-50/50 dark:bg-slate-950 text-slate-800 dark:text-slate-100 transition"
            />
            <button
              onClick={handleAddSpecies}
              disabled={addBusy || !newSciName.trim()}
              className="w-full px-4 py-2 bg-emerald-600 hover:bg-emerald-700 disabled:opacity-50 disabled:cursor-not-allowed text-white text-xs font-semibold rounded-xl shadow-sm transition cursor-pointer"
            >
              {addBusy ? "Adding…" : "Add Species"}
            </button>
          </div>

          {addError && (
            <p className="text-red-500 dark:text-red-400 text-xs font-semibold">
              <span className="material-symbols-outlined text-sm align-middle mr-1 select-none">warning</span> {addError}
            </p>
          )}
          {addSuccess && (
            <p className="text-emerald-600 dark:text-emerald-400 text-xs font-semibold">
              <span className="material-symbols-outlined text-sm align-middle mr-1 select-none">check_circle</span> {addSuccess}
            </p>
          )}
        </div>
      </div>

      {/* Main library section (flex layout for slide-out drawer) */}
      <div className="flex flex-col lg:flex-row gap-6 items-start">
        {/* Cards Grid */}
        <div className="flex-1 bg-white dark:bg-slate-900 rounded-2xl border border-slate-200/60 dark:border-slate-800/80 overflow-hidden shadow-sm w-full">
          <div className="px-5 py-4 border-b border-slate-100 dark:border-slate-800 flex items-center justify-between flex-wrap gap-2">
            <span className="font-bold text-slate-800 dark:text-white text-sm">
              All Registered Species ({species.length})
            </span>
            <input
              placeholder="Search by name, order, family…"
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              className="border border-slate-200 dark:border-slate-800 rounded-xl px-3 py-1.5 text-xs focus:outline-none focus:ring-2 focus:ring-emerald-500/20 focus:border-emerald-500 bg-slate-50/50 dark:bg-slate-950 text-slate-800 dark:text-slate-100 min-w-[240px] transition"
            />
          </div>

          {loading ? (
            <div className="text-center py-20 text-slate-400 animate-pulse">
              Loading library database...
            </div>
          ) : (
            <div className="p-5">
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                {filtered.map((s, i) => {
                  const sciName = String(s.name ?? s.scientific_name ?? "Unknown");
                  const commonName = String(s.common_name ?? "Vernacular name unmapped");
                  const taxonClass = String(s.class ?? s.category ?? "Mammalia");
                  const orderName = String(s.order ?? "");
                  const familyName = String(s.family ?? "");

                  const isSelected = selectedSp && (selectedSp.scientific_name === sciName || selectedSp.name === sciName);

                  return (
                    <div
                      key={i}
                      onClick={() => setSelectedSp(s)}
                      className={`p-4 rounded-xl border transition-all cursor-pointer flex flex-col justify-between h-36 ${
                        isSelected
                          ? "bg-emerald-50/40 dark:bg-emerald-950/20 border-emerald-500 dark:border-emerald-700 ring-1 ring-emerald-500 shadow-sm"
                          : "bg-white dark:bg-slate-950 border-slate-200 dark:border-slate-850 hover:border-slate-350 dark:hover:border-slate-700 hover:shadow-sm"
                      }`}
                    >
                      <div className="flex justify-between items-start">
                        <div className="space-y-1">
                          <p className="font-bold text-slate-800 dark:text-slate-100 text-sm leading-snug line-clamp-1">{commonName}</p>
                          <p className="text-xs text-slate-500 dark:text-slate-400 italic font-semibold line-clamp-1">{sciName}</p>
                        </div>
                        <span className="text-slate-400 dark:text-slate-500 select-none p-1.5 bg-slate-50 dark:bg-slate-900 border border-slate-200/20 dark:border-slate-800/40 rounded-lg" title={taxonClass}>
                          {getTaxonIcon(taxonClass)}
                        </span>
                      </div>

                      <div className="flex flex-wrap gap-1.5 mt-3">
                        {orderName && (
                          <span className="px-2 py-0.5 bg-slate-50 dark:bg-slate-900 border border-slate-200/30 dark:border-slate-800/60 text-slate-500 dark:text-slate-400 rounded-full text-[9px] font-bold uppercase tracking-wider">
                            {orderName}
                          </span>
                        )}
                        {familyName && (
                          <span className="px-2 py-0.5 bg-slate-50 dark:bg-slate-900 border border-slate-200/30 dark:border-slate-800/60 text-slate-500 dark:text-slate-400 rounded-full text-[9px] font-bold uppercase tracking-wider">
                            {familyName}
                          </span>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
              {filtered.length === 0 && (
                <div className="text-center py-20 text-slate-400">
                  No species match your query filters.
                </div>
              )}
            </div>
          )}
        </div>

        {/* Detailed slide-out profile drawer */}
        {selectedSp && (() => {
          const sciName = String(selectedSp.name ?? selectedSp.scientific_name ?? "Unknown");
          const commonName = String(selectedSp.common_name ?? "Vernacular name unmapped");
          
          return (
            <div className="w-full lg:w-80 xl:w-96 bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 p-5 space-y-5 shadow-sm animate-fade-in shrink-0">
              <div className="flex items-start justify-between border-b border-slate-100 dark:border-slate-800 pb-3">
                <div className="space-y-1">
                  <h3 className="font-bold text-slate-900 dark:text-white text-base leading-snug">{commonName}</h3>
                  <p className="text-xs text-emerald-600 dark:text-emerald-400 italic font-semibold">{sciName}</p>
                </div>
                <button
                  onClick={() => setSelectedSp(null)}
                  className="text-slate-400 hover:text-slate-600 dark:hover:text-slate-250 cursor-pointer font-semibold text-xs flex items-center gap-0.5"
                >
                  <span className="material-symbols-outlined text-sm leading-none select-none">close</span> Close
                </button>
              </div>

              {/* Taxonomy lineage */}
              <div className="space-y-2.5">
                <p className="text-[10px] uppercase font-bold text-slate-400 dark:text-slate-500 tracking-wider">Taxonomic Hierarchy</p>
                <div className="space-y-1.5 border-l-2 border-slate-100 dark:border-slate-800 pl-3.5 ml-1.5 text-xs font-semibold">
                  {[
                    { rank: "Kingdom", val: selectedSp.kingdom ?? "Animalia" },
                    { rank: "Phylum", val: selectedSp.phylum ?? "Chordata" },
                    { rank: "Class", val: selectedSp.class ?? selectedSp.category ?? "Mammalia" },
                    { rank: "Order", val: selectedSp.order },
                    { rank: "Family", val: selectedSp.family },
                    { rank: "Genus", val: selectedSp.genus },
                    { rank: "Species", val: selectedSp.species ?? sciName.split(" ").slice(-1)[0] },
                  ].map((tax, tIdx) => {
                    if (!tax.val) return null;
                    return (
                      <div key={tIdx} className="flex justify-between items-center py-0.5">
                        <span className="text-slate-400 dark:text-slate-500">{tax.rank}</span>
                        <span className="text-slate-800 dark:text-slate-300 font-bold">{String(tax.val)}</span>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Ecological Status card */}
              <div className="space-y-2.5 border-t border-slate-100 dark:border-slate-800 pt-4">
                <p className="text-[10px] uppercase font-bold text-slate-400 dark:text-slate-500 tracking-wider">Conservation & Ecology</p>
                <div className="bg-slate-50 dark:bg-slate-950 border border-slate-100 dark:border-slate-900 p-3.5 rounded-xl text-xs space-y-2 font-semibold">
                  <div className="flex justify-between">
                    <span className="text-slate-400">Regional Presence</span>
                    <span className="text-slate-700 dark:text-slate-300 font-bold">Ethiopia (Gambella)</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">IUCN Red List Status</span>
                    <span className="text-amber-600 dark:text-amber-450 font-bold bg-amber-50 dark:bg-amber-950/40 px-2 py-0.5 rounded border border-amber-100 dark:border-amber-900/30">
                      Least Concern (LC)
                    </span>
                  </div>
                </div>
              </div>
            </div>
          );
        })()}
      </div>
    </div>
  );
}
