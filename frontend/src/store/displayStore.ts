import { create } from "zustand";

const STORAGE_KEY = "cameraTrapsHideNonWildlife";

function readHideNonWildlife(): boolean {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw === null) return true;
    return raw !== "false";
  } catch {
    return true;
  }
}

interface DisplayStore {
  /** When true, blank / person / vehicle rows are hidden in lists and charts. */
  hideNonWildlife: boolean;
  setHideNonWildlife: (hide: boolean) => void;
  showNonWildlife: () => void;
}

export const useDisplayStore = create<DisplayStore>((set) => ({
  hideNonWildlife: readHideNonWildlife(),

  setHideNonWildlife: (hide) => {
    try {
      localStorage.setItem(STORAGE_KEY, String(hide));
    } catch {
      /* ignore */
    }
    set({ hideNonWildlife: hide });
  },

  showNonWildlife: () => {
    try {
      localStorage.setItem(STORAGE_KEY, "false");
    } catch {
      /* ignore */
    }
    set({ hideNonWildlife: false });
  },
}));
