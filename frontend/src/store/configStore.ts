import { create } from "zustand";
import { getConfig, updateConfig } from "../api/client";

export interface AppConfig {
  enable_ocr: boolean;
  enable_detection: boolean;
  enable_day_night: boolean;
  enable_scrubbing: boolean;
  enable_low_spec: boolean;
  cpu_threads: number;
  detection_confidence: number;
  brightness_threshold: number;
  ocr_strip_height: number;
  blur_strength: number;
  review_confidence_threshold: number;
  reviewer_id: string;
  default_station_id: string;
  independence_window: number;
  trap_nights_default: number;
  speciesnet_lat: number;
  speciesnet_lng: number;
  speciesnet_country: string;
  speciesnet_bypass_threshold: number;
}

interface ConfigStore {
  config: AppConfig | null;
  loading: boolean;
  fetch: () => Promise<void>;
  patch: (patch: Partial<AppConfig>) => Promise<void>;
}

export const useConfigStore = create<ConfigStore>((set) => ({
  config: null,
  loading: false,

  fetch: async () => {
    set({ loading: true });
    try {
      const data = await getConfig();
      set({ config: data });
    } finally {
      set({ loading: false });
    }
  },

  patch: async (patch) => {
    const updated = await updateConfig(patch as Record<string, unknown>);
    set({ config: updated });
  },
}));
