import { create } from "zustand";
import { getSession, setSessionUsername, clearSession } from "../api/client";

interface SessionStore {
  username: string | null;
  loading: boolean;
  fetch: () => Promise<void>;
  login: (username: string) => Promise<void>;
  logout: () => Promise<void>;
}

export const useSessionStore = create<SessionStore>((set) => ({
  username: null,
  loading: true,

  fetch: async () => {
    set({ loading: true });
    try {
      const data = await getSession();
      set({ username: data.username ?? null });
    } catch {
      set({ username: null });
    } finally {
      set({ loading: false });
    }
  },

  login: async (username: string) => {
    const data = await setSessionUsername(username);
    set({ username: data.username ?? username });
  },

  logout: async () => {
    await clearSession();
    set({ username: null });
  },
}));
