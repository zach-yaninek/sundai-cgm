/// <reference types="vite/client" />

interface ImportMetaEnv {
  /**
   * Absolute origin of the Python API, e.g. https://sundai-cgm-api.onrender.com
   *
   * Unset in development — Vite proxies /api to the local backend instead.
   * Set on Vercel so the static build knows where the API actually lives.
   */
  readonly VITE_API_BASE?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
