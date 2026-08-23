import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// /api is proxied to the Python backend so the browser sees one origin and CORS
// never enters the picture. Point it at the stub today, at serve.py later —
// nothing in src/ changes when you switch.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: process.env.VITE_API_TARGET ?? "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
});
