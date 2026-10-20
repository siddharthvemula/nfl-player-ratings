import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In development, /api/* goes to the local FastAPI server (uvicorn api.index:app on port 8000).
// On Vercel, the same paths are served by the Python function, so the frontend code never changes.
export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/api": "http://127.0.0.1:8000" } },
});
