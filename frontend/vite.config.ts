/// <reference types="vitest/config" />
import { defineConfig, loadEnv, type Plugin } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";
import tailwindcss from "@tailwindcss/vite";

// Fail the production build when the backend URL is missing or malformed,
// so a misconfigured deploy never ships a client that throws on first load
// (see resolveApiBaseUrl in src/services/client.ts for the runtime twin).
function requireApiUrl(): Plugin {
  return {
    name: "papermind-require-api-url",
    apply: "build",
    config(_, { mode }) {
      const env = loadEnv(mode, process.cwd(), "");
      // loadEnv already merges process.env, but read it explicitly so the
      // gate never depends on that merge behavior.
      const raw = ((process.env.VITE_API_URL ?? env.VITE_API_URL) ?? "")
        .trim()
        .replace(/\/+$/, "");
      if (!raw) {
        throw new Error(
          "VITE_API_URL is not set. Copy frontend/.env.example to .env.local and set it."
        );
      }
      let parsed: URL;
      try {
        parsed = new URL(raw);
      } catch {
        throw new Error(`VITE_API_URL is invalid: "${raw}". Expected an http(s) URL.`);
      }
      if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
        throw new Error(`VITE_API_URL uses unsupported protocol: ${parsed.protocol}`);
      }
    },
  };
}

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss(), requireApiUrl()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  test: {
    environment: "jsdom",
    env: {
      VITE_API_URL: "http://127.0.0.1:3000",
    },
  },
});
