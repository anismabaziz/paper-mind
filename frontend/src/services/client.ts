import axios from "axios";

export function resolveApiBaseUrl(raw?: string): string {
  const value = (raw ?? import.meta.env.VITE_API_URL ?? "").trim().replace(/\/+$/, "");
  if (!value) {
    throw new Error("VITE_API_URL is not set. Copy frontend/.env.example to .env.local and set it.");
  }
  try {
    const parsed = new URL(value);
    if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
      throw new Error(`Unsupported protocol: ${parsed.protocol}`);
    }
  } catch (e) {
    if (e instanceof Error && e.message.startsWith("Unsupported protocol")) throw e;
    throw new Error(`VITE_API_URL is invalid: "${value}". Expected an http(s) URL.`);
  }
  return value;
}

export const apiBaseUrl = resolveApiBaseUrl();

const client = axios.create({
  baseURL: apiBaseUrl,
});

export default client;
