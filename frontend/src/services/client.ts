import axios from "axios";

import { toApiError } from "@/lib/api-error";

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

/**
 * Turn every failed response into the failure the backend actually sent.
 *
 * Axios reports a failed request as "Request failed with status code NNN" and
 * leaves the server's explanation in the response body, where nothing reads it.
 * That loses the one thing a reader can act on — "This document is being
 * deleted" versus "Only PDF files are allowed" — and replaces it with a status
 * code. Rewriting the rejection here rather than at each call site is what makes
 * every caller show the real reason without each having to remember to.
 */
client.interceptors.response.use(
  (response) => response,
  (error: unknown) => Promise.reject(toApiError(error)),
);

export default client;
