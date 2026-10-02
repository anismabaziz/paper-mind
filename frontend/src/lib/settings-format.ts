// How a settings number or failure reaches the reader. The server's own
// wording is preferred where it sent one: an error it explained itself is more
// useful than anything this app could invent about it.

import { ApiError } from "./api-error";

/**
 * The wording to show for a failed settings request.
 *
 * A failure that came from the backend arrives as an `ApiError` already holding
 * what the server said, so this mostly just hands that back. Anything else — a
 * request that never reached the server, a bug in this app — falls back to
 * wording that at least says which of the two happened.
 */
export function errorMessage(e: unknown, fallback: string): string {
  if (e instanceof ApiError) return e.message;
  if (e instanceof Error && e.message) return e.message;
  return fallback;
}

export function formatTokenCount(value: number): string {
  return new Intl.NumberFormat("en-US").format(value);
}

export function formatUsd(value: number): string {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 2,
    maximumFractionDigits: 3,
  }).format(value);
}